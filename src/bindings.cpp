// SPDX-License-Identifier: AGPL-3.0-only
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "maskops_build_profile.h"
#include "geometry.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cstddef>
#include <limits>
#include <mutex>
#include <vector>
namespace py = pybind11;
template <typename T> using Array = py::array_t<T, py::array::c_style>;

// Inclusive original-coordinate extents. Delay widening/clipping until a raster
// size is known, so immutable packed geometry can be reused at different sizes.
struct Extents {
    int32_t x0=INT_MAX,y0=INT_MAX,x1=INT_MIN,y1=INT_MIN;
    bool empty() const { return x1<x0; }
    void add(int32_t x,int32_t y) {
        x0=std::min(x0,x); x1=std::max(x1,x);
        y0=std::min(y0,y); y1=std::max(y1,y);
    }
    void merge(const Extents &other) {
        if(!other.empty()) { add(other.x0,other.y0); add(other.x1,other.y1); }
    }
    cv::Rect clipped(int h,int w) const {
        if(empty()) return {};
        const auto left=std::clamp(int64_t(x0),int64_t(0),int64_t(w));
        const auto right=std::clamp(int64_t(x1)+1,int64_t(0),int64_t(w));
        const auto top=std::clamp(int64_t(y0),int64_t(0),int64_t(h));
        const auto bottom=std::clamp(int64_t(y1)+1,int64_t(0),int64_t(h));
        return right>left && bottom>top ? cv::Rect(int(left),int(top),int(right-left),int(bottom-top)) : cv::Rect();
    }
};

struct Polygons {
    std::vector<cv::Point> points;
    std::vector<int64_t> offsets;
    std::vector<Extents> extents;
    Extents combined;
    Polygons(Array<int32_t> xy, Array<int64_t> off) {
        if (xy.ndim() != 2 || xy.shape(1) != 2 || off.ndim() != 1 || off.size() < 1)
            throw py::value_error("expected points[P,2] and offsets[N+1]");
        // A C-contiguous NumPy array need not be naturally aligned. Copy bytes
        // into aligned owned vectors before any typed reads, including offsets.
        std::vector<int64_t> source_offsets(size_t(off.size()));
        std::memcpy(source_offsets.data(),static_cast<const py::array&>(off).data(),size_t(off.nbytes()));
        if (source_offsets.front() != 0 || source_offsets.back() != xy.shape(0))
            throw py::value_error("offsets must span points");
        for (size_t i=1; i<source_offsets.size(); ++i)
            if (source_offsets[i] < source_offsets[i-1] || source_offsets[i]-source_offsets[i-1] > INT_MAX)
                throw py::value_error("invalid contour length or offset order");
        static_assert(std::is_trivially_copyable_v<cv::Point> && std::is_standard_layout_v<cv::Point>);
        static_assert(sizeof(cv::Point)==2*sizeof(int32_t) && offsetof(cv::Point,x)==0 && offsetof(cv::Point,y)==sizeof(int32_t));
        points.reserve(size_t(xy.shape(0)));
        offsets.resize(source_offsets.size());
        offsets[0]=0;
        extents.resize(size());
        const auto *bytes=static_cast<const unsigned char*>(static_cast<const py::array&>(xy).data());
        for(size_t i=0;i<size();++i) {
            auto &box=extents[i];
            const size_t begin=points.size();
            for(auto j=source_offsets[i];j<source_offsets[i+1];++j) {
                cv::Point point;
                std::memcpy(&point,bytes+size_t(j)*sizeof(point),sizeof(point));
                // Only remove consecutive identical integer vertices. They add
                // zero-length LINE_8 edges whose pixel is already covered by
                // an incident edge. An all-identical contour keeps one point.
                // Nonzero edges (including collinear edges) stay unchanged.
                if(points.size()==begin || point!=points.back()) {
                    points.push_back(point);
                    box.add(point.x,point.y);
                }
            }
            // Keep a terminal copy of the first vertex. Removing it would
            // rotate the nonzero edge insertion order in CollectPolyEdges.
            offsets[i+1]=int64_t(points.size());
            combined.merge(box);
        }
    }
    size_t size() const { return offsets.size()-1; }
};

struct Rasterizer {
    cv::Mat scratch;
    cv::Rect dirty;
    std::mutex mutex;
    size_t budget;
    explicit Rasterizer(size_t limit): budget(limit) {}
    size_t dimensions(int h, int w, int r) const {
        if (h <= 0 || w <= 0 || r <= 0 || h/r == 0 || w/r == 0)
            throw py::value_error("positive dimensions and nonzero downsampled size required");
        const size_t full = size_t(h)*size_t(w), reduced = size_t(h/r)*size_t(w/r);
        if (full > size_t(PY_SSIZE_T_MAX) || reduced > size_t(PY_SSIZE_T_MAX))
            throw py::value_error("dimensions overflow");
        if (full > budget || reduced > budget-full)
            throw py::value_error("scratch_limit_bytes cannot hold one full and one resized mask");
        return reduced;
    }
    void prepare(int h,int w) {
        if(scratch.rows!=h || scratch.cols!=w) {
            scratch.create(h,w,CV_8UC1);
            scratch.setTo(0);
        } else if(!dirty.empty()) {
            scratch(dirty).setTo(0);
        }
        dirty=cv::Rect();
    }
    cv::Rect resize_support(cv::Mat &dst, bool area_support=true) {
        const cv::Rect full(0,0,dst.cols,dst.rows);
        const int h=scratch.rows, w=scratch.cols;
        // Mask-only callers do not consume a reduced area-scan rectangle.
        // At unit scale OpenCV already takes its whole-image copy path; avoid
        // clearing the destination and then copying a crop over part of it.
        if(!area_support && dst.rows==h && dst.cols==w) {
            cv::resize(scratch,dst,dst.size(),0,0,cv::INTER_LINEAR);
            return full;
        }
        // Preserve the full-image sampling lattice. Power-of-two integer
        // scales have exact reciprocal coefficients; the dimension bound keeps
        // half-integer source coordinates representable in resize.cpp's float.
        const int scale=w/dst.cols;
        const bool exact=scale>0 && (scale&(scale-1))==0 &&
            w%dst.cols==0 && h%dst.rows==0 && h/dst.rows==scale &&
            h<=(1<<23) && w<=(1<<23);
        if(!exact) {
            cv::resize(scratch,dst,dst.size(),0,0,cv::INTER_LINEAR);
            return full;
        }
        if(dirty.empty()) {
            dst.setTo(0);
            return {};
        }
        int left=dirty.x/scale, top=dirty.y/scale;
        const int right=(dirty.x+dirty.width+scale-1)/scale;
        const int bottom=(dirty.y+dirty.height+scale-1)/scale;
        int width=right-left, height=bottom-top;
        // Carotene's single-channel linear HAL requires both output dimensions
        // >=8. Preserve that eligibility: switching a small crop to the generic
        // path changes integer rounding at scales >=4, even with exact sampling
        // coordinates. Extend into known-zero scratch, retaining aligned origins.
        if(dst.cols>=8 && dst.rows>=8) {
            width=std::max(width,8); height=std::max(height,8);
            left=std::min(left,dst.cols-width); top=std::min(top,dst.rows-height);
        }
        const cv::Rect reduced(left,top,width,height);
        // Zeroing a full output plus rewriting a large ROI can lose to a single
        // full resize. This initial cost guard is fixed before measurement.
        if(size_t(reduced.width)*reduced.height*2>=size_t(dst.cols)*dst.rows) {
            cv::resize(scratch,dst,dst.size(),0,0,cv::INTER_LINEAR);
            return full;
        }
        const cv::Rect source(left*scale,top*scale,reduced.width*scale,reduced.height*scale);
        dst.setTo(0);
        cv::Mat target=dst(reduced);
        cv::resize(scratch(source),target,target.size(),0,0,cv::INTER_LINEAR);
        return reduced;
    }
    cv::Rect render(const Polygons &p, size_t i, int h, int w, cv::Mat &dst, int color, bool area_support=true) {
        prepare(h,w);
        int count = int(p.offsets[i+1]-p.offsets[i]);
        if (count) {
            const cv::Point *ptr = p.points.data()+p.offsets[i];
            dirty=p.extents[i].clipped(h,w);
            cv::fillPoly(scratch, &ptr, &count, 1, cv::Scalar(color));
        }
        return resize_support(dst,area_support);
    }
    py::tuple raster(const Polygons &p, int h, int w, int r, int color, bool keep) {
        const auto a=dimensions(h,w,r), n=p.size();
        if (color<0 || color>255) throw py::value_error("color must be in [0,255]");
        if (n && a > size_t(PY_SSIZE_T_MAX)/n) throw py::value_error("output size overflow");
        Array<uint8_t> masks({py::ssize_t(keep?n:0),py::ssize_t(h/r),py::ssize_t(w/r)});
        Array<uint64_t> areas{py::ssize_t(n)};
        auto *out=masks.mutable_data(); auto *sums=areas.mutable_data();
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(mutex);
            cv::Mat small;
            if (!keep) small.create(h/r,w/r,CV_8UC1);
            for (size_t i=0;i<n;++i) {
                cv::Mat dst=keep?cv::Mat(h/r,w/r,CV_8UC1,out+i*a):small;
                const auto support=render(p,i,h,w,dst,color);
                sums[i]=support.empty()?uint64_t(0):uint64_t(cv::sum(dst(support))[0]);
            }
        }
        return py::make_tuple(masks,areas);
    }
    Array<uint8_t> masks(const Polygons &p, int h, int w, int r, int color) {
        const auto a=dimensions(h,w,r), n=p.size();
        if (color<0 || color>255) throw py::value_error("color must be in [0,255]");
        if (n && a > size_t(PY_SSIZE_T_MAX)/n) throw py::value_error("output size overflow");
        Array<uint8_t> result({py::ssize_t(n),py::ssize_t(h/r),py::ssize_t(w/r)});
        auto *out=result.mutable_data();
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(mutex);
            for (size_t i=0;i<n;++i) {
                cv::Mat dst(h/r,w/r,CV_8UC1,out+i*a);
                render(p,i,h,w,dst,color,false);
            }
        }
        return result;
    }
    Array<uint8_t> single(const Polygons &p,int h,int w,int r,int color) {
        dimensions(h,w,r);
        if (color<0 || color>255) throw py::value_error("color must be in [0,255]");
        Array<uint8_t> result({h/r,w/r});
        auto *out=result.mutable_data();
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(mutex);
            prepare(h,w);
            std::vector<const cv::Point*> ptrs; std::vector<int> counts;
            for(size_t i=0;i<p.size();++i) {
                int count=int(p.offsets[i+1]-p.offsets[i]);
                if(count) {ptrs.push_back(p.points.data()+p.offsets[i]); counts.push_back(count);}
            }
            if(!ptrs.empty()) {
                dirty=p.combined.clipped(h,w);
                cv::fillPoly(scratch,ptrs.data(),counts.data(),int(ptrs.size()),cv::Scalar(color));
            }
            cv::Mat dst(h/r,w/r,CV_8UC1,out); resize_support(dst,false);
        }
        return result;
    }
    template<typename T> py::array compose_t(const Polygons &p,Array<int64_t> order,int h,int w,int r,
                                            Array<uint8_t> masks,bool retained) {
        const auto a=dimensions(h,w,r), n=p.size();
        if(order.ndim()!=1 || size_t(order.size())!=n || n>INT_MAX)
            throw py::value_error("invalid order length");
        std::vector<int64_t> indices(n);
        if(n) std::memcpy(indices.data(),static_cast<const py::array&>(order).data(),size_t(order.nbytes()));
        std::vector<bool> seen(n,false);
        for(auto i:indices) {
            if(i<0 || size_t(i)>=n || seen[i]) throw py::value_error("order must be a permutation");
            seen[i]=true;
        }
        if(retained && (masks.ndim()!=3 || size_t(masks.shape(0))!=n || masks.shape(1)!=h/r || masks.shape(2)!=w/r))
            throw py::value_error("invalid retained mask shape");
        Array<T> result({h/r,w/r});
        auto *out=result.mutable_data(); const auto *input=masks.data();
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(mutex);
            std::fill(out,out+a,T(0));
            cv::Mat small;
            if(!retained) small.create(h/r,w/r,CV_8UC1);
            for(size_t rank=0;rank<n;++rank) {
                const uint8_t *pixels;
                if(retained) pixels=input+indices[rank]*a;
                else {render(p,indices[rank],h,w,small,1); pixels=small.data;}
                const size_t source=size_t(indices[rank]);
                const auto box=p.extents[source].clipped(h,w);
                // Conservative support of LINEAR resize. Two destination pixels
                // of padding include interpolation neighbors and odd-size rounding.
                // Pixels outside this region are known zero, so leave output intact.
                if(!box.empty()) {
                    const int sw=w/r, sh=h/r;
                    const int left=std::max(0,int(int64_t(box.x)*sw/w)-2);
                    const int top=std::max(0,int(int64_t(box.y)*sh/h)-2);
                    const int right=int(std::min(int64_t(sw),(int64_t(box.x+box.width)*sw+w-1)/w+2));
                    const int bottom=int(std::min(int64_t(sh),(int64_t(box.y+box.height)*sh+h-1)/h+2));
                    for(int y=top;y<bottom;++y) {
                        const size_t row=size_t(y)*sw;
                        for(int x=left;x<right;++x) {
                            const size_t j=row+size_t(x);
                            out[j]=pixels[j]?T(rank+1):out[j];
                        }
                    }
                }
            }
        }
        return result;
    }
    py::array compose(const Polygons &p,Array<int64_t> order,int h,int w,int r,Array<uint8_t> masks,bool retained) {
        return p.size()>255?compose_t<int32_t>(p,order,h,w,r,masks,retained):compose_t<uint8_t>(p,order,h,w,r,masks,retained);
    }
};

PYBIND11_MODULE(_native,m) {
    // This statically linked, symbol-private OpenCV copy has an explicit one
    // thread policy. It must not alter the separately imported Python cv2 state.
    cv::setNumThreads(0); // disable internal parallel regions (one calling worker)
    m.attr("opencv_version")=CV_VERSION;
    m.def("build_profile",[](){
        py::dict info;
        info["bindings_sha256"]=MASKOPS_BINDINGS_SHA256;
        info["geometry_sha256"]=MASKOPS_GEOMETRY_SHA256;
        info["cmake_sha256"]=MASKOPS_CMAKE_SHA256;
        info["template_sha256"]=MASKOPS_PROFILE_SHA256;
        info["compiler"]=MASKOPS_COMPILER;
        info["build_type"]=MASKOPS_BUILD_TYPE;
        return info;
    });
    m.def("opencv_threads",[](){return cv::getNumThreads();});
    m.def("opencv_build_info",[](){return cv::getBuildInformation();});
    maskops_geometry::register_geometry(m);
    py::class_<Polygons>(m,"Polygons").def(py::init<Array<int32_t>,Array<int64_t>>()).def("__len__",&Polygons::size);
    py::class_<Rasterizer>(m,"Rasterizer").def(py::init<size_t>()).def("raster",&Rasterizer::raster)
        .def("masks",&Rasterizer::masks).def("single",&Rasterizer::single).def("compose",&Rasterizer::compose);
}
