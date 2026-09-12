// SPDX-License-Identifier: AGPL-3.0-only
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>
namespace py = pybind11;
template <typename T> using Array = py::array_t<T, py::array::c_style>;

struct Polygons {
    std::vector<cv::Point> points;
    std::vector<int64_t> offsets;
    Polygons(Array<int32_t> xy, Array<int64_t> off) {
        if (xy.ndim() != 2 || xy.shape(1) != 2 || off.ndim() != 1 || off.size() < 1)
            throw py::value_error("expected points[P,2] and offsets[N+1]");
        offsets.assign(off.data(), off.data() + off.size());
        if (offsets.front() != 0 || offsets.back() != xy.shape(0))
            throw py::value_error("offsets must span points");
        for (size_t i=1; i<offsets.size(); ++i)
            if (offsets[i] < offsets[i-1] || offsets[i]-offsets[i-1] > INT_MAX)
                throw py::value_error("invalid contour length or offset order");
        points.reserve(xy.shape(0));
        for (py::ssize_t i=0; i<xy.shape(0); ++i) points.emplace_back(xy.data()[2*i], xy.data()[2*i+1]);
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
    static cv::Rect bounds(const cv::Point *points,size_t count,int h,int w) {
        if(!count) return {};
        int64_t x0=points[0].x,x1=x0,y0=points[0].y,y1=y0;
        for(size_t i=1;i<count;++i) {
            x0=std::min(x0,int64_t(points[i].x)); x1=std::max(x1,int64_t(points[i].x));
            y0=std::min(y0,int64_t(points[i].y)); y1=std::max(y1,int64_t(points[i].y));
        }
        // LINE_8 integer fill cannot write outside inclusive vertex bounds.
        // Widen before +1 so INT_MAX coordinates cannot overflow this check.
        x0=std::clamp(x0,int64_t(0),int64_t(w)); x1=std::clamp(x1+1,int64_t(0),int64_t(w));
        y0=std::clamp(y0,int64_t(0),int64_t(h)); y1=std::clamp(y1+1,int64_t(0),int64_t(h));
        if(x1>x0 && y1>y0) return cv::Rect(int(x0),int(y0),int(x1-x0),int(y1-y0));
        return {};
    }
    void render(const Polygons &p, size_t i, int h, int w, cv::Mat &dst, int color) {
        prepare(h,w);
        int count = int(p.offsets[i+1]-p.offsets[i]);
        if (count) {
            const cv::Point *ptr = p.points.data()+p.offsets[i];
            dirty=bounds(ptr,count,h,w);
            cv::fillPoly(scratch, &ptr, &count, 1, cv::Scalar(color));
        }
        cv::resize(scratch, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
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
                render(p,i,h,w,dst,color);
                sums[i]=uint64_t(cv::sum(dst)[0]);
            }
        }
        return py::make_tuple(masks,areas);
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
                dirty=bounds(p.points.data(),p.points.size(),h,w);
                cv::fillPoly(scratch,ptrs.data(),counts.data(),int(ptrs.size()),cv::Scalar(color));
            }
            cv::Mat dst(h/r,w/r,CV_8UC1,out); cv::resize(scratch,dst,dst.size());
        }
        return result;
    }
    template<typename T> py::array compose_t(const Polygons &p,Array<int64_t> order,int h,int w,int r,
                                            Array<uint8_t> masks,bool retained) {
        const auto a=dimensions(h,w,r), n=p.size();
        if(order.ndim()!=1 || size_t(order.size())!=n || n>INT_MAX)
            throw py::value_error("invalid order length");
        std::vector<int64_t> indices(order.data(),order.data()+n);
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
                const size_t count=size_t(p.offsets[source+1]-p.offsets[source]);
                const auto box=bounds(count?p.points.data()+p.offsets[source]:nullptr,count,h,w);
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
    m.def("opencv_threads",[](){return cv::getNumThreads();});
    m.def("opencv_build_info",[](){return cv::getBuildInformation();});
    py::class_<Polygons>(m,"Polygons").def(py::init<Array<int32_t>,Array<int64_t>>()).def("__len__",&Polygons::size);
    py::class_<Rasterizer>(m,"Rasterizer").def(py::init<size_t>()).def("raster",&Rasterizer::raster)
        .def("single",&Rasterizer::single).def("compose",&Rasterizer::compose);
}
