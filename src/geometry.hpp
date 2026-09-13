// SPDX-License-Identifier: AGPL-3.0-only
// Exact batched replicas of the pinned Ultralytics segment geometry.
//
// The reference runs segment2box once per instance and resample_segments once
// per polygon; each iteration dispatches a dozen small NumPy calls, so the
// interpreter, not arithmetic, dominates. These kernels process a whole sample
// per call. Arithmetic follows the reference operation by operation in the
// input precision. The target is compiled without floating-point contraction,
// so each multiply and add rounds separately, as separate NumPy ufuncs do.
// np.interp's own multiply-add may be fused by the NumPy build; that choice is
// a parameter the Python layer calibrates against the installed NumPy.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace maskops_geometry {
namespace py = pybind11;

template <typename T> using Input = py::array_t<T, py::array::c_style>;

template <typename T> const T *aligned_data(const py::array &a, const char *name) {
    if (reinterpret_cast<uintptr_t>(a.data()) % alignof(T))
        throw py::value_error(std::string(name) + " must be naturally aligned");
    return static_cast<const T *>(a.data());
}

// ---- resample_segments ----------------------------------------------------

// numpy.linspace(0.0, stop, num), float64, endpoint=True (function_base.py).
inline void linspace(double stop, int64_t num, std::vector<double> &y) {
    y.resize(size_t(num));
    const int64_t div = num - 1;
    const double delta = stop - 0.0;
    if (div > 0) {
        const double step = delta / double(div);
        if (step == 0.0) {
            for (int64_t i = 0; i < num; ++i) y[size_t(i)] = (double(i) / double(div)) * delta;
        } else {
            for (int64_t i = 0; i < num; ++i) y[size_t(i)] = double(i) * step;
        }
    } else {
        for (int64_t i = 0; i < num; ++i) y[size_t(i)] = double(i) * delta;
    }
    for (auto &v : y) v = v + 0.0;  // y += start
    if (num > 1) y[size_t(num - 1)] = stop;
}

// numpy.interp(x, arange(L), fp) (compiled_base.c) for one query.
template <bool Fused> inline double interp(double x, const std::vector<double> &fp) {
    const int64_t n = int64_t(fp.size());
    if (x > double(n - 1)) return fp[size_t(n - 1)];
    if (x < 0.0) return fp[0];
    const int64_t j = std::min<int64_t>(int64_t(std::floor(x)), n - 1);  // binary_search_with_guess
    if (j == n - 1) return fp[size_t(j)];
    const double x0 = double(j), x1 = double(j + 1);
    if (x0 == x) return fp[size_t(j)];
    const double y0 = fp[size_t(j)], y1 = fp[size_t(j + 1)];
    const double slope = (y1 - y0) / (x1 - x0);
    double r = Fused ? std::fma(slope, x - x0, y0) : slope * (x - x0) + y0;
    if (std::isnan(r)) {
        r = Fused ? std::fma(slope, x - x1, y1) : slope * (x - x1) + y1;
        if (std::isnan(r) && y0 == y1) r = y0;
    }
    return r;
}

struct ResampleScratch {
    std::vector<double> xs, lin, fx, fy;
};

// resample_segments for one (m, 2) float32 polygon into n float32 points.
template <bool Fused> void resample_one(const float *p, int64_t m, int64_t n, float *out, ResampleScratch &s) {
    if (m == n) {
        std::copy(p, p + 2 * n, out);  // the reference leaves it untouched
        return;
    }
    const int64_t count = m + 1;  // closed polygon
    s.fx.resize(size_t(count));
    s.fy.resize(size_t(count));
    for (int64_t k = 0; k < m; ++k) {
        s.fx[size_t(k)] = double(p[2 * k]);
        s.fy[size_t(k)] = double(p[2 * k + 1]);
    }
    s.fx[size_t(m)] = s.fx[0];
    s.fy[size_t(m)] = s.fy[0];
    const double stop = double(count - 1);
    if (count < n) {
        // np.insert(x, np.searchsorted(x, xp), xp): each integer precedes the
        // first linspace value >= it; equal values place the integer first.
        linspace(stop, n - count, s.lin);
        s.xs.clear();
        size_t i = 0;
        for (int64_t k = 0; k < count; ++k) {
            const double xp = double(k);
            while (i < s.lin.size() && s.lin[i] < xp) s.xs.push_back(s.lin[i++]);
            s.xs.push_back(xp);
        }
        while (i < s.lin.size()) s.xs.push_back(s.lin[i++]);
    } else {
        linspace(stop, n, s.xs);
    }
    for (int64_t t = 0; t < n; ++t) {
        out[2 * t] = static_cast<float>(interp<Fused>(s.xs[size_t(t)], s.fx));
        out[2 * t + 1] = static_cast<float>(interp<Fused>(s.xs[size_t(t)], s.fy));
    }
}

// np.stack(resample_segments(segments, n), 0) for packed float32 polygons.
inline py::array_t<float> resample_stack(Input<float> points, Input<int64_t> offsets, int64_t n, bool fused) {
    if (points.ndim() != 2 || points.shape(1) != 2 || offsets.ndim() != 1 || offsets.size() < 1)
        throw py::value_error("expected points[P,2] and offsets[N+1]");
    if (n < 1) throw py::value_error("n must be positive");
    const auto *xy = aligned_data<float>(points, "points");
    const auto *off = aligned_data<int64_t>(offsets, "offsets");
    const int64_t count = offsets.size() - 1;
    if (off[0] != 0 || off[count] != points.shape(0)) throw py::value_error("offsets must span points");
    for (int64_t i = 0; i < count; ++i)
        if (off[i + 1] <= off[i]) throw py::value_error("empty or unordered polygon");
    py::array_t<float> result({py::ssize_t(count), py::ssize_t(n), py::ssize_t(2)});
    float *out = result.mutable_data();
    {
        py::gil_scoped_release release;
        ResampleScratch scratch;
        for (int64_t i = 0; i < count; ++i) {
            const float *p = xy + 2 * off[i];
            float *o = out + 2 * n * i;
            if (fused)
                resample_one<true>(p, off[i + 1] - off[i], n, o, scratch);
            else
                resample_one<false>(p, off[i + 1] - off[i], n, o, scratch);
        }
    }
    return result;
}

// ---- segment2box and apply_segments' clip ---------------------------------

// cv::pointPolygonTest(contour, pt, measureDist=false) >= 0 for a CV_32F
// contour (imgproc/src/geometry.cpp). Each product multiplies two float
// differences, which is exact in double, so contraction cannot change it.
inline bool inside_or_on(const std::vector<float> &c, float px, float py) {
    const size_t total = c.size() / 2;
    float vx = c[2 * (total - 1)], vy = c[2 * (total - 1) + 1];
    int counter = 0;
    for (size_t i = 0; i < total; ++i) {
        const float v0x = vx, v0y = vy;
        vx = c[2 * i];
        vy = c[2 * i + 1];
        if ((v0y <= py && vy <= py) || (v0y > py && vy > py) || (v0x < px && vx < px)) {
            if (py == vy && (px == vx || (py == v0y && ((v0x <= px && px <= vx) || (vx <= px && px <= v0x)))))
                return true;
            continue;
        }
        const float a = py - v0y, b = vx - v0x, c0 = px - v0x, d = vy - v0y;
        double dist = double(a) * double(b) - double(c0) * double(d);
        if (dist == 0) return true;
        if (vy < v0y) dist = -dist;
        counter += dist > 0;
    }
    return counter % 2 != 0;
}

// segment2box for one instance of finite values without negative zeros. With
// such inputs no box coordinate can be a negative zero, so every min, max and
// clip below is independent of evaluation order and of NumPy's loop choice.
template <typename T>
void segment_box(const T *seg, int64_t m, int width, int height, T box[4], std::vector<float> &contour) {
    if (m == 0) {
        box[0] = box[1] = box[2] = box[3] = T(0);
        return;
    }
    T xmin = seg[0], ymin = seg[1], xmax = seg[0], ymax = seg[1];
    for (int64_t i = 1; i < m; ++i) {
        xmin = std::min(xmin, seg[2 * i]);
        xmax = std::max(xmax, seg[2 * i]);
        ymin = std::min(ymin, seg[2 * i + 1]);
        ymax = std::max(ymax, seg[2 * i + 1]);
    }
    const T W = T(width), H = T(height);
    if (xmin >= 0 && ymin >= 0 && xmax <= W && ymax <= H) {
        box[0] = xmin, box[1] = ymin, box[2] = xmax, box[3] = ymax;
        return;
    }
    const int axes[4] = {0, 0, 1, 1};
    const T bounds[4] = {T(0), W, T(0), H};
    const T lims[4] = {H, H, W, W};
    bool any = false;
    T lo_x = 0, lo_y = 0, hi_x = 0, hi_y = 0;
    auto add = [&](T x, T y) {
        if (!any) {
            lo_x = hi_x = x, lo_y = hi_y = y, any = true;
            return;
        }
        lo_x = std::min(lo_x, x), hi_x = std::max(hi_x, x);
        lo_y = std::min(lo_y, y), hi_y = std::max(hi_y, y);
    };
    for (int64_t i = 0; i < m; ++i) {
        const T x = seg[2 * i], y = seg[2 * i + 1];
        if (x >= 0 && y >= 0 && x <= W && y <= H) add(x, y);
    }
    for (int64_t i = 0; i < m; ++i) {
        const int64_t next = i + 1 == m ? 0 : i + 1;  // np.roll(segment, -1, axis=0)
        const T sx = seg[2 * i], sy = seg[2 * i + 1];
        const T dx = seg[2 * next] - sx, dy = seg[2 * next + 1] - sy;
        for (int k = 0; k < 4; ++k) {
            const T t = (bounds[k] - (axes[k] ? sy : sx)) / (axes[k] ? dy : dx);
            const T px = t * dx, py = t * dy;  // t[:, :, None] * delta[:, None, :]
            const T ix = sx + px, iy = sy + py;
            const T other = axes[k] ? ix : iy;  // inter[:, k, 1 - axes[k]]
            if (t >= 0 && t <= 1 && other >= 0 && other <= lims[k]) add(ix, iy);
        }
    }
    contour.resize(size_t(2 * m));
    for (int64_t i = 0; i < 2 * m; ++i) contour[size_t(i)] = static_cast<float>(seg[i]);
    const T corners[4][2] = {{T(0), T(0)}, {W, T(0)}, {T(0), H}, {W, H}};
    for (const auto &corner : corners)
        if (inside_or_on(contour, float(double(corner[0])), float(double(corner[1])))) add(corner[0], corner[1]);
    if (!any) {
        box[0] = box[1] = box[2] = box[3] = T(0);
    } else {
        box[0] = lo_x, box[1] = lo_y, box[2] = hi_x, box[3] = hi_y;
    }
}

// np.stack([segment2box(s, width, height) for s in segments]) and, when clip,
// the reference's in-place clip of x and y to each box. Returns None, without
// touching the input, when any value is NaN, infinite or a negative zero:
// NumPy's own reductions and clip loops then differ in the sign of zero by
// array shape, so only the unchanged reference reproduces them.
template <typename T> py::object segment_boxes(py::array_t<T, py::array::c_style> segments, int width, int height, bool clip) {
    if (segments.ndim() != 3 || segments.shape(2) != 2) throw py::value_error("expected segments[N,M,2]");
    if (!segments.writeable()) throw py::value_error("segments must be writable");
    if (reinterpret_cast<uintptr_t>(segments.data()) % alignof(T)) throw py::value_error("segments must be aligned");
    const int64_t n = segments.shape(0), m = segments.shape(1);
    T *seg = segments.mutable_data();
    bool exact = true;
    {
        py::gil_scoped_release release;
        for (int64_t i = 0; i < 2 * n * m && exact; ++i)
            exact = std::isfinite(seg[i]) && !(seg[i] == T(0) && std::signbit(seg[i]));
    }
    if (!exact) return py::none();
    py::array_t<T> boxes({py::ssize_t(n), py::ssize_t(4)});
    T *out = boxes.mutable_data();
    {
        py::gil_scoped_release release;
        std::vector<float> contour;
        for (int64_t i = 0; i < n; ++i) {
            T *s = seg + 2 * m * i;
            T *b = out + 4 * i;
            segment_box<T>(s, m, width, height, b, contour);
            if (clip) {
                for (int64_t k = 0; k < m; ++k) {
                    s[2 * k] = std::min(std::max(s[2 * k], b[0]), b[2]);
                    s[2 * k + 1] = std::min(std::max(s[2 * k + 1], b[1]), b[3]);
                }
            }
        }
    }
    return std::move(boxes);
}

// numpy.interp(x, arange(len(fp)), fp) for calibration against NumPy.
inline py::array_t<double> interp_values(Input<double> x, Input<double> fp, bool fused) {
    if (x.ndim() != 1 || fp.ndim() != 1 || fp.size() < 2) throw py::value_error("expected 1-D x and fp (len >= 2)");
    std::vector<double> table(aligned_data<double>(fp, "fp"), aligned_data<double>(fp, "fp") + fp.size());
    const double *xs = aligned_data<double>(x, "x");
    py::array_t<double> out(x.size());
    double *o = out.mutable_data();
    for (py::ssize_t i = 0; i < x.size(); ++i) o[i] = fused ? interp<true>(xs[i], table) : interp<false>(xs[i], table);
    return out;
}

inline void register_geometry(py::module_ &m) {
    m.def("resample_stack", &resample_stack, py::arg("points"), py::arg("offsets"), py::arg("n"), py::arg("fused"));
    m.def("interp_values", &interp_values, py::arg("x"), py::arg("fp"), py::arg("fused"));
    m.def("segment_boxes_f32", &segment_boxes<float>, py::arg("segments"), py::arg("width"), py::arg("height"),
          py::arg("clip"));
    m.def("segment_boxes_f64", &segment_boxes<double>, py::arg("segments"), py::arg("width"), py::arg("height"),
          py::arg("clip"));
}

}  // namespace maskops_geometry
