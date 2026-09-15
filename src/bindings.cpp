// SPDX-License-Identifier: AGPL-3.0-only
// Python bindings. The extension depends on nothing but pybind11: the mask
// kernel (sampled.hpp) replicates cv::fillPoly and the 4x linear downscale in
// integer arithmetic, and the segment geometry (geometry.hpp) replicates the
// NumPy and OpenCV calls of the pinned Ultralytics functions.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "maskops_build_profile.h"
#include "geometry.hpp"
#include "rfdetr.hpp"
#include "sampled.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
namespace py = pybind11;
template <typename T> using Array = py::array_t<T, py::array::c_style>;

// An owned, immutable snapshot of int32 contours: one per instance.
struct Polygons {
    std::vector<int32_t> points;  // x, y pairs
    std::vector<int64_t> offsets;  // vertex index of each contour's start; N + 1 entries
    Polygons(Array<int32_t> xy, Array<int64_t> off) {
        if (xy.ndim() != 2 || xy.shape(1) != 2 || off.ndim() != 1 || off.size() < 1)
            throw py::value_error("expected points[P,2] and offsets[N+1]");
        // A C-contiguous NumPy array need not be naturally aligned. Copy bytes
        // into aligned owned vectors before any typed reads, including offsets.
        std::vector<int64_t> source_offsets(size_t(off.size()));
        std::memcpy(source_offsets.data(), static_cast<const py::array &>(off).data(), size_t(off.nbytes()));
        if (source_offsets.front() != 0 || source_offsets.back() != xy.shape(0))
            throw py::value_error("offsets must span points");
        for (size_t i = 1; i < source_offsets.size(); ++i)
            if (source_offsets[i] < source_offsets[i - 1] || source_offsets[i] - source_offsets[i - 1] > INT_MAX)
                throw py::value_error("invalid contour length or offset order");
        points.reserve(2 * size_t(xy.shape(0)));
        offsets.resize(source_offsets.size());
        offsets[0] = 0;
        const auto *bytes = static_cast<const unsigned char *>(static_cast<const py::array &>(xy).data());
        for (size_t i = 0; i < size(); ++i) {
            const size_t begin = points.size();
            for (auto j = source_offsets[i]; j < source_offsets[i + 1]; ++j) {
                int32_t point[2];
                std::memcpy(point, bytes + size_t(j) * sizeof(point), sizeof(point));
                // Only remove consecutive identical vertices. They add
                // zero-length LINE_8 edges whose pixel an incident edge
                // already draws. An all-identical contour keeps one point.
                if (points.size() == begin || point[0] != points[points.size() - 2] || point[1] != points.back())
                    points.insert(points.end(), point, point + 2);
            }
            offsets[i + 1] = int64_t(points.size() / 2);
        }
    }
    size_t size() const { return offsets.size() - 1; }
};

// Per-worker mask engine: one contour source at a time, serialized by its
// mutex. The Python layer calibrates the pattern table against the installed
// cv2.resize and decides when this engine applies (4x ratio, sides divisible
// by 4); anything else runs the reference Python code.
struct Rasterizer {
    std::mutex mutex;
    maskops_sampled::Sampler sampler;
    maskops_sampled::Vec<int32_t> sampled_points;
    std::vector<int64_t> sampled_offsets;
    maskops_sampled::Vec<uint8_t> sampled_arena, sampled_scratch;
    std::vector<size_t> sampled_starts;
    std::vector<maskops_sampled::Box> sampled_boxes;
    bool sampled_ready = false, sampled_kept = false;

    static size_t reduced_size(int h, int w) {
        if (!maskops_sampled::Sampler::eligible(h, w))
            throw py::value_error("sampled rasterization requires sides divisible by 4 and at most 32768");
        const size_t reduced = size_t(h / 4) * size_t(w / 4);
        if (reduced > size_t(PY_SSIZE_T_MAX)) throw py::value_error("dimensions overflow");
        return reduced;
    }
    static std::array<uint8_t, 16> sampled_table(int h, int w, const py::bytes &lut, bool binary) {
        reduced_size(h, w);
        const std::string bytes = lut;
        if (bytes.size() != 16) throw py::value_error("pattern table must hold 16 bytes");
        std::array<uint8_t, 16> table;
        std::memcpy(table.data(), bytes.data(), 16);
        if (binary && std::any_of(table.begin(), table.end(), [](uint8_t v) { return v > 1; }))
            throw py::value_error("overlap pattern table must be binary");
        return table;
    }
    // Source of contours: a Polygons snapshot, or a C-contiguous aligned
    // float32/float64 (N,M,2) array converted like np.asarray(..., np.int32).
    struct SampledSource {
        const Polygons *packed = nullptr;
        py::array array;
        int kind = 0;  // 1 float32, 2 float64
        size_t count = 0;
    };
    static bool sampled_source(const py::object &source, SampledSource &s) {
        if (py::isinstance<Polygons>(source)) {
            s.packed = &source.cast<const Polygons &>();
            s.count = s.packed->size();
            return true;
        }
        if (!py::isinstance<py::array>(source)) return false;
        s.array = py::reinterpret_borrow<py::array>(source);
        const auto &a = s.array;
        if (a.ndim() != 3 || a.shape(2) != 2 || a.shape(1) > INT_MAX || (a.shape(0) > 0 && a.shape(1) == 0))
            return false;
        const auto address = reinterpret_cast<uintptr_t>(a.data());
        if (py::isinstance<Array<float>>(a) && address % alignof(float) == 0)
            s.kind = 1;
        else if (py::isinstance<Array<double>>(a) && address % alignof(double) == 0)
            s.kind = 2;
        else
            return false;
        s.count = size_t(a.shape(0));
        return true;
    }
    // Loads contours into the padded layout described at kPadVertices (engine
    // lock held, GIL released). False when a coordinate is non-finite or
    // outside the exact envelope.
    bool sampled_load(const SampledSource &s) {
        using maskops_sampled::kPadVertices;
        if (s.packed) {
            const auto &points = s.packed->points;
            const auto &offsets = s.packed->offsets;
            const size_t n = offsets.size() - 1;
            sampled_points.resize(2 * (1 + points.size() / 2 + n * size_t(kPadVertices)) + maskops_sampled::kPointSlack);
            sampled_offsets.resize(n + 1);
            bool ok = true;
            int64_t start = 1;
            sampled_offsets[0] = start;
            for (size_t i = 0; i < n; ++i) {
                const int64_t count = offsets[i + 1] - offsets[i];
                int32_t *out = sampled_points.data() + 2 * start;
                for (int64_t j = 0; j < 2 * count; ++j) {
                    const int32_t value = points[size_t(2 * offsets[i] + j)];
                    ok &= std::abs(int64_t(value)) < (int64_t(1) << 24);
                    out[j] = value;
                }
                if (count) maskops_sampled::pad_contour(out, count);
                start += count + kPadVertices;
                sampled_offsets[i + 1] = start;
            }
            return ok;
        }
        const auto n = size_t(s.array.shape(0)), m = size_t(s.array.shape(1));
        sampled_points.resize(2 * (1 + n * (m + size_t(kPadVertices))) + maskops_sampled::kPointSlack);
        sampled_offsets.resize(n + 1);
        return s.kind == 1 ? maskops_sampled::load_contours(static_cast<const float *>(s.array.data()), n, m,
                                                            sampled_points.data(), sampled_offsets.data())
                           : maskops_sampled::load_contours(static_cast<const double *>(s.array.data()), n, m,
                                                            sampled_points.data(), sampled_offsets.data());
    }
    maskops_sampled::Box sampled_render(size_t i) {
        const int64_t count = sampled_offsets[i + 1] - sampled_offsets[i] - maskops_sampled::kPadVertices;
        return sampler.render(sampled_points.data() + 2 * sampled_offsets[i], int(count));
    }
    // Areas of each contour's downscaled mask; None when the source is not
    // one this engine handles. Retains the masks (keep) or only the contours,
    // which sampled_compose then renders again.
    py::object sampled_raster(const py::object &source, int h, int w, const py::bytes &lut, bool keep) {
        const auto table = sampled_table(h, w, lut, true);
        SampledSource s;
        if (!sampled_source(source, s)) return py::none();
        Array<uint64_t> areas{py::ssize_t(s.count)};
        auto *sums = areas.mutable_data();
        bool ok;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(mutex);
            sampler.configure(h, w, table.data());
            uint64_t since = 0;
#ifdef MASKOPS_ABLATE
            since = maskops_sampled::now_ns();
#endif
            ok = sampled_ready = sampled_load(s);
            MASKOPS_TICK(0, since);
            sampled_kept = keep;
            if (ok) {
                const size_t n = s.count;
                sampled_boxes.resize(n);
                sampled_starts.assign(n + 1, 0);
                sampled_arena.clear();
                for (size_t i = 0; i < n; ++i) {
                    const auto box = sampled_render(i);
                    auto &store = keep ? sampled_arena : sampled_scratch;
                    const size_t start = keep ? store.size() : 0;
                    store.resize(start + box.size());
                    MASKOPS_SKIP(32) {
                        sums[i] = 0, sampled_boxes[i] = box, sampled_starts[i + 1] = store.size();
                        continue;
                    }
#ifdef MASKOPS_ABLATE
                    since = maskops_sampled::now_ns();
#endif
                    sums[i] = sampler.emit(box, store.data() + start, ptrdiff_t(box.width()));
                    MASKOPS_TICK(7, since);
                    sampled_boxes[i] = box;
                    sampled_starts[i + 1] = store.size();
                }
            }
        }
        if (!ok) return py::none();
        return std::move(areas);
    }
    template <typename T> py::array sampled_compose_t(const std::vector<int64_t> &indices) {
        // The engine lock is never held while the GIL is reacquired, so taking
        // it briefly with the GIL held cannot deadlock.
        int hh, ww;
        {
            std::lock_guard<std::mutex> guard(mutex);
            if (!sampled_ready || sampled_boxes.size() != indices.size())
                throw py::value_error("sampled_compose requires a preceding sampled_raster of the same contours");
            hh = sampler.hh, ww = sampler.ww;
        }
        Array<T> result({hh, ww});
        auto *out = result.mutable_data();
        bool changed;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(mutex);
            changed = !sampled_ready || sampled_boxes.size() != indices.size() || sampler.hh != hh || sampler.ww != ww;
            if (!changed) std::fill(out, out + size_t(hh) * size_t(ww), T(0));
            for (size_t rank = 0; !changed && rank < indices.size(); ++rank) {
                const auto i = size_t(indices[rank]);
                maskops_sampled::Box box;
                const uint8_t *mask;
                if (sampled_kept) {
                    box = sampled_boxes[i];
                    mask = sampled_arena.data() + sampled_starts[i];
                } else {
                    box = sampled_render(i);
                    sampled_scratch.resize(box.size());
                    sampler.emit(box, sampled_scratch.data(), ptrdiff_t(box.width()));
                    mask = sampled_scratch.data();
                }
                if (!box.empty()) maskops_sampled::paint(out, ww, box, mask, T(rank + 1));
            }
        }
        if (changed) throw py::value_error("sampled state changed during sampled_compose");
        return result;
    }
    py::array sampled_compose(Array<int64_t> order) {
        if (order.ndim() != 1) throw py::value_error("invalid order length");
        const auto n = size_t(order.size());
        std::vector<int64_t> indices(n);
        if (n) std::memcpy(indices.data(), static_cast<const py::array &>(order).data(), size_t(order.nbytes()));
        std::vector<bool> seen(n, false);
        for (auto i : indices) {
            if (i < 0 || size_t(i) >= n || seen[i]) throw py::value_error("order must be a permutation");
            seen[i] = true;
        }
        return n > 255 ? sampled_compose_t<int32_t>(indices) : sampled_compose_t<uint8_t>(indices);
    }
    // One mask per contour, each the downscaled table value of its patterns.
    py::object sampled_masks(const py::object &source, int h, int w, const py::bytes &lut) {
        const auto table = sampled_table(h, w, lut, false);
        const auto a = reduced_size(h, w);
        SampledSource s;
        if (!sampled_source(source, s)) return py::none();
        if (s.count && a > size_t(PY_SSIZE_T_MAX) / s.count) throw py::value_error("output size overflow");
        Array<uint8_t> result({py::ssize_t(s.count), py::ssize_t(h / 4), py::ssize_t(w / 4)});
        auto *out = result.mutable_data();
        bool ok;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(mutex);
            sampler.configure(h, w, table.data());
            sampled_ready = false;
            ok = sampled_load(s);
            if (ok) {
                std::memset(out, 0, s.count * a);
                for (size_t i = 0; i < s.count; ++i) {
                    const auto box = sampled_render(i);
                    if (!box.empty())
                        sampler.emit(box, out + i * a + size_t(box.y0) * size_t(sampler.ww) + size_t(box.x0), sampler.ww);
                }
            }
        }
        if (!ok) return py::none();
        return std::move(result);
    }
};

// pycocotools masks of RF-DETR instances at the pixels a transform chain
// keeps. instances holds, per instance, a sequence of polygons; each polygon
// is anything np.asarray(..., float64) accepts, read as consecutive x, y
// pairs (an odd trailing coordinate is ignored, as frPoly does). rows and
// cols map every output row and column to a source row and column, -1 where
// a crop padded. fused selects the contracted double arithmetic (see
// rfdetr.hpp). Returns a bool array (N, len(rows), len(cols)).
py::array rfdetr_masks(const py::sequence &instances, int h, int w, Array<int32_t> rows, Array<int32_t> cols, bool fused) {
    if (rows.ndim() != 1 || cols.ndim() != 1) throw py::value_error("rows and cols must be one-dimensional");
    if (h < 0 || w < 0 || h > INT_MAX / 8 || w > INT_MAX / 8) throw py::value_error("invalid source size");
    std::vector<int32_t> row_map(size_t(rows.size())), col_map(size_t(cols.size()));
    if (rows.size()) std::memcpy(row_map.data(), static_cast<const py::array &>(rows).data(), size_t(rows.nbytes()));
    if (cols.size()) std::memcpy(col_map.data(), static_cast<const py::array &>(cols).data(), size_t(cols.nbytes()));
    maskops_rfdetr::Chain chain;
    if (!chain.configure(h, w, row_map.data(), int(row_map.size()), col_map.data(), int(col_map.size())))
        throw py::value_error("rows and cols are not a nearest-resize, crop and flip composition");
    using Coordinates = py::array_t<double, py::array::c_style | py::array::forcecast>;
    std::vector<std::vector<Coordinates>> polygons;
    polygons.reserve(size_t(py::len(instances)));
    for (const auto &instance : instances) {
        std::vector<Coordinates> own;
        for (const auto &polygon : py::reinterpret_borrow<py::sequence>(instance)) own.push_back(polygon.cast<Coordinates>());
        polygons.push_back(std::move(own));
    }
    const size_t n = polygons.size(), plane = size_t(chain.H) * size_t(chain.W);
    if (n && plane > size_t(PY_SSIZE_T_MAX) / n) throw py::value_error("output size overflow");
    py::array_t<bool> result({py::ssize_t(n), py::ssize_t(chain.H), py::ssize_t(chain.W)});
    auto *out = reinterpret_cast<uint8_t *>(result.mutable_data());
    {
        py::gil_scoped_release release;
        std::memset(out, 0, n * plane);
        maskops_rfdetr::Renderer renderer;
        for (size_t i = 0; i < n; ++i)
            for (const auto &polygon : polygons[i])
                renderer.render(chain, polygon.data(), size_t(polygon.size()) / 2, out + i * plane, fused);
    }
    return std::move(result);
}

PYBIND11_MODULE(_native, m) {
    m.def("build_profile", []() {
        py::dict info;
        info["bindings_sha256"] = MASKOPS_BINDINGS_SHA256;
        info["geometry_sha256"] = MASKOPS_GEOMETRY_SHA256;
        info["sampled_sha256"] = MASKOPS_SAMPLED_SHA256;
        info["rfdetr_sha256"] = MASKOPS_RFDETR_SHA256;
        info["cmake_sha256"] = MASKOPS_CMAKE_SHA256;
        info["template_sha256"] = MASKOPS_PROFILE_SHA256;
        info["compiler"] = MASKOPS_COMPILER;
        const std::string build_type = MASKOPS_BUILD_TYPE;  // empty under multi-config generators
        info["build_type"] = build_type.empty() ? std::string(MASKOPS_CONFIG) : build_type;
        return info;
    });
    m.def("simd_mode", []() { return std::string(maskops_sampled::simd_mode()); });
#ifdef MASKOPS_ABLATE
    m.def("phase_ns", []() {
        py::list out;
        for (auto t : maskops_sampled::phase_ns()) out.append(double(t) * 41.6667);
        return out;
    });
#endif
    m.def("rfdetr_masks", &rfdetr_masks, py::arg("instances"), py::arg("h"), py::arg("w"), py::arg("rows"), py::arg("cols"),
          py::arg("fused") = false);
    maskops_geometry::register_geometry(m);
    py::class_<Polygons>(m, "Polygons").def(py::init<Array<int32_t>, Array<int64_t>>()).def("__len__", &Polygons::size);
    py::class_<Rasterizer>(m, "Rasterizer")
        .def(py::init<>())
        .def("sampled_raster", &Rasterizer::sampled_raster)
        .def("sampled_compose", &Rasterizer::sampled_compose)
        .def("sampled_masks", &Rasterizer::sampled_masks);
}
