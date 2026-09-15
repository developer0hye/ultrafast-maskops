// SPDX-License-Identifier: AGPL-3.0-only
// pycocotools polygon masks (rleFrPoly followed by decode), evaluated only at
// the source pixels that a chain of torchvision nearest resizes, crops and
// horizontal flips keeps.
//
// rleFrPoly scales the vertices by 5, walks every edge as a dense integer
// line, and keeps the points where that walk changes column and the scaled
// column maps back to an exact pixel column. Each kept point is a toggle at
// column-major position x*h + y; decode then alternates 0/1 between sorted
// positions, so pixel (y, x) is set exactly when an odd number of toggles lie
// at or before x*h + y. Columns with an odd number of toggles leak their
// parity into every later column, and a toggle at y == h counts for later
// columns only; both follow from the position formula and are reproduced.
//
// The chain of transforms is summarized by two index maps: output row ->
// source row (non-decreasing, -1 where a crop padded) and output column ->
// source column (monotone in either direction, -1 where padded). A polygon is
// rendered by sweeping the sampled source rows once, flipping one bit per
// column as its toggles pass, and painting each 1-run of columns into the
// output row through the column map. Only ones are written, so the polygons
// of one instance union by rendering into the same zeroed plane.
//
// Integer conversions follow the C source as compiled for x86-64: a NaN or
// out-of-range double converts to INT_MIN, the "integer indefinite" value.
#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace maskops_rfdetr {

struct Toggle {
    int32_t x, y;  // 0 <= x < w, 0 <= y <= h
};

// (int)value in maskApi.c: truncation, INT_MIN when not representable.
inline int c_int(double value) {
    if (!(value > -2147483649.0 && value < 2147483648.0)) return std::numeric_limits<int>::min();
    return int(value);
}

// The boundary toggles of one polygon of k vertices at 2k doubles, unsorted.
inline void polygon_toggles(const double *xy, size_t k, int h, int w, std::vector<int> &x, std::vector<int> &y,
                            std::vector<Toggle> &out) {
    const double scale = 5;
    x.resize(k + 1), y.resize(k + 1);
    for (size_t j = 0; j < k; ++j) x[j] = c_int(scale * xy[j * 2 + 0] + .5);
    for (size_t j = 0; j < k; ++j) y[j] = c_int(scale * xy[j * 2 + 1] + .5);
    if (k == 0) return;
    x[k] = x[0], y[k] = y[0];
    // Visit the dense points in emission order and keep the column changes
    // between consecutive points, as the second loop of rleFrPoly does over
    // its u/v arrays; only the previous point is needed.
    bool have_previous = false;
    int up = 0, vp = 0;
    auto visit = [&](int u, int v) {
        if (have_previous && u != up) {
            double xd = double(u < up ? u : u - 1);
            xd = (xd + .5) / scale - .5;
            if (!(std::floor(xd) != xd || xd < 0 || xd > double(w - 1))) {
                double yd = double(v < vp ? v : vp);
                yd = (yd + .5) / scale - .5;
                if (yd < 0)
                    yd = 0;
                else if (yd > double(h))
                    yd = double(h);
                yd = std::ceil(yd);
                out.push_back({int32_t(c_int(xd)), int32_t(c_int(yd))});
            }
        }
        have_previous = true, up = u, vp = v;
    };
    for (size_t j = 0; j < k; ++j) {
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1];
        const int dx = std::abs(xe - xs), dy = std::abs(ys - ye);
        const bool flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) std::swap(xs, xe), std::swap(ys, ye);
        const double s = dx >= dy ? double(ye - ys) / dx : double(xe - xs) / dy;
        if (dx >= dy) {
            for (int d = 0; d <= dx; ++d) {
                const int t = flip ? dx - d : d;
                visit(t + xs, c_int(ys + s * t + .5));
            }
        } else {
            for (int d = 0; d <= dy; ++d) {
                const int t = flip ? dy - d : d;
                visit(c_int(xs + s * t + .5), t + ys);
            }
        }
    }
}

// Output geometry shared by every instance of one sample.
struct Chain {
    int h = 0, w = 0;  // source size
    int H = 0, W = 0;  // output size
    std::vector<int32_t> rows;  // H entries: source row or -1
    // w + 1 entries: the first output column, in unflipped order, that reads
    // a source column >= s. A source run [a, b) covers output [first[a], first[b]).
    std::vector<int32_t> col_first;
    bool flip = false;
    int mirror = 0;  // with flip, unflipped column i is output column mirror - 1 - i

    // False when the maps are not a nearest-resize/crop/flip composition.
    bool configure(int h_, int w_, const int32_t *rows_, int H_, const int32_t *cols, int W_) {
        h = h_, w = w_, H = H_, W = W_;
        if (h < 0 || w < 0 || H < 0 || W < 0) return false;
        rows.assign(rows_, rows_ + H);
        int32_t last = -1;
        for (auto r : rows) {
            if (r >= h || (r >= 0 && r < last)) return false;
            if (r >= 0) last = r;
        }
        int lo = 0, hi = W;
        while (lo < W && cols[lo] < 0) ++lo;
        while (hi > lo && cols[hi - 1] < 0) --hi;
        for (int i = lo; i < hi; ++i)
            if (cols[i] < 0 || cols[i] >= w) return false;
        flip = hi - lo >= 2 && cols[lo] > cols[hi - 1];
        mirror = lo + hi;
        auto ordered = [&](int i) { return flip ? cols[mirror - 1 - i] : cols[i]; };
        for (int i = lo + 1; i < hi; ++i)
            if (ordered(i) < ordered(i - 1)) return false;
        col_first.resize(size_t(w) + 1);
        int i = lo;
        for (int s = 0; s <= w; ++s) {
            while (i < hi && ordered(i) < s) ++i;
            col_first[size_t(s)] = int32_t(i);
        }
        return true;
    }
};

// Renders polygons into out[H][W] (row-major bytes), writing only ones.
struct Renderer {
    std::vector<int> x, y;
    std::vector<Toggle> toggles;
    std::vector<uint64_t> state;
    std::vector<int32_t> counts;

    void render(const Chain &c, const double *xy, size_t k, uint8_t *out) {
        toggles.clear();
        polygon_toggles(xy, k, c.h, c.w, x, y, toggles);
        if (toggles.empty()) return;
        // Initial state of each column: the parity of the toggles before it.
        counts.assign(size_t(c.w) + 1, 0);
        for (const auto &t : toggles) ++counts[size_t(t.x)];
        const size_t words = (size_t(c.w) + 63) / 64;
        state.assign(words, 0);
        uint32_t parity = 0;
        for (int col = 0; col < c.w; ++col) {
            if (parity) state[size_t(col) / 64] |= uint64_t(1) << (col % 64);
            parity ^= uint32_t(counts[size_t(col)]) & 1;
        }
        std::sort(toggles.begin(), toggles.end(),
                  [](const Toggle &a, const Toggle &b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
        size_t next = 0;
        int32_t current = -1;  // source row the state describes
        int painted = -1;  // output row already painted for it
        for (int rf = 0; rf < c.H; ++rf) {
            const int32_t sr = c.rows[size_t(rf)];
            if (sr < 0) continue;
            uint8_t *row = out + size_t(rf) * size_t(c.W);
            if (sr != current) {
                for (; next < toggles.size() && toggles[next].y <= sr; ++next)
                    state[size_t(toggles[next].x) / 64] ^= uint64_t(1) << (toggles[next].x % 64);
                current = sr, painted = -1;
            }
            if (painted >= 0)
                std::memcpy(row, out + size_t(painted) * size_t(c.W), size_t(c.W));
            else
                paint_row(c, row);
            painted = rf;
        }
    }

    // Index of the first bit >= pos with the wanted value, or w.
    int next_bit(int pos, bool wanted, int w) const {
        for (size_t wi = size_t(pos) / 64; wi < state.size(); ++wi) {
            uint64_t bits = wanted ? state[wi] : ~state[wi];
            if (wi == size_t(pos) / 64) bits &= ~uint64_t(0) << (pos % 64);
            if (bits) return std::min(w, int(wi * 64) + std::countr_zero(bits));
        }
        return w;
    }

    // Paints the 1-runs of the column state through the column map.
    void paint_row(const Chain &c, uint8_t *row) const {
        for (int pos = 0; pos < c.w;) {
            const int start = next_bit(pos, true, c.w);
            if (start >= c.w) break;
            const int end = next_bit(start, false, c.w);
            const int a = c.col_first[size_t(start)], b = c.col_first[size_t(end)];
            if (b > a) {
                const int lo = c.flip ? c.mirror - b : a, hi = c.flip ? c.mirror - a : b;
                std::memset(row + lo, 1, size_t(hi - lo));
            }
            pos = end;
        }
    }
};

}  // namespace maskops_rfdetr
