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
// The double arithmetic comes in two flavours: the C source as written (every
// product rounded before the addition, pycocotools' x86-64 wheels) and the
// contracted form (scale*v + .5 and ys + s*t fused into one rounding, which
// compilers emit for arm64 wheels). The Python layer picks the flavour that
// reproduces the installed pycocotools.
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

// Floor division by 5 for any int.
inline int div5_floor(int v) { return v >= 0 ? v / 5 : -((-v + 4) / 5); }

// The boundary toggles of one polygon of k vertices at 2k doubles, unsorted.
//
// rleFrPoly keeps a dense point when its column u differs from the previous
// point's and xd = ((u < up ? u : u - 1) + .5) / 5 - .5 is an integer in
// [0, w - 1]; then y = ceil(clamp((min(v, vp) + .5) / 5 - .5, 0, h)). Both
// doubles are exact rationals with fractional parts in multiples of 1/5, so
// the tests are the integer ones below: xd is integral exactly when
// u_adj = 5n + 2, and the ceil is the floor division of (yv - 2 + 4) by 5.
// Along an x-major edge u advances by one per step, so only every fifth step
// can toggle; those steps and their predecessors are evaluated directly with
// the C code's own double formula. Along a y-major edge u rarely changes, so
// every step is visited but only a column change runs the test.
template <bool Fused>
inline void polygon_toggles(const double *xy, size_t k, int h, int w, std::vector<int> &x, std::vector<int> &y,
                            std::vector<Toggle> &out) {
    const double scale = 5;
    // scale * v + .5 and base + s * t, each as one rounding when Fused.
    auto scaled = [](double v) { return Fused ? std::fma(5.0, v, .5) : 5.0 * v + .5; };
    auto along = [](int base, double s, int t) { return Fused ? std::fma(s, double(t), double(base)) : base + s * t; };
    x.resize(k + 1), y.resize(k + 1);
    for (size_t j = 0; j < k; ++j) x[j] = c_int(scaled(xy[j * 2 + 0]));
    for (size_t j = 0; j < k; ++j) y[j] = c_int(scaled(xy[j * 2 + 1]));
    if (k == 0) return;
    x[k] = x[0], y[k] = y[0];
    bool have_previous = false;
    int up = 0, vp = 0;
    // The pair (previous point, this point) in emission order.
    auto visit = [&](int u, int v) {
        if (have_previous && u != up) {
            const int adj = u < up ? u : u - 1;
            const int n = div5_floor(adj - 2);
            if (adj - 5 * n == 2 && n >= 0 && n <= w - 1) {
                const int yv = v < vp ? v : vp;
                int yd = div5_floor(yv + 2);  // ceil((yv - 2) / 5)
                yd = yd < 0 ? 0 : yd > h ? h : yd;
                out.push_back({int32_t(n), int32_t(yd)});
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
            auto v_at = [&](int t) { return c_int(along(ys, s, t) + .5); };
            if (!flip) {
                // Emission order t = 0 .. dx, u = xs + t. The first point pairs
                // with the previous edge; afterwards only u - 1 = 5n + 2 can toggle.
                visit(xs, v_at(0));
                // Smallest t >= 1 with (xs + t - 1) mod 5 == 2, i.e. t = 3 - xs (mod 5).
                int t = ((3 - xs) % 5 + 5) % 5;
                if (t < 1) t += 5;
                for (; t <= dx; t += 5) {
                    up = xs + t - 1, vp = v_at(t - 1);
                    visit(xs + t, v_at(t));
                }
                up = xs + dx, vp = v_at(dx);
            } else {
                // Emission order t = dx .. 0, u = xs + t decreasing: u = 5n + 2 toggles.
                visit(xs + dx, v_at(dx));
                // Largest t <= dx - 1 with (xs + t) mod 5 == 2.
                int t = dx - 1;
                t -= ((xs + t - 2) % 5 + 5) % 5;
                for (; t >= 0; t -= 5) {
                    up = xs + t + 1, vp = v_at(t + 1);
                    visit(xs + t, v_at(t));
                }
                up = xs, vp = v_at(0);
            }
            have_previous = true;
        } else if (dy < 4 * dx || std::abs(xs) > (1 << 30) || std::abs(xe) > (1 << 30)) {
            for (int d = 0; d <= dy; ++d) {
                const int t = flip ? dy - d : d;
                visit(c_int(along(xs, s, t) + .5), t + ys);
            }
        } else {
            // A steep edge: u is a monotone function of the step (a correctly
            // rounded product and sum of a fixed s, then a monotone truncation;
            // the magnitude guard above keeps INT_MIN out), so the next column
            // change is found by galloping and bisection instead of one step
            // at a time. Every pair that can toggle is still evaluated with
            // the C formula at its own step.
            auto u_at = [&](int d) { return c_int(along(xs, s, flip ? dy - d : d) + .5); };
            int d = 0, u0 = u_at(0);
            visit(u0, (flip ? dy : 0) + ys);
            while (d < dy) {
                int lo = d + 1, hi = dy + 1, step = 1;
                while (true) {
                    const int probe = d + step;
                    if (probe > dy) break;
                    if (u_at(probe) != u0) {
                        hi = probe;
                        break;
                    }
                    lo = probe + 1, step *= 2;
                }
                while (lo < hi) {
                    const int mid = lo + (hi - lo) / 2;
                    if (u_at(mid) != u0)
                        hi = mid;
                    else
                        lo = mid + 1;
                }
                if (hi > dy) break;  // constant to the end of the edge
                up = u0, vp = (flip ? dy - (hi - 1) : hi - 1) + ys;
                const int u1 = u_at(hi);
                visit(u1, (flip ? dy - hi : hi) + ys);
                u0 = u1, d = hi;
            }
            up = u0, vp = (flip ? 0 : dy) + ys, have_previous = true;
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
    std::vector<Toggle> toggles, sorted;
    std::vector<uint32_t> row_start;
    std::vector<uint64_t> state;
    std::vector<int32_t> counts;
    int64_t set_bits = 0;  // bits set in state
    size_t word_lo = 0, word_hi = 0;  // words of state that can hold set bits
    int span_lo = 0, span_hi = 0;  // output columns painted on the last painted row

    void render(const Chain &c, const double *xy, size_t k, uint8_t *out, bool fused) {
        toggles.clear();
        if (fused)
            polygon_toggles<true>(xy, k, c.h, c.w, x, y, toggles);
        else
            polygon_toggles<false>(xy, k, c.h, c.w, x, y, toggles);
        if (toggles.empty()) return;
        // Initial state of each column: the parity of the toggles before it.
        // Only columns from the first toggle on can ever be set; a polygon
        // with an odd number of toggles leaks up to the last column.
        int32_t x_lo = c.w, x_hi = -1;
        for (const auto &t : toggles) x_lo = std::min(x_lo, t.x), x_hi = std::max(x_hi, t.x);
        counts.assign(size_t(x_hi - x_lo) + 1, 0);
        for (const auto &t : toggles) ++counts[size_t(t.x - x_lo)];
        const bool leak = toggles.size() % 2 != 0;
        const size_t words = (size_t(c.w) + 63) / 64;
        state.assign(words, 0);
        set_bits = 0;
        uint32_t parity = 0;
        for (int col = x_lo; col <= x_hi; ++col) {
            if (parity) state[size_t(col) / 64] |= uint64_t(1) << (col % 64), ++set_bits;
            parity ^= uint32_t(counts[size_t(col - x_lo)]) & 1;
        }
        if (leak)
            for (int col = x_hi + 1; col < c.w; ++col) state[size_t(col) / 64] |= uint64_t(1) << (col % 64), ++set_bits;
        word_lo = size_t(x_lo) / 64, word_hi = leak ? words - 1 : size_t(x_hi) / 64;
        // Order the toggles by row (a counting sort; the order within a row
        // does not matter, the flips commute).
        row_start.assign(size_t(c.h) + 2, 0);
        for (const auto &t : toggles) ++row_start[size_t(t.y) + 1];
        for (size_t r = 1; r < row_start.size(); ++r) row_start[r] += row_start[r - 1];
        sorted.resize(toggles.size());
        for (const auto &t : toggles) sorted[size_t(row_start[size_t(t.y)]++)] = t;
        toggles.swap(sorted);
        size_t next = 0;
        int32_t current = -1;  // source row the state describes
        int painted = -1;  // output row already painted for it
        for (int rf = 0; rf < c.H; ++rf) {
            const int32_t sr = c.rows[size_t(rf)];
            if (sr < 0) continue;
            if (sr != current) {
                for (; next < toggles.size() && toggles[next].y <= sr; ++next) {
                    uint64_t &word = state[size_t(toggles[next].x) / 64];
                    const uint64_t bit = uint64_t(1) << (toggles[next].x % 64);
                    set_bits += (word & bit) ? -1 : 1;
                    word ^= bit;
                }
                current = sr, painted = -1;
            }
            if (set_bits == 0) continue;  // the row stays zero: nothing to paint or copy
            uint8_t *row = out + size_t(rf) * size_t(c.W);
            if (painted >= 0) {
                if (span_hi > span_lo) std::memcpy(row + span_lo, out + size_t(painted) * size_t(c.W) + span_lo, size_t(span_hi - span_lo));
            } else {
                span_lo = c.W, span_hi = 0;
                paint_row(c, row);
            }
            painted = rf;
        }
    }

    // Index of the first bit >= pos with the wanted value, or w. Set bits
    // live in words word_lo..word_hi; a run's end may be the zero bit just past
    // them, so the search for a zero looks one word further.
    int next_bit(int pos, bool wanted, int w) const {
        const size_t last = wanted ? word_hi : std::min(word_hi + 1, state.size() - 1);
        for (size_t wi = size_t(pos) / 64; wi <= last; ++wi) {
            uint64_t bits = wanted ? state[wi] : ~state[wi];
            if (wi == size_t(pos) / 64) bits &= ~uint64_t(0) << (pos % 64);
            if (bits) return std::min(w, int(wi * 64) + std::countr_zero(bits));
        }
        return w;
    }

    // Paints the 1-runs of the column state through the column map.
    void paint_row(const Chain &c, uint8_t *row) {
        for (int pos = int(word_lo * 64); pos < c.w;) {
            const int start = next_bit(pos, true, c.w);
            if (start >= c.w) break;
            const int end = next_bit(start, false, c.w);
            const int a = c.col_first[size_t(start)], b = c.col_first[size_t(end)];
            if (b > a) {
                const int lo = c.flip ? c.mirror - b : a, hi = c.flip ? c.mirror - a : b;
                std::memset(row + lo, 1, size_t(hi - lo));
                span_lo = std::min(span_lo, lo), span_hi = std::max(span_hi, hi);
            }
            pos = end;
        }
    }
};

}  // namespace maskops_rfdetr
