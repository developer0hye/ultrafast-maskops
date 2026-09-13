// SPDX-License-Identifier: AGPL-3.0-only
// cv::fillPoly (one contour, LINE_8, shift 0) followed by a 4x INTER_LINEAR
// downscale, evaluated only where the downscale reads.
//
// At exactly 4x, output pixel (j, k) reads source rows 4k+1, 4k+2 and columns
// 4j+1, 4j+2 with equal weights, so its value is a function of those four
// pixels. That function (a 16-entry table) is calibrated against the installed
// cv2.resize by the caller for each output size, because OpenCV's Arm HAL and
// its generic path round differently. fillPoly is integer arithmetic: every
// edge is drawn as an 8-connected line, then each row is filled between pairs
// of active edges sorted by x. Both steps are evaluated here only on sampled
// rows and columns, following OpenCV 4.13's clipLine, LineIterator,
// CollectPolyEdges and FillEdgeCollection. The full-resolution scratch image,
// its clearing, the resize and the full-size area pass disappear.
//
// FillEdgeCollection keeps an active edge list sorted by x (merge insertion
// plus bubble sort) and advances each edge by dx per row, so at row y it pairs
// the values x0 + (y - y0) * dx of the edges with y0 <= y < y1 in ascending
// order. A closed contour crosses every row an even number of times, so each
// edge is advanced on every row it is active and the pairs are exactly the
// consecutive sorted values computed directly below. Its early exits (fewer
// than two edges, or all edges above, below, left or right of the image) only
// skip rows or spans that clipping would discard anyway.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>
#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(__ARM_BIG_ENDIAN)
#include <arm_neon.h>
#define MASKOPS_SAMPLED_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
// SSE2 is the x86-64 baseline; the SSSE3 table lookup is selected at run time.
#include <immintrin.h>
#define MASKOPS_SAMPLED_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define MASKOPS_TARGET_SSSE3
#else
#define MASKOPS_TARGET_SSSE3 __attribute__((target("ssse3")))
#endif
#endif

namespace maskops_sampled {

constexpr int kShift = 16;  // XY_SHIFT
constexpr int64_t kOne = int64_t(1) << kShift;
// Envelope in which every fixed-point product below stays far inside int64 and
// every clipLine double product is exact: |coordinate| < 2^24, side <= 2^15.
constexpr double kCoordLimit = 16777216.0;
constexpr int kSideLimit = 1 << 15;
// int32 values load_contours may write past the last kept vertex.
constexpr size_t kPointSlack = 8;

// A std::vector whose growth leaves new elements uninitialized; every element
// is written before it is read.
template <typename T> struct DefaultInit : std::allocator<T> {
    template <typename U> struct rebind {
        using other = DefaultInit<U>;
    };
    DefaultInit() = default;
    template <typename U> DefaultInit(const DefaultInit<U> &) noexcept {}
    template <typename U> void construct(U *p) noexcept { ::new (static_cast<void *>(p)) U; }
    template <typename U, typename... A> void construct(U *p, A &&...a) {
        ::new (static_cast<void *>(p)) U(std::forward<A>(a)...);
    }
};
template <typename T> using Vec = std::vector<T, DefaultInit<T>>;

struct Point64 {
    int64_t x, y;
};

// Inclusive box in output pixels; empty when x0 > x1.
struct Box {
    int x0 = INT_MAX, y0 = INT_MAX, x1 = INT_MIN, y1 = INT_MIN;
    bool empty() const { return x0 > x1 || y0 > y1; }
    size_t width() const { return empty() ? 0 : size_t(x1 - x0 + 1); }
    size_t size() const { return empty() ? 0 : width() * size_t(y1 - y0 + 1); }
};

// cv::clipLine(Size2l, Point2l&, Point2l&). Like OpenCV, it may modify the
// points even when it returns false; CollectPolyEdges relies on that.
inline bool clip_line(int64_t width, int64_t height, Point64 &p1, Point64 &p2) {
    if (width <= 0 || height <= 0) return false;
    const int64_t right = width - 1, bottom = height - 1;
    int64_t &x1 = p1.x, &y1 = p1.y, &x2 = p2.x, &y2 = p2.y;
    int c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
    int c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;
    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
        int64_t a;
        if (c1 & 12) {
            a = c1 < 8 ? 0 : bottom;
            x1 += (int64_t)((double)(a - y1) * (x2 - x1) / (y2 - y1));
            y1 = a;
            c1 = (x1 < 0) + (x1 > right) * 2;
        }
        if (c2 & 12) {
            a = c2 < 8 ? 0 : bottom;
            x2 += (int64_t)((double)(a - y2) * (x2 - x1) / (y2 - y1));
            y2 = a;
            c2 = (x2 < 0) + (x2 > right) * 2;
        }
        if ((c1 & c2) == 0 && (c1 | c2) != 0) {
            if (c1) {
                a = c1 == 1 ? 0 : right;
                y1 += (int64_t)((double)(a - x1) * (y2 - y1) / (x2 - x1));
                x1 = a;
                c1 = 0;
            }
            if (c2) {
                a = c2 == 1 ? 0 : right;
                y2 += (int64_t)((double)(a - x2) * (y2 - y1) / (x2 - x1));
                x2 = a;
                c2 = 0;
            }
        }
    }
    return (c1 | c2) == 0;
}

// The pixels of LineIterator (connectivity 8, leftToRight) from (x1, y1) to
// (x2, y2), both inside the image; with interior_only, all but the two ends.
template <bool interior_only, typename Visit>
inline void line8_unclipped(int x1, int y1, int x2, int y2, Visit &&visit) {
    int delta_x = 1, delta_y = 1;
    int dx = x2 - x1, dy = y2 - y1;
    int px = x1, py = y1;
    if (dx < 0) {
        dx = -dx, dy = -dy, px = x2, py = y2;
    }
    if (dy < 0) {
        dy = -dy, delta_y = -1;
    }
    const bool vert = dy > dx;
    if (vert) {
        std::swap(dx, dy);
        std::swap(delta_x, delta_y);
    }
    int err = dx - (dy + dy);
    const int plus_delta = dx + dx, minus_delta = -(dy + dy);
    int minus_shift = delta_x, plus_shift = 0, minus_step = 0, plus_step = delta_y;
    if (vert) {
        std::swap(plus_step, plus_shift);
        std::swap(minus_step, minus_shift);
    }
    auto step = [&] {
        const int mask = err < 0 ? -1 : 0;
        err += minus_delta + (plus_delta & mask);
        px += minus_shift + (plus_shift & mask);
        py += minus_step + (plus_step & mask);
    };
    const int count = dx + 1;
    if (interior_only) {
        for (int i = 1; i + 1 < count; ++i) {
            step();
            visit(px, py);
        }
    } else {
        for (int i = 0; i < count; ++i) {
            visit(px, py);
            step();
        }
    }
}

// Every pixel cv::Line(img, a, b, color, LINE_8) writes: the image-clipped
// line's LineIterator pixels.
template <typename Visit> inline void line8(int width, int height, int x1, int y1, int x2, int y2, Visit &&visit) {
    if (unsigned(x1) >= unsigned(width) || unsigned(x2) >= unsigned(width) || unsigned(y1) >= unsigned(height) ||
        unsigned(y2) >= unsigned(height)) {
        Point64 c1{x1, y1}, c2{x2, y2};
        if (!clip_line(width, height, c1, c2)) return;
        x1 = int(c1.x), y1 = int(c1.y), x2 = int(c2.x), y2 = int(c2.y);
    }
    line8_unclipped<false>(x1, y1, x2, y2, visit);
}

// kPrefix.m[len]: 16 bytes, the first len of them 0xFF.
struct PrefixMasks {
    uint8_t m[17][16];
};
constexpr PrefixMasks make_prefix_masks() {
    PrefixMasks t{};
    for (int len = 0; len <= 16; ++len)
        for (int i = 0; i < len; ++i) t.m[len][i] = 0xFF;
    return t;
}
inline constexpr PrefixMasks kPrefix = make_prefix_masks();

// p[j] |= bits for j in [j0, j1]. May read and rewrite unchanged up to 15
// bytes past j1, so buffers keep 16 bytes of slack.
inline void or_run(uint8_t *p, int j0, int j1, uint8_t bits) {
    int len = j1 - j0 + 1;
    if (len <= 0) return;
    uint8_t *q = p + j0;
#ifdef MASKOPS_SAMPLED_NEON
    const uint8x16_t v = vdupq_n_u8(bits);
    for (; len >= 16; len -= 16, q += 16) vst1q_u8(q, vorrq_u8(vld1q_u8(q), v));
    if (len) vst1q_u8(q, vorrq_u8(vld1q_u8(q), vandq_u8(v, vld1q_u8(kPrefix.m[len]))));
#elif defined(MASKOPS_SAMPLED_X86)
    const __m128i v = _mm_set1_epi8(char(bits));
    auto *w = reinterpret_cast<__m128i *>(q);
    for (; len >= 16; len -= 16, ++w) _mm_storeu_si128(w, _mm_or_si128(_mm_loadu_si128(w), v));
    if (len) {
        const __m128i prefix = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kPrefix.m[len]));
        _mm_storeu_si128(w, _mm_or_si128(_mm_loadu_si128(w), _mm_and_si128(v, prefix)));
    }
#else
    for (; len > 0; --len, ++q) *q |= bits;
#endif
}

// Pattern bit of pixel (x, y) indexed by (y & 3) * 4 + (x & 3). Rows 4k+1/4k+2
// and columns 4j+1/4j+2 give bits 8, 4 (upper row) and 2, 1 (lower row).
constexpr uint8_t kBit[16] = {0, 0, 0, 0, 0, 8, 4, 0, 0, 2, 1, 0, 0, 0, 0, 0};
// Distance from y (by y & 3) to the first sampled row at or below it.
constexpr int kFirstSampled[4] = {1, 0, 0, 2};

inline void bounds(const int32_t *v, int n, int &x0, int &y0, int &x1, int &y1) {
    x0 = y0 = INT_MAX, x1 = y1 = INT_MIN;
    int i = 0;
#ifdef MASKOPS_SAMPLED_NEON
    if (n >= 2) {
        int32x4_t lo = vdupq_n_s32(INT_MAX), hi = vdupq_n_s32(INT_MIN);
        for (; i + 2 <= n; i += 2) {
            const int32x4_t p = vld1q_s32(v + 2 * i);
            lo = vminq_s32(lo, p), hi = vmaxq_s32(hi, p);
        }
        const int32x2_t l = vmin_s32(vget_low_s32(lo), vget_high_s32(lo));
        const int32x2_t u = vmax_s32(vget_low_s32(hi), vget_high_s32(hi));
        x0 = vget_lane_s32(l, 0), y0 = vget_lane_s32(l, 1), x1 = vget_lane_s32(u, 0), y1 = vget_lane_s32(u, 1);
    }
#endif
    for (; i < n; ++i) {
        x0 = std::min(x0, v[2 * i]), x1 = std::max(x1, v[2 * i]);
        y0 = std::min(y0, v[2 * i + 1]), y1 = std::max(y1, v[2 * i + 1]);
    }
}

class Sampler {
  public:
    int h = 0, w = 0, hh = 0, ww = 0;

    static bool eligible(int height, int width) {
        return height > 0 && width > 0 && height % 4 == 0 && width % 4 == 0 && height <= kSideLimit &&
               width <= kSideLimit;
    }

    void configure(int height, int width, const uint8_t *table) {
        if (height != h || width != w) {
            h = height, w = width, hh = height / 4, ww = width / 4;
            pattern_.assign(size_t(hh) * size_t(ww) + 16, 0);  // or_run slack
            slots_.resize(size_t(2 * hh) + 2);
        }
        std::memcpy(lut_, table, sizeof(lut_));
    }

    // ORs one contour's sampled pixels into the pattern image and returns the
    // box that may hold nonzero patterns. Vertices must lie in the envelope.
    Box render(const int32_t *v, int n) {
        Box box;
        if (n <= 0) return box;
        int vx0, vy0, vx1, vy1;
        bounds(v, n, vx0, vy0, vx1, vy1);
        // A contour entirely beside the image draws nothing: its lines stay in
        // the vertex box, clipLine leaves its edges unchanged, and every span
        // falls outside the image.
        if (vx1 < 0 || vx0 >= w || vy1 < 0 || vy0 >= h) return box;
        box = {std::max(vx0, 0) >> 2, std::max(vy0, 0) >> 2, std::min(vx1, w - 1) >> 2, std::min(vy1, h - 1) >> 2};
        if (vx0 >= 0 && vy0 >= 0 && vx1 < w && vy1 < h) {
            outline_inside(v, n);
            if (vy0 != vy1 && crossings_inside(v, n)) fill(box);
        } else {
            outline(v, n);
            if (vy0 != vy1 && crossings(v, n)) fill(box);
        }
        return box;
    }

    // Writes table[pattern] for the box to dst (row stride in bytes), clears
    // the box in the pattern image and returns the sum of written values.
    uint64_t emit(const Box &b, uint8_t *dst, ptrdiff_t stride) {
        if (b.empty()) return 0;
#ifdef MASKOPS_SAMPLED_X86
        if (has_ssse3()) return emit_ssse3(b, dst, stride);
#endif
        uint64_t area = 0;
        const int bw = int(b.width());
#ifdef MASKOPS_SAMPLED_NEON
        const uint8x16_t table = vld1q_u8(lut_), zero = vdupq_n_u8(0);
#endif
        for (int y = b.y0; y <= b.y1; ++y) {
            uint8_t *p = pattern_.data() + size_t(y) * size_t(ww) + size_t(b.x0);
            uint8_t *d = dst + ptrdiff_t(y - b.y0) * stride;
            int x = 0;
#ifdef MASKOPS_SAMPLED_NEON
            while (x + 16 <= bw) {
                // Each u16 lane gains at most 2 * 255 per step: 128 steps fit.
                uint16x8_t acc = vdupq_n_u16(0);
                for (int s = 0; s < 128 && x + 16 <= bw; ++s, x += 16) {
                    const uint8x16_t o = vqtbl1q_u8(table, vld1q_u8(p + x));
                    vst1q_u8(d + x, o);
                    vst1q_u8(p + x, zero);
                    acc = vpadalq_u8(acc, o);
                }
                area += vaddlvq_u16(acc);
            }
#endif
            for (; x < bw; ++x) {
                const uint8_t o = lut_[p[x]];
                d[x] = o;
                area += o;
                p[x] = 0;
            }
        }
        return area;
    }

  private:
#ifdef MASKOPS_SAMPLED_X86
    static bool has_ssse3() {
        static const bool supported = [] {
#if defined(_MSC_VER) && !defined(__clang__)
            int info[4];
            __cpuid(info, 1);
            return ((info[2] >> 9) & 1) != 0;
#else
            return __builtin_cpu_supports("ssse3") != 0;
#endif
        }();
        return supported;
    }

    // emit() with pshufb as the 16-entry table lookup and psadbw for the sum.
    MASKOPS_TARGET_SSSE3 uint64_t emit_ssse3(const Box &b, uint8_t *dst, ptrdiff_t stride) {
        const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lut_)), zero = _mm_setzero_si128();
        __m128i sums = zero;
        uint64_t area = 0;
        const int bw = int(b.width());
        for (int y = b.y0; y <= b.y1; ++y) {
            uint8_t *p = pattern_.data() + size_t(y) * size_t(ww) + size_t(b.x0);
            uint8_t *d = dst + ptrdiff_t(y - b.y0) * stride;
            int x = 0;
            for (; x + 16 <= bw; x += 16) {
                auto *source = reinterpret_cast<__m128i *>(p + x);
                const __m128i o = _mm_shuffle_epi8(table, _mm_loadu_si128(source));
                _mm_storeu_si128(reinterpret_cast<__m128i *>(d + x), o);
                _mm_storeu_si128(source, zero);
                sums = _mm_add_epi64(sums, _mm_sad_epu8(o, zero));
            }
            for (; x < bw; ++x) {
                const uint8_t o = lut_[p[x]];
                d[x] = o;
                area += o;
                p[x] = 0;
            }
        }
        return area + uint64_t(_mm_cvtsi128_si64(sums)) + uint64_t(_mm_cvtsi128_si64(_mm_unpackhi_epi64(sums, sums)));
    }
#endif

    // Pixels of every edge's LINE_8 on sampled rows and columns. When both
    // ends are inside, the ends are vertices, each also the end of the edge
    // arriving at it, so only b and the interior are drawn here. A vertex b
    // inside the image with a strictly lower neighbour is the upper end of
    // that edge, whose crossing at row b.y is exactly b.x whether or not the
    // other end is clipped; every crossing is an end of a filled span, so the
    // fill draws b and its mark is skipped.
    void outline(const int32_t *v, int n) {
        uint8_t *const pat = pattern_.data();
        const int W = w, H = h, WW = ww;
        auto mark = [pat, WW](int x, int y) { pat[(y >> 2) * WW + (x >> 2)] |= kBit[((y & 3) << 2) | (x & 3)]; };
        int ax = v[2 * n - 2], ay = v[2 * n - 1];
        for (int i = 0; i < n; ++i) {
            const int bx = v[2 * i], by = v[2 * i + 1];
            if (unsigned(ax) < unsigned(W) && unsigned(bx) < unsigned(W) && unsigned(ay) < unsigned(H) &&
                unsigned(by) < unsigned(H)) {
                const int next_y = v[i + 1 < n ? 2 * i + 3 : 1];
                if (ay <= by && next_y <= by) mark(bx, by);
                if (unsigned(bx - ax + 1) > 2u || unsigned(by - ay + 1) > 2u)
                    line8_unclipped<true>(ax, ay, bx, by, mark);
            } else {
                line8(W, H, ax, ay, bx, by, mark);
            }
            ax = bx, ay = by;
        }
    }

    // outline() for a contour entirely inside the image. Marks are collected
    // without branches and applied afterwards, so neither unpredictable
    // skips nor read-modify-writes of one byte serialize the vertex loop.
    void outline_inside(const int32_t *v, int n) {
        uint8_t *const pat = pattern_.data();
        const int WW = ww;
        auto mark = [pat, WW](int x, int y) { pat[(y >> 2) * WW + (x >> 2)] |= kBit[((y & 3) << 2) | (x & 3)]; };
        if (marks_.size() < size_t(n)) marks_.resize(size_t(n));
        uint32_t *const marks = marks_.data();
        size_t k = 0;
        int ax = v[2 * n - 2], ay = v[2 * n - 1];
        for (int i = 0; i < n; ++i) {
            const int bx = v[2 * i], by = v[2 * i + 1];
            const int next_y = v[i + 1 < n ? 2 * i + 3 : 1];
            const uint32_t bit = kBit[((by & 3) << 2) | (bx & 3)];
            marks[k] = (uint32_t((by >> 2) * WW + (bx >> 2)) << 4) | bit;
            k += (ay <= by) & (next_y <= by) & (bit != 0);
            if (unsigned(bx - ax + 1) > 2u || unsigned(by - ay + 1) > 2u) line8_unclipped<true>(ax, ay, bx, by, mark);
            ax = bx, ay = by;
        }
        for (size_t j = 0; j < k; ++j) pat[marks[j] >> 4] |= uint8_t(marks[j] & 15);
    }

    // crossings() for a contour entirely inside the image. An edge with
    // |dy| = 1 is active on one row, its upper end's, at that end's x; those
    // are collected without branches.
    size_t crossings_inside(const int32_t *v, int n) {
        if (slot_of_.size() < size_t(n) + 1) slot_of_.resize(size_t(n) + 64), x_of_.resize(size_t(n) + 64);
        size_t c = 0;
        int ax = v[2 * n - 2], ay = v[2 * n - 1];
        for (int i = 0; i < n; ++i) {
            const int bx = v[2 * i], by = v[2 * i + 1];
            const int dy = by - ay;
            if (unsigned(dy + 1) <= 2u) {
                const int y0 = std::min(ay, by);
                slot_of_[c] = ((y0 >> 2) << 1) | ((y0 >> 1) & 1);
                x_of_[c] = int64_t(dy > 0 ? ax : bx) * kOne;
                c += (dy != 0) & (unsigned((y0 & 3) - 1) < 2u);
            } else {
                const int64_t dx = (int64_t(bx - ax) * kOne) / dy;
                const int y0 = std::min(ay, by), y1 = std::max(ay, by);
                const int64_t x = int64_t(dy > 0 ? ax : bx) * kOne;
                int y = y0 + kFirstSampled[y0 & 3];
                // Room for this edge and one crossing for each remaining edge.
                const size_t need = c + size_t(std::max(y1 - y, 0)) / 2 + 2 + size_t(n - i);
                if (need > slot_of_.size()) {
                    const size_t grown = std::max(need, 2 * slot_of_.size());
                    slot_of_.resize(grown), x_of_.resize(grown);
                }
                for (; y < y1; y += (y & 1) ? 1 : 3) {
                    slot_of_[c] = ((y >> 2) << 1) | ((y >> 1) & 1);
                    x_of_[c++] = x + int64_t(y - y0) * dx;
                }
            }
            ax = bx, ay = by;
        }
        crossings_ = c;
        return c;
    }

    // CollectPolyEdges' edges evaluated at the sampled rows they are active
    // on. Returns the number of crossings collected.
    size_t crossings(const int32_t *v, int n) {
        const int W = w, H = h;
        size_t c = 0;
        int ax = v[2 * n - 2], ay = v[2 * n - 1];
        for (int i = 0; i < n; ++i) {
            const int bx = v[2 * i], by = v[2 * i + 1];
            if (ay != by) {
                // pt0c/pt1c: x in 16.16 fixed point from the clipped line ends.
                int64_t cx0, cx1, cy0 = ay, cy1 = by;
                if (unsigned(ax) < unsigned(W) && unsigned(bx) < unsigned(W) && unsigned(ay) < unsigned(H) &&
                    unsigned(by) < unsigned(H)) {
                    if (unsigned(by - ay + 1) <= 2u) {
                        // One active row, the upper end's, where x is that end's.
                        const int y0 = std::min(ay, by);
                        if (unsigned((y0 & 3) - 1) < 2u) {
                            if (c == slot_of_.size()) slot_of_.resize(2 * c + 64), x_of_.resize(2 * c + 64);
                            slot_of_[c] = ((y0 >> 2) << 1) | ((y0 >> 1) & 1);
                            x_of_[c++] = int64_t(ay < by ? ax : bx) * kOne;
                        }
                        ax = bx, ay = by;
                        continue;
                    }
                    cx0 = int64_t(ax) * kOne, cx1 = int64_t(bx) * kOne;
                } else {
                    Point64 t0{ax, ay}, t1{bx, by};
                    clip_line(W, H, t0, t1);
                    if (t0.y != t1.y) cy0 = t0.y, cy1 = t1.y;
                    cx0 = t0.x * kOne, cx1 = t1.x * kOne;
                }
                const int64_t dx = (cx1 - cx0) / (cy1 - cy0);
                int y0, y1;
                int64_t x;
                if (ay < by)
                    y0 = ay, y1 = by, x = cx0 + (ay - cy0) * dx;
                else
                    y0 = by, y1 = ay, x = cx1 + (by - cy1) * dx;
                int y = std::max(y0, 0);
                const int end = std::min(y1, H);
                if (y < end && (y += kFirstSampled[y & 3]) < end) {
                    const size_t most = size_t(end - y) / 2 + 2;
                    if (c + most > slot_of_.size()) {
                        const size_t grown = std::max(c + most, 2 * slot_of_.size());
                        slot_of_.resize(grown), x_of_.resize(grown);
                    }
                    for (; y < end; y += (y & 1) ? 1 : 3) {
                        slot_of_[c] = ((y >> 2) << 1) | ((y >> 1) & 1);
                        x_of_[c++] = x + int64_t(y - y0) * dx;
                    }
                }
            }
            ax = bx, ay = by;
        }
        crossings_ = c;
        return c;
    }

    // Fills the sampled rows between consecutive sorted crossings.
    void fill(Box &box) {
        const size_t m = crossings_;
        int smin = INT_MAX, smax = INT_MIN;
        for (size_t c = 0; c < m; ++c) smin = std::min(smin, slot_of_[c]), smax = std::max(smax, slot_of_[c]);
        const int ns = smax - smin + 1;
        uint32_t *pos = slots_.data();
        std::fill(pos, pos + ns + 1, 0u);
        for (size_t c = 0; c < m; ++c) ++pos[slot_of_[c] - smin + 1];
        for (int k = 0; k < ns; ++k) pos[k + 1] += pos[k];
        if (sorted_.size() < m) sorted_.resize(m);
        for (size_t c = 0; c < m; ++c) sorted_[pos[slot_of_[c] - smin]++] = x_of_[c];
        uint8_t *const pat = pattern_.data();
        const int W = w;
        size_t begin = 0;
        for (int k = 0; k < ns; ++k) {
            const size_t end = pos[k], count = end - begin;
            int64_t *xs = sorted_.data() + begin;
            begin = end;
            if (count < 2) continue;
            for (size_t i = 1; i < count; ++i) {
                const int64_t value = xs[i];
                size_t j = i;
                for (; j > 0 && xs[j - 1] > value; --j) xs[j] = xs[j - 1];
                xs[j] = value;
            }
            const int slot = smin + k, row = slot >> 1;
            const uint8_t a = (slot & 1) ? 2 : 8, b = (slot & 1) ? 1 : 4, ab = uint8_t(a | b);
            uint8_t *line = pat + size_t(row) * size_t(ww);
            for (size_t i = 0; i + 1 < count; i += 2) {
                int x1 = int((xs[i] + (kOne - 1)) >> kShift), x2 = int(xs[i + 1] >> kShift);
                if (x1 >= W || x2 < 0) continue;
                x1 = std::max(x1, 0), x2 = std::min(x2, W - 1);
                // Output columns whose source column 4j+1 (a) or 4j+2 (b) is
                // in [x1, x2]. ja0 - jb0 and ja1 - jb1 are 0 or 1, so the
                // columns of [ja0, jb1] take both bits and at most one column
                // at each end takes one.
                const int ja0 = (x1 + 2) >> 2, ja1 = (x2 - 1) >> 2, jb0 = (x1 + 1) >> 2, jb1 = (x2 - 2) >> 2;
                or_run(line, ja0, jb1, ab);
                if (jb0 < ja0 && jb0 <= jb1) line[jb0] |= b;
                if (ja1 > jb1 && ja0 <= ja1) line[ja1] |= a;
                const int lo = std::min(ja0 <= ja1 ? ja0 : INT_MAX, jb0 <= jb1 ? jb0 : INT_MAX);
                const int hi = std::max(ja0 <= ja1 ? ja1 : INT_MIN, jb0 <= jb1 ? jb1 : INT_MIN);
                if (lo <= hi) {
                    box.x0 = std::min(box.x0, lo), box.x1 = std::max(box.x1, hi);
                    box.y0 = std::min(box.y0, row), box.y1 = std::max(box.y1, row);
                }
            }
        }
    }

    uint8_t lut_[16] = {};
    std::vector<uint8_t> pattern_;  // hh x ww, all zero between contours
    Vec<uint32_t> slots_, marks_;
    Vec<int32_t> slot_of_;
    Vec<int64_t> x_of_, sorted_;
    size_t crossings_ = 0;
};

#ifdef MASKOPS_SAMPLED_NEON
// For each 4-bit keep mask, byte indices gathering the kept 8-byte vertices of
// a 32-byte block to its front, and the number kept.
struct CompactTable {
    uint8_t index[16][32];
    uint8_t count[16];
};
constexpr CompactTable make_compact_table() {
    CompactTable t{};
    for (int keep = 0; keep < 16; ++keep) {
        int k = 0;
        for (int p = 0; p < 4; ++p) {
            if (!(keep & (1 << p))) continue;
            for (int b = 0; b < 8; ++b) t.index[keep][8 * k + b] = uint8_t(8 * p + b);
            ++k;
        }
        t.count[keep] = uint8_t(k);
    }
    return t;
}
inline constexpr CompactTable kCompact = make_compact_table();

// load_contours for one float32 contour, four vertices per step: convert,
// check the envelope, and move the vertices that differ from their
// predecessor to the front with one table shuffle. Writes up to 8 int32
// values past the kept vertices.
inline size_t load_contour_f32(const float *p, size_t m, int32_t *out, uint32x4_t &ok) {
    const float32x4_t limit = vdupq_n_f32(float(kCoordLimit));
    const uint32x4_t lane_bits = {1, 2, 4, 8};
    const float limit1 = float(kCoordLimit);
    auto key_of = [&](float fx, float fy, bool &inside) {
        inside = std::fabs(fx) < limit1 && std::fabs(fy) < limit1;
        const int32_t x = int32_t(inside ? fx : 0.f), y = int32_t(inside ? fy : 0.f);
        return uint64_t(uint32_t(x)) | (uint64_t(uint32_t(y)) << 32);
    };
    bool inside;
    // A predecessor different from the first vertex keeps it.
    uint64x2_t previous = vdupq_n_u64(~key_of(p[0], p[1], inside));
    size_t k = 0, j = 0;
    for (; j + 4 <= m; j += 4) {
        const float32x4_t a = vld1q_f32(p + 2 * j), b = vld1q_f32(p + 2 * j + 4);
        ok = vandq_u32(ok, vandq_u32(vcaltq_f32(a, limit), vcaltq_f32(b, limit)));
        const uint64x2_t pa = vreinterpretq_u64_s32(vcvtq_s32_f32(a)), pb = vreinterpretq_u64_s32(vcvtq_s32_f32(b));
        const uint32x4_t same = vcombine_u32(vmovn_u64(vceqq_u64(pa, vextq_u64(previous, pa, 1))),
                                             vmovn_u64(vceqq_u64(pb, vextq_u64(pa, pb, 1))));
        const unsigned keep = ~vaddvq_u32(vandq_u32(same, lane_bits)) & 15u;
        const uint8x16x2_t block = {{vreinterpretq_u8_u64(pa), vreinterpretq_u8_u64(pb)}};
        uint8_t *dst = reinterpret_cast<uint8_t *>(out + 2 * k);
        vst1q_u8(dst, vqtbl2q_u8(block, vld1q_u8(kCompact.index[keep])));
        vst1q_u8(dst + 16, vqtbl2q_u8(block, vld1q_u8(kCompact.index[keep] + 16)));
        k += kCompact.count[keep];
        previous = pb;
    }
    uint64_t last = vgetq_lane_u64(previous, 1);
    for (; j < m; ++j) {
        const uint64_t key = key_of(p[2 * j], p[2 * j + 1], inside);
        if (!inside) ok = vdupq_n_u32(0);
        std::memcpy(out + 2 * k, &key, sizeof(key));
        k += key != last;
        last = key;
    }
    return k;
}
#endif

// Truncates float coordinates to int32 as NumPy's cast does and drops
// consecutive repeated vertices (zero-length LINE_8 edges whose pixel an
// incident edge already draws). points needs 2 * n * m + kPointSlack values
// and offsets n + 1. Returns false, leaving the outputs unusable, if any value
// is non-finite or outside the exact envelope. m must be positive if n is.
template <typename F>
inline bool load_contours(const F *xy, size_t n, size_t m, int32_t *points, int64_t *offsets) {
    offsets[0] = 0;
    size_t k = 0;
#ifdef MASKOPS_SAMPLED_NEON
    if constexpr (std::is_same_v<F, float>) {
        uint32x4_t ok = vdupq_n_u32(~0u);
        for (size_t i = 0; i < n; ++i) {
            k += load_contour_f32(xy + 2 * i * m, m, points + 2 * k, ok);
            offsets[i + 1] = int64_t(k);
        }
        return vminvq_u32(ok) != 0;
    }
#elif defined(MASKOPS_SAMPLED_X86)
    if constexpr (std::is_same_v<F, float>) {
        // Two vertices per step: one cvttps2dq, then a compaction whose only
        // loop-carried dependency is the running count.
        const __m128 limit = _mm_set1_ps(float(kCoordLimit));
        const __m128 magnitude = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        __m128 inside = _mm_castsi128_ps(_mm_set1_epi32(-1));
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            const float *p = xy + 2 * i * m;
            uint64_t previous = 0;
            size_t j = 0;
            for (; j + 2 <= m; j += 2) {
                const __m128 v = _mm_loadu_ps(p + 2 * j);
                inside = _mm_and_ps(inside, _mm_cmplt_ps(_mm_and_ps(v, magnitude), limit));  // false for NaN
                const __m128i c = _mm_cvttps_epi32(v);
                const auto a = uint64_t(_mm_cvtsi128_si64(c)), b = uint64_t(_mm_cvtsi128_si64(_mm_unpackhi_epi64(c, c)));
                std::memcpy(points + 2 * k, &a, sizeof(a));
                k += (a != previous) | (j == 0);
                std::memcpy(points + 2 * k, &b, sizeof(b));
                k += b != a;
                previous = b;
            }
            for (; j < m; ++j) {
                const float fx = p[2 * j], fy = p[2 * j + 1];
                const bool in = std::fabs(fx) < float(kCoordLimit) && std::fabs(fy) < float(kCoordLimit);
                ok &= in;
                const int32_t x = int32_t(in ? fx : 0.f), y = int32_t(in ? fy : 0.f);
                const uint64_t key = uint64_t(uint32_t(x)) | (uint64_t(uint32_t(y)) << 32);
                points[2 * k] = x, points[2 * k + 1] = y;
                k += (key != previous) | (j == 0);
                previous = key;
            }
            offsets[i + 1] = int64_t(k);
        }
        return ok && _mm_movemask_ps(inside) == 0xF;
    }
#endif
    bool ok = true;
    const F limit = F(kCoordLimit);
    for (size_t i = 0; i < n; ++i) {
        const F *p = xy + 2 * i * m;
        uint64_t previous = 0;
        for (size_t j = 0; j < m; ++j) {
            const F fx = p[2 * j], fy = p[2 * j + 1];
            const bool inside = std::fabs(fx) < limit && std::fabs(fy) < limit;  // false for NaN
            ok &= inside;
            const int32_t x = int32_t(inside ? fx : F(0)), y = int32_t(inside ? fy : F(0));
            const uint64_t key = uint64_t(uint32_t(x)) | (uint64_t(uint32_t(y)) << 32);
            points[2 * k] = x, points[2 * k + 1] = y;
            // The first vertex of each contour is always kept.
            k += (key != previous) | (j == 0);
            previous = key;
        }
        offsets[i + 1] = int64_t(k);
    }
    return ok;
}

// out[box] = mask ? value : out[box], i.e. np.maximum(out, mask * value) for
// binary masks and a value above every earlier one.
template <typename T>
inline void paint(T *__restrict out, int ww, const Box &b, const uint8_t *__restrict mask, T value) {
    const size_t bw = b.width();
    for (int y = b.y0; y <= b.y1; ++y) {
        T *__restrict o = out + size_t(y) * size_t(ww) + size_t(b.x0);
        const uint8_t *__restrict s = mask + size_t(y - b.y0) * bw;
        size_t x = 0;
#ifdef MASKOPS_SAMPLED_NEON
        if constexpr (std::is_same_v<T, uint8_t>) {
            const uint8x16_t fill = vdupq_n_u8(value);
            for (; x + 16 <= bw; x += 16) {
                const uint8x16_t m = vld1q_u8(s + x);
                vst1q_u8(o + x, vbslq_u8(vtstq_u8(m, m), fill, vld1q_u8(o + x)));
            }
        }
#elif defined(MASKOPS_SAMPLED_X86)
        if constexpr (std::is_same_v<T, uint8_t>) {
            const __m128i fill = _mm_set1_epi8(char(value)), zero = _mm_setzero_si128();
            for (; x + 16 <= bw; x += 16) {
                auto *target = reinterpret_cast<__m128i *>(o + x);
                const __m128i keep = _mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(s + x)), zero);
                _mm_storeu_si128(target, _mm_or_si128(_mm_and_si128(keep, _mm_loadu_si128(target)),
                                                      _mm_andnot_si128(keep, fill)));
            }
        }
#endif
        for (; x < bw; ++x) o[x] = s[x] ? value : o[x];
    }
}

}  // namespace maskops_sampled
