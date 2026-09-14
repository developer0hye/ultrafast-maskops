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
#include <array>
#include <bit>
#include <climits>
#include <concepts>
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
#if defined(__aarch64__) || defined(_M_ARM64)
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

// The vector paths reinterpret int32 lanes as bytes (table lookups, key
// packing and compaction), which assumes little-endian lanes.
static_assert(std::endian::native == std::endian::little, "the sampled kernel assumes a little-endian host");

// Every vector path below has a scalar twin that computes the same bytes.
// ULTRAFAST_MASKOPS_SCALAR=1 in the environment at import selects the scalar
// twins everywhere, so the tests can cover them on hosts that have SIMD.
inline bool simd_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("ULTRAFAST_MASKOPS_SCALAR");
        return value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0;
    }();
    return enabled;
}

inline const char *simd_mode() {
    if (!simd_enabled()) return "scalar";
#if defined(MASKOPS_SAMPLED_NEON)
    return "neon";
#elif defined(MASKOPS_SAMPLED_X86)
    return "sse2";
#else
    return "scalar";
#endif
}


#ifdef MASKOPS_ABLATE
#include <mach/mach_time.h>
// Profiling builds only: phase timers (ns) and MASKOPS_ABLATE=<bits> skips.
inline uint64_t (&phase_ns())[16] { static uint64_t t[16]; return t; }
inline uint64_t now_ns() { return mach_absolute_time(); }  // 1 tick = 41.67 ns on Apple silicon
#define MASKOPS_TICK(i, since) do { const uint64_t _n = ::maskops_sampled::now_ns(); ::maskops_sampled::phase_ns()[i] += _n - (since); (since) = _n; } while (0)
inline unsigned ablate() {
    static const unsigned bits = [] { const char *v = std::getenv("MASKOPS_ABLATE"); return v ? unsigned(std::atoi(v)) : 0u; }();
    return bits;
}
#define MASKOPS_SKIP(bit) if (::maskops_sampled::ablate() & (bit))
#define MASKOPS_ABLATED(bit) (::maskops_sampled::ablate() & (bit))
#else
#define MASKOPS_SKIP(bit) if (false)
#define MASKOPS_ABLATED(bit) false
#define MASKOPS_TICK(i, since) do {} while (0)
#endif

constexpr int kShift = 16;  // XY_SHIFT
constexpr int64_t kOne = int64_t(1) << kShift;
// Envelope in which every fixed-point product below stays far inside int64 and
// every clipLine double product is exact: |coordinate| < 2^24, side <= 2^15.
constexpr double kCoordLimit = 16777216.0;
constexpr int kSideLimit = 1 << 15;
// Contour layout in the point buffer: each contour is preceded by a copy of
// its last vertex and followed by a copy of its first (so a pass may read the
// neighbours of every vertex without branches), and the buffer ends with
// kPointSlack readable int32 values that vector passes may read or, during
// loading, overwrite before the pads are written.
constexpr int64_t kPadVertices = 2;
constexpr size_t kPointSlack = 16;

// Writes the two pad vertices of a contour of n vertices starting at v.
inline void pad_contour(int32_t *v, int64_t n) {
    v[-2] = v[2 * n - 2], v[-1] = v[2 * n - 1];
    v[2 * n] = v[0], v[2 * n + 1] = v[1];
}

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
    if (simd_enabled()) {
        const uint8x16_t v = vdupq_n_u8(bits);
        for (; len >= 16; len -= 16, q += 16) vst1q_u8(q, vorrq_u8(vld1q_u8(q), v));
        if (len) vst1q_u8(q, vorrq_u8(vld1q_u8(q), vandq_u8(v, vld1q_u8(kPrefix.m[len]))));
        return;
    }
#elif defined(MASKOPS_SAMPLED_X86)
    if (simd_enabled()) {
        const __m128i v = _mm_set1_epi8(char(bits));
        auto *w = reinterpret_cast<__m128i *>(q);
        for (; len >= 16; len -= 16, ++w) _mm_storeu_si128(w, _mm_or_si128(_mm_loadu_si128(w), v));
        if (len) {
            const __m128i prefix = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kPrefix.m[len]));
            _mm_storeu_si128(w, _mm_or_si128(_mm_loadu_si128(w), _mm_and_si128(v, prefix)));
        }
        return;
    }
#endif
    for (; len > 0; --len, ++q) *q |= bits;
}

// The pattern bits of one filled span [x1, x2] (source columns) on a
// sampled row: output columns whose source column 4j+1 lies in the span take
// bit a, those whose 4j+2 does take bit b. The two column ranges start and
// end at most one apart, so the run is a middle of both bits with at most one
// single-bit column at each end. One vector OR per 16 columns; the bytes
// outside the run receive zero bits (the row buffers keep 16 bytes of slack).
inline void or_span(uint8_t *line, int x1, int x2, uint8_t a, uint8_t b, int &lo, int &hi) {
    const int ja0 = (x1 + 2) >> 2, ja1 = (x2 - 1) >> 2, jb0 = (x1 + 1) >> 2, jb1 = (x2 - 2) >> 2;
    const bool has_a = ja0 <= ja1, has_b = jb0 <= jb1;
    if (!has_a && !has_b) return;
    lo = has_b ? jb0 : ja0, hi = has_a ? ja1 : jb1;  // jb0 <= ja0 and jb1 <= ja1
    const int a0 = has_a ? ja0 : INT_MAX, a1 = has_a ? ja1 : INT_MIN, b0 = has_b ? jb0 : INT_MAX, b1 = has_b ? jb1 : INT_MIN;
#ifdef MASKOPS_SAMPLED_NEON
    if (simd_enabled()) {
        const int8x16_t lane = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        const uint8x16_t va = vdupq_n_u8(a), vb = vdupq_n_u8(b);
        for (int j = lo; j <= hi; j += 16) {
            auto rel = [j](int bound) { return int8_t(std::clamp(bound - j, -1, 16)); };
            const uint8x16_t in_a = vandq_u8(vcgeq_s8(lane, vdupq_n_s8(rel(a0))), vcleq_s8(lane, vdupq_n_s8(rel(a1))));
            const uint8x16_t in_b = vandq_u8(vcgeq_s8(lane, vdupq_n_s8(rel(b0))), vcleq_s8(lane, vdupq_n_s8(rel(b1))));
            const uint8x16_t bits = vorrq_u8(vandq_u8(in_a, va), vandq_u8(in_b, vb));
            vst1q_u8(line + j, vorrq_u8(vld1q_u8(line + j), bits));
        }
        return;
    }
#elif defined(MASKOPS_SAMPLED_X86)
    if (simd_enabled()) {
        const __m128i lane = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const __m128i va = _mm_set1_epi8(char(a)), vb = _mm_set1_epi8(char(b));
        for (int j = lo; j <= hi; j += 16) {
            auto rel = [j](int bound) { return char(std::clamp(bound - j, -1, 16)); };
            // lane >= r  <=>  !(lane < r)  <=>  !(r > lane)
            auto within = [&](int r0, int r1) {
                const __m128i below = _mm_cmpgt_epi8(_mm_set1_epi8(rel(r0)), lane), above = _mm_cmpgt_epi8(lane, _mm_set1_epi8(rel(r1)));
                return _mm_andnot_si128(_mm_or_si128(below, above), _mm_set1_epi8(-1));
            };
            const __m128i bits = _mm_or_si128(_mm_and_si128(within(a0, a1), va), _mm_and_si128(within(b0, b1), vb));
            auto *p = reinterpret_cast<__m128i *>(line + j);
            _mm_storeu_si128(p, _mm_or_si128(_mm_loadu_si128(p), bits));
        }
        return;
    }
#endif
    for (int j = lo; j <= hi; ++j) line[j] |= uint8_t(((j >= a0 && j <= a1) ? a : 0) | ((j >= b0 && j <= b1) ? b : 0));
}

// Pattern bit of pixel (x, y) indexed by (y & 3) * 4 + (x & 3). Rows 4k+1/4k+2
// and columns 4j+1/4j+2 give bits 8, 4 (upper row) and 2, 1 (lower row).
constexpr uint8_t kBit[16] = {0, 0, 0, 0, 0, 8, 4, 0, 0, 2, 1, 0, 0, 0, 0, 0};
// Distance from y (by y & 3) to the first sampled row at or below it.
constexpr int kFirstSampled[4] = {1, 0, 0, 2};

// For each 4-bit keep mask, byte indices gathering the kept 4-byte lanes of a
// 16-byte vector to its front, and the number kept.
struct Compact4 {
    uint8_t index[16][16];
    uint8_t count[16];
};
constexpr Compact4 make_compact4() {
    Compact4 t{};
    for (int keep = 0; keep < 16; ++keep) {
        int k = 0;
        for (int lane = 0; lane < 4; ++lane) {
            if (!(keep & (1 << lane))) continue;
            for (int b = 0; b < 4; ++b) t.index[keep][4 * k + b] = uint8_t(4 * lane + b);
            ++k;
        }
        for (int b = 4 * k; b < 16; ++b) t.index[keep][b] = 0xFF;  // zero lanes
        t.count[keep] = uint8_t(k);
    }
    return t;
}
inline constexpr Compact4 kCompact4 = make_compact4();

// Sampled-row crossings of one contour, bucketed by slot (2 per output row).
// A slot holding more than kBucket crossings makes the contour fall back to
// the sorting path; dense self-intersecting outlines can do that.
constexpr int kBucket = 8;

inline void bounds(const int32_t *v, int n, int &x0, int &y0, int &x1, int &y1) {
    x0 = y0 = INT_MAX, x1 = y1 = INT_MIN;
    int i = 0;
#ifdef MASKOPS_SAMPLED_NEON
    if (n >= 2 && simd_enabled()) {
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
            bucket_n_.assign(size_t(2 * hh) + 3, 0);  // + three dummy slots
            bucket_.resize((size_t(2 * hh) + 3) * kBucket);
            masks_.assign(4 * (size_t(ww) / 64 + 1), 0);  // four column bit sets, zero between rows
        }
        std::memcpy(lut_, table, sizeof(lut_));
    }

    // ORs one contour's sampled pixels into the pattern image and returns the
    // box that may hold nonzero patterns. Vertices must lie in the envelope
    // and in the padded layout described at kPadVertices.
    Box render(const int32_t *v, int n) {
        Box box;
        if (n <= 0) return box;
        int vx0, vy0, vx1, vy1;
        uint64_t since = 0;
#ifdef MASKOPS_ABLATE
        since = now_ns();
#endif
        bounds(v, n, vx0, vy0, vx1, vy1);
        MASKOPS_TICK(1, since);
        // A contour entirely beside the image draws nothing: its lines stay in
        // the vertex box, clipLine leaves its edges unchanged, and every span
        // falls outside the image.
        if (vx1 < 0 || vx0 >= w || vy1 < 0 || vy0 >= h) return box;
        box = {std::max(vx0, 0) >> 2, std::max(vy0, 0) >> 2, std::min(vx1, w - 1) >> 2, std::min(vy1, h - 1) >> 2};
        MASKOPS_SKIP(16) return box;
        if (!render_fast(v, n, box) && crossings(v, n)) fill(box);
        return box;
    }

    // Writes table[pattern] for the box to dst (row stride in bytes), clears
    // the box in the pattern image and returns the sum of written values.
    uint64_t emit(const Box &b, uint8_t *dst, ptrdiff_t stride) {
        if (b.empty()) return 0;
#ifdef MASKOPS_SAMPLED_X86
        if (simd_enabled() && has_ssse3()) return emit_ssse3(b, dst, stride);
#endif
        uint64_t area = 0;
        const int bw = int(b.width());
#ifdef MASKOPS_SAMPLED_NEON
        const bool vector = simd_enabled();
        const uint8x16_t table = vld1q_u8(lut_), zero = vdupq_n_u8(0);
#endif
        for (int y = b.y0; y <= b.y1; ++y) {
            uint8_t *p = pattern_.data() + size_t(y) * size_t(ww) + size_t(b.x0);
            uint8_t *d = dst + ptrdiff_t(y - b.y0) * stride;
            int x = 0;
#ifdef MASKOPS_SAMPLED_NEON
            while (vector && x + 16 <= bw) {
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

    // One pass over the contour. For every vertex b, with its neighbours a
    // before and c after read through the pads, it collects in contour order
    // and without branches:
    // - the mark of b when b is inside the image, on a sampled row and column,
    //   and not the upper end of an edge to a strictly lower neighbour
    //   (a.y <= b.y and c.y <= b.y). Such an upper end is drawn by the fill:
    //   the edge's crossing on b's row is exactly b.x whether or not the
    //   other end is clipped, since clipLine leaves an inside endpoint alone;
    // - the crossing of the edge a->b on its single sampled row when both
    //   ends are inside and the edge is short (|dx| <= 1 and |dy| <= 1, the
    //   common case of resampled outlines) and not horizontal: at row
    //   min(a.y, b.y), x = the upper end's x in 16.16 fixed point;
    // - the index of b for every other edge (longer, or with an end outside
    //   the image), for the scalar line and crossing code.
    // NEON handles four vertices per step and compacts each stream with a
    // table shuffle; the scalar twin computes the same streams.
#ifdef MASKOPS_SAMPLED_X86
    // Lambdas do not inherit a function's target attribute under GCC, so the
    // SSSE3 compaction is a separate attributed function.
    MASKOPS_TARGET_SSSE3 static void compact_ssse3(__m128i values, unsigned keep, int32_t *out) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out),
                         _mm_shuffle_epi8(values, _mm_loadu_si128(reinterpret_cast<const __m128i *>(kCompact4.index[keep]))));
    }
    // pass() with SSE2 and the SSSE3 byte shuffle (table lookup and
    // compaction), four vertices per step, the same streams as the NEON code.
    MASKOPS_TARGET_SSSE3 void pass_ssse3(const int32_t *v, int n, uint32_t *marks, int32_t *slots, int32_t *xs,
                                         int32_t *longs) {
        const int WW = ww, W = w, H = h;
        const __m128i zero = _mm_setzero_si128(), one = _mm_set1_epi32(1), three = _mm_set1_epi32(3);
        const __m128i two = _mm_set1_epi32(2), lanes = _mm_setr_epi32(0, 1, 2, 3), count = _mm_set1_epi32(n);
        const __m128i vw = _mm_set1_epi32(W), vh = _mm_set1_epi32(H), x_limit = _mm_set1_epi32(1 << 15);
        const __m128i bit_table = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kBit));
        const __m128i width = _mm_set1_epi32(WW), all = _mm_set1_epi32(-1);
        size_t nm = 0, nc = 0, nl = 0;
        for (int i = 0; i < n; i += 4) {
            // Four vertices and their neighbours, de-interleaved into x and y lanes.
            auto split = [](const int32_t *p, __m128i &x, __m128i &y) {
                const __m128 a = _mm_loadu_ps(reinterpret_cast<const float *>(p));
                const __m128 b = _mm_loadu_ps(reinterpret_cast<const float *>(p + 4));
                x = _mm_castps_si128(_mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)));
                y = _mm_castps_si128(_mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)));
            };
            __m128i x, y, px, py, nx, ny;
            split(v + 2 * i, x, y), split(v + 2 * (i - 1), px, py), split(v + 2 * (i + 1), nx, ny);
            const __m128i valid = _mm_cmplt_epi32(_mm_add_epi32(_mm_set1_epi32(i), lanes), count);
            auto inside = [&](__m128i cx, __m128i cy) {  // 0 <= cx < W and 0 <= cy < H
                const __m128i ox = _mm_andnot_si128(_mm_cmplt_epi32(cx, zero), _mm_cmplt_epi32(cx, vw));
                const __m128i oy = _mm_andnot_si128(_mm_cmplt_epi32(cy, zero), _mm_cmplt_epi32(cy, vh));
                return _mm_and_si128(ox, oy);
            };
            const __m128i b_in = _mm_and_si128(inside(x, y), valid), both_in = _mm_and_si128(b_in, inside(px, py));
            const __m128i dx = _mm_sub_epi32(x, px), dy = _mm_sub_epi32(y, py);
            const __m128i is_short = _mm_and_si128(_mm_cmplt_epi32(_mm_abs_epi32(dx), two), _mm_cmplt_epi32(_mm_abs_epi32(dy), two));
            const __m128i above = _mm_and_si128(_mm_cmplt_epi32(y, zero), _mm_cmplt_epi32(py, zero));
            const __m128i below = _mm_andnot_si128(_mm_or_si128(_mm_cmplt_epi32(y, vh), _mm_cmplt_epi32(py, vh)), all);
            const __m128i left = _mm_and_si128(_mm_cmplt_epi32(x, zero), _mm_cmplt_epi32(px, zero));
            const __m128i right = _mm_andnot_si128(_mm_or_si128(_mm_cmplt_epi32(x, vw), _mm_cmplt_epi32(px, vw)), all);
            const __m128i small = _mm_and_si128(_mm_cmplt_epi32(_mm_abs_epi32(x), x_limit), _mm_cmplt_epi32(_mm_abs_epi32(px), x_limit));
            const __m128i beside = _mm_andnot_si128(_mm_or_si128(above, below), _mm_and_si128(_mm_or_si128(left, right), small));
            const __m128i discard = _mm_and_si128(valid, _mm_or_si128(above, below));
            // Marks through the byte table; bytes 1..3 of each lane index entry 0.
            const __m128i index = _mm_or_si128(_mm_slli_epi32(_mm_and_si128(y, three), 2), _mm_and_si128(x, three));
            const __m128i bit = _mm_shuffle_epi8(bit_table, index);
            __m128i mark_ok = _mm_andnot_si128(_mm_or_si128(_mm_cmpgt_epi32(py, y), _mm_cmpgt_epi32(ny, y)), b_in);
            mark_ok = _mm_andnot_si128(_mm_cmpeq_epi32(bit, zero), mark_ok);
            // (y >> 2) * WW with SSE2: two 32x32->64 products of the even and odd lanes.
            const __m128i row = _mm_srai_epi32(y, 2);
            const __m128i even = _mm_mul_epu32(row, width), odd = _mm_mul_epu32(_mm_srli_si128(row, 4), width);
            const __m128i product = _mm_unpacklo_epi32(_mm_shuffle_epi32(even, _MM_SHUFFLE(0, 0, 2, 0)),
                                                       _mm_shuffle_epi32(odd, _MM_SHUFFLE(0, 0, 2, 0)));
            const __m128i cell = _mm_add_epi32(product, _mm_srai_epi32(x, 2));
            unsigned keep = unsigned(_mm_movemask_ps(_mm_castsi128_ps(mark_ok)));
            compact_ssse3(_mm_or_si128(_mm_slli_epi32(cell, 4), bit), keep, reinterpret_cast<int32_t *>(marks + nm));
            nm += kCompact4.count[keep];
            // Crossings of short, non-horizontal edges on a sampled row inside the image.
            const __m128i y_lt = _mm_cmplt_epi32(y, py);
            const __m128i y0 = _mm_or_si128(_mm_and_si128(y_lt, y), _mm_andnot_si128(y_lt, py));
            const __m128i slot = _mm_or_si128(_mm_slli_epi32(_mm_srai_epi32(y0, 2), 1), _mm_and_si128(_mm_srai_epi32(y0, 1), one));
            const __m128i phase = _mm_sub_epi32(_mm_and_si128(y0, three), one);
            const __m128i sampled = _mm_andnot_si128(_mm_cmplt_epi32(phase, zero), _mm_cmplt_epi32(phase, two));
            const __m128i dy_pos = _mm_cmpgt_epi32(dy, zero);
            const __m128i xc = _mm_slli_epi32(_mm_or_si128(_mm_and_si128(dy_pos, px), _mm_andnot_si128(dy_pos, x)), kShift);
            const __m128i easy = _mm_and_si128(_mm_and_si128(is_short, _mm_or_si128(both_in, beside)), valid);
            const __m128i row_in = _mm_andnot_si128(_mm_cmplt_epi32(y0, zero), _mm_cmplt_epi32(y0, vh));
            const __m128i cross_ok = _mm_andnot_si128(_mm_cmpeq_epi32(dy, zero), _mm_and_si128(_mm_and_si128(easy, sampled), row_in));
            keep = unsigned(_mm_movemask_ps(_mm_castsi128_ps(cross_ok)));
            compact_ssse3(slot, keep, slots + nc);
            compact_ssse3(xc, keep, xs + nc);
            nc += kCompact4.count[keep];
            // Indices of the vertices ending every other edge.
            keep = unsigned(_mm_movemask_ps(_mm_castsi128_ps(_mm_andnot_si128(_mm_or_si128(easy, discard), valid))));
            compact_ssse3(_mm_add_epi32(_mm_set1_epi32(i), lanes), keep, longs + nl);
            nl += kCompact4.count[keep];
        }
        marks_n_ = nm, cross_n_ = nc, long_n_ = nl;
    }
#endif

    void pass(const int32_t *v, int n) {
        const size_t room = size_t(n) + 8;
        if (marks_.size() < room) marks_.resize(room);
        if (cslot_.size() < room) cslot_.resize(room), cx_.resize(room);
        if (long_.size() < room) long_.resize(room);
        uint32_t *marks = marks_.data();
        int32_t *slots = cslot_.data(), *xs = cx_.data(), *longs = long_.data();
        size_t nm = 0, nc = 0, nl = 0;
        const int WW = ww, W = w, H = h;
#ifdef MASKOPS_SAMPLED_NEON
        if (simd_enabled()) {
            const int32x4_t one = vdupq_n_s32(1), three = vdupq_n_s32(3), zero = vdupq_n_s32(0);
            const int32x4_t lanes = {0, 1, 2, 3}, count = vdupq_n_s32(n), width = vdupq_n_s32(WW);
            const uint32x4_t lane_bits = {1, 2, 4, 8}, two = vdupq_n_u32(2);
            const uint32x4_t uw = vdupq_n_u32(unsigned(W)), uh = vdupq_n_u32(unsigned(H));
            const int32x4_t x_limit = vdupq_n_s32(1 << 15);  // x << 16 must fit int32
            const uint8x16_t bit_table = vld1q_u8(kBit);
            auto keep_of = [&](uint32x4_t m) { return vaddvq_u32(vandq_u32(m, lane_bits)); };
            auto compact = [&](int32x4_t values, unsigned keep, int32_t *out) {
                vst1q_u8(reinterpret_cast<uint8_t *>(out),
                         vqtbl1q_u8(vreinterpretq_u8_s32(values), vld1q_u8(kCompact4.index[keep])));
            };
            auto inside = [&](int32x4_t x, int32x4_t y) {  // 0 <= x < W and 0 <= y < H, unsigned
                return vandq_u32(vcltq_u32(vreinterpretq_u32_s32(x), uw), vcltq_u32(vreinterpretq_u32_s32(y), uh));
            };
            for (int i = 0; i < n; i += 4) {
                const int32x4x2_t cur = vld2q_s32(v + 2 * i), prev = vld2q_s32(v + 2 * (i - 1));
                const int32x4x2_t next = vld2q_s32(v + 2 * (i + 1));
                const int32x4_t x = cur.val[0], y = cur.val[1], px = prev.val[0], py = prev.val[1];
                const int32x4_t ny = next.val[1];
                const uint32x4_t valid = vcltq_s32(vaddq_s32(vdupq_n_s32(i), lanes), count);
                const uint32x4_t b_in = vandq_u32(inside(x, y), valid), both_in = vandq_u32(b_in, inside(px, py));
                const int32x4_t dx = vsubq_s32(x, px), dy = vsubq_s32(y, py);
                const uint32x4_t is_short = vandq_u32(vcleq_s32(vabsq_s32(dx), one), vcleq_s32(vabsq_s32(dy), one));
                // Edges entirely above or below the image touch no row, and
                // edges entirely left or right of it only need a crossing on
                // that side (its exact value never affects a clipped span).
                const uint32x4_t above = vandq_u32(vcltq_s32(y, zero), vcltq_s32(py, zero));
                const uint32x4_t below = vandq_u32(vcgeq_s32(y, vreinterpretq_s32_u32(uh)), vcgeq_s32(py, vreinterpretq_s32_u32(uh)));
                const uint32x4_t left = vandq_u32(vcltq_s32(x, zero), vcltq_s32(px, zero));
                const uint32x4_t right = vandq_u32(vcgeq_s32(x, vreinterpretq_s32_u32(uw)), vcgeq_s32(px, vreinterpretq_s32_u32(uw)));
                const uint32x4_t small = vandq_u32(vcltq_s32(vabsq_s32(x), x_limit), vcltq_s32(vabsq_s32(px), x_limit));
                const uint32x4_t beside = vandq_u32(vandq_u32(vorrq_u32(left, right), small), vbicq_u32(vdupq_n_u32(~0u), vorrq_u32(above, below)));
                const uint32x4_t discard = vandq_u32(valid, vorrq_u32(above, below));
                // Marks: the pattern bit of (x, y) through a byte table; bytes
                // 1..3 of each lane index entry 0, which is zero.
                const int32x4_t index = vorrq_s32(vshlq_n_s32(vandq_s32(y, three), 2), vandq_s32(x, three));
                const int32x4_t bit = vreinterpretq_s32_u8(vqtbl1q_u8(bit_table, vreinterpretq_u8_s32(index)));
                uint32x4_t mark_ok = vandq_u32(vcleq_s32(py, y), vcleq_s32(ny, y));
                mark_ok = vandq_u32(vandq_u32(mark_ok, vtstq_s32(bit, bit)), b_in);
                const int32x4_t cell = vaddq_s32(vmulq_s32(vshrq_n_s32(y, 2), width), vshrq_n_s32(x, 2));
                unsigned keep = keep_of(mark_ok);
                compact(vorrq_s32(vshlq_n_s32(cell, 4), bit), keep, reinterpret_cast<int32_t *>(marks + nm));
                nm += kCompact4.count[keep];
                // Crossings of short, non-horizontal, inside edges on a sampled row.
                const int32x4_t y0 = vminq_s32(y, py);
                const int32x4_t slot = vorrq_s32(vshlq_n_s32(vshrq_n_s32(y0, 2), 1), vandq_s32(vshrq_n_s32(y0, 1), one));
                const uint32x4_t sampled = vcltq_u32(vreinterpretq_u32_s32(vsubq_s32(vandq_s32(y0, three), one)), two);
                const int32x4_t xc = vshlq_n_s32(vbslq_s32(vcgtq_s32(dy, zero), px, x), kShift);
                const uint32x4_t easy = vandq_u32(vandq_u32(is_short, vorrq_u32(both_in, beside)), valid);
                const uint32x4_t row_in = vcltq_u32(vreinterpretq_u32_s32(y0), uh);
                const uint32x4_t cross_ok = vandq_u32(vandq_u32(easy, sampled), vandq_u32(vtstq_s32(dy, dy), row_in));
                keep = keep_of(cross_ok);
                compact(slot, keep, slots + nc);
                compact(xc, keep, xs + nc);
                nc += kCompact4.count[keep];
                // Indices of the vertices ending every other edge.
                keep = keep_of(vbicq_u32(valid, vorrq_u32(easy, discard)));
                compact(vaddq_s32(vdupq_n_s32(i), lanes), keep, longs + nl);
                nl += kCompact4.count[keep];
            }
            marks_n_ = nm, cross_n_ = nc, long_n_ = nl;
            return;
        }
#elif defined(MASKOPS_SAMPLED_X86)
        if (simd_enabled() && has_ssse3()) {
            pass_ssse3(v, n, marks, slots, xs, longs);
            return;
        }
#endif
        for (int i = 0; i < n; ++i) {
            const int x = v[2 * i], y = v[2 * i + 1], px = v[2 * i - 2], py = v[2 * i - 1], ny = v[2 * i + 3];
            const bool b_in = unsigned(x) < unsigned(W) && unsigned(y) < unsigned(H);
            const bool both_in = b_in && unsigned(px) < unsigned(W) && unsigned(py) < unsigned(H);
            const int dx = x - px, dy = y - py;
            const bool above_below = (y < 0 && py < 0) || (y >= H && py >= H);
            const bool beside = ((x < 0 && px < 0) || (x >= W && px >= W)) && !above_below &&
                                std::abs(x) < (1 << 15) && std::abs(px) < (1 << 15);
            const bool is_short = unsigned(dx + 1) <= 2u && unsigned(dy + 1) <= 2u;
            const bool easy = is_short && (both_in || beside);
            const uint32_t bit = kBit[((y & 3) << 2) | (x & 3)];
            marks[nm] = (uint32_t((y >> 2) * WW + (x >> 2)) << 4) | bit;
            nm += (py <= y) & (ny <= y) & (bit != 0) & b_in;
            const int y0 = std::min(y, py);
            slots[nc] = ((y0 >> 2) << 1) | ((y0 >> 1) & 1);
            xs[nc] = (dy > 0 ? px : x) << kShift;
            nc += easy & (unsigned((y0 & 3) - 1) < 2u) & (dy != 0) & (unsigned(y0) < unsigned(H));
            longs[nl] = i;
            nl += !(easy || above_below);
        }
        marks_n_ = nm, cross_n_ = nc, long_n_ = nl;
    }

    // Pixels, crossings and fill of a contour via pass() and the crossing
    // buckets. False when a bucket overflowed; the caller then computes the
    // crossings with crossings() and fills with fill(). The marks and line
    // pixels already applied are the same ones those apply.
    bool render_fast(const int32_t *v, int n, Box &box) {
        uint64_t since = 0;
#ifdef MASKOPS_ABLATE
        since = now_ns();
#endif
        pass(v, n);
        MASKOPS_TICK(2, since);
        MASKOPS_SKIP(64) return true;
        uint8_t *const pat = pattern_.data();
        const uint32_t *marks = marks_.data();
        MASKOPS_SKIP(1) marks_n_ = 0;
        for (size_t j = 0; j < marks_n_; ++j) pat[marks[j] >> 4] |= uint8_t(marks[j] & 15);
        MASKOPS_TICK(3, since);
        int64_t *const bucket = bucket_.data();
        uint8_t *const bucket_n = bucket_n_.data();
        int smin = INT_MAX, smax = INT_MIN;
        unsigned overflow = 0;
        auto push = [&](int slot, int64_t x) {
            const unsigned k = bucket_n[slot];
            bucket[size_t(slot) * kBucket + (k & (kBucket - 1))] = x;
            bucket_n[slot] = uint8_t(k + 1);
            overflow |= k >> 3;  // k >= kBucket
            smin = std::min(smin, slot), smax = std::max(smax, slot);
        };
        const int32_t *slots = cslot_.data(), *xs = cx_.data();
        MASKOPS_SKIP(2) cross_n_ = 0;
        for (size_t c = 0; c < cross_n_; ++c) push(slots[c], xs[c]);
        MASKOPS_TICK(4, since);
        // Every other edge: CollectPolyEdges' line, clipping and crossings.
        const int WW = ww, W = w, H = h;
        auto mark = [pat, WW](int x, int y) { pat[(y >> 2) * WW + (x >> 2)] |= kBit[((y & 3) << 2) | (x & 3)]; };
        MASKOPS_SKIP(4) long_n_ = 0;
        // Edges the pass left out. Most are inside and only a few pixels
        // long, so their pixels and crossings are computed without branches:
        // LineIterator's set-up with selects, four interior pixels whose
        // writes go to a dummy byte when past the end, and three candidate
        // sampled rows whose pushes go to a dummy slot when past the edge.
        // Distinct dummies keep the invalid candidates' read-modify-writes off
        // one another's store-to-load chain.
        const int dummy_slot = 2 * hh;
        uint8_t dummy_bytes[4] = {0, 0, 0, 0};
        auto push_if = [&](bool valid, int slot, int64_t x, int dummy) {
            slot = valid ? slot : dummy_slot + dummy;
            const unsigned k = bucket_n[slot];
            bucket[size_t(slot) * kBucket + (k & (kBucket - 1))] = x;
            bucket_n[slot] = uint8_t(k + 1);
            overflow |= (k >> 3) & unsigned(valid);
            smin = std::min(smin, valid ? slot : INT_MAX), smax = std::max(smax, valid ? slot : INT_MIN);
        };
        for (size_t l = 0; l < long_n_; ++l) {
            const int i = long_[l];
            const int ax = v[2 * i - 2], ay = v[2 * i - 1], bx = v[2 * i], by = v[2 * i + 1];
            const int dx0 = bx - ax, dy0 = by - ay;
            const bool both_in = unsigned(ax) < unsigned(W) && unsigned(bx) < unsigned(W) &&
                                 unsigned(ay) < unsigned(H) && unsigned(by) < unsigned(H);
#ifdef MASKOPS_ABLATE
            phase_ns()[8 + (both_in ? (std::max(std::abs(dx0), std::abs(dy0)) <= 5 ? 0 : 1) : 2)] += 24;  // counts in ticks
#endif
            if (both_in && std::max(std::abs(dx0), std::abs(dy0)) <= 5 && !MASKOPS_ABLATED(128u)) [[likely]] {
                // LineIterator(connectivity 8, leftToRight): the left end
                // starts; the major axis advances every pixel, the minor
                // axis when the error term is negative.
                const bool swap_ends = dx0 < 0;
                const int x0 = swap_ends ? bx : ax, y0 = swap_ends ? by : ay;
                const int adx = swap_ends ? -dx0 : dx0, dy1 = swap_ends ? -dy0 : dy0;
                const int sy = dy1 < 0 ? -1 : 1, ady = dy1 < 0 ? -dy1 : dy1;
                const bool vert = ady > adx;
                const int major = vert ? ady : adx, minor = vert ? adx : ady;
                const int mjx = vert ? 0 : 1, mjy = vert ? sy : 0, mnx = vert ? 1 : 0, mny = vert ? 0 : sy;
                int err = major - 2 * minor, x = x0, y = y0;
                for (int s = 1; s <= 4; ++s) {  // interior pixels 1 .. major - 1
                    const int mask = err < 0 ? -1 : 0;
                    err += -2 * minor + ((2 * major) & mask);
                    x += mjx + (mnx & mask), y += mjy + (mny & mask);
                    uint8_t *target = s < major ? pat + (y >> 2) * WW + (x >> 2) : dummy_bytes + (s - 1);
                    *target |= kBit[((y & 3) << 2) | (x & 3)];
                }
                const int ymin = std::min(ay, by), ymax = std::max(ay, by);
                const int64_t dx = (int64_t(dx0) * kOne) / (dy0 | (dy0 == 0));  // unused when horizontal
                const int64_t xu = int64_t(dy0 > 0 ? ax : bx) * kOne;
                int row = ymin + kFirstSampled[ymin & 3];
                for (int s = 0; s < 3; ++s) {  // at most three sampled rows in five
                    push_if(row < ymax, ((row >> 2) << 1) | ((row >> 1) & 1), xu + int64_t(row - ymin) * dx, s);
                    row += (row & 1) ? 1 : 3;
                }
                continue;
            }
            int64_t cx0, cx1, cy0 = ay, cy1 = by;
            if (both_in) {
                line8_unclipped<true>(ax, ay, bx, by, mark);  // the ends are marked or filled
                cx0 = int64_t(ax) * kOne, cx1 = int64_t(bx) * kOne;
            } else {
                // Line() clips the same way; an invisible line draws nothing
                // but its clipped ends still shape the edge.
                Point64 t0{ax, ay}, t1{bx, by};
                if (clip_line(W, H, t0, t1)) line8_unclipped<false>(int(t0.x), int(t0.y), int(t1.x), int(t1.y), mark);
                if (t0.y != t1.y) cy0 = t0.y, cy1 = t1.y;
                cx0 = t0.x * kOne, cx1 = t1.x * kOne;
            }
            if (ay == by) continue;
            const int64_t dx = (cx1 - cx0) / (cy1 - cy0);
            int y0, y1;
            int64_t x;
            if (ay < by)
                y0 = ay, y1 = by, x = cx0 + (ay - cy0) * dx;
            else
                y0 = by, y1 = ay, x = cx1 + (by - cy1) * dx;
            int y = std::max(y0, 0);
            const int end = std::min(y1, H);
            if (y >= end) continue;
            for (y += kFirstSampled[y & 3]; y < end; y += (y & 1) ? 1 : 3)
                push(((y >> 2) << 1) | ((y >> 1) & 1), x + int64_t(y - y0) * dx);
        }
        bucket_n[dummy_slot] = bucket_n[dummy_slot + 1] = bucket_n[dummy_slot + 2] = 0;
        MASKOPS_TICK(5, since);
        if (smin > smax) return true;  // no crossings on sampled rows
        MASKOPS_SKIP(8) { std::fill(bucket_n + smin, bucket_n + smax + 1, uint8_t(0)); return true; }
        if (overflow) {
            std::fill(bucket_n + smin, bucket_n + smax + 1, uint8_t(0));
            return false;
        }
        // Fill one output row at a time: the spans of its two slots set bits in
        // four column bit sets (one per pattern bit), which are then expanded
        // to pattern bytes 16 columns at a time. Each pattern byte is read and
        // written once per row, so no vector store is reloaded by the next
        // span (a partial overlap the core cannot forward).
        // W declared above.
        uint64_t *const masks = masks_.data();
        const size_t words = masks_.size() / 4;
        for (int slot = smin & ~1; slot <= smax; slot += 2) {
            const int row = slot >> 1;
            int lo = INT_MAX, hi = INT_MIN;
            for (int half = 0; half < 2; ++half) {
                const unsigned count = bucket_n[slot + half];
                bucket_n[slot + half] = 0;
                if (count < 2) continue;
                int64_t *b = bucket + size_t(slot + half) * kBucket;
                if (count == 2) [[likely]] {  // a simple outline: one span per row
                    const int64_t p = b[0], q = b[1];
                    b[0] = std::min(p, q), b[1] = std::max(p, q);
                } else {
                    for (unsigned i = 1; i < count; ++i) {  // at most kBucket values
                        const int64_t value = b[i];
                        unsigned j = i;
                        for (; j > 0 && b[j - 1] > value; --j) b[j] = b[j - 1];
                        b[j] = value;
                    }
                }
                // Bit 8/4 for row 4k+1 (half 0), 2/1 for row 4k+2: masks 0..3.
                uint64_t *const ma = masks + size_t(2 * half) * words, *const mb = ma + words;
                for (unsigned i = 0; i + 1 < count; i += 2) {
                    int x1 = int((b[i] + (kOne - 1)) >> kShift), x2 = int(b[i + 1] >> kShift);
                    if (x1 >= W || x2 < 0) continue;
                    x1 = std::max(x1, 0), x2 = std::min(x2, W - 1);
                    // Output columns whose source column 4j+1 (a) or 4j+2 (b) lies in [x1, x2].
                    const int ja0 = (x1 + 2) >> 2, ja1 = (x2 - 1) >> 2, jb0 = (x1 + 1) >> 2, jb1 = (x2 - 2) >> 2;
                    if (ja0 <= ja1) set_bits(ma, ja0, ja1), lo = std::min(lo, ja0), hi = std::max(hi, ja1);
                    if (jb0 <= jb1) set_bits(mb, jb0, jb1), lo = std::min(lo, jb0), hi = std::max(hi, jb1);
                }
            }
            if (lo > hi) continue;
            expand_row(pat + size_t(row) * size_t(ww), masks, words, lo, hi);
            for (size_t wd = size_t(lo) >> 6; wd <= size_t(hi) >> 6; ++wd)
                masks[wd] = masks[words + wd] = masks[2 * words + wd] = masks[3 * words + wd] = 0;
            box.x0 = std::min(box.x0, lo), box.x1 = std::max(box.x1, hi);
            box.y0 = std::min(box.y0, row), box.y1 = std::max(box.y1, row);
        }
        MASKOPS_TICK(6, since);
        return true;
    }

    // bits[c] = 1 for c in [c0, c1] of a column bit set.
    static void set_bits(uint64_t *bits, int c0, int c1) {
        for (int wd = c0 >> 6; wd <= c1 >> 6; ++wd) {
            const int lo = std::max(c0 - wd * 64, 0), hi = std::min(c1 - wd * 64, 63);
            bits[wd] |= (hi - lo == 63 ? ~uint64_t(0) : (uint64_t(2) << (hi - lo)) - 1) << lo;
        }
    }

    // line[c] |= 8*m0[c] | 4*m1[c] | 2*m2[c] | m3[c] for c in [lo, hi], where
    // m0..m3 are the four column bit sets of `words` words each. Writes whole
    // 16-column groups (zero bits outside the range).
    static void expand_row(uint8_t *line, const uint64_t *masks, size_t words, int lo, int hi) {
#ifdef MASKOPS_SAMPLED_NEON
        if (simd_enabled()) {
            const uint8x16_t bit_of_lane = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
            for (int c = lo & ~15; c <= hi; c += 16) {
                // The two mask bytes covering these 16 columns, spread over the lanes.
                const size_t wd = size_t(c) >> 6;
                const int shift = c & 63;  // 0, 16, 32 or 48
                uint8x16_t bits = vdupq_n_u8(0);
                for (int k = 0; k < 4; ++k) {
                    const uint16_t pair = uint16_t(masks[k * words + wd] >> shift);
                    const uint8x16_t spread = vcombine_u8(vdup_n_u8(uint8_t(pair)), vdup_n_u8(uint8_t(pair >> 8)));
                    bits = vorrq_u8(bits, vandq_u8(vtstq_u8(spread, bit_of_lane), vdupq_n_u8(uint8_t(8 >> k))));
                }
                vst1q_u8(line + c, vorrq_u8(vld1q_u8(line + c), bits));
            }
            return;
        }
#elif defined(MASKOPS_SAMPLED_X86)
        if (simd_enabled()) {
            const __m128i bit_of_lane = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
            for (int c = lo & ~15; c <= hi; c += 16) {
                const size_t wd = size_t(c) >> 6;
                const int shift = c & 63;
                __m128i bits = _mm_setzero_si128();
                for (int k = 0; k < 4; ++k) {
                    const uint16_t pair = uint16_t(masks[k * words + wd] >> shift);
                    const __m128i spread = _mm_unpacklo_epi64(_mm_set1_epi8(char(pair)), _mm_set1_epi8(char(pair >> 8)));
                    const __m128i set = _mm_cmpeq_epi8(_mm_and_si128(spread, bit_of_lane), bit_of_lane);
                    bits = _mm_or_si128(bits, _mm_and_si128(set, _mm_set1_epi8(char(8 >> k))));
                }
                auto *p = reinterpret_cast<__m128i *>(line + c);
                _mm_storeu_si128(p, _mm_or_si128(_mm_loadu_si128(p), bits));
            }
            return;
        }
#endif
        for (int c = lo; c <= hi; ++c) {
            uint8_t bits = 0;
            for (int k = 0; k < 4; ++k) bits |= uint8_t(((masks[k * words + (size_t(c) >> 6)] >> (c & 63)) & 1) << (3 - k));
            line[c] |= bits;
        }
    }

    // Fills the pairs of sorted crossings of one slot.
    void spans(const int64_t *xs, size_t count, int slot, Box &box) {
        const int row = slot >> 1, W = w;
        const uint8_t a = (slot & 1) ? 2 : 8, b = (slot & 1) ? 1 : 4;
        uint8_t *line = pattern_.data() + size_t(row) * size_t(ww);
        int lo = INT_MAX, hi = INT_MIN;
        for (size_t i = 0; i + 1 < count; i += 2) {
            int x1 = int((xs[i] + (kOne - 1)) >> kShift), x2 = int(xs[i + 1] >> kShift);
            if (x1 >= W || x2 < 0) continue;
            x1 = std::max(x1, 0), x2 = std::min(x2, W - 1);
            int l, r;
            or_span(line, x1, x2, a, b, l = INT_MAX, r = INT_MIN);
            lo = std::min(lo, l), hi = std::max(hi, r);
        }
        if (lo <= hi) {
            box.x0 = std::min(box.x0, lo), box.x1 = std::max(box.x1, hi);
            box.y0 = std::min(box.y0, row), box.y1 = std::max(box.y1, row);
        }
    }

    // CollectPolyEdges' edges evaluated at the sampled rows they are active
    // on, for the sorting fill. Returns the number of crossings collected.
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
            spans(xs, count, smin + k, box);
        }
    }

    uint8_t lut_[16] = {};
    std::vector<uint8_t> pattern_;  // hh x ww, all zero between contours
    Vec<uint32_t> slots_, marks_;
    Vec<int32_t> slot_of_, cslot_, cx_, long_;
    Vec<int64_t> x_of_, sorted_, bucket_;
    std::vector<uint64_t> masks_;
    Vec<uint8_t> bucket_n_;  // all zero between contours
    size_t crossings_ = 0, marks_n_ = 0, cross_n_ = 0, long_n_ = 0;
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
        return std::bit_cast<uint64_t>(std::array<int32_t, 2>{x, y});
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
        const int kept = 4 + vaddvq_s32(vreinterpretq_s32_u32(same));  // lanes are 0 or -1
        const uint8x16x2_t block = {{vreinterpretq_u8_u64(pa), vreinterpretq_u8_u64(pb)}};
        uint8_t *dst = reinterpret_cast<uint8_t *>(out + 2 * k);
        vst1q_u8(dst, vqtbl2q_u8(block, vld1q_u8(kCompact.index[keep])));
        vst1q_u8(dst + 16, vqtbl2q_u8(block, vld1q_u8(kCompact.index[keep] + 16)));
        k += size_t(kept);
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
// incident edge already draws), writing the padded layout described at
// kPadVertices: offsets[i] is the vertex index of contour i's first vertex
// and its count is offsets[i + 1] - offsets[i] - kPadVertices. points needs
// 2 * (1 + n * (m + kPadVertices)) + kPointSlack values and offsets n + 1.
// Returns false, leaving the outputs unusable, if any value is non-finite or
// outside the exact envelope. m must be positive if n is.
template <std::floating_point F>
inline bool load_contours(const F *xy, size_t n, size_t m, int32_t *points, int64_t *offsets) {
    int64_t start = 1;  // points[0..1] is the first contour's leading pad
    offsets[0] = start;
#ifdef MASKOPS_SAMPLED_NEON
    if constexpr (std::same_as<F, float>) {
        if (simd_enabled()) {
            uint32x4_t ok = vdupq_n_u32(~0u);
            for (size_t i = 0; i < n; ++i) {
                int32_t *out = points + 2 * start;
                const size_t k = load_contour_f32(xy + 2 * i * m, m, out, ok);
                pad_contour(out, int64_t(k));
                start += int64_t(k) + kPadVertices;
                offsets[i + 1] = start;
            }
            return vminvq_u32(ok) != 0;
        }
    }
#elif defined(MASKOPS_SAMPLED_X86)
    if constexpr (std::same_as<F, float>) {
      if (simd_enabled()) {
        // Two vertices per step: one cvttps2dq, then a compaction whose only
        // loop-carried dependency is the running count.
        const __m128 limit = _mm_set1_ps(float(kCoordLimit));
        const __m128 magnitude = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        __m128 inside = _mm_castsi128_ps(_mm_set1_epi32(-1));
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            const float *p = xy + 2 * i * m;
            int32_t *out = points + 2 * start;
            uint64_t previous = 0;
            size_t j = 0, k = 0;
            for (; j + 2 <= m; j += 2) {
                const __m128 v = _mm_loadu_ps(p + 2 * j);
                inside = _mm_and_ps(inside, _mm_cmplt_ps(_mm_and_ps(v, magnitude), limit));  // false for NaN
                const __m128i c = _mm_cvttps_epi32(v);
                const auto a = uint64_t(_mm_cvtsi128_si64(c)), b = uint64_t(_mm_cvtsi128_si64(_mm_unpackhi_epi64(c, c)));
                std::memcpy(out + 2 * k, &a, sizeof(a));
                k += (a != previous) | (j == 0);
                std::memcpy(out + 2 * k, &b, sizeof(b));
                k += b != a;
                previous = b;
            }
            for (; j < m; ++j) {
                const float fx = p[2 * j], fy = p[2 * j + 1];
                const bool in = std::fabs(fx) < float(kCoordLimit) && std::fabs(fy) < float(kCoordLimit);
                ok &= in;
                const int32_t x = int32_t(in ? fx : 0.f), y = int32_t(in ? fy : 0.f);
                const uint64_t key = std::bit_cast<uint64_t>(std::array<int32_t, 2>{x, y});
                out[2 * k] = x, out[2 * k + 1] = y;
                k += (key != previous) | (j == 0);
                previous = key;
            }
            pad_contour(out, int64_t(k));
            start += int64_t(k) + kPadVertices;
            offsets[i + 1] = start;
        }
        return ok && _mm_movemask_ps(inside) == 0xF;
      }
    }
#endif
    bool ok = true;
    const F limit = F(kCoordLimit);
    for (size_t i = 0; i < n; ++i) {
        const F *p = xy + 2 * i * m;
        int32_t *out = points + 2 * start;
        uint64_t previous = 0;
        size_t k = 0;
        for (size_t j = 0; j < m; ++j) {
            const F fx = p[2 * j], fy = p[2 * j + 1];
            const bool inside = std::fabs(fx) < limit && std::fabs(fy) < limit;  // false for NaN
            ok &= inside;
            const int32_t x = int32_t(inside ? fx : F(0)), y = int32_t(inside ? fy : F(0));
            const uint64_t key = std::bit_cast<uint64_t>(std::array<int32_t, 2>{x, y});
            out[2 * k] = x, out[2 * k + 1] = y;
            // The first vertex of each contour is always kept.
            k += (key != previous) | (j == 0);
            previous = key;
        }
        pad_contour(out, int64_t(k));
        start += int64_t(k) + kPadVertices;
        offsets[i + 1] = start;
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
        if constexpr (std::same_as<T, uint8_t>) {
            const uint8x16_t fill = vdupq_n_u8(value);
            for (; simd_enabled() && x + 16 <= bw; x += 16) {
                const uint8x16_t m = vld1q_u8(s + x);
                vst1q_u8(o + x, vbslq_u8(vtstq_u8(m, m), fill, vld1q_u8(o + x)));
            }
        }
#elif defined(MASKOPS_SAMPLED_X86)
        if constexpr (std::same_as<T, uint8_t>) {
            const __m128i fill = _mm_set1_epi8(char(value)), zero = _mm_setzero_si128();
            for (; simd_enabled() && x + 16 <= bw; x += 16) {
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
