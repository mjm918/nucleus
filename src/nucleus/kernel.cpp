/*
 * ************************************************************************
 * Copyright (c) 2026 Mohammad Julfikar.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * *************************************************************************
 */

#include "kernel.h"

#include <cstdlib>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nucleus {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)

    static inline f32 hsum_f32x8(__m256 v) {
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        return _mm_cvtss_f32(s);
    }

    // One 32-group of i8 x i8 products -> 8 i32 partial sums. maddubs wants one
    // unsigned operand, so pass |w| and a with w's sign folded in. Safe from i16
    // overflow because both sides are clamped to [-127, 127] at quantization:
    // each pair sum is at most 2*127*127 < 32767.
    static inline __m256i dot_i8_pairs(__m256i w, __m256i a) {
        const __m256i absw = _mm256_sign_epi8(w, w);
        const __m256i sa = _mm256_sign_epi8(a, w);
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        return _mm256_dpbusd_epi32(_mm256_setzero_si256(), absw, sa);
#else
        return _mm256_madd_epi16(_mm256_maddubs_epi16(absw, sa), _mm256_set1_epi16(1));
#endif
    }

#endif // __AVX2__ && __FMA__ && __F16C__

    bool exact_mode() {
        static const bool on = [] {
            const char *v = std::getenv("NUCLEUS_EXACT");
            return v && v[0] == '1';
        }();
        return on;
    }

    void quantize_act(const f32 *x, int n, i8 *q, f32 *s) {
        NUC_CHECK(n % kGroup == 0, "activation length not group-aligned");
        for (int g = 0; g < n / kGroup; ++g) {
            const f32 *xs = x + g * kGroup;
            f32 amax = 0.f;
            for (int j = 0; j < kGroup; ++j)
                amax = std::max(amax, std::fabs(xs[j]));
            f32 scale = amax / 127.0f;
            f32 inv = scale > 0.f ? 1.0f / scale : 0.f;
            s[g] = scale;
            i8 *qs = q + g * kGroup;
            for (int j = 0; j < kGroup; ++j) {
                i32 v = (i32) lrintf(xs[j] * inv);
                qs[j] = (i8) std::min(127, std::max(-127, v));
            }
        }
    }

    // ---- q8: one row dot --------------------------------------------------------

    static inline f32 dot_row_q8(const u16 *scales, const u8 *quants, u32 cols, const i8 *xq, const f32 *xs) {
        const i8 *wq = reinterpret_cast<const i8 *>(quants);
        const u32 groups = cols / kGroup;
        f32 acc = 0.f;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        for (u32 g = 0; g < groups; ++g) {
            const i8 *w = wq + g * kGroup;
            const i8 *a = xq + g * kGroup;
            int32x4_t sum = vdupq_n_s32(0);
            sum = vdotq_s32(sum, vld1q_s8(w), vld1q_s8(a));
            sum = vdotq_s32(sum, vld1q_s8(w + 16), vld1q_s8(a + 16));
            acc += (f32) vaddvq_s32(sum) * f16_to_f32(scales[g]) * xs[g];
        }
#elif defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
        __m256 accv = _mm256_setzero_ps();
        for (u32 g = 0; g < groups; ++g) {
            const __m256i w = _mm256_loadu_si256((const __m256i *) (wq + g * kGroup));
            const __m256i a = _mm256_loadu_si256((const __m256i *) (xq + g * kGroup));
            const __m256 s = _mm256_set1_ps(f16_to_f32(scales[g]) * xs[g]);
            accv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8_pairs(w, a)), s, accv);
        }
        acc = hsum_f32x8(accv);
#else
        for (u32 g = 0; g < groups; ++g) {
            const i8 *w = wq + g * kGroup;
            const i8 *a = xq + g * kGroup;
            i32 sum = 0;
            for (int j = 0; j < kGroup; ++j)
                sum += (i32) w[j] * (i32) a[j];
            acc += (f32) sum * f16_to_f32(scales[g]) * xs[g];
        }
#endif
        return acc;
    }

    // ---- q4: one row dot --------------------------------------------------------

    static inline f32 dot_row_q4(const u16 *scales, const u8 *packed, u32 cols, const i8 *xq, const f32 *xs) {
        const u32 groups = cols / kGroup;
        f32 acc = 0.f;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        const int8x16_t off = vdupq_n_s8(8);
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        for (u32 g = 0; g < groups; ++g) {
            uint8x16_t b = vld1q_u8(packed + g * (kGroup / 2));
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, mask)), off);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), off);
            const i8 *a = xq + g * kGroup;
            int32x4_t sum = vdupq_n_s32(0);
            sum = vdotq_s32(sum, lo, vld1q_s8(a));
            sum = vdotq_s32(sum, hi, vld1q_s8(a + 16));
            acc += (f32) vaddvq_s32(sum) * f16_to_f32(scales[g]) * xs[g];
        }
#elif defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
        const __m128i mask = _mm_set1_epi8(0x0F);
        const __m256i off = _mm256_set1_epi8(8);
        __m256 accv = _mm256_setzero_ps();
        for (u32 g = 0; g < groups; ++g) {
            const __m128i b = _mm_loadu_si128((const __m128i *) (packed + g * (kGroup / 2)));
            const __m128i lo = _mm_and_si128(b, mask); // elems 0..15
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), mask); // elems 16..31
            const __m256i w = _mm256_sub_epi8(_mm256_set_m128i(hi, lo), off);
            const __m256i a = _mm256_loadu_si256((const __m256i *) (xq + g * kGroup));
            const __m256 s = _mm256_set1_ps(f16_to_f32(scales[g]) * xs[g]);
            accv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8_pairs(w, a)), s, accv);
        }
        acc = hsum_f32x8(accv);
#else
        for (u32 g = 0; g < groups; ++g) {
            const u8 *b = packed + g * (kGroup / 2);
            const i8 *a = xq + g * kGroup;
            i32 sum = 0;
            for (int j = 0; j < kGroup / 2; ++j) {
                i32 lo = (i32) (b[j] & 0x0F) - 8;
                i32 hi = (i32) (b[j] >> 4) - 8;
                sum += lo * (i32) a[j] + hi * (i32) a[j + kGroup / 2];
            }
            acc += (f32) sum * f16_to_f32(scales[g]) * xs[g];
        }
#endif
        return acc;
    }

    // ---- exact (dequantized f32) row dots ----------------------------------------

    static f32 dot_row_exact(const QView &w, u32 r, const f32 *x) {
        const u32 groups = w.cols / kGroup;
        const u16 *scales = w.scales + (u64) r * groups;
        f32 acc = 0.f;
        if (w.dtype == DType::Q8) {
            const i8 *wq = reinterpret_cast<const i8 *>(w.quants) + (u64) r * w.cols;
            for (u32 g = 0; g < groups; ++g) {
                f32 s = f16_to_f32(scales[g]);
                const i8 *wg = wq + g * kGroup;
                const f32 *xg = x + g * kGroup;
                for (int j = 0; j < kGroup; ++j)
                    acc += (f32) wg[j] * s * xg[j];
            }
        } else {
            const u8 *wp = w.quants + (u64) r * (w.cols / 2);
            for (u32 g = 0; g < groups; ++g) {
                f32 s = f16_to_f32(scales[g]);
                const u8 *b = wp + g * (kGroup / 2);
                const f32 *xg = x + g * kGroup;
                for (int j = 0; j < kGroup / 2; ++j) {
                    acc += (f32) ((i32) (b[j] & 0x0F) - 8) * s * xg[j];
                    acc += (f32) ((i32) (b[j] >> 4) - 8) * s * xg[j + kGroup / 2];
                }
            }
        }
        return acc;
    }

    // ---- public matvecs -----------------------------------------------------------

    void matvec_q8(const QView &w, const i8 *xq, const f32 *xs, f32 *out, ThreadPool &pool) {
        const u32 groups = w.cols / kGroup;
        pool.parallel_for(w.rows, [&](i64 r0, i64 r1) {
            for (i64 r = r0; r < r1; ++r) {
                out[r] = dot_row_q8(w.scales + r * groups, w.quants + r * (u64) w.cols, w.cols, xq, xs);
            }
        });
    }

    void matvec_q4_rows(const QView &w, const i8 *xq, const f32 *xs, f32 *out, u32 row_begin, u32 row_end) {
        const u32 groups = w.cols / kGroup;
        for (u32 r = row_begin; r < row_end; ++r) {
            out[r - row_begin] =
                    dot_row_q4(w.scales + (u64) r * groups, w.quants + (u64) r * (w.cols / 2), w.cols, xq, xs);
        }
    }

    void matvec_q4(const QView &w, const i8 *xq, const f32 *xs, f32 *out, ThreadPool &pool) {
        pool.parallel_for(w.rows, [&](i64 r0, i64 r1) { matvec_q4_rows(w, xq, xs, out + r0, (u32) r0, (u32) r1); });
    }

    void matvec_exact_rows(const QView &w, const f32 *x, f32 *out, u32 row_begin, u32 row_end) {
        for (u32 r = row_begin; r < row_end; ++r)
            out[r - row_begin] = dot_row_exact(w, r, x);
    }

    void matvec_exact(const QView &w, const f32 *x, f32 *out, ThreadPool &pool) {
        pool.parallel_for(w.rows, [&](i64 r0, i64 r1) { matvec_exact_rows(w, x, out + r0, (u32) r0, (u32) r1); });
    }

    void matvec_f32(const f32 *w, u32 rows, u32 cols, const f32 *x, f32 *out, ThreadPool &pool) {
        pool.parallel_for(rows, [&](i64 r0, i64 r1) {
            for (i64 r = r0; r < r1; ++r) {
                const f32 *wr = w + r * cols;
                f32 acc = 0.f;
                for (u32 c = 0; c < cols; ++c)
                    acc += wr[c] * x[c];
                out[r] = acc;
            }
        });
    }

    void dequant_row(const QView &w, u32 row, f32 *out) {
        const u32 groups = w.cols / kGroup;
        const u16 *scales = w.scales + (u64) row * groups;
        if (w.dtype == DType::Q8) {
            const i8 *wq = reinterpret_cast<const i8 *>(w.quants) + (u64) row * w.cols;
            for (u32 g = 0; g < groups; ++g) {
                f32 s = f16_to_f32(scales[g]);
                for (int j = 0; j < kGroup; ++j)
                    out[g * kGroup + j] = wq[g * kGroup + j] * s;
            }
        } else {
            const u8 *wp = w.quants + (u64) row * (w.cols / 2);
            for (u32 g = 0; g < groups; ++g) {
                f32 s = f16_to_f32(scales[g]);
                const u8 *b = wp + g * (kGroup / 2);
                for (int j = 0; j < kGroup / 2; ++j) {
                    out[g * kGroup + j] = ((i32) (b[j] & 0x0F) - 8) * s;
                    out[g * kGroup + j + kGroup / 2] = ((i32) (b[j] >> 4) - 8) * s;
                }
            }
        }
    }
} // namespace nucleus
