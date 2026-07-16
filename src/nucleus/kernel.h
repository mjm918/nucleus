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

#pragma once
#ifndef NUCLEUS_KERNEL_H
#define NUCLEUS_KERNEL_H

#include <cmath>
#include "common.h"
#include "threadpool.h"
#include "view.h"


namespace nucleus {

    // ---- activation quantization ------------------------------------------------

    // x[n] -> q[n] (int8), s[n/32] (f32 scale per group). n must be a multiple of 32.
    void quantize_act(const f32 *x, int n, i8 *q, f32 *s);

    // ---- matrix-vector products (out[r] = dot(W[r, :], x)) -----------------------

    // Fast paths: quantized weights x int8 activations.
    void matvec_q8(const QView &w, const i8 *xq, const f32 *xs, f32 *out, ThreadPool &pool);
    void matvec_q4(const QView &w, const i8 *xq, const f32 *xs, f32 *out, ThreadPool &pool);
    // Row-range variants (used for expert sub-blocks living inside one blob).
    void matvec_q4_rows(const QView &w, const i8 *xq, const f32 *xs, f32 *out, u32 row_begin, u32 row_end);

    // Exact paths: dequantize weights, dot with f32 activations (slow, for tests).
    void matvec_exact(const QView &w, const f32 *x, f32 *out, ThreadPool &pool);
    void matvec_exact_rows(const QView &w, const f32 *x, f32 *out, u32 row_begin, u32 row_end);

    // Plain f32 weights (router projection).
    void matvec_f32(const f32 *w, u32 rows, u32 cols, const f32 *x, f32 *out, ThreadPool &pool);

    // Dequantize a single row (embedding lookup).
    void dequant_row(const QView &w, u32 row, f32 *out);

    bool exact_mode(); // NUCLEUS_EXACT=1

    // ---- small vector math -------------------------------------------------------

    // Gemma 4 RMSNorm: y = x / sqrt(mean(x^2) + eps) * w   (w == nullptr -> no scale)
    inline void rmsnorm(f32 *out, const f32 *x, const f32 *w, int n, f32 eps) {
        f32 ss = 0.f;
        for (int i = 0; i < n; ++i)
            ss += x[i] * x[i];
        f32 inv = 1.0f / std::sqrt(ss / n + eps);
        if (w) {
            for (int i = 0; i < n; ++i)
                out[i] = x[i] * inv * w[i];
        } else {
            for (int i = 0; i < n; ++i)
                out[i] = x[i] * inv;
        }
    }

    inline void softmax(f32 *x, int n) {
        f32 mx = x[0];
        for (int i = 1; i < n; ++i)
            mx = std::max(mx, x[i]);
        f32 sum = 0.f;
        for (int i = 0; i < n; ++i) {
            x[i] = std::exp(x[i] - mx);
            sum += x[i];
        }
        f32 inv = 1.0f / sum;
        for (int i = 0; i < n; ++i)
            x[i] *= inv;
    }

    // gelu_pytorch_tanh
    inline f32 gelu_tanh(f32 x) {
        constexpr f32 k = 0.7978845608028654f; // sqrt(2/pi)
        return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
    }

    // HF half-rotation RoPE. inv_freq has head_dim/2 entries (zeros = no rotation,
    // which is how Gemma 4's proportional RoPE encodes partial rotation).
    inline void rope(f32 *v, int head_dim, const f32 *inv_freq, int pos) {
        int half = head_dim / 2;
        for (int j = 0; j < half; ++j) {
            if (inv_freq[j] == 0.0f)
                continue; // pass-through dims
            f32 f = static_cast<f32>(pos) * inv_freq[j];
            f32 c = std::cos(f), s = std::sin(f);
            f32 a = v[j], b = v[j + half];
            v[j] = a * c - b * s;
            v[j + half] = b * c + a * s;
        }
    }

} // namespace nucleus

#endif // NUCLEUS_KERNEL_H
