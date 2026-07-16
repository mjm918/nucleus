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
#ifndef NUCLEUS_VIEW_H
#define NUCLEUS_VIEW_H

#include "common.h"
#include "page.h"

namespace nucleus {
    struct QView {
        const u16 *scales = nullptr;
        const u8 *quants = nullptr;
        u32 rows = 0, cols = 0;
        DType dtype = DType::F32;
    };

    inline QView qview(const NucFile &f, const TensorEntry &e) {
        QView v;
        v.dtype = static_cast<DType>(e.dtype);
        NUC_CHECK(v.dtype == DType::Q8 || v.dtype == DType::Q4, std::string("not quantized: ") + e.name);
        NUC_CHECK(e.ndim == 2, std::string("expected 2-D tensor: ") + e.name);
        v.rows = e.shape[0];
        v.cols = e.shape[1];
        NUC_CHECK(v.cols % kGroup == 0, std::string("cols not group-aligned: ") + e.name);
        const u8 *base = f.data(e);
        v.scales = reinterpret_cast<const u16 *>(base);
        v.quants = base + q_scales_bytes(v.rows, v.cols);
        u64 expect = v.dtype == DType::Q8 ? q8_bytes(v.rows, v.cols) : q4_bytes(v.rows, v.cols);
        NUC_CHECK(e.nbytes == expect, std::string("size mismatch: ") + e.name);
        return v;
    }

    inline const f32 *f32view(const NucFile &f, const TensorEntry &e, u64 expect_count) {
        NUC_CHECK(static_cast<DType>(e.dtype) == DType::F32, std::string("expected f32: ") + e.name);
        NUC_CHECK(e.nbytes == expect_count * 4, std::string("f32 size mismatch: ") + e.name);
        return reinterpret_cast<const f32 *>(f.data(e));
    }
} // namespace nucleus

#endif // NUCLEUS_VIEW_H
