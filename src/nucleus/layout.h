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
#ifndef NUCLEUS_LAYOUT_H
#define NUCLEUS_LAYOUT_H

#include "common.h"

namespace nucleus {
    constexpr u32 kMagic = 0x3143554E;
    constexpr u32 kVersion = 1;
    constexpr int kGroup = 32;
    constexpr int kMaxLayers = 256;

    enum class DType : u8 { F32 = 0, F16 = 1, Q8 = 2, Q4 = 3 };

#pragma pack(push, 1)
    struct ModelConfigBin {
        u32 hidden;
        u32 n_layers;
        u32 n_heads;
        u32 n_kv_heads;
        u32 head_dim;
        u32 global_head_dim;
        u32 n_global_kv_heads;
        u32 ffn_dim;
        u32 moe_ffn_dim;
        u32 n_experts;
        u32 top_k;
        u32 vocab;
        u32 sliding_window;
        u32 max_pos;
        f32 rope_theta_local;
        f32 rope_theta_global;
        f32 partial_rotary_global;
        f32 rms_eps;
        f32 final_softcap;
        u32 full_attn_bits[kMaxLayers / 32];
        u32 bos_id, eos_id, pad_id;
        u32 tok_add_dummy_prefix;
    };

    struct FileHeader {
        u32 magic;
        u32 version;
        ModelConfigBin cfg;
        u64 tok_offset;
        u64 tok_bytes;
        u64 table_offset;
        u32 tensor_count;
        u32 reserved;
    };

    struct TensorEntry {
        char name[96];
        u8 dtype;
        u8 ndim;
        u16 pad;
        u32 shape[4];
        u64 offset;
        u64 nbytes;
    };
#pragma pack(pop)

    static_assert(sizeof(TensorEntry) == 96 + 4 + 16 + 16, "TensorEntry layout");

    inline bool layer_is_full_attn(const ModelConfigBin &c, u32 layer) {
        return (c.full_attn_bits[layer / 32] >> (layer % 32)) & 1u;
    }

    // Byte sizes of quantized [rows, cols] tensors
    inline u64 q_scales_bytes(u64 rows, u64 cols) { return rows * (cols / kGroup) * 2; }
    inline u64 q8_bytes(u64 rows, u64 cols) { return q_scales_bytes(rows, cols) + rows * cols; }
    inline u64 q4_bytes(u64 rows, u64 cols) { return q_scales_bytes(rows, cols) + rows * cols / 2; }
} // namespace nucleus

#endif // NUCLEUS_LAYOUT_H
