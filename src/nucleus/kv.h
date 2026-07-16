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
#ifndef NUCLEUS_KV_H
#define NUCLEUS_KV_H

#include <algorithm>
#include "common.h"

namespace nucleus {
    class LayerKV {
    public:
        LayerKV(bool sliding, u32 window, u32 max_ctx, u32 heads, u32 dim) :
            _sliding(sliding), _window(window), _slots(sliding ? std::min(window, max_ctx) : max_ctx),
            _distance(heads * dim), _k(_slots * _distance), _v(_slots * _distance) {}

        void push(u32 pos, const f32 *k, const f32 *v) {
            u32 slot = _sliding ? pos % _slots : pos;
            NUC_CHECK(_slots > slot, "context overflown. adjust --ctx");

            std::memcpy(_k.data() + slot * _distance, k, _distance * sizeof(f32));
            std::memcpy(_v.data() + slot * _distance, v, _distance * sizeof(f32));
        }

        u32 attn_start(u32 pos) const {
            if (!_sliding || pos + 1 <= _window)
                return 0;
            return pos + 1 - _window;
        }

        const f32 *k_at(u32 pos) const { return _k.data() + (_sliding ? pos % _slots : pos) * _distance; }

        const f32 *v_at(u32 pos) const { return _v.data() + (_sliding ? pos % _slots : pos) * _distance; }

        u32 distance() const { return _distance; }

    private:
        bool _sliding;
        u32 _window;
        u32 _slots;
        u32 _distance;
        AlignedBuf<f32> _k, _v;
    };
} // namespace nucleus

#endif // NUCLEUS_KV_H
