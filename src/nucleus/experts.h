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
#ifndef NUCLEUS_EXPERTS_H
#define NUCLEUS_EXPERTS_H

#include <algorithm>
#include <numeric>


#include "meminfo.h"
#include "view.h"

namespace nucleus {
    struct ExpertView {
        QView gate_up; // [2*moe_ffn, hidden] q4
        QView down; // [hidden, moe_ffn] q4
    };

    class ExpertStore {
    public:
        ExpertStore(const NucFile &file, const ModelConfigBin &cfg) : _file(file), _cfg(cfg) {
            _entries.resize(cfg.n_layers * cfg.n_experts);
            _hist_count.assign(_entries.size(), 0);
            _resident.assign(_entries.size(), 0);
            _used_at.assign(_entries.size(), 0);
            const u64 gu_sc = q_scales_bytes(2ull * cfg.moe_ffn_dim, cfg.hidden);
            const u64 gu_q = 2ull * cfg.moe_ffn_dim * cfg.hidden / 2;
            const u64 d_sc = q_scales_bytes(cfg.hidden, cfg.moe_ffn_dim);
            const u64 d_q = (u64) cfg.hidden * cfg.moe_ffn_dim / 2;
            _blob_bytes = gu_sc + gu_q + d_sc + d_q;
            for (u32 l = 0; l < cfg.n_layers; ++l) {
                for (u32 e = 0; e < cfg.n_experts; ++e) {
                    const TensorEntry &t = file.tensor("l" + std::to_string(l) + ".exp" + std::to_string(e));
                    NUC_CHECK(t.nbytes == _blob_bytes, "expert blob size mismatch");
                    _entries[l * cfg.n_experts + e] = &t;
                }
            }
        }

        ExpertView view(u32 layer, u32 e) {
            const size_t i = (size_t) layer * _cfg.n_experts + e;
            const TensorEntry &t = *_entries[i];
            ++_hist_count[i];
            _used_at[i] = ++_clock;
            _resident[i] = 1;
            const u8 *base = _file.data(t);
            const u64 gu_sc = q_scales_bytes(2ull * _cfg.moe_ffn_dim, _cfg.hidden);
            const u64 gu_q = 2ull * _cfg.moe_ffn_dim * _cfg.hidden / 2;
            const u64 d_sc = q_scales_bytes(_cfg.hidden, _cfg.moe_ffn_dim);
            ExpertView v;
            v.gate_up = {reinterpret_cast<const u16 *>(base), base + gu_sc, 2 * _cfg.moe_ffn_dim, _cfg.hidden,
                         DType::Q4};
            const u8 *d = base + gu_sc + gu_q;
            v.down = {reinterpret_cast<const u16 *>(d), d + d_sc, _cfg.hidden, _cfg.moe_ffn_dim, DType::Q4};
            return v;
        }

        void prefetch(u32 layer, const u32 *ids, u32 k) {
            for (u32 i = 0; i < k; ++i) {
                const size_t idx = layer * _cfg.n_experts + ids[i];
#if !defined(__APPLE__)
                if (_resident[idx])
                    continue;
#endif
                const TensorEntry &t = *_entries[idx];
                _file.map().prefetch(t.offset, t.nbytes);
                _resident[idx] = 1;
            }
        }

        void warm_hot_experts(u32 max_experts) {
            std::vector<u32> order(_entries.size());
            std::iota(order.begin(), order.end(), 0);
            std::ranges::partial_sort(order, order.begin() + std::min<size_t>(max_experts, order.size()),
                                      [this](u32 a, u32 b) { return _hist_count[a] > _hist_count[b]; });
            for (u32 i = 0; i < max_experts && i < order.size(); ++i) {
                if (_hist_count[order[i]] == 0)
                    break;
                const TensorEntry &t = *_entries[order[i]];
                _file.map().prefetch(t.offset, t.nbytes);
                _resident[order[i]] = 1;
            }
        }

        u64 expert_bytes() const { return _blob_bytes; }

        // -- RSS Cap --
        void set_max_rss(u64 bytes) { _max_rss = bytes; }
        bool capped() const { return _max_rss != 0; }

        bool enforce_max_rss(u64 reserve = 0) {
            if (!_max_rss)
                return true;
            const u64 target = _max_rss > reserve ? _max_rss - reserve : 0;
            u64 rss = rss_bytes();
            if (rss == 0)
                return true;
            if (rss <= target)
                return true;

            _scratch.clear();
            for (size_t i = 0; i < _entries.size(); ++i)
                if (_resident[i])
                    _scratch.push_back((u32) i);
            std::sort(_scratch.begin(), _scratch.end(), [this](u32 a, u32 b) { return _used_at[a] < _used_at[b]; });

            size_t next = 0;
            while (rss > target && next < _scratch.size()) {
                const u64 over = rss - target;
                const size_t batch = (size_t) ((over + _blob_bytes - 1) / _blob_bytes);
                for (size_t n = 0; n < batch && next < _scratch.size(); ++n, ++next) {
                    const TensorEntry &t = *_entries[_scratch[next]];
                    _file.map().evict(t.offset, t.nbytes);
                    _resident[_scratch[next]] = 0;
                    ++_evictions;
                }
                rss = rss_bytes();
            }
            return rss <= target;
        }

        u64 evictions() const { return _evictions; }
        u64 resident_experts() const { return std::ranges::count(_resident, (u8) 1); }

        void load_usage(const std::string &path) {
            FILE *f = std::fopen(path.c_str(), "rb");
            if (!f)
                return;
            u32 magic = 0, n = 0;
            if (std::fread(&magic, 4, 1, f) == 1 && magic == 0x47535543 && std::fread(&n, 4, 1, f) == 1 &&
                n == _hist_count.size()) {
                size_t got = std::fread(_hist_count.data(), sizeof(u64), n, f);
                if (got != n)
                    std::fill(_hist_count.begin(), _hist_count.end(), 0);
            }
            std::fclose(f);
        }

        void save_usage(const std::string &path) const {
            FILE *f = std::fopen(path.c_str(), "wb");
            if (!f)
                return;
            u32 magic = 0x47535543, n = _hist_count.size();
            std::fwrite(&magic, 4, 1, f);
            std::fwrite(&n, 4, 1, f);
            std::fwrite(_hist_count.data(), sizeof(u64), n, f);
            std::fclose(f);
        }

    private:
        const NucFile &_file;
        const ModelConfigBin &_cfg;
        std::vector<const TensorEntry *> _entries;
        std::vector<u64> _hist_count;
        std::vector<u8> _resident;
        std::vector<u64> _used_at;
        std::vector<u32> _scratch;
        u64 _clock = 0;
        u64 _max_rss = 0; // 0 = uncapped
        u64 _evictions = 0;
        u64 _blob_bytes = 0;
    };
} // namespace nucleus

#endif // NUCLEUS_EXPERTS_H
