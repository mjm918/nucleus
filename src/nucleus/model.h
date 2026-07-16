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
#ifndef NUCLEUS_MODEL_H
#define NUCLEUS_MODEL_H

#include "common.h"
#include "experts.h"
#include "kv.h"
#include "threadpool.h"
#include "tokenizer.h"
#include "view.h"


namespace nucleus {
    struct LayerWeights {
        bool full_attn = false;
        QView attn_q, attn_k, attn_o;
        QView attn_v; // sliding layers only; full layers reuse attn_k (k == v)
        bool has_v = false;
        const f32 *q_norm = nullptr; // [head_dim of this layer]
        const f32 *k_norm = nullptr; // [head_dim of this layer]
        const f32 *ln_in = nullptr; // [hidden] input_layernorm
        const f32 *ln_postattn = nullptr;
        const f32 *ln_preffw = nullptr;
        const f32 *ln_postffw = nullptr;
        const f32 *ln_postffw1 = nullptr; // after dense MLP (MoE merge)
        const f32 *ln_postffw2 = nullptr; // after expert sum
        const f32 *ln_preffw2 = nullptr; // before experts
        QView mlp_gate, mlp_up, mlp_down;
        const f32 *router_w = nullptr; // [n_experts, hidden] f32
        const f32 *router_scale = nullptr; // [hidden]
        const f32 *expert_scale = nullptr; // [n_experts]
        f32 layer_scalar = 1.0f;
    };

    struct ForwardStats {
        u64 tokens = 0;
        double total_s = 0.0;
    };

    class Model {
    public:
        Model(const std::string &path, u32 max_ctx, int n_threads);
        ~Model();

        void forward(i32 token, u32 pos, bool want_logits);

        const f32 *logits() const { return _logits.data(); }
        const ModelConfigBin &cfg() const { return _file.cfg(); }
        const Tokenizer &tokenizer() const { return _tokenizer; }
        u32 max_ctx() const { return _max_ctx; }
        i32 eot_id() const { return _eot_id; } // "<turn|>" end-of-turn, -1 if absent

        void set_max_rss(u64 bytes) { _experts.set_max_rss(bytes); }
        bool cap_holding() const { return _cap_holding; }
        u64 evictions() const { return _experts.evictions(); }
        u64 resident_experts() const { return _experts.resident_experts(); }

    private:
        void attention(const LayerWeights &lw, u32 layer, u32 pos, const f32 *x, f32 *out);
        void moe_ffn(const LayerWeights &lw, u32 layer, const f32 *residual, f32 *out);
        void prepare_mv(const f32 *x, u32 n);
        void mv_prepared(const QView &w, const f32 *x, f32 *out);
        void mv(const QView &w, const f32 *x, f32 *out);

        NucFile _file;
        ThreadPool _pool;
        Tokenizer _tokenizer;
        ExpertStore _experts;
        std::vector<LayerWeights> _layers;
        std::vector<std::unique_ptr<LayerKV>> _kv;
        QView _token_emb;
        const f32 *_out_norm = nullptr;
        f32 _embed_scale = 1.0f;
        u32 _max_ctx = 0;
        i32 _eot_id = -1;
        bool _cap_holding = true;
        bool _logits_warmed = false;
        u64 _token_emb_bytes = 0;
        std::string _usage_path;

        std::vector<f32> _inv_freq_local, _inv_freq_global;

        AlignedBuf<f32> _h, _t1, _t2, _xa, _xb;
        AlignedBuf<f32> _qbuf, _kbuf, _vbuf, _att_out, _scores;
        AlignedBuf<f32> _ffn_a, _ffn_b, _moe_gu, _moe_h, _moe_acc, _router_logits;
        AlignedBuf<f32> _moe_out, _moe_acts;
        AlignedBuf<i8> _moe_actq;
        AlignedBuf<f32> _logits;
        AlignedBuf<i8> _actq;
        AlignedBuf<f32> _acts;
    };
} // namespace nucleus

#endif // NUCLEUS_MODEL_H
