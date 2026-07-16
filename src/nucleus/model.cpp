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

#include "model.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "kernel.h"

namespace nucleus {
    namespace prof {
        enum Sec { kAttn, kDenseMlp, kRouter, kExperts, kLogitsMv, kSoftcap, kSecs };
        static const char *kNames[kSecs] = {"attention", "dense_mlp", "router+topk", "experts", "logits_mv", "softcap"};
        static double acc[kSecs];
        static u64 fwd_calls, logit_calls;
        static bool on() {
            static const bool v = [] {
                const char *e = std::getenv("NUCLEUS_PROF");
                return e && e[0] == '1';
            }();
            return v;
        }
        static double now() {
            return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }
        struct Timer {
            Sec s;
            double t0;
            explicit Timer(Sec sec) : s(sec), t0(on() ? now() : 0) {}
            ~Timer() {
                if (on())
                    acc[s] += now() - t0;
            }
        };
        struct Report {
            ~Report() {
                if (!on() || !fwd_calls)
                    return;
                std::fprintf(stderr, "[prof] %llu forward calls (%llu with logits)\n", (unsigned long long) fwd_calls,
                             logit_calls);
                double sum = 0;
                for (int i = 0; i < kSecs; ++i) {
                    const u64 n = (i == kLogitsMv || i == kSoftcap) ? logit_calls : fwd_calls;
                    if (n)
                        std::fprintf(stderr, "[prof] %-12s %8.2f ms/token\n", kNames[i], acc[i] / n * 1e3);
                    sum += acc[i];
                }
                std::fprintf(stderr, "[prof] sections total %.2f ms/token averaged over all calls\n",
                             sum / fwd_calls * 1e3);
            }
        } report_at_exit;
    } // namespace prof

    static void add_into(f32 *dst, const f32 *src, u32 n) {
        for (u32 i = 0; i < n; ++i)
            dst[i] += src[i];
    }

    Model::Model(const std::string &path, u32 max_ctx, int n_threads) :
        _file(path), _pool(n_threads), _experts(_file, _file.cfg()) {
        const ModelConfigBin &c = _file.cfg();
        NUC_CHECK(c.n_experts > 0 && c.top_k > 0, "nucleus requires a MoE checkpoint");
        _max_ctx = std::min<u32>(max_ctx, c.max_pos);

        _tokenizer.load(_file.tokenizer_blob(), _file.tokenizer_bytes(), c.tok_add_dummy_prefix != 0);
        _eot_id = _tokenizer.find("<turn|>");

        _embed_scale = bf16_round(std::sqrt((f32) c.hidden));

        _inv_freq_local.resize(c.head_dim / 2);
        for (u32 j = 0; j < c.head_dim / 2; ++j) {
            _inv_freq_local[j] = std::pow(c.rope_theta_local, -((f32) (2 * j) / (f32) c.head_dim));
        }
        _inv_freq_global.assign(c.global_head_dim / 2, 0.0f);
        u32 rope_angles = (u32) (c.partial_rotary_global * (f32) c.global_head_dim) / 2;
        for (u32 j = 0; j < rope_angles; ++j) {
            _inv_freq_global[j] = std::pow(c.rope_theta_global, -((f32) (2 * j) / (f32) c.global_head_dim));
        }

        _token_emb = qview(_file, _file.tensor("tok_emb"));
        _token_emb_bytes = _file.tensor("tok_emb").nbytes;
        NUC_CHECK(_token_emb.rows == c.vocab && _token_emb.cols == c.hidden, "tok_emb shape");
        _out_norm = f32view(_file, _file.tensor("out_norm"), c.hidden);

        _layers.resize(c.n_layers);
        _kv.resize(c.n_layers);
        for (u32 l = 0; l < c.n_layers; ++l) {
            LayerWeights &w = _layers[l];
            w.full_attn = layer_is_full_attn(c, l);
            const u32 hd = w.full_attn ? c.global_head_dim : c.head_dim;
            const u32 kvh = w.full_attn ? c.n_global_kv_heads : c.n_kv_heads;
            const std::string p = "l" + std::to_string(l) + ".";

            w.attn_q = qview(_file, _file.tensor(p + "attn_q"));
            NUC_CHECK(w.attn_q.rows == c.n_heads * hd && w.attn_q.cols == c.hidden, p + "attn_q shape");
            w.attn_k = qview(_file, _file.tensor(p + "attn_k"));
            NUC_CHECK(w.attn_k.rows == kvh * hd && w.attn_k.cols == c.hidden, p + "attn_k shape");
            if (const TensorEntry *ve = _file.tensor_or_null(p + "attn_v")) {
                w.attn_v = qview(_file, *ve);
                NUC_CHECK(w.attn_v.rows == kvh * hd && w.attn_v.cols == c.hidden, p + "attn_v shape");
                w.has_v = true;
            } else {
                NUC_CHECK(w.full_attn, p + "attn_v missing on a sliding layer");
            }
            w.attn_o = qview(_file, _file.tensor(p + "attn_o"));
            NUC_CHECK(w.attn_o.rows == c.hidden && w.attn_o.cols == c.n_heads * hd, p + "attn_o shape");

            w.q_norm = f32view(_file, _file.tensor(p + "q_norm"), hd);
            w.k_norm = f32view(_file, _file.tensor(p + "k_norm"), hd);
            w.ln_in = f32view(_file, _file.tensor(p + "ln_in"), c.hidden);
            w.ln_postattn = f32view(_file, _file.tensor(p + "ln_postattn"), c.hidden);
            w.ln_preffw = f32view(_file, _file.tensor(p + "ln_preffw"), c.hidden);
            w.ln_postffw = f32view(_file, _file.tensor(p + "ln_postffw"), c.hidden);
            w.ln_postffw1 = f32view(_file, _file.tensor(p + "ln_postffw1"), c.hidden);
            w.ln_postffw2 = f32view(_file, _file.tensor(p + "ln_postffw2"), c.hidden);
            w.ln_preffw2 = f32view(_file, _file.tensor(p + "ln_preffw2"), c.hidden);

            w.mlp_gate = qview(_file, _file.tensor(p + "mlp_gate"));
            w.mlp_up = qview(_file, _file.tensor(p + "mlp_up"));
            w.mlp_down = qview(_file, _file.tensor(p + "mlp_down"));
            NUC_CHECK(w.mlp_gate.rows == c.ffn_dim && w.mlp_down.cols == c.ffn_dim, p + "mlp shape");

            w.router_w = f32view(_file, _file.tensor(p + "router_w"), (u64) c.n_experts * c.hidden);
            w.router_scale = f32view(_file, _file.tensor(p + "router_scale"), c.hidden);
            w.expert_scale = f32view(_file, _file.tensor(p + "expert_scale"), c.n_experts);
            w.layer_scalar = *f32view(_file, _file.tensor(p + "layer_scalar"), 1);

            _kv[l] = std::make_unique<LayerKV>(!w.full_attn, c.sliding_window, _max_ctx, kvh, hd);
        }

        // scratch
        const u32 max_hd = std::max(c.head_dim, c.global_head_dim);
        const u32 q_dim = c.n_heads * max_hd;
        const u32 kv_dim = std::max(c.n_kv_heads * c.head_dim, c.n_global_kv_heads * c.global_head_dim);
        const u32 max_cols = std::max({c.hidden, c.ffn_dim, c.moe_ffn_dim, q_dim});
        _h.resize(c.hidden);
        _t1.resize(c.hidden);
        _t2.resize(c.hidden);
        _xa.resize(c.hidden);
        _xb.resize(c.hidden);
        _qbuf.resize(q_dim);
        _kbuf.resize(kv_dim);
        _vbuf.resize(kv_dim);
        _att_out.resize(q_dim);
        _scores.resize((size_t) _max_ctx * c.n_heads);
        _ffn_a.resize(c.ffn_dim);
        _ffn_b.resize(c.ffn_dim);
        _moe_gu.resize((size_t) c.top_k * 2 * c.moe_ffn_dim);
        _moe_h.resize((size_t) c.top_k * c.moe_ffn_dim);
        _moe_actq.resize((size_t) c.top_k * c.moe_ffn_dim);
        _moe_acts.resize((size_t) c.top_k * c.moe_ffn_dim / kGroup);
        _moe_out.resize((size_t) c.top_k * c.hidden);
        _moe_acc.resize(c.hidden);
        _router_logits.resize(c.n_experts);
        _logits.resize(c.vocab);
        _actq.resize(max_cols);
        _acts.resize(max_cols / kGroup);

        _usage_path = path + ".usage";
        _experts.load_usage(_usage_path);
        _experts.warm_hot_experts(256);
    }

    Model::~Model() { _experts.save_usage(_usage_path); }

    void Model::mv(const QView &w, const f32 *x, f32 *out) {
        if (exact_mode()) {
            matvec_exact(w, x, out, _pool);
            return;
        }
        quantize_act(x, w.cols, _actq.data(), _acts.data());
        if (w.dtype == DType::Q8) {
            matvec_q8(w, _actq.data(), _acts.data(), out, _pool);
        } else {
            matvec_q4(w, _actq.data(), _acts.data(), out, _pool);
        }
    }

    void Model::attention(const LayerWeights &lw, u32 layer, u32 pos, const f32 *x, f32 *out) {
        const ModelConfigBin &c = _file.cfg();
        const u32 hd = lw.full_attn ? c.global_head_dim : c.head_dim;
        const u32 kvh = lw.full_attn ? c.n_global_kv_heads : c.n_kv_heads;
        const u32 n_rep = c.n_heads / kvh;
        const f32 *inv_freq = lw.full_attn ? _inv_freq_global.data() : _inv_freq_local.data();

        mv(lw.attn_q, x, _qbuf.data());
        mv(lw.attn_k, x, _kbuf.data());
        if (lw.has_v) {
            mv(lw.attn_v, x, _vbuf.data());
        } else {
            // k == v weights: V is the raw K projection, before k_norm and RoPE.
            std::memcpy(_vbuf.data(), _kbuf.data(), (size_t) kvh * hd * sizeof(f32));
        }

        for (u32 i = 0; i < kvh; ++i) {
            rmsnorm(_vbuf.data() + i * hd, _vbuf.data() + i * hd, nullptr, hd, c.rms_eps);
            rmsnorm(_kbuf.data() + i * hd, _kbuf.data() + i * hd, lw.k_norm, hd, c.rms_eps);
            rope(_kbuf.data() + i * hd, hd, inv_freq, pos);
        }
        for (u32 hq = 0; hq < c.n_heads; ++hq) {
            rmsnorm(_qbuf.data() + hq * hd, _qbuf.data() + hq * hd, lw.q_norm, hd, c.rms_eps);
            rope(_qbuf.data() + hq * hd, hd, inv_freq, pos);
        }

        LayerKV &kv = *_kv[layer];
        kv.push(pos, _kbuf.data(), _vbuf.data());

        const u32 start = kv.attn_start(pos);
        const u32 n_ctx = pos - start + 1;

        _pool.parallel_for(c.n_heads, [&](i64 h0, i64 h1) {
            for (i64 hq = h0; hq < h1; ++hq) {
                const f32 *q = _qbuf.data() + hq * hd;
                const u32 kv_head = (u32) hq / n_rep;
                f32 *s = _scores.data() + (size_t) hq * _max_ctx;
                for (u32 p = start; p <= pos; ++p) {
                    const f32 *k = kv.k_at(p) + kv_head * hd;
                    f32 acc = 0.f;
                    for (u32 d = 0; d < hd; ++d)
                        acc += q[d] * k[d];
                    s[p - start] = acc; // Gemma 4 uses attention scale 1.0 (QK-norm instead)
                }
                softmax(s, (int) n_ctx);
                f32 *o = _att_out.data() + hq * hd;
                std::memset(o, 0, hd * sizeof(f32));
                for (u32 p = start; p <= pos; ++p) {
                    const f32 *v = kv.v_at(p) + kv_head * hd;
                    const f32 sw = s[p - start];
                    for (u32 d = 0; d < hd; ++d)
                        o[d] += sw * v[d];
                }
            }
        });

        mv(lw.attn_o, _att_out.data(), out);
    }

    void Model::moe_ffn(const LayerWeights &lw, u32 layer, const f32 *residual, f32 *out) {
        const ModelConfigBin &c = _file.cfg();
        const u32 k = c.top_k;

        u32 ids[64];
        f32 wts[64];
        {
            prof::Timer _pt(prof::kRouter);
            // --- router (on the pre-FFN residual), before the dense MLP so the
            // madvise readahead overlaps with the dense compute below.
            rmsnorm(_t1.data(), residual, nullptr, c.hidden, c.rms_eps);
            const f32 root = std::pow((f32) c.hidden, -0.5f);
            for (u32 i = 0; i < c.hidden; ++i)
                _t1[i] *= lw.router_scale[i] * root;
            matvec_f32(lw.router_w, c.n_experts, c.hidden, _t1.data(), _router_logits.data(), _pool);
            softmax(_router_logits.data(), (int) c.n_experts);

            NUC_CHECK(k <= 64, "top_k too large");
            for (u32 j = 0; j < k; ++j) {
                u32 best = 0;
                f32 bv = -1.f;
                for (u32 e = 0; e < c.n_experts; ++e) {
                    bool taken = false;
                    for (u32 t = 0; t < j; ++t)
                        taken |= (ids[t] == e);
                    if (!taken && _router_logits[e] > bv) {
                        bv = _router_logits[e];
                        best = e;
                    }
                }
                ids[j] = best;
                wts[j] = bv;
            }
            f32 wsum = 0.f;
            for (u32 j = 0; j < k; ++j)
                wsum += wts[j];
            for (u32 j = 0; j < k; ++j)
                wts[j] = wts[j] / wsum * lw.expert_scale[ids[j]];

            if (!_experts.enforce_max_rss())
                _cap_holding = false;
            _experts.prefetch(layer, ids, k);
        }
        {
            prof::Timer _pt(prof::kDenseMlp);
            rmsnorm(_t1.data(), residual, lw.ln_preffw, c.hidden, c.rms_eps);
            mv(lw.mlp_gate, _t1.data(), _ffn_a.data());
            mv(lw.mlp_up, _t1.data(), _ffn_b.data());
            f32 *fa = _ffn_a.data();
            const f32 *fb = _ffn_b.data();
            _pool.parallel_for(c.ffn_dim, [&](i64 i0, i64 i1) {
                for (i64 i = i0; i < i1; ++i)
                    fa[i] = gelu_tanh(fa[i]) * fb[i];
            });
            mv(lw.mlp_down, _ffn_a.data(), _xa.data());
            rmsnorm(_xa.data(), _xa.data(), lw.ln_postffw1, c.hidden, c.rms_eps); // x1
        }
        {
            prof::Timer _pt(prof::kExperts);
            rmsnorm(_t1.data(), residual, lw.ln_preffw2, c.hidden, c.rms_eps);
            const u32 moe = c.moe_ffn_dim;
            const u32 gu_rows = 2 * moe;
            const bool exact = exact_mode();
            ExpertView evs[64];
            for (u32 j = 0; j < k; ++j)
                evs[j] = _experts.view(layer, ids[j]);

            if (!exact)
                quantize_act(_t1.data(), (int) c.hidden, _actq.data(), _acts.data());
            _pool.parallel_for((i64) k * gu_rows, [&](i64 r0, i64 r1) {
                while (r0 < r1) {
                    const u32 j = (u32) (r0 / gu_rows);
                    const u32 lo = (u32) (r0 % gu_rows);
                    const u32 hi = (u32) std::min<i64>(gu_rows, lo + (r1 - r0));
                    f32 *out = _moe_gu.data() + (size_t) j * gu_rows + lo;
                    if (exact) {
                        matvec_exact_rows(evs[j].gate_up, _t1.data(), out, lo, hi);
                    } else {
                        matvec_q4_rows(evs[j].gate_up, _actq.data(), _acts.data(), out, lo, hi);
                    }
                    r0 += hi - lo;
                }
            });

            _pool.parallel_for((i64) k * moe, [&](i64 f0, i64 f1) {
                for (i64 f = f0; f < f1; ++f) {
                    const size_t j = (size_t) (f / moe), i = (size_t) (f % moe);
                    const f32 *gu = _moe_gu.data() + j * gu_rows;
                    _moe_h[j * moe + i] = gelu_tanh(gu[i]) * gu[i + moe];
                }
            });
            if (!exact) {
                for (u32 j = 0; j < k; ++j) {
                    quantize_act(_moe_h.data() + (size_t) j * moe, (int) moe, _moe_actq.data() + (size_t) j * moe,
                                 _moe_acts.data() + (size_t) j * (moe / kGroup));
                }
            }

            _pool.parallel_for((i64) k * c.hidden, [&](i64 r0, i64 r1) {
                while (r0 < r1) {
                    const u32 j = (u32) (r0 / c.hidden);
                    const u32 lo = (u32) (r0 % c.hidden);
                    const u32 hi = (u32) std::min<i64>(c.hidden, lo + (r1 - r0));
                    f32 *out = _moe_out.data() + (size_t) j * c.hidden + lo;
                    if (exact) {
                        matvec_exact_rows(evs[j].down, _moe_h.data() + (size_t) j * moe, out, lo, hi);
                    } else {
                        matvec_q4_rows(evs[j].down, _moe_actq.data() + (size_t) j * moe,
                                       _moe_acts.data() + (size_t) j * (moe / kGroup), out, lo, hi);
                    }
                    r0 += hi - lo;
                }
            });
            std::memset(_moe_acc.data(), 0, c.hidden * sizeof(f32));
            for (u32 j = 0; j < k; ++j) {
                const f32 wj = wts[j];
                const f32 *oj = _moe_out.data() + (size_t) j * c.hidden;
                for (u32 i = 0; i < c.hidden; ++i)
                    _moe_acc[i] += wj * oj[i];
            }
            rmsnorm(_moe_acc.data(), _moe_acc.data(), lw.ln_postffw2, c.hidden, c.rms_eps); // x2
        }

        for (u32 i = 0; i < c.hidden; ++i)
            out[i] = _xa[i] + _moe_acc[i];
        rmsnorm(out, out, lw.ln_postffw, c.hidden, c.rms_eps);
    }

    void Model::forward(i32 token, u32 pos, bool want_logits) {
        const ModelConfigBin &c = _file.cfg();
        NUC_CHECK(token >= 0 && (u32) token < c.vocab, "token out of range");
        NUC_CHECK(pos < _max_ctx, "position beyond --ctx");

        if (prof::on()) {
            ++prof::fwd_calls;
            if (want_logits)
                ++prof::logit_calls;
        }
        dequant_row(_token_emb, (u32) token, _h.data());
        for (u32 i = 0; i < c.hidden; ++i)
            _h[i] *= _embed_scale;

        for (u32 l = 0; l < c.n_layers; ++l) {
            const LayerWeights &lw = _layers[l];

            {
                prof::Timer _pt(prof::kAttn);
                rmsnorm(_t1.data(), _h.data(), lw.ln_in, c.hidden, c.rms_eps);
                attention(lw, l, pos, _t1.data(), _t2.data());
                rmsnorm(_t2.data(), _t2.data(), lw.ln_postattn, c.hidden, c.rms_eps);
                add_into(_h.data(), _t2.data(), c.hidden);
            }

            moe_ffn(lw, l, _h.data(), _t2.data());
            add_into(_h.data(), _t2.data(), c.hidden);

            if (lw.layer_scalar != 1.0f) {
                for (u32 i = 0; i < c.hidden; ++i)
                    _h[i] *= lw.layer_scalar;
            }
        }

        if (!want_logits)
            return;

        if (!_logits_warmed) {
            if (!_experts.enforce_max_rss(_token_emb_bytes))
                _cap_holding = false;
            _logits_warmed = true;
        }
        rmsnorm(_t1.data(), _h.data(), _out_norm, c.hidden, c.rms_eps);
        {
            prof::Timer _pt(prof::kLogitsMv);
            mv(_token_emb, _t1.data(), _logits.data());
        }
        if (c.final_softcap > 0.f) {
            prof::Timer _pt(prof::kSoftcap);
            const f32 cap = c.final_softcap;
            f32 *lg = _logits.data();
            _pool.parallel_for(c.vocab, [&](i64 i0, i64 i1) {
                for (i64 i = i0; i < i1; ++i)
                    lg[i] = cap * std::tanh(lg[i] / cap);
            });
        }
    }
} // namespace nucleus
