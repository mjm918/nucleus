#  *************************************************************************
#  Copyright (c) 2026 Mohammad Julfikar.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  **************************************************************************

#!/usr/bin/env python3
"""Independent NumPy implementation of the nucleus forward pass.

Reads a .nuc file, dequantizes the weights, runs the Gemma 4 forward pass in
float32, and dumps per-step logits for a fixed token sequence:

  python3 tools/reference.py toy.nuc toy.expected [tok tok ...]

Output: u32 n_steps, u32 vocab, i32 tokens[n_steps], f32 logits[n_steps*vocab].
The C++ engine must reproduce these logits (tests/test_parity.cpp).
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert as C  # noqa: E402  (shared format constants)

CFG_KEYS_U32 = ["hidden", "n_layers", "n_heads", "n_kv_heads", "head_dim",
                "global_head_dim", "n_global_kv_heads", "ffn_dim", "moe_ffn_dim",
                "n_experts", "top_k", "vocab", "sliding_window", "max_pos"]
CFG_KEYS_F32 = ["rope_theta_local", "rope_theta_global", "partial_rotary_global",
                "rms_eps", "final_softcap"]


class Nuc:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        vals = struct.unpack_from(C.HDR_FMT, self.buf, 0)
        assert vals[0] == C.MAGIC and vals[1] == C.VERSION
        i = 2
        self.cfg = {}
        for k in CFG_KEYS_U32:
            self.cfg[k] = vals[i]; i += 1
        for k in CFG_KEYS_F32:
            self.cfg[k] = vals[i]; i += 1
        self.cfg["full_attn_bits"] = list(vals[i : i + 8]); i += 8
        for k in ["bos_id", "eos_id", "pad_id", "tok_add_dummy_prefix"]:
            self.cfg[k] = vals[i]; i += 1
        tok_off, tok_bytes, table_off, count, _ = vals[i : i + 5]
        self.tensors = {}
        sz = struct.calcsize(C.TENSOR_FMT)
        for t in range(count):
            e = struct.unpack_from(C.TENSOR_FMT, self.buf, table_off + t * sz)
            name = e[0].rstrip(b"\0").decode()
            self.tensors[name] = dict(dtype=e[1], ndim=e[2], shape=e[4:8],
                                      offset=e[8], nbytes=e[9])

    def full_attn(self, l):
        return (self.cfg["full_attn_bits"][l // 32] >> (l % 32)) & 1

    def raw(self, name):
        t = self.tensors[name]
        return self.buf[t["offset"] : t["offset"] + t["nbytes"]], t

    def f32(self, name):
        raw, t = self.raw(name)
        return np.frombuffer(raw, np.float32).copy()

    def _dq_q8(self, raw, rows, cols):
        gs = cols // C.GROUP
        scales = np.frombuffer(raw, np.float16, rows * gs).astype(np.float32).reshape(rows, gs)
        q = np.frombuffer(raw, np.int8, rows * cols, offset=rows * gs * 2)
        q = q.reshape(rows, gs, C.GROUP).astype(np.float32)
        return (q * scales[:, :, None]).reshape(rows, cols)

    def _dq_q4(self, raw, rows, cols):
        gs = cols // C.GROUP
        scales = np.frombuffer(raw, np.float16, rows * gs).astype(np.float32).reshape(rows, gs)
        p = np.frombuffer(raw, np.uint8, rows * cols // 2, offset=rows * gs * 2)
        p = p.reshape(rows, gs, C.GROUP // 2)
        lo = (p & 0x0F).astype(np.int32) - 8
        hi = (p >> 4).astype(np.int32) - 8
        q = np.concatenate([lo, hi], axis=-1).astype(np.float32)  # [rows, gs, 32]
        return (q * scales[:, :, None]).reshape(rows, cols)

    def dequant(self, name):
        raw, t = self.raw(name)
        rows, cols = t["shape"][0], t["shape"][1]
        if t["dtype"] == C.DT_Q8:
            return self._dq_q8(raw, rows, cols)
        if t["dtype"] == C.DT_Q4:
            return self._dq_q4(raw, rows, cols)
        raise ValueError(name)

    def expert(self, l, e):
        """Returns (gate_up [2*moe, H], down [H, moe]) dequantized."""
        raw, _ = self.raw(f"l{l}.exp{e}")
        H, moe = self.cfg["hidden"], self.cfg["moe_ffn_dim"]
        gu_nb = C.q4_nbytes(2 * moe, H)
        return self._dq_q4(raw[:gu_nb], 2 * moe, H), self._dq_q4(raw[gu_nb:], H, moe)


def bf16_round(x):
    b = np.float32(x).view(np.uint32)
    b = np.uint32((int(b) + 0x7FFF + ((int(b) >> 16) & 1)) & 0xFFFF0000)
    return b.view(np.float32)


def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    inv = 1.0 / np.sqrt(np.mean(x * x) + eps)
    y = x * np.float32(inv)
    return y * w if w is not None else y


def gelu_tanh(x):
    k = np.float32(0.7978845608028654)
    return 0.5 * x * (1.0 + np.tanh(k * (x + np.float32(0.044715) * x * x * x)))


def rope(v, inv_freq, pos):
    hd = v.shape[-1]
    half = hd // 2
    f = np.float32(pos) * inv_freq
    c, s = np.cos(f).astype(np.float32), np.sin(f).astype(np.float32)
    out = v.copy()
    out[..., :half] = v[..., :half] * c - v[..., half:] * s
    out[..., half:] = v[..., half:] * c + v[..., :half] * s
    return out


def forward_all(nuc, tokens):
    c = nuc.cfg
    H = c["hidden"]
    emb = nuc.dequant("tok_emb")
    out_norm = nuc.f32("out_norm")
    escale = bf16_round(np.sqrt(np.float32(H)))

    inv_local = (c["rope_theta_local"] **
                 -(np.arange(0, c["head_dim"], 2, dtype=np.float32) / c["head_dim"]))
    ghd = c["global_head_dim"]
    rope_angles = int(c["partial_rotary_global"] * ghd) // 2
    inv_global = np.zeros(ghd // 2, np.float32)
    inv_global[:rope_angles] = (c["rope_theta_global"] **
                                -(np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / ghd))

    L = []
    for l in range(c["n_layers"]):
        n = f"l{l}."
        full = nuc.full_attn(l)
        lw = dict(
            full=full,
            q=nuc.dequant(n + "attn_q"), k=nuc.dequant(n + "attn_k"),
            v=nuc.dequant(n + "attn_v") if not full else None,
            o=nuc.dequant(n + "attn_o"),
            q_norm=nuc.f32(n + "q_norm"), k_norm=nuc.f32(n + "k_norm"),
            ln_in=nuc.f32(n + "ln_in"), ln_postattn=nuc.f32(n + "ln_postattn"),
            ln_preffw=nuc.f32(n + "ln_preffw"), ln_postffw=nuc.f32(n + "ln_postffw"),
            ln_postffw1=nuc.f32(n + "ln_postffw1"), ln_postffw2=nuc.f32(n + "ln_postffw2"),
            ln_preffw2=nuc.f32(n + "ln_preffw2"),
            gate=nuc.dequant(n + "mlp_gate"), up=nuc.dequant(n + "mlp_up"),
            down=nuc.dequant(n + "mlp_down"),
            router_w=nuc.f32(n + "router_w").reshape(c["n_experts"], H),
            router_scale=nuc.f32(n + "router_scale"),
            expert_scale=nuc.f32(n + "expert_scale"),
            layer_scalar=nuc.f32(n + "layer_scalar")[0],
            kcache=[], vcache=[],
        )
        L.append(lw)

    logits_out = []
    for pos, tok in enumerate(tokens):
        h = emb[tok] * escale
        for l, lw in enumerate(L):
            full = lw["full"]
            hd = ghd if full else c["head_dim"]
            kvh = c["n_global_kv_heads"] if full else c["n_kv_heads"]
            inv = inv_global if full else inv_local
            n_rep = c["n_heads"] // kvh

            x = rmsnorm(h, lw["ln_in"], c["rms_eps"])
            q = (lw["q"] @ x).reshape(c["n_heads"], hd)
            kraw = (lw["k"] @ x).reshape(kvh, hd)
            vraw = (lw["v"] @ x).reshape(kvh, hd) if lw["v"] is not None else kraw.copy()

            k = np.stack([rope(rmsnorm(kraw[i], lw["k_norm"], c["rms_eps"]), inv, pos)
                          for i in range(kvh)])
            v = np.stack([rmsnorm(vraw[i], None, c["rms_eps"]) for i in range(kvh)])
            q = np.stack([rope(rmsnorm(q[i], lw["q_norm"], c["rms_eps"]), inv, pos)
                          for i in range(c["n_heads"])])
            lw["kcache"].append(k)
            lw["vcache"].append(v)

            start = 0
            if not full and pos + 1 > c["sliding_window"]:
                start = pos + 1 - c["sliding_window"]
            ks = np.stack(lw["kcache"][start : pos + 1])  # [T, kvh, hd]
            vs = np.stack(lw["vcache"][start : pos + 1])

            att = np.zeros(c["n_heads"] * hd, np.float32)
            for hq in range(c["n_heads"]):
                kvi = hq // n_rep
                s = ks[:, kvi, :] @ q[hq]  # scale 1.0
                s = s - s.max()
                p = np.exp(s); p /= p.sum()
                att[hq * hd : (hq + 1) * hd] = p @ vs[:, kvi, :]
            attn_out = lw["o"] @ att
            h = h + rmsnorm(attn_out, lw["ln_postattn"], c["rms_eps"])

            residual = h
            # router
            r = rmsnorm(residual, None, c["rms_eps"])
            r = r * lw["router_scale"] * np.float32(H ** -0.5)
            rl = lw["router_w"] @ r
            rl = rl - rl.max()
            probs = np.exp(rl.astype(np.float32)); probs /= probs.sum()
            order = np.argsort(-probs, kind="stable")[: c["top_k"]]
            wts = probs[order]
            wts = wts / wts.sum() * lw["expert_scale"][order]

            # dense MLP
            xm = rmsnorm(residual, lw["ln_preffw"], c["rms_eps"])
            mlp = lw["down"] @ (gelu_tanh(lw["gate"] @ xm) * (lw["up"] @ xm))
            x1 = rmsnorm(mlp, lw["ln_postffw1"], c["rms_eps"])

            # experts
            xe = rmsnorm(residual, lw["ln_preffw2"], c["rms_eps"])
            acc = np.zeros(H, np.float32)
            moe = c["moe_ffn_dim"]
            for j, e in enumerate(order):
                gu, dn = nuc.expert(l, int(e))
                z = gu @ xe
                hexp = gelu_tanh(z[:moe]) * z[moe:]
                acc = acc + wts[j] * (dn @ hexp)
            x2 = rmsnorm(acc, lw["ln_postffw2"], c["rms_eps"])

            h = h + rmsnorm(x1 + x2, lw["ln_postffw"], c["rms_eps"])
            h = h * lw["layer_scalar"]

        xf = rmsnorm(h, out_norm, c["rms_eps"])
        logits = emb @ xf
        cap = c["final_softcap"]
        if cap > 0:
            logits = cap * np.tanh(logits / cap)
        logits_out.append(logits.astype(np.float32))
    return logits_out


def main():
    nuc_path, out_path = sys.argv[1], sys.argv[2]
    nuc = Nuc(nuc_path)
    if len(sys.argv) > 3:
        tokens = [int(t) for t in sys.argv[3:]]
    else:
        tokens = [2, 265, 270, 267, 268, 269, 100, 42, 55, 56, 57, 58]
    assert all(0 <= t < nuc.cfg["vocab"] for t in tokens)
    logits = forward_all(nuc, tokens)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", len(tokens), nuc.cfg["vocab"]))
        f.write(np.asarray(tokens, np.int32).tobytes())
        for lg in logits:
            f.write(lg.tobytes())
    print(f"wrote {out_path}: {len(tokens)} steps, vocab {nuc.cfg['vocab']}")
    print("last-step top5:", np.argsort(-logits[-1])[:5], np.sort(logits[-1])[::-1][:5])


if __name__ == "__main__":
    main()