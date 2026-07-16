#!/usr/bin/env python3
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

import argparse
import json
import mmap
import os
import re
import struct
import sys

import numpy as np

MAGIC = 0x3143554E  # "NUC1"
VERSION = 1
GROUP = 32
ALIGN = 64

DT_F32, DT_F16, DT_Q8, DT_Q4 = 0, 1, 2, 3

CFG_FMT = "<14I5f8I4I"
HDR_FMT = "<II" + CFG_FMT[1:] + "QQQII"
TENSOR_FMT = "<96sBBH4IQQ"

# sentencepiece piece types
SP_NORMAL, SP_UNKNOWN, SP_CONTROL, SP_USER_DEFINED, SP_UNUSED, SP_BYTE = 1, 2, 3, 4, 5, 6


# ---------------------------------------------------------------------------
# quantization


def quant_q8(w: np.ndarray) -> bytes:
    rows, cols = w.shape
    assert cols % GROUP == 0, f"cols {cols} not divisible by {GROUP}"
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    amax = np.abs(g).max(axis=-1)
    scales = (amax / 127.0).astype(np.float16)
    s32 = scales.astype(np.float32)
    inv = np.where(s32 > 0, 1.0 / np.where(s32 > 0, s32, 1), 0.0).astype(np.float32)
    q = np.clip(np.rint(g * inv[..., None]), -127, 127).astype(np.int8)
    return scales.tobytes() + q.tobytes()


def quant_q4(w: np.ndarray) -> bytes:
    rows, cols = w.shape
    assert cols % GROUP == 0, f"cols {cols} not divisible by {GROUP}"
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    amax = np.abs(g).max(axis=-1)
    scales = (amax / 7.0).astype(np.float16)
    s32 = scales.astype(np.float32)
    inv = np.where(s32 > 0, 1.0 / np.where(s32 > 0, s32, 1), 0.0).astype(np.float32)
    q = np.clip(np.rint(g * inv[..., None]), -8, 7).astype(np.int8) + 8
    lo = q[..., :16].astype(np.uint8)
    hi = q[..., 16:].astype(np.uint8)
    packed = (lo | (hi << 4)).astype(np.uint8)
    return scales.tobytes() + packed.tobytes()


def q8_nbytes(rows, cols):
    return rows * (cols // GROUP) * 2 + rows * cols


def q4_nbytes(rows, cols):
    return rows * (cols // GROUP) * 2 + rows * cols // 2


# ---------------------------------------------------------------------------
# .nuc writer


class NucWriter:
    def __init__(self, path, cfg: dict, tok_blob: bytes):
        self.f = open(path, "wb")
        self.cfg = cfg
        self.entries = []
        hdr_size = struct.calcsize(HDR_FMT)
        self.f.write(b"\0" * hdr_size)
        self.tok_offset = self._align()
        self.tok_bytes = len(tok_blob)
        self.f.write(tok_blob)

    def _align(self):
        off = self.f.tell()
        pad = (-off) % ALIGN
        if pad:
            self.f.write(b"\0" * pad)
        return self.f.tell()

    def add(self, name: str, dtype: int, shape, data: bytes):
        off = self._align()
        self.f.write(data)
        shp = list(shape) + [0] * (4 - len(shape))
        self.entries.append((name, dtype, len(shape), shp, off, len(data)))

    def reserve(self, name: str, dtype: int, shape, nbytes: int):
        """Reserve space; returns file offset. Caller seeks+writes chunks."""
        off = self._align()
        self.f.seek(off + nbytes - 1)
        self.f.write(b"\0")
        shp = list(shape) + [0] * (4 - len(shape))
        self.entries.append((name, dtype, len(shape), shp, off, nbytes))
        return off

    def finish(self):
        self.f.seek(0, os.SEEK_END)
        table_offset = self._align()
        for name, dtype, ndim, shp, off, nb in self.entries:
            nb_name = name.encode()
            assert len(nb_name) < 96, name
            self.f.write(struct.pack(TENSOR_FMT, nb_name, dtype, ndim, 0, *shp, off, nb))
        c = self.cfg
        hdr = struct.pack(
            HDR_FMT, MAGIC, VERSION,
            c["hidden"], c["n_layers"], c["n_heads"], c["n_kv_heads"], c["head_dim"],
            c["global_head_dim"], c["n_global_kv_heads"], c["ffn_dim"], c["moe_ffn_dim"],
            c["n_experts"], c["top_k"], c["vocab"], c["sliding_window"], c["max_pos"],
            c["rope_theta_local"], c["rope_theta_global"], c["partial_rotary_global"],
            c["rms_eps"], c["final_softcap"],
            *c["full_attn_bits"],
            c["bos_id"], c["eos_id"], c["pad_id"], c["tok_add_dummy_prefix"],
            self.tok_offset, self.tok_bytes, table_offset, len(self.entries), 0)
        self.f.seek(0)
        self.f.write(hdr)
        self.f.close()


def build_tok_blob(pieces):
    """pieces: list of (text:str|bytes, score:float, type:int)"""
    out = [struct.pack("<I", len(pieces))]
    for text, score, typ in pieces:
        b = text.encode() if isinstance(text, str) else text
        out.append(struct.pack("<fBBH", score, typ, 0, len(b)) + b)
    return b"".join(out)


def full_attn_bits(layer_types):
    bits = [0] * 8
    for i, t in enumerate(layer_types):
        if t == "full_attention":
            bits[i // 32] |= 1 << (i % 32)
    return bits


# ---------------------------------------------------------------------------
# safetensors (manual, mmap-backed)


class Shards:
    def __init__(self, model_dir):
        self.dir = model_dir
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            idx = json.load(open(idx_path))
            self.weight_map = idx["weight_map"]
        else:
            self.weight_map = None  # single file
        self.files = {}

    def _open(self, fname):
        if fname not in self.files:
            f = open(os.path.join(self.dir, fname), "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            (hlen,) = struct.unpack("<Q", mm[:8])
            hdr = json.loads(mm[8 : 8 + hlen].decode())
            self.files[fname] = (mm, hdr, 8 + hlen)
        return self.files[fname]

    def get(self, name) -> np.ndarray:
        """Returns float32 array (bf16/f32/f16 supported)."""
        fname = self.weight_map[name] if self.weight_map else "model.safetensors"
        mm, hdr, base = self._open(fname)
        meta = hdr[name]
        b0, b1 = meta["data_offsets"]
        raw = np.frombuffer(mm, dtype=np.uint8, count=b1 - b0, offset=base + b0)
        shape = meta["shape"]
        dt = meta["dtype"]
        if dt == "BF16":
            u16 = raw.view(np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            return f32.reshape(shape)
        if dt == "F32":
            return raw.view(np.float32).reshape(shape)
        if dt == "F16":
            return raw.view(np.float16).astype(np.float32).reshape(shape)
        raise ValueError(f"unsupported dtype {dt} for {name}")


# ---------------------------------------------------------------------------
# sentencepiece model parser (hand-rolled protobuf)


def _read_varint(buf, i):
    v, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, i
        shift += 7


def _fields(buf):
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        fno, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _read_varint(buf, i)
        elif wire == 1:
            v, i = buf[i : i + 8], i + 8
        elif wire == 2:
            ln, i = _read_varint(buf, i)
            v, i = buf[i : i + ln], i + ln
        elif wire == 5:
            v, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield fno, wire, v


def parse_sentencepiece(path):
    """Returns (pieces [(bytes, score, type)], add_dummy_prefix)."""
    buf = open(path, "rb").read()
    pieces = []
    add_dummy_prefix = True
    for fno, wire, v in _fields(buf):
        if fno == 1 and wire == 2:  # SentencePiece
            text, score, typ = b"", 0.0, SP_NORMAL
            for f2, w2, v2 in _fields(v):
                if f2 == 1:
                    text = v2
                elif f2 == 2:
                    (score,) = struct.unpack("<f", v2)
                elif f2 == 3:
                    typ = v2
            pieces.append((text, score, typ))
        elif fno == 3 and wire == 2:  # NormalizerSpec
            for f2, w2, v2 in _fields(v):
                if f2 == 3:
                    add_dummy_prefix = bool(v2)
    return pieces, add_dummy_prefix


# ---------------------------------------------------------------------------
# tokenizer.json parser (HF BPE)

BYTE_PIECE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def _has_prepend(norm):
    if not isinstance(norm, dict):
        return False
    if norm.get("type") == "Prepend":
        return True
    if norm.get("type") == "Sequence":
        return any(_has_prepend(n) for n in norm.get("normalizers", []))
    return False


def parse_tokenizer_json(path):
    """Returns (pieces [(str, score, type)], add_dummy_prefix).

    Gemma 4 ships tokenizer.json (HF BPE) and no tokenizer.model. The two carry
    the same information for us: sentencepiece BPE merges the adjacent pair
    whose *concatenation* scores best, and HF derives `merges` by sorting those
    same concatenations by that score, descending. So rank is the score order,
    and score = -rank reproduces the engine's merge decisions exactly. Only the
    ordering matters -- Tokenizer::spm_encode just takes the max.
    """
    d = json.load(open(path))
    m = d["model"]
    assert m.get("type") == "BPE", f"expected a BPE tokenizer.json, got {m.get('type')}"

    vocab = m["vocab"]
    n = len(vocab)
    texts = [None] * n
    for text, i in vocab.items():
        assert 0 <= i < n, f"vocab id {i} out of range for {text!r}"
        assert texts[i] is None, f"duplicate vocab id {i}"
        texts[i] = text

    # Pieces that are never a merge result are unreachable by the merge loop
    # (it only looks up concatenations), so park them below every real merge.
    merges = m["merges"]
    floor = -float(len(merges) + 1)
    scores = [floor] * n
    for rank, pair in enumerate(merges):
        a, b = pair if isinstance(pair, (list, tuple)) else pair.split(" ", 1)
        i = vocab.get(a + b)
        if i is not None and scores[i] == floor:
            scores[i] = -float(rank)  # first (best-scoring) rank wins

    specials = {t["id"] for t in d.get("added_tokens", []) if t.get("special")}
    unk = m.get("unk_token")
    controls = {"<pad>", "<eos>", "<bos>", "<mask>"}

    types = [SP_NORMAL] * n
    n_bytes = 0
    for i, text in enumerate(texts):
        if text == unk:
            types[i] = SP_UNKNOWN
        elif i in specials:
            types[i] = SP_CONTROL if text in controls else SP_USER_DEFINED
        elif BYTE_PIECE_RE.match(text):
            types[i] = SP_BYTE
            n_bytes += 1

    assert n_bytes == 256, f"expected 256 <0xNN> byte pieces, found {n_bytes}"
    assert SP_UNKNOWN in types, f"unk_token {unk!r} not in vocab"

    # sentencepiece prepends the dummy prefix in the normalizer; HF encodes that
    # as a Prepend normalizer. Gemma 4 has only Replace(" " -> U+2581).
    return list(zip(texts, scores, types)), _has_prepend(d.get("normalizer"))


def load_tokenizer(model_dir):
    """Prefer sentencepiece (Gemma 3 and earlier); fall back to tokenizer.json
    (Gemma 4 ships no tokenizer.model)."""
    sp_path = os.path.join(model_dir, "tokenizer.model")
    tj_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(sp_path):
        return parse_sentencepiece(sp_path)
    assert os.path.exists(tj_path), "no tokenizer.model or tokenizer.json in model dir"
    return parse_tokenizer_json(tj_path)


# ---------------------------------------------------------------------------
# real model conversion


def convert_real(model_dir, out_path, dense_q4=False):
    cfg_json = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg_json["text_config"] if "text_config" in cfg_json else cfg_json
    assert tc.get("enable_moe_block", False), "nucleus expects the MoE Gemma 4 variant"
    assert tc.get("num_kv_shared_layers", 0) == 0, "kv-shared layers not supported"
    assert not tc.get("hidden_size_per_layer_input", 0), "per-layer inputs not supported"

    layer_types = tc["layer_types"]
    rope = tc["rope_parameters"]
    cfg = dict(
        hidden=tc["hidden_size"],
        n_layers=tc["num_hidden_layers"],
        n_heads=tc["num_attention_heads"],
        n_kv_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        global_head_dim=tc.get("global_head_dim") or tc["head_dim"],
        n_global_kv_heads=tc.get("num_global_key_value_heads") or tc["num_key_value_heads"],
        ffn_dim=tc["intermediate_size"],
        moe_ffn_dim=tc["moe_intermediate_size"],
        n_experts=tc["num_experts"],
        top_k=tc["top_k_experts"],
        vocab=tc["vocab_size"],
        sliding_window=tc["sliding_window"],
        max_pos=tc["max_position_embeddings"],
        rope_theta_local=float(rope["sliding_attention"]["rope_theta"]),
        rope_theta_global=float(rope["full_attention"]["rope_theta"]),
        partial_rotary_global=float(rope["full_attention"].get("partial_rotary_factor", 1.0)),
        rms_eps=float(tc["rms_norm_eps"]),
        final_softcap=float(tc.get("final_logit_softcapping") or 0.0),
        full_attn_bits=full_attn_bits(layer_types),
        bos_id=tc["bos_token_id"],
        eos_id=tc["eos_token_id"],
        pad_id=tc.get("pad_token_id", 0),
    )

    pieces, add_dummy_prefix = load_tokenizer(model_dir)
    assert len(pieces) == cfg["vocab"], f"vocab mismatch: {len(pieces)} vs {cfg['vocab']}"
    cfg["tok_add_dummy_prefix"] = int(add_dummy_prefix)
    tok_blob = build_tok_blob(pieces)

    shards = Shards(model_dir)
    P = "model.language_model."
    w = NucWriter(out_path, cfg, tok_blob)
    H, moe = cfg["hidden"], cfg["moe_ffn_dim"]

    # The dense set defaults to q8; --dense-q4 halves its read traffic per
    # token (the experts are q4 already) at some quality cost.
    dq, dq_nbytes, dq_dt, dq_tag = (
        (quant_q4, q4_nbytes, DT_Q4, "q4") if dense_q4 else
        (quant_q8, q8_nbytes, DT_Q8, "q8"))

    def put_dense(name, hf_name):
        t = shards.get(hf_name)
        w.add(name, dq_dt, t.shape, dq(t))
        print(f"  {name:24s} {dq_tag} {list(t.shape)}", flush=True)

    def put_f32(name, hf_name, expect=None):
        t = shards.get(hf_name).astype(np.float32)
        if expect is not None:
            assert t.size == expect, f"{hf_name}: {t.size} != {expect}"
        w.add(name, DT_F32, t.shape, t.tobytes())

    # embeddings: chunked (262k x 2816 doesn't fit comfortably in RAM as f32)
    emb = shards.get(P + "embed_tokens.weight")
    rows, cols = emb.shape
    row_qbytes = cols // 2 if dense_q4 else cols
    off = w.reserve("tok_emb", dq_dt, (rows, cols), dq_nbytes(rows, cols))
    scales_bytes = rows * (cols // GROUP) * 2
    chunk = 16384
    for r in range(0, rows, chunk):
        blk = np.ascontiguousarray(emb[r : r + chunk])
        data = dq(blk)
        n = blk.shape[0]
        sb = n * (cols // GROUP) * 2
        w.f.seek(off + r * (cols // GROUP) * 2)
        w.f.write(data[:sb])
        w.f.seek(off + scales_bytes + r * row_qbytes)
        w.f.write(data[sb:])
        print(f"  tok_emb rows {r}..{r + n}", flush=True)
    w.f.seek(0, os.SEEK_END)

    put_f32("out_norm", P + "norm.weight", cfg["hidden"])

    for l in range(cfg["n_layers"]):
        lp = f"{P}layers.{l}."
        n = f"l{l}."
        full = layer_types[l] == "full_attention"
        print(f"layer {l} ({'full' if full else 'sliding'})", flush=True)
        put_dense(n + "attn_q", lp + "self_attn.q_proj.weight")
        put_dense(n + "attn_k", lp + "self_attn.k_proj.weight")
        if not full:
            put_dense(n + "attn_v", lp + "self_attn.v_proj.weight")
        put_dense(n + "attn_o", lp + "self_attn.o_proj.weight")
        put_f32(n + "q_norm", lp + "self_attn.q_norm.weight")
        put_f32(n + "k_norm", lp + "self_attn.k_norm.weight")
        put_f32(n + "ln_in", lp + "input_layernorm.weight", H)
        put_f32(n + "ln_postattn", lp + "post_attention_layernorm.weight", H)
        put_f32(n + "ln_preffw", lp + "pre_feedforward_layernorm.weight", H)
        put_f32(n + "ln_postffw", lp + "post_feedforward_layernorm.weight", H)
        put_f32(n + "ln_postffw1", lp + "post_feedforward_layernorm_1.weight", H)
        put_f32(n + "ln_postffw2", lp + "post_feedforward_layernorm_2.weight", H)
        put_f32(n + "ln_preffw2", lp + "pre_feedforward_layernorm_2.weight", H)
        put_dense(n + "mlp_gate", lp + "mlp.gate_proj.weight")
        put_dense(n + "mlp_up", lp + "mlp.up_proj.weight")
        put_dense(n + "mlp_down", lp + "mlp.down_proj.weight")
        put_f32(n + "router_w", lp + "router.proj.weight", cfg["n_experts"] * H)
        put_f32(n + "router_scale", lp + "router.scale", H)
        put_f32(n + "expert_scale", lp + "router.per_expert_scale", cfg["n_experts"])
        put_f32(n + "layer_scalar", lp + "layer_scalar", 1)

    # experts last: grouped by layer, one contiguous blob per expert
    for l in range(cfg["n_layers"]):
        lp = f"{P}layers.{l}."
        gu = shards.get(lp + "experts.gate_up_proj")   # [E, 2*moe, H] (mmap-lazy)
        dn = shards.get(lp + "experts.down_proj")      # [E, H, moe]
        assert gu.shape == (cfg["n_experts"], 2 * moe, H), gu.shape
        assert dn.shape == (cfg["n_experts"], H, moe), dn.shape
        for e in range(cfg["n_experts"]):
            blob = quant_q4(np.ascontiguousarray(gu[e])) + quant_q4(np.ascontiguousarray(dn[e]))
            w.add(f"l{l}.exp{e}", DT_Q4, (2 * moe + H,), blob)
        print(f"experts layer {l}: {cfg['n_experts']} blobs", flush=True)

    w.finish()
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# toy model (same architecture, tiny dims, random weights)


TOY_CFG = dict(
    hidden=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16,
    global_head_dim=32, n_global_kv_heads=1, ffn_dim=64, moe_ffn_dim=64,
    n_experts=8, top_k=2, vocab=271, sliding_window=8, max_pos=64,
    rope_theta_local=10000.0, rope_theta_global=1000000.0,
    partial_rotary_global=0.25, rms_eps=1e-6, final_softcap=30.0,
    full_attn_bits=full_attn_bits(["s", "s", "full_attention", "s"]),
    bos_id=2, eos_id=1, pad_id=0, tok_add_dummy_prefix=0,
)


def toy_tokenizer():
    pieces = [("<pad>", 0.0, SP_CONTROL), ("<eos>", 0.0, SP_CONTROL),
              ("<bos>", 0.0, SP_CONTROL), ("<unk>", 0.0, SP_UNKNOWN)]
    for b in range(256):
        pieces.append((f"<0x{b:02X}>", 0.0, SP_BYTE))
    pieces += [("<|turn>", 0.0, SP_USER_DEFINED), ("<turn|>", 0.0, SP_USER_DEFINED),
               ("<|channel>", 0.0, SP_USER_DEFINED), ("<channel|>", 0.0, SP_USER_DEFINED),
               ("<|think|>", 0.0, SP_USER_DEFINED),
               ("▁hello", -1.0, SP_NORMAL), ("hello", -2.0, SP_NORMAL),
               ("he", -3.0, SP_NORMAL), ("ll", -4.0, SP_NORMAL), ("o", -5.0, SP_NORMAL),
               ("▁world", -1.5, SP_NORMAL)]
    assert len(pieces) == TOY_CFG["vocab"], len(pieces)
    return pieces


def convert_toy(out_path, seed=1234, dense_q4=False):
    rng = np.random.default_rng(seed)
    dq, dq_dt = (quant_q4, DT_Q4) if dense_q4 else (quant_q8, DT_Q8)
    cfg = dict(TOY_CFG)
    c = cfg
    w = NucWriter(out_path, cfg, build_tok_blob(toy_tokenizer()))
    H = c["hidden"]

    def rand(shape, scale=0.05):
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    def norm_w(n):
        return (1.0 + rng.normal(0, 0.05, size=n)).astype(np.float32)

    w.add("tok_emb", dq_dt, (c["vocab"], H), dq(rand((c["vocab"], H), 0.5)))
    w.add("out_norm", DT_F32, (H,), norm_w(H).tobytes())

    for l in range(c["n_layers"]):
        full = (c["full_attn_bits"][l // 32] >> (l % 32)) & 1
        hd = c["global_head_dim"] if full else c["head_dim"]
        kvh = c["n_global_kv_heads"] if full else c["n_kv_heads"]
        n = f"l{l}."
        w.add(n + "attn_q", dq_dt, (c["n_heads"] * hd, H), dq(rand((c["n_heads"] * hd, H))))
        w.add(n + "attn_k", dq_dt, (kvh * hd, H), dq(rand((kvh * hd, H))))
        if not full:
            w.add(n + "attn_v", dq_dt, (kvh * hd, H), dq(rand((kvh * hd, H))))
        w.add(n + "attn_o", dq_dt, (H, c["n_heads"] * hd), dq(rand((H, c["n_heads"] * hd))))
        w.add(n + "q_norm", DT_F32, (hd,), norm_w(hd).tobytes())
        w.add(n + "k_norm", DT_F32, (hd,), norm_w(hd).tobytes())
        for ln in ["ln_in", "ln_postattn", "ln_preffw", "ln_postffw",
                   "ln_postffw1", "ln_postffw2", "ln_preffw2"]:
            w.add(n + ln, DT_F32, (H,), norm_w(H).tobytes())
        w.add(n + "mlp_gate", dq_dt, (c["ffn_dim"], H), dq(rand((c["ffn_dim"], H))))
        w.add(n + "mlp_up", dq_dt, (c["ffn_dim"], H), dq(rand((c["ffn_dim"], H))))
        w.add(n + "mlp_down", dq_dt, (H, c["ffn_dim"]), dq(rand((H, c["ffn_dim"]))))
        w.add(n + "router_w", DT_F32, (c["n_experts"], H), rand((c["n_experts"], H), 0.5).tobytes())
        w.add(n + "router_scale", DT_F32, (H,), norm_w(H).tobytes())
        w.add(n + "expert_scale", DT_F32, (c["n_experts"],), norm_w(c["n_experts"]).tobytes())
        w.add(n + "layer_scalar", DT_F32, (1,), np.float32(1.0).tobytes())

    moe = c["moe_ffn_dim"]
    for l in range(c["n_layers"]):
        for e in range(c["n_experts"]):
            blob = quant_q4(rand((2 * moe, H))) + quant_q4(rand((H, moe)))
            w.add(f"l{l}.exp{e}", DT_Q4, (2 * moe + H,), blob)

    w.finish()
    print(f"wrote toy model {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?", help="HF checkpoint dir")
    ap.add_argument("out", help="output .nuc path")
    ap.add_argument("--toy", action="store_true", help="generate tiny random model")
    ap.add_argument("--dense-q4", action="store_true",
                    help="quantize the dense set (attention, MLPs, embeddings) to q4 "
                         "like the experts; ~35%% less read traffic per token, some "
                         "quality cost")
    ap.add_argument("--dump-tok", action="store_true",
                    help="write only the tokenizer blob (for tests/test_tokenizer)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    if args.toy:
        convert_toy(args.out, args.seed, args.dense_q4)
    elif args.dump_tok:
        if not args.model_dir:
            ap.error("model_dir required with --dump-tok")
        pieces, add_dummy_prefix = load_tokenizer(args.model_dir)
        with open(args.out, "wb") as f:
            f.write(build_tok_blob(pieces))
        print(f"wrote {args.out}: {len(pieces)} pieces, "
              f"add_dummy_prefix={int(add_dummy_prefix)}")
    else:
        if not args.model_dir:
            ap.error("model_dir required unless --toy")
        convert_real(args.model_dir, args.out, args.dense_q4)


if __name__ == "__main__":
    main()