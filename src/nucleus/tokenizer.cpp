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


#include "tokenizer.h"
#include <algorithm>

namespace nucleus {
    static const char *kSpaceMark = "\xE2\x96\x81"; // U+2581 LOWER ONE EIGHTH BLOCK

    void Tokenizer::load(const u8 *blob, u64 bytes, bool add_dummy_prefix) {
        _add_dummy_prefix = add_dummy_prefix;
        NUC_CHECK(bytes >= 4, "tokenizer blob truncated");
        u32 n;
        std::memcpy(&n, blob, 4);
        u64 off = 4;
        _pieces.reserve(n);
        _scores.reserve(n);
        _types.reserve(n);
        for (i32 i = 0; i < 256; ++i)
            _byte_ids[i] = -1;
        for (u32 i = 0; i < n; ++i) {
            NUC_CHECK(off + 8 <= bytes, "tokenizer blob truncated");
            f32 score;
            u8 type;
            u16 len;
            std::memcpy(&score, blob + off, 4);
            type = blob[off + 4];
            std::memcpy(&len, blob + off + 6, 2);
            off += 8;
            NUC_CHECK(off + len <= bytes, "tokenizer blob truncated");
            std::string text(reinterpret_cast<const char *>(blob + off), len);
            off += len;

            i32 id = (i32) _pieces.size();
            if (type == kByte && text.size() == 6 && text.rfind("<0x", 0) == 0) {
                int v = std::stoi(text.substr(3, 2), nullptr, 16);
                _byte_ids[v] = id;
            }
            if (type == kUnknown)
                _unk_id = id;

            if (type == kUnknown || type == kControl || type == kUserDefined)
                _specials.push_back(id);
            _index.emplace(text, id);
            _pieces.push_back(std::move(text));
            _scores.push_back(score);
            _types.push_back(type);
        }
        std::sort(_specials.begin(), _specials.end(),
                  [this](i32 a, i32 b) { return _pieces[a].size() > _pieces[b].size(); });
    }

    i32 Tokenizer::find(const std::string &piece) const {
        auto it = _index.find(piece);
        return it == _index.end() ? -1 : it->second;
    }

    static size_t utf8_len(u8 c) {
        if (c < 0x80)
            return 1;
        if ((c >> 5) == 0x6)
            return 2;
        if ((c >> 4) == 0xE)
            return 3;
        if ((c >> 3) == 0x1E)
            return 4;
        return 1;
    }

    void Tokenizer::spm_encode(const std::string &raw, std::vector<i32> &out) const {
        if (raw.empty())
            return;

        std::string text;
        text.reserve(raw.size() + 4);
        if (_add_dummy_prefix)
            text += kSpaceMark;
        for (char c: raw) {
            if (c == ' ')
                text += kSpaceMark;
            else
                text += c;
        }

        std::vector<i32> ids;
        std::vector<std::string> syms;
        for (size_t i = 0; i < text.size();) {
            size_t l = std::min(utf8_len((u8) text[i]), text.size() - i);
            std::string ch = text.substr(i, l);
            i32 id = find(ch);
            if (id >= 0 && _types[id] != kControl && _types[id] != kUserDefined) {
                ids.push_back(id);
                syms.push_back(std::move(ch));
            } else {
                for (size_t b = 0; b < l; ++b) {
                    i32 bid = _byte_ids[(u8) ch[b]];
                    ids.push_back(bid >= 0 ? bid : _unk_id);
                    syms.push_back(std::string(1, ch[b]));
                }
            }
            i += l;
        }

        // Greedy merges: always merge the adjacent pair with the best score.
        for (;;) {
            f32 best_score = -1e30f;
            i32 best_i = -1, best_id = -1;
            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                std::string cat = syms[i] + syms[i + 1];
                i32 id = find(cat);
                if (id >= 0 && _types[id] == kNormal && _scores[id] > best_score) {
                    best_score = _scores[id];
                    best_i = (i32) i;
                    best_id = id;
                }
            }
            if (best_i < 0)
                break;
            syms[best_i] += syms[best_i + 1];
            ids[best_i] = best_id;
            syms.erase(syms.begin() + best_i + 1);
            ids.erase(ids.begin() + best_i + 1);
        }
        out.insert(out.end(), ids.begin(), ids.end());
    }

    std::vector<i32> Tokenizer::encode(const std::string &text, bool add_bos, u32 bos_id, bool parse_specials) const {
        std::vector<i32> out;
        if (add_bos)
            out.push_back((i32) bos_id);

        if (!parse_specials || _specials.empty()) {
            spm_encode(text, out);
            return out;
        }

        // Scan for special-token strings (longest first), SPM-encode the gaps.
        size_t seg_start = 0, i = 0;
        while (i < text.size()) {
            i32 matched = -1;
            for (i32 id: _specials) {
                const std::string &s = _pieces[id];
                if (s.empty() || s.size() > text.size() - i)
                    continue;
                if (std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                    matched = id;
                    break;
                }
            }
            if (matched >= 0) {
                spm_encode(text.substr(seg_start, i - seg_start), out);
                out.push_back(matched);
                i += _pieces[matched].size();
                seg_start = i;
            } else {
                ++i;
            }
        }
        spm_encode(text.substr(seg_start), out);
        return out;
    }

    std::string Tokenizer::decode_piece(i32 id, bool show_specials) const {
        if (id < 0 || id >= (i32) _pieces.size())
            return "";
        u8 t = _types[id];
        if (t == kByte) {
            int v = std::stoi(_pieces[id].substr(3, 2), nullptr, 16);
            return std::string(1, (char) v);
        }
        if (t == kControl || t == kUserDefined) {
            return show_specials ? _pieces[id] : "";
        }

        std::string out;
        const std::string &p = _pieces[id];
        for (size_t i = 0; i < p.size();) {
            if (p.size() - i >= 3 && std::memcmp(p.data() + i, kSpaceMark, 3) == 0) {
                out += ' ';
                i += 3;
            } else {
                out += p[i++];
            }
        }
        return out;
    }
} // namespace nucleus
