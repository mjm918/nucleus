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
#ifndef NUCLEUS_TOKENIZER_H
#define NUCLEUS_TOKENIZER_H

#include <string>
#include <unordered_map>
#include <vector>
#include "common.h"

namespace nucleus {
    enum PieceType : u8 {
        kNormal = 1,
        kUnknown = 2,
        kControl = 3,
        kUserDefined = 4,
        kByte = 6,
    };
    class Tokenizer {
    public:
        // blob layout: see format.hpp
        void load(const u8 *blob, u64 bytes, bool add_dummy_prefix);

        std::vector<i32> encode(const std::string &text, bool add_bos, u32 bos_id, bool parse_specials) const;

        std::string decode_piece(i32 id, bool show_specials = false) const;

        i32 find(const std::string &piece) const;
        i32 n_pieces() const { return _pieces.size(); }
        u8 piece_type(i32 id) const { return _types[id]; }

    private:
        void spm_encode(const std::string &text, std::vector<i32> &out) const;

        std::vector<std::string> _pieces;
        std::vector<f32> _scores;
        std::vector<u8> _types;
        std::unordered_map<std::string, i32> _index;
        std::vector<i32> _specials;
        i32 _byte_ids[256];
        i32 _unk_id = -1;
        bool _add_dummy_prefix = false;
    };

} // namespace nucleus

#endif // NUCLEUS_TOKENIZER_H
