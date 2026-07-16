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
#ifndef NUCLEUS_PAGE_H
#define NUCLEUS_PAGE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "layout.h"
#include "mmap.h"

namespace nucleus {

    class NucFile {
    public:
        explicit NucFile(const std::string &path) : _map(path) {
            NUC_CHECK(_map.size() >= sizeof(FileHeader), "file too small");
            std::memcpy(&_hdr, _map.data(), sizeof(_hdr));
            NUC_CHECK(_hdr.magic == kMagic, "bad magic (not a .nuc file)");
            NUC_CHECK(_hdr.version == kVersion, "unsupported .nuc version");
            NUC_CHECK(_hdr.table_offset + _hdr.tensor_count * sizeof(TensorEntry) <= _map.size(),
                      "tensor table out of range");
            _entries.resize(_hdr.tensor_count);
            std::memcpy(_entries.data(), _map.data() + _hdr.table_offset, _hdr.tensor_count * sizeof(TensorEntry));
            for (const auto &e: _entries) {
                NUC_CHECK(e.offset + e.nbytes <= _map.size(), std::string("tensor out of range: ") + e.name);
                _index.emplace(e.name, &e);
            }
        }

        const ModelConfigBin &cfg() const { return _hdr.cfg; }
        const FileHeader &header() const { return _hdr; }
        const Mmap &map() const { return _map; }

        const TensorEntry &tensor(const std::string &name) const {
            const auto it = _index.find(name);
            NUC_CHECK(it != _index.end(), "missing tensor: " + name);
            return *it->second;
        }
        const TensorEntry *tensor_or_null(const std::string &name) const {
            const auto it = _index.find(name);
            return it == _index.end() ? nullptr : it->second;
        }

        const u8 *data(const TensorEntry &e) const { return _map.data() + e.offset; }

        const u8 *tokenizer_blob() const { return _map.data() + _hdr.tok_offset; }
        u64 tokenizer_bytes() const { return _hdr.tok_bytes; }

    private:
        Mmap _map;
        FileHeader _hdr{};
        std::vector<TensorEntry> _entries;
        std::unordered_map<std::string, const TensorEntry *> _index;
    };
} // namespace nucleus

#endif // NUCLEUS_PAGE_H
