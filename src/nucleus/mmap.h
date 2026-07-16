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
#ifndef NUCLEUS_MMAP_H
#define NUCLEUS_MMAP_H

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "common.h"

namespace nucleus {
    class Mmap {
    public:
        Mmap() = default;
        Mmap(const Mmap &) = delete;
        Mmap &operator=(const Mmap &) = delete;
        Mmap(Mmap &&m) noexcept { *this = std::move(m); };
        Mmap &operator=(Mmap &&o) noexcept {
            release();
            _data = std::exchange(o._data, nullptr);
            _size = std::exchange(o._size, 0);
            _fd = std::exchange(o._fd, -1);
            return *this;
        }

        explicit Mmap(const std::string &path) {
            _fd = open(path.c_str(), O_RDONLY);
            NUC_CHECK(_fd >= 0, "cannot open " + path);
            struct stat st{};
            NUC_CHECK(::fstat(_fd, &st) == 0, "fstat failed on " + path);
            _size = static_cast<size_t>(st.st_size);
            const void *p = mmap(nullptr, _size, PROT_READ, MAP_PRIVATE, _fd, 0);
            NUC_CHECK(p != MAP_FAILED, "mmap failed on " + path);
            _data = static_cast<const u8 *>(p);
        }

        ~Mmap() { release(); }

        const u8 *data() const { return _data; }
        size_t size() const { return _size; }
        bool valid() const { return _data != nullptr; }

        void prefetch(size_t offset, size_t len) const {
            if (!_data || offset + len > _size)
                return;
            madvise(const_cast<u8 *>(_data) + page_floor(offset), len + (offset - page_floor(offset)), MADV_WILLNEED);
        }

        void advise_sequential(size_t offset, size_t len) const {
            if (!_data || offset + len > _size)
                return;
            madvise(const_cast<u8 *>(_data) + page_floor(offset), len + (offset - page_floor(offset)), MADV_SEQUENTIAL);
        }

        size_t evict(size_t offset, size_t len) const {
            if (!_data || offset + len > _size)
                return 0;
            const size_t lo = page_ceil(offset), hi = page_floor(offset + len);
            if (hi <= lo)
                return 0;
            if (madvise(const_cast<u8 *>(_data) + lo, hi - lo, MADV_DONTNEED) != 0)
                return 0;
            return hi - lo;
        }

    private:
        static size_t page_size() {
            static const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
            return page;
        }
        static size_t page_floor(const size_t x) { return x & ~(page_size() - 1); }
        static size_t page_ceil(const size_t x) { return page_floor(x + page_size() - 1); }

        const u8 *_data = nullptr;
        size_t _size = 0;
        int _fd = -1;

        void release() {
            if (_data)
                munmap(const_cast<u8 *>(_data), _size);
            if (_fd >= 0)
                close(_fd);
            _data = nullptr;
            _size = 0;
            _fd = -1;
        }
    };
} // namespace nucleus

#endif // NUCLEUS_MMAP_H
