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

#ifndef NUCLEUS_COMMON_H
#define NUCLEUS_COMMON_H
#include <stdexcept>
#include <string>

namespace nucleus {
    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using i8 = int8_t;
    using i32 = int32_t;
    using i64 = int64_t;
    using f32 = float;

    [[noreturn]] inline void fail(const std::string& msg) { throw std::runtime_error(msg); }

#define NUC_CHECK(cond, msg) \
do { if (!(cond)) ::nucleus::fail(std::string("nucleus: ") + (msg)); } while (0)

    inline f32 f16_to_f32(u16 h) {
#if defined(__ARM_FP16_FORMAT_IEEE)
        __fp16 v;
        std::memcpy(&v, &h, sizeof(v));
        return static_cast<f32>(v);
#elif defined(__F16C__)
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
        u32 sign = (u32)(h & 0x8000) << 16;
        u32 exp = (h >> 10) & 0x1F;
        u32 man = h & 0x3FF;
        u32 bits;
        if (exp == 0) {
            if (man == 0) {
                bits = sign;
            } else {  // subnormal
                exp = 1;
                while (!(man & 0x400)) { man <<= 1; --exp; }
                man &= 0x3FF;
                bits = sign | ((exp + 112) << 23) | (man << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000 | (man << 13);
        } else {
            bits = sign | ((exp + 112) << 23) | (man << 13);
        }
        f32 out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
#endif
    }

    inline u16 f32_to_f16(f32 f) {
#if defined(__ARM_FP16_FORMAT_IEEE)
        const __fp16 v = static_cast<__fp16>(f);
        u16 out;
        std::memcpy(&out, &v, sizeof(out));
        return out;
#else
        u32 bits;
        std::memcpy(&bits, &f, sizeof(bits));
        u32 sign = (bits >> 16) & 0x8000;
        i32 exp = (i32)((bits >> 23) & 0xFF) - 127 + 15;
        u32 man = bits & 0x7FFFFF;
        if (exp <= 0) return (u16)sign;
        if (exp >= 31) return (u16)(sign | 0x7C00);
        return (u16)(sign | (exp << 10) | (man >> 13));
#endif
    }

    inline f32 bf16_round(f32 f) {
        u32 bits;
        std::memcpy(&bits, &f, sizeof(bits));
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000u;
        f32 out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    struct FreeDeleter {
        void operator()(void* p) const { std::free(p); }
    };

    template <typename T>
    class AlignedBuf {
    public:
        AlignedBuf() = default;
        explicit AlignedBuf(size_t count) { resize(count); }

        void resize(size_t count) {
            if (count == 0) { _ptr.reset(); _size = 0; return; }
            void* p = nullptr;
            if (posix_memalign(&p, 64, count * sizeof(T)) != 0) fail("out of memory");
            std::memset(p, 0, count * sizeof(T));
            _ptr.reset(static_cast<T*>(p));
            _size = count;
        }

        T* data() { return _ptr.get(); }
        const T* data() const { return _ptr.get(); }
        T& operator[](size_t i) { return _ptr.get()[i]; }
        const T& operator[](size_t i) const { return _ptr.get()[i]; }
        [[nodiscard]] size_t size() const { return _size; }

    private:
        std::unique_ptr<T[], FreeDeleter> _ptr;
        size_t _size = 0;
    };
}

#endif //NUCLEUS_COMMON_H
