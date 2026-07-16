#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "nucleus/layout.h"

namespace nucleus::test {

class TempNucFile {
public:
    TempNucFile() : _path(make_path()) { write(); }

    TempNucFile(const TempNucFile&) = delete;
    TempNucFile& operator=(const TempNucFile&) = delete;

    ~TempNucFile() {
        std::error_code error;
        std::filesystem::remove(_path, error);
    }

    const std::filesystem::path& path() const { return _path; }

private:
    static std::filesystem::path make_path() {
        static std::atomic<unsigned int> counter{0};
        return std::filesystem::temp_directory_path() /
               ("nucleus-test-" + std::to_string(getpid()) + "-" + std::to_string(counter++));
    }

    static TensorEntry entry(const char* name, DType dtype, u32 rows, u32 cols, u64 offset, u64 nbytes) {
        TensorEntry result{};
        std::strncpy(result.name, name, sizeof(result.name) - 1);
        result.dtype = static_cast<u8>(dtype);
        result.ndim = 2;
        result.shape[0] = rows;
        result.shape[1] = cols;
        result.offset = offset;
        result.nbytes = nbytes;
        return result;
    }

    void write() const {
        constexpr u32 hidden = 32;
        constexpr u32 moe_ffn = 32;
        const u64 q4_size = q4_bytes(hidden, hidden);
        const u64 f32_size = 4 * sizeof(f32);
        const u64 expert_size = q_scales_bytes(2ULL * moe_ffn, hidden) +
                                2ULL * moe_ffn * hidden / 2 +
                                q_scales_bytes(hidden, moe_ffn) +
                                static_cast<u64>(hidden) * moe_ffn / 2;

        FileHeader header{};
        header.magic = kMagic;
        header.version = kVersion;
        header.cfg.hidden = hidden;
        header.cfg.n_layers = 1;
        header.cfg.moe_ffn_dim = moe_ffn;
        header.cfg.n_experts = 2;
        header.table_offset = sizeof(FileHeader);
        header.tensor_count = 4;

        u64 offset = header.table_offset + header.tensor_count * sizeof(TensorEntry);
        std::array<TensorEntry, 4> entries{
            entry("weight", DType::Q4, hidden, hidden, offset, q4_size),
            entry("bias", DType::F32, 1, 4, offset + q4_size, f32_size),
            entry("l0.exp0", DType::Q4, 1, 1, offset + q4_size + f32_size, expert_size),
            entry("l0.exp1", DType::Q4, 1, 1, offset + q4_size + f32_size + expert_size, expert_size),
        };
        header.tok_offset = entries.back().offset + entries.back().nbytes;
        header.tok_bytes = 4;

        std::vector<u8> bytes(header.tok_offset + header.tok_bytes, 0);
        std::memcpy(bytes.data(), &header, sizeof(header));
        std::memcpy(bytes.data() + header.table_offset, entries.data(), sizeof(entries));

        const u16 scale = 0x3c00;
        std::memcpy(bytes.data() + entries[0].offset, &scale, sizeof(scale));
        const std::array<f32, 4> bias = {1.0F, -2.0F, 3.0F, 4.0F};
        std::memcpy(bytes.data() + entries[1].offset, bias.data(), sizeof(bias));
        std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(entries[2].offset),
                  bytes.begin() + static_cast<std::ptrdiff_t>(entries[2].offset + entries[2].nbytes), 0x11);
        std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(entries[3].offset),
                  bytes.begin() + static_cast<std::ptrdiff_t>(entries[3].offset + entries[3].nbytes), 0x22);
        bytes[header.tok_offset] = 't';
        bytes[header.tok_offset + 1] = 'o';
        bytes[header.tok_offset + 2] = 'k';

        std::ofstream output(_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    std::filesystem::path _path;
};

} // namespace nucleus::test
