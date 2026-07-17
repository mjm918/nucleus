#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "nucleus/kernel.h"

namespace {

    constexpr nucleus::u32 kRows = 3;
    constexpr nucleus::u32 kCols = 64;
    constexpr nucleus::u32 kGroups = kCols / nucleus::kGroup;

    std::vector<nucleus::f32> make_input() {
        std::vector<nucleus::f32> x(kCols);
        for (nucleus::u32 i = 0; i < kCols; ++i)
            x[i] = static_cast<nucleus::f32>(static_cast<int>(i % 13) - 6) * 0.17F + 0.03F;
        return x;
    }

    void expect_close(nucleus::f32 actual, nucleus::f32 expected) {
        const nucleus::f32 tolerance = std::max(1e-3F, std::fabs(expected) * 2e-5F);
        EXPECT_NEAR(actual, expected, tolerance);
    }

} // namespace

TEST(Kernel, Q4MatvecMatchesScalarQuantizedDot) {
    using namespace nucleus;
    std::vector<u16> scales(kRows * kGroups);
    std::vector<u8> packed(kRows * kCols / 2);
    for (u32 r = 0; r < kRows; ++r) {
        for (u32 g = 0; g < kGroups; ++g) {
            scales[r * kGroups + g] = f32_to_f16(0.25F + 0.0625F * static_cast<f32>(r + g));
            u8 *dst = packed.data() + (r * kGroups + g) * (kGroup / 2);
            for (u32 j = 0; j < kGroup / 2; ++j) {
                const i32 lo = static_cast<i32>((r * 3 + g * 5 + j * 7) % 16) - 8;
                const i32 hi = static_cast<i32>((r * 11 + g * 3 + (j + 16) * 5) % 16) - 8;
                dst[j] = static_cast<u8>((lo + 8) | ((hi + 8) << 4));
            }
        }
    }

    const std::vector<f32> x = make_input();
    std::vector<i8> xq(kCols);
    std::vector<f32> xs(kGroups);
    quantize_act(x.data(), static_cast<int>(kCols), xq.data(), xs.data());

    const QView view{scales.data(), packed.data(), kRows, kCols, DType::Q4};
    std::vector<f32> out(kRows);
    ThreadPool pool(3);
    matvec_q4(view, xq.data(), xs.data(), out.data(), pool);

    for (u32 r = 0; r < kRows; ++r) {
        f32 expected = 0.F;
        for (u32 g = 0; g < kGroups; ++g) {
            const u8 *src = packed.data() + (r * kGroups + g) * (kGroup / 2);
            i32 dot = 0;
            for (u32 j = 0; j < kGroup / 2; ++j) {
                dot += (static_cast<i32>(src[j] & 0x0f) - 8) * xq[g * kGroup + j];
                dot += (static_cast<i32>(src[j] >> 4) - 8) * xq[g * kGroup + j + kGroup / 2];
            }
            expected += static_cast<f32>(dot) * f16_to_f32(scales[r * kGroups + g]) * xs[g];
        }
        expect_close(out[r], expected);
    }
}

TEST(Kernel, Q8MatvecMatchesScalarQuantizedDot) {
    using namespace nucleus;
    std::vector<u16> scales(kRows * kGroups);
    std::vector<u8> quants(kRows * kCols);
    for (u32 r = 0; r < kRows; ++r) {
        for (u32 g = 0; g < kGroups; ++g) {
            scales[r * kGroups + g] = f32_to_f16(0.125F + 0.0625F * static_cast<f32>(r + g));
            for (u32 j = 0; j < kGroup; ++j) {
                const i8 value = static_cast<i8>(static_cast<i32>((r * 17 + g * 7 + j * 11) % 255) - 127);
                quants[(r * kGroups + g) * kGroup + j] = static_cast<u8>(value);
            }
        }
    }

    const std::vector<f32> x = make_input();
    std::vector<i8> xq(kCols);
    std::vector<f32> xs(kGroups);
    quantize_act(x.data(), static_cast<int>(kCols), xq.data(), xs.data());

    const QView view{scales.data(), quants.data(), kRows, kCols, DType::Q8};
    std::vector<f32> out(kRows);
    ThreadPool pool(3);
    matvec_q8(view, xq.data(), xs.data(), out.data(), pool);

    for (u32 r = 0; r < kRows; ++r) {
        f32 expected = 0.F;
        for (u32 g = 0; g < kGroups; ++g) {
            const i8 *w = reinterpret_cast<const i8 *>(quants.data()) + (r * kGroups + g) * kGroup;
            i32 dot = 0;
            for (u32 j = 0; j < kGroup; ++j)
                dot += static_cast<i32>(w[j]) * xq[g * kGroup + j];
            expected += static_cast<f32>(dot) * f16_to_f32(scales[r * kGroups + g]) * xs[g];
        }
        expect_close(out[r], expected);
    }
}
