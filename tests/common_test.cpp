#include <cstdint>

#include <gtest/gtest.h>

#include "nucleus/common.h"

TEST(HalfPrecision, ConvertsNormalValues) {
    EXPECT_FLOAT_EQ(nucleus::f16_to_f32(0x3c00), 1.0F);
    EXPECT_FLOAT_EQ(nucleus::f16_to_f32(0xc000), -2.0F);
    EXPECT_EQ(nucleus::f32_to_f16(1.0F), 0x3c00);
    EXPECT_EQ(nucleus::f32_to_f16(-2.0F), 0xc000);
}

TEST(AlignedBuffer, IsAlignedAndZeroInitialized) {
    nucleus::AlignedBuf<std::uint32_t> buffer(4);

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % 64, 0U);
    EXPECT_EQ(buffer.size(), 4U);
    EXPECT_EQ(buffer[0], 0U);
    EXPECT_EQ(buffer[3], 0U);
}
