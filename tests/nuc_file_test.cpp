#include <cstring>
#include <stdexcept>

#include <gtest/gtest.h>

#include "nuc_test_file.h"
#include "nucleus/page.h"
#include "nucleus/view.h"

TEST(NucFile, ReadsTensorMetadataAndData) {
    nucleus::test::TempNucFile fixture;
    nucleus::NucFile file(fixture.path().string());

    EXPECT_EQ(file.cfg().hidden, 32U);
    EXPECT_EQ(file.header().tensor_count, 4U);
    EXPECT_EQ(std::memcmp(file.tokenizer_blob(), "tok", 3), 0);
    EXPECT_EQ(file.tokenizer_bytes(), 4U);

    const nucleus::TensorEntry& weight = file.tensor("weight");
    EXPECT_EQ(weight.dtype, static_cast<nucleus::u8>(nucleus::DType::Q4));
    EXPECT_EQ(file.tensor_or_null("missing"), nullptr);
    EXPECT_THROW(file.tensor("missing"), std::runtime_error);
}

TEST(NucFile, BuildsQuantizedAndFloatViews) {
    nucleus::test::TempNucFile fixture;
    nucleus::NucFile file(fixture.path().string());

    const nucleus::QView weight = nucleus::qview(file, file.tensor("weight"));
    EXPECT_EQ(weight.dtype, nucleus::DType::Q4);
    EXPECT_EQ(weight.rows, 32U);
    EXPECT_EQ(weight.cols, 32U);
    EXPECT_EQ(weight.scales[0], 0x3c00);

    const nucleus::f32* bias = nucleus::f32view(file, file.tensor("bias"), 4);
    EXPECT_FLOAT_EQ(bias[0], 1.0F);
    EXPECT_FLOAT_EQ(bias[1], -2.0F);
    EXPECT_FLOAT_EQ(bias[3], 4.0F);
}
