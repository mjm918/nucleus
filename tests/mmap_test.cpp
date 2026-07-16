#include <utility>

#include <gtest/gtest.h>

#include "nuc_test_file.h"
#include "nucleus/mmap.h"

TEST(Mmap, MapsReadOnlyDataAndTransfersOwnership) {
    nucleus::test::TempNucFile file;
    nucleus::Mmap map(file.path().string());

    ASSERT_TRUE(map.valid());
    ASSERT_GT(map.size(), 0U);
    EXPECT_EQ(map.data()[0], static_cast<nucleus::u8>(nucleus::kMagic & 0xffU));

    nucleus::Mmap moved(std::move(map));
    EXPECT_FALSE(map.valid());
    EXPECT_TRUE(moved.valid());
    EXPECT_EQ(moved.data()[0], static_cast<nucleus::u8>(nucleus::kMagic & 0xffU));

    moved.prefetch(moved.size(), 1);
    moved.advise_sequential(moved.size(), 1);
    EXPECT_EQ(moved.evict(1, 0), 0U);
}
