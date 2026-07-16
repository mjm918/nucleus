#include <filesystem>

#include <gtest/gtest.h>

#include "nuc_test_file.h"
#include "nucleus/experts.h"

TEST(ExpertStore, TracksResidentExpertsAndBuildsViews) {
    nucleus::test::TempNucFile fixture;
    nucleus::NucFile file(fixture.path().string());
    nucleus::ExpertStore store(file, file.cfg());

    EXPECT_EQ(store.expert_bytes(), 1728U);
    EXPECT_EQ(store.resident_experts(), 0U);

    const nucleus::ExpertView expert = store.view(0, 0);
    EXPECT_EQ(expert.gate_up.dtype, nucleus::DType::Q4);
    EXPECT_EQ(expert.gate_up.rows, 64U);
    EXPECT_EQ(expert.gate_up.cols, 32U);
    EXPECT_EQ(expert.down.rows, 32U);
    EXPECT_EQ(expert.down.cols, 32U);
    EXPECT_EQ(store.resident_experts(), 1U);

    const nucleus::u32 ids[] = {1};
    store.prefetch(0, ids, 1);
    EXPECT_EQ(store.resident_experts(), 2U);
    EXPECT_FALSE(store.capped());
    store.set_max_rss(1);
    EXPECT_TRUE(store.capped());
}

TEST(ExpertStore, PersistsUsageForWarmingHotExperts) {
    nucleus::test::TempNucFile fixture;
    const std::filesystem::path usage = fixture.path().string() + ".usage";
    nucleus::NucFile file(fixture.path().string());

    {
        nucleus::ExpertStore store(file, file.cfg());
        store.view(0, 0);
        store.view(0, 0);
        store.view(0, 1);
        store.save_usage(usage.string());
    }

    nucleus::ExpertStore restored(file, file.cfg());
    restored.load_usage(usage.string());
    restored.warm_hot_experts(1);

    EXPECT_EQ(restored.resident_experts(), 1U);
    std::error_code error;
    std::filesystem::remove(usage, error);
}
