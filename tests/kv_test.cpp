#include <gtest/gtest.h>

#include "nucleus/kv.h"

TEST(LayerKV, StoresValuesInAFullContextCache) {
    nucleus::LayerKV cache(false, 0, 3, 1, 2);
    const nucleus::f32 key[] = {1.0F, 2.0F};
    const nucleus::f32 value[] = {3.0F, 4.0F};

    cache.push(2, key, value);

    EXPECT_EQ(cache.distance(), 2U);
    EXPECT_FLOAT_EQ(cache.k_at(2)[0], 1.0F);
    EXPECT_FLOAT_EQ(cache.k_at(2)[1], 2.0F);
    EXPECT_FLOAT_EQ(cache.v_at(2)[0], 3.0F);
    EXPECT_FLOAT_EQ(cache.v_at(2)[1], 4.0F);
    EXPECT_EQ(cache.attn_start(2), 0U);
}

TEST(LayerKV, WrapsSlotsForASlidingWindow) {
    nucleus::LayerKV cache(true, 3, 8, 1, 1);
    const nucleus::f32 initial_key[] = {1.0F};
    const nucleus::f32 replacement_key[] = {9.0F};
    const nucleus::f32 value[] = {2.0F};

    cache.push(0, initial_key, value);
    cache.push(3, replacement_key, value);

    EXPECT_FLOAT_EQ(cache.k_at(0)[0], 9.0F);
    EXPECT_FLOAT_EQ(cache.k_at(3)[0], 9.0F);
    EXPECT_EQ(cache.attn_start(2), 0U);
    EXPECT_EQ(cache.attn_start(5), 3U);
}
