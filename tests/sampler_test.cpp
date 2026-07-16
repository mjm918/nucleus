#include <gtest/gtest.h>

#include "nucleus/sampler.h"

TEST(Sampler, SelectsArgmaxAtZeroTemperature) {
    const nucleus::f32 logits[] = {-2.0F, 1.0F, 4.0F, 3.0F};
    nucleus::Sampler sampler(0.0F, 0, 1.0F, 42);

    EXPECT_EQ(sampler.sample(logits, 4), 2);
}

TEST(Sampler, TopKOneAlwaysSelectsTheBestCandidate) {
    const nucleus::f32 logits[] = {-2.0F, 1.0F, 4.0F, 3.0F};
    nucleus::Sampler sampler(1.0F, 1, 1.0F, 42);

    EXPECT_EQ(sampler.sample(logits, 4), 2);
}

TEST(Sampler, TopPExcludesCandidatesOutsideTheNucleus) {
    const nucleus::f32 logits[] = {4.0F, 0.0F, -2.0F};
    nucleus::Sampler sampler(1.0F, 0, 0.5F, 42);

    EXPECT_EQ(sampler.sample(logits, 3), 0);
}
