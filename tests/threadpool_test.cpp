#include <atomic>
#include <vector>

#include <gtest/gtest.h>

#include "nucleus/threadpool.h"

TEST(ThreadPool, RunsWorkAcrossTheWholeInput) {
    nucleus::ThreadPool pool(3);
    std::vector<nucleus::i64> result(96, -1);
    std::atomic<int> calls{0};

    pool.parallel_for(static_cast<nucleus::i64>(result.size()), [&](nucleus::i64 begin, nucleus::i64 end) {
        ++calls;
        for (nucleus::i64 i = begin; i < end; ++i) {
            result[i] = i * 2;
        }
    });

    EXPECT_EQ(pool.size(), 3);
    EXPECT_EQ(calls.load(), 3);
    for (nucleus::i64 i = 0; i < static_cast<nucleus::i64>(result.size()); ++i) {
        EXPECT_EQ(result[i], i * 2);
    }
}

TEST(ThreadPool, HandlesSmallAndEmptyInputsOnTheCallingThread) {
    nucleus::ThreadPool pool(0);
    int calls = 0;

    pool.parallel_for(1, [&](nucleus::i64 begin, nucleus::i64 end) {
        ++calls;
        EXPECT_EQ(begin, 0);
        EXPECT_EQ(end, 1);
    });
    pool.parallel_for(0, [&](nucleus::i64, nucleus::i64) { ++calls; });

    EXPECT_EQ(pool.size(), 1);
    EXPECT_EQ(calls, 1);
}
