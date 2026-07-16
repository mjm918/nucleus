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

#ifndef NUCLEUS_THREADPOOL_H
#define NUCLEUS_THREADPOOL_H

#include <sched.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common.h"

namespace nucleus {

    inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
    }

    class ThreadPool {
    public:
        explicit ThreadPool(int n_threads) {
            _n = n_threads < 1 ? 1 : n_threads;
            _workers.reserve(_n - 1);
            for (int t = 1; t < _n; ++t) {
                _workers.emplace_back([this, t] { worker_loop(t); });
            }
        }

        ~ThreadPool() {
            {
                std::lock_guard<std::mutex> lk(_mu);
                _stop = true;
                _epoch.fetch_add(1, std::memory_order_release);
            }
            _cv.notify_all();
            for (auto &w: _workers)
                w.join();
        }

        ThreadPool(const ThreadPool &) = delete;
        ThreadPool &operator=(const ThreadPool &) = delete;

        int size() const { return _n; }

        void parallel_for(i64 n, const std::function<void(i64, i64)> &fn) {
            if (n <= 0)
                return;
            i64 chunk = (n + _n - 1) / _n;
            if (_n == 1 || n < 2 * _n) {
                fn(0, n);
                return;
            }
            {
                std::lock_guard<std::mutex> lk(_mu);
                _job = &fn;
                _n_job = n;
                _job_chunk = chunk;
                _pending.store(_n - 1, std::memory_order_relaxed);
                _epoch.fetch_add(1, std::memory_order_release);
            }
            _cv.notify_all();
            fn(0, std::min(chunk, n));
            if (!spin_until([this] { return _pending.load(std::memory_order_acquire) == 0; })) {
                std::unique_lock<std::mutex> lk(_mu);
                _cv_done.wait(lk, [this] { return _pending.load(std::memory_order_acquire) == 0; });
            }
            _job = nullptr;
        }

    private:
        static constexpr int kPauseBurst = 512;
        static constexpr int kRounds = 64;

        template<class Pred>
        static bool spin_until(Pred pred) {
            for (int r = 0; r < kRounds; ++r) {
                for (int i = 0; i < kPauseBurst; ++i) {
                    if (pred())
                        return true;
                    cpu_relax();
                }
                sched_yield();
            }
            return pred();
        }

        void worker_loop(int tid) {
            u64 seen = 0;
            for (;;) {
                if (!spin_until([&] { return _epoch.load(std::memory_order_acquire) != seen; })) {
                    std::unique_lock<std::mutex> lk(_mu);
                    _cv.wait(lk, [&] { return _epoch.load(std::memory_order_relaxed) != seen; });
                }
                seen = _epoch.load(std::memory_order_acquire);
                if (_stop)
                    return;

                i64 begin = tid * _job_chunk;
                i64 end = std::min(begin + _job_chunk, _n_job);
                if (begin < end)
                    (*_job)(begin, end);
                if (_pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> lk(_mu);
                    _cv_done.notify_one();
                }
            }
        }

        int _n = 1;
        std::vector<std::thread> _workers;
        std::mutex _mu;
        std::condition_variable _cv, _cv_done;
        const std::function<void(i64, i64)> *_job = nullptr;
        i64 _n_job = 0, _job_chunk = 0;
        std::atomic<int> _pending{0};
        std::atomic<u64> _epoch{0};
        bool _stop = false;
    };

} // namespace nucleus

#endif // NUCLEUS_THREADPOOL_H
