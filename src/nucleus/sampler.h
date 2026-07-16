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
#ifndef NUCLEUS_SAMPLER_H
#define NUCLEUS_SAMPLER_H

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "common.h"

namespace nucleus {
    class Sampler {
    public:
        Sampler(f32 temp, i32 top_k, f32 top_p, u64 seed) : _temp(temp), _top_k(top_k), _top_p(top_p), _rng(seed) {};

        i32 sample(const f32 *logits, i32 vocab) {
            if (_temp <= 0.f) {
                i32 best = 0;
                for (i32 i = 1; i < vocab; ++i) {
                    if (logits[i] > logits[best]) {
                        best = i;
                    }
                }
                return best;
            }

            const i32 k = std::min(_top_k > 0 ? _top_k : vocab, vocab);
            _candidates.clear();
            _candidates.reserve(k);

            for (i32 i = 0; i < vocab; ++i) {
                if (_candidates.size() < k) {
                    _candidates.emplace_back(logits[i], i);
                    if (_candidates.size() == k) {
                        std::ranges::make_heap(_candidates, _gt);
                    }
                } else if (logits[i] > _candidates.front().first) {
                    std::ranges::pop_heap(_candidates.begin(), _candidates.end(), _gt);
                    _candidates.back() = {logits[i], i};
                    std::ranges::push_heap(_candidates.begin(), _candidates.end(), _gt);
                }
            }

            std::ranges::sort(_candidates, _gt);

            f32 mx = _candidates.front().first;
            f32 sum = 0.f;

            _probs.resize(_candidates.size());

            for (i32 i = 0; i < _candidates.size(); ++i) {
                _probs[i] = std::exp((_candidates[i].first - mx) / _temp);
                sum += _probs[i];
            }

            for (auto &p: _probs) {
                p /= sum;
            }

            const size_t p = _probs.size();
            size_t n = p;
            if (_top_p > 0.f && _top_p < 1.f) {
                f32 c = 0.f;
                for (size_t i = 0; i < p; ++i) {
                    c += _probs[i];
                    if (c >= _top_p) {
                        n = i + 1;
                        break;
                    }
                }
            }

            f32 cut = 0.f;
            for (i32 i = 0; i < n; ++i) {
                cut += _probs[i];
            }
            std::uniform_real_distribution dist(0.f, cut);

            const f32 r = dist(_rng);
            f32 cum = 0.f;
            for (size_t i = 0; i < n; ++i) {
                cum += _probs[i];
                if (r <= cum) {
                    return _candidates[i].second;
                }
            }
            return _candidates[n - 1].second;
        }

    private:
        struct Gt {
            bool operator()(const std::pair<f32, i32> &x, const std::pair<f32, i32> &y) const {
                return x.first > y.first;
            }
        };
        f32 _temp;
        i32 _top_k;
        f32 _top_p;
        std::mt19937_64 _rng;
        std::vector<std::pair<f32, i32>> _candidates;
        std::vector<f32> _probs;
        Gt _gt;
    };
} // namespace nucleus

#endif // NUCLEUS_SAMPLER_H
