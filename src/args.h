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

#ifndef NUCLEUS_ARGS_H
#define NUCLEUS_ARGS_H
#include <string>
#include <thread>

#include "nucleus/common.h"
using namespace nucleus;

namespace cli {
    struct Args {
        std::string model;
        std::string prompt;
        std::string system;
        bool chat = false;
        bool show_specials = false;
        i32 max_tokens = 512;
        f32 temp = 1.0f;
        i32 top_k = 64;
        f32 top_p = 0.95f;
        u64 seed = 42;
        u32 ctx = 4096;
        double max_rss_gb = 0.0;
        int threads = std::thread::hardware_concurrency();
    };


    inline void usage() {
        std::fprintf(stderr, "usage: nucleus MODEL.nuc [options]\n"
                             "  --prompt TEXT     one-shot completion of raw text -- for base\n"
                             "                    checkpoints; -it ones want --chat\n"
                             "  --chat            interactive chat (Gemma 4 turn format)\n"
                             "  --system TEXT     system prompt (chat mode)\n"
                             "  --max-tokens N    max new tokens (default 512)\n"
                             "  --temp F          temperature, 0 = greedy (default 1.0)\n"
                             "  --top-k N         top-k (default 64)\n"
                             "  --top-p F         top-p (default 0.95)\n"
                             "  --seed N          rng seed (default 42)\n"
                             "  --ctx N           max context positions (default 4096)\n"
                             "  --max-rss GB      cap resident memory by evicting experts\n"
                             "                    (default: uncapped -- let the page cache decide)\n"
                             "  --threads N       worker threads (default: hw)\n"
                             "  --show-specials   print special tokens\n");
    }
} // namespace cli

#endif // NUCLEUS_ARGS_H
