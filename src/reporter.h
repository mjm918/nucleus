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
#ifndef NUCLEUS_REPORTER_H
#define NUCLEUS_REPORTER_H

#include "nucleus/common.h"
#include "nucleus/meminfo.h"
#include "nucleus/model.h"
#include "nucleus/sampler.h"
using namespace nucleus;

namespace cli {
    struct MemBase {
        u64 rss;
        u64 major_faults;
    };

    inline std::string fmt_bytes(u64 b) {
        char buf[32];
        const double gb = (double) b / (1ull << 30);
        if (gb >= 1.0)
            std::snprintf(buf, sizeof(buf), "%.2f GB", gb);
        else
            std::snprintf(buf, sizeof(buf), "%.0f MB", (double) b / (1ull << 20));
        return buf;
    }

    inline std::string fmt_delta(i64 d) {
        const u64 mag = (u64) (d < 0 ? -d : d);
        return (d < 0 ? "-" : "+") + fmt_bytes(mag);
    }

    inline void report_mem(const MemBase &base, const Model &m, u64 max_rss) {
        const u64 rss = rss_bytes();
        const u64 major = page_faults().major;
        std::fprintf(stderr, "[nucleus] rss %s (peak %s) | %s and %llu major faults while generating\n",
                     fmt_bytes(rss).c_str(), fmt_bytes(peak_rss_bytes()).c_str(),
                     fmt_delta((i64) rss - (i64) base.rss).c_str(), (unsigned long long) (major - base.major_faults));
        if (!max_rss)
            return;
        std::fprintf(stderr, "[nucleus] cap %s | %llu evictions, %llu experts resident%s\n", fmt_bytes(max_rss).c_str(),
                     (unsigned long long) m.evictions(), (unsigned long long) m.resident_experts(),
                     m.cap_holding() ? ""
                                     : " | OVER CAP: nothing left to evict, "
                                       "dense set + KV cache are the floor");
    }

    inline void prefill(Model &m, const std::vector<i32> &toks, u32 &pos) {
        for (size_t i = 0; i < toks.size(); ++i) {
            m.forward(toks[i], pos++, i + 1 == toks.size());
        }
    }

    inline i32 decode_loop(Model &m, Sampler &s, u32 &pos, i32 max_new, bool show_specials) {
        i32 made = 0;
        while (made < max_new && pos < m.max_ctx()) {
            i32 tok = s.sample(m.logits(), (i32) m.cfg().vocab);
            ++made;
            if (tok == (i32) m.cfg().eos_id || tok == m.eot_id()) {
                m.forward(tok, pos++, false);
                break;
            }
            std::string piece = m.tokenizer().decode_piece(tok, show_specials);
            if (!piece.empty()) {
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
            }
            if (made == max_new || pos + 1 >= m.max_ctx())
                break;
            m.forward(tok, pos++, true);
        }
        return made;
    }
} // namespace cli

#endif // NUCLEUS_REPORTER_H
