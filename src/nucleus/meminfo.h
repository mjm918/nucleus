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
#ifndef NUCLEUS_MEMINFO_H
#define NUCLEUS_MEMINFO_H
#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#else
#include <unistd.h>

#include <cstdio>
#endif

#include "common.h"

namespace nucleus {
    inline u64 rss_bytes() {
#if defined(__APPLE__)
        mach_task_basic_info info{};
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
            KERN_SUCCESS) {
            return 0;
            }
        return info.resident_size;
#else
        std::FILE* f = std::fopen("/proc/self/statm", "r");
        if (!f) return 0;
        unsigned long total = 0, resident = 0;
        const int n = std::fscanf(f, "%lu %lu", &total, &resident);
        std::fclose(f);
        if (n != 2) return 0;
        return (u64)resident * (u64)sysconf(_SC_PAGESIZE);
#endif
    }

    inline u64 peak_rss_bytes() {
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
        return static_cast<u64>(ru.ru_maxrss);
#else
        return (u64)ru.ru_maxrss * 1024;
#endif
    }

    struct PageFaults {
        u64 minor;
        u64 major;
    };

    inline PageFaults page_faults() {
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) != 0) return {0, 0};
        return {static_cast<u64>(ru.ru_minflt), static_cast<u64>(ru.ru_majflt)};
    }

}

#endif //NUCLEUS_MEMINFO_H
