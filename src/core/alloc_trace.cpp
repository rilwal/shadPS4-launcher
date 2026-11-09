// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/singleton.h"
#include "core/alloc_trace.h"
#include "core/linker.h"
#include "core/tls.h"

namespace Core::AllocTrace {
namespace {

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_ALLOC_TRACE");
        return env && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

enum Fn : u32 {
    Malloc,
    Free,
    Calloc,
    Realloc,
    Memalign,
    AlignedAlloc,
    PosixMemalign,
    New,
    NewArray,
    NewNothrow,
    NewArrayNothrow,
    NewAligned,
    NewArrayAligned,
    Delete,
    DeleteArray,
    DeleteSized,
    DeleteArraySized,
    DeleteNothrow,
    DeleteArrayNothrow,
    DeleteAligned,
    DeleteArrayAligned,
    DeleteSizedAligned,
    DeleteArraySizedAligned,
    Count,
};

std::array<u64, Count> originals{};

struct Site {
    s64 live_bytes;
    s64 live_count;
    u64 allocs;
};

struct State {
    std::mutex mutex;
    std::unordered_map<u64, std::pair<u64, u64>> live; // pointer -> size, site
    std::unordered_map<u64, Site> sites;
    std::unordered_map<u64, s64> reported; // site -> live bytes at the last report
    s64 live_bytes{};
    u64 calls{};
    u64 last_report_ns{};
};

State& GetState() {
    static State state;
    return state;
}

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string DescribeSite(u64 site) {
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    if (const Module* module = linker->FindByAddress(site)) {
        return fmt::format("{}+{:#x}", module->name, site - module->GetBaseAddress());
    }
    return fmt::format("{:#x}", site);
}

void Report(State& state, u64 now) {
    struct Row {
        u64 site;
        Site stats;
        s64 growth;
    };
    std::vector<Row> rows;
    s64 total{};
    u64 entries{};
    {
        rows.reserve(state.sites.size());
        for (const auto& [site, stats] : state.sites) {
            const auto it = state.reported.find(site);
            const s64 before = it != state.reported.end() ? it->second : 0;
            rows.push_back({site, stats, stats.live_bytes - before});
            state.reported[site] = stats.live_bytes;
        }
        total = state.live_bytes;
        entries = state.live.size();
    }
    constexpr double MiB = 1024.0 * 1024.0;
    LOG_INFO(Core, "Alloc trace: {:.1f} MiB live in {} blocks from {} sites", total / MiB, entries,
             rows.size());
    const auto log_top = [&](const char* what, auto key) {
        std::ranges::sort(rows, [&](const Row& a, const Row& b) { return key(a) > key(b); });
        for (size_t i = 0; i < std::min<size_t>(rows.size(), 10); ++i) {
            const Row& row = rows[i];
            if (key(row) <= 0) {
                break;
            }
            LOG_INFO(Core,
                     "Alloc trace {} #{}: {} live {:.2f} MiB in {} blocks ({:+.2f} MiB in 5 s), "
                     "{} allocations",
                     what, i + 1, DescribeSite(row.site), row.stats.live_bytes / MiB,
                     row.stats.live_count, row.growth / MiB, row.stats.allocs);
        }
    };
    log_top("largest", [](const Row& row) { return row.stats.live_bytes; });
    log_top("growing", [](const Row& row) { return row.growth; });
    state.last_report_ns = now;
}

void OnAlloc(void* pointer, u64 size, u64 site) {
    if (!pointer) {
        return;
    }
    State& state = GetState();
    std::scoped_lock lk{state.mutex};
    auto [it, inserted] = state.live.try_emplace(reinterpret_cast<u64>(pointer), size, site);
    if (!inserted) {
        // Freed by a path we don't see (inside libc): drop the stale record.
        Site& old = state.sites[it->second.second];
        old.live_bytes -= static_cast<s64>(it->second.first);
        --old.live_count;
        state.live_bytes -= static_cast<s64>(it->second.first);
        it->second = {size, site};
    }
    Site& stats = state.sites[site];
    stats.live_bytes += static_cast<s64>(size);
    ++stats.live_count;
    ++stats.allocs;
    state.live_bytes += static_cast<s64>(size);
    if ((++state.calls & 4095) == 0) {
        const u64 now = NowNs();
        if (state.last_report_ns == 0) {
            state.last_report_ns = now;
        } else if (now - state.last_report_ns >= 5'000'000'000ULL) {
            Report(state, now);
        }
    }
}

void OnFree(void* pointer) {
    if (!pointer) {
        return;
    }
    State& state = GetState();
    std::scoped_lock lk{state.mutex};
    const auto it = state.live.find(reinterpret_cast<u64>(pointer));
    if (it == state.live.end()) {
        return;
    }
    Site& stats = state.sites[it->second.second];
    stats.live_bytes -= static_cast<s64>(it->second.first);
    --stats.live_count;
    state.live_bytes -= static_cast<s64>(it->second.first);
    state.live.erase(it);
}

template <Fn fn, typename R, typename... Args>
R CallOriginal(Args... args) {
    using Type = PS4_SYSV_ABI R (*)(Args...);
    return reinterpret_cast<Type>(originals[fn])(args...);
}

#define SITE reinterpret_cast<u64>(__builtin_return_address(0))

template <Fn fn>
PS4_SYSV_ABI void* AllocOne(u64 size) {
    void* p = CallOriginal<fn, void*>(size);
    OnAlloc(p, size, SITE);
    return p;
}

template <Fn fn>
PS4_SYSV_ABI void* AllocTwo(u64 a, u64 b) {
    void* p = CallOriginal<fn, void*>(a, b);
    // calloc(count, size), memalign and aligned_alloc(align, size), new(size, nothrow or align)
    const u64 size = fn == Calloc ? a * b : (fn == Memalign || fn == AlignedAlloc) ? b : a;
    OnAlloc(p, size, SITE);
    return p;
}

PS4_SYSV_ABI void* ReallocHook(void* old, u64 size) {
    void* p = CallOriginal<Realloc, void*>(old, size);
    if (p || size == 0) {
        OnFree(old);
    }
    OnAlloc(p, size, SITE);
    return p;
}

PS4_SYSV_ABI s32 PosixMemalignHook(void** out, u64 align, u64 size) {
    const s32 result = CallOriginal<PosixMemalign, s32>(out, align, size);
    if (result == 0 && out) {
        OnAlloc(*out, size, SITE);
    }
    return result;
}

template <Fn fn>
PS4_SYSV_ABI void FreeOne(void* p) {
    OnFree(p);
    CallOriginal<fn, void>(p);
}

template <Fn fn>
PS4_SYSV_ABI void FreeTwo(void* p, u64 a) {
    OnFree(p);
    CallOriginal<fn, void>(p, a);
}

template <Fn fn>
PS4_SYSV_ABI void FreeThree(void* p, u64 a, u64 b) {
    OnFree(p);
    CallOriginal<fn, void>(p, a, b);
}

struct Entry {
    std::string_view name;
    Fn fn;
    u64 hook;
};

template <typename F>
u64 Address(F f) {
    return reinterpret_cast<u64>(f);
}

const std::array<Entry, Count> entries{{
    {"malloc", Malloc, Address(&AllocOne<Malloc>)},
    {"free", Free, Address(&FreeOne<Free>)},
    {"calloc", Calloc, Address(&AllocTwo<Calloc>)},
    {"realloc", Realloc, Address(&ReallocHook)},
    {"memalign", Memalign, Address(&AllocTwo<Memalign>)},
    {"aligned_alloc", AlignedAlloc, Address(&AllocTwo<AlignedAlloc>)},
    {"posix_memalign", PosixMemalign, Address(&PosixMemalignHook)},
    {"_Znwm", New, Address(&AllocOne<New>)},
    {"_Znam", NewArray, Address(&AllocOne<NewArray>)},
    {"_ZnwmRKSt9nothrow_t", NewNothrow, Address(&AllocTwo<NewNothrow>)},
    {"_ZnamRKSt9nothrow_t", NewArrayNothrow, Address(&AllocTwo<NewArrayNothrow>)},
    {"_ZnwmSt11align_val_t", NewAligned, Address(&AllocTwo<NewAligned>)},
    {"_ZnamSt11align_val_t", NewArrayAligned, Address(&AllocTwo<NewArrayAligned>)},
    {"_ZdlPv", Delete, Address(&FreeOne<Delete>)},
    {"_ZdaPv", DeleteArray, Address(&FreeOne<DeleteArray>)},
    {"_ZdlPvm", DeleteSized, Address(&FreeTwo<DeleteSized>)},
    {"_ZdaPvm", DeleteArraySized, Address(&FreeTwo<DeleteArraySized>)},
    {"_ZdlPvRKSt9nothrow_t", DeleteNothrow, Address(&FreeTwo<DeleteNothrow>)},
    {"_ZdaPvRKSt9nothrow_t", DeleteArrayNothrow, Address(&FreeTwo<DeleteArrayNothrow>)},
    {"_ZdlPvSt11align_val_t", DeleteAligned, Address(&FreeTwo<DeleteAligned>)},
    {"_ZdaPvSt11align_val_t", DeleteArrayAligned, Address(&FreeTwo<DeleteArrayAligned>)},
    {"_ZdlPvmSt11align_val_t", DeleteSizedAligned, Address(&FreeThree<DeleteSizedAligned>)},
    {"_ZdaPvmSt11align_val_t", DeleteArraySizedAligned,
     Address(&FreeThree<DeleteArraySizedAligned>)},
}};

} // namespace

u64 Interpose(std::string_view name, u64 original) {
    if (!Enabled() || original == 0) {
        return original;
    }
    for (const Entry& entry : entries) {
        if (entry.name != name) {
            continue;
        }
        if (originals[entry.fn] != 0 && originals[entry.fn] != original) {
            LOG_WARNING(Core, "Alloc trace: {} resolves to two addresses; tracing the first", name);
            return original;
        }
        originals[entry.fn] = original;
        LOG_INFO(Core, "Alloc trace: tracing {}", name);
        return entry.hook;
    }
    return original;
}

} // namespace Core::AllocTrace
