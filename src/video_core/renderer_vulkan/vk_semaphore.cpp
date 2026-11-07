// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <limits>
#include <mutex>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

#include "common/assert.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

namespace {
std::mutex registry_mutex;
std::vector<const Semaphore*> registry;
} // namespace

Semaphore::Semaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
    std::scoped_lock lock{registry_mutex};
    registry.push_back(this);
}

Semaphore::~Semaphore() {
    std::scoped_lock lock{registry_mutex};
    std::erase(registry, this);
}

std::string Semaphore::DescribeAll(const Instance& instance) {
    std::scoped_lock lock{registry_mutex};
    std::string out;
    for (const auto* sem : registry) {
        const auto [result, reached] =
            instance.GetDevice().getSemaphoreCounterValue(*sem->semaphore);
        out +=
            fmt::format("{}{} reached {} of {}", out.empty() ? "" : "; ", sem->name,
                        result == vk::Result::eSuccess ? reached : u64{0}, sem->CurrentTick() - 1);
    }
    return out;
}

void Semaphore::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void Semaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    // A wait this long means the GPU is stuck, or the queue is blocked by a submission waiting on
    // something that never comes: say which timelines got how far.
    constexpr u64 LongWaitNs = 2'000'000'000;
    bool logged = false;
    while (true) {
        const vk::Result result =
            instance.GetDevice().waitSemaphores(&wait_info, logged ? WAIT_TIMEOUT : LongWaitNs);
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result == vk::Result::eTimeout && !logged) {
            LOG_ERROR(Render_Vulkan,
                      "GPU wait for {} tick {} over 2 s. Timelines (reached of last handed "
                      "out): {}",
                      name, tick, DescribeAll(instance));
            logged = true;
        }
    }
    if (logged) {
        LOG_ERROR(Render_Vulkan, "GPU wait for {} tick {} finished after all", name, tick);
    }
    Refresh();
}

} // namespace Vulkan
