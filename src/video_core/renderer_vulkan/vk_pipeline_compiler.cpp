// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fmt/format.h>

#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_pipeline_compiler.h"

namespace Vulkan {

PipelineCompiler::PipelineCompiler(u32 num_threads) {
    threads.reserve(num_threads);
    for (u32 i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i] { Run(i); });
    }
}

PipelineCompiler::~PipelineCompiler() {
    {
        std::scoped_lock lock{mutex};
        stopping = true;
    }
    work_cv.notify_all();
    for (auto& thread : threads) {
        thread.join();
    }
    // Workers drop what's left at shutdown, so this only runs if there were no workers.
    for (auto& job : queue) {
        job(true);
    }
}

void PipelineCompiler::Submit(Job job) {
    {
        std::scoped_lock lock{mutex};
        queue.push_back(std::move(job));
    }
    work_cv.notify_one();
}

void PipelineCompiler::WaitIdle() {
    std::unique_lock lock{mutex};
    done_cv.wait(lock, [this] { return queue.empty() && running == 0; });
}

void PipelineCompiler::Notify() {
    // Taking the lock orders this after a waiter's check, so the wakeup can't be lost.
    { std::scoped_lock lock{mutex}; }
    done_cv.notify_all();
}

size_t PipelineCompiler::Pending() {
    std::scoped_lock lock{mutex};
    return queue.size() + running;
}

void PipelineCompiler::Run(u32 index) {
    const auto name = fmt::format("shadPS4:PipelineCompiler{}", index);
    Common::SetCurrentThreadName(name.c_str());
    while (true) {
        Job job;
        bool drop;
        {
            std::unique_lock lock{mutex};
            work_cv.wait(lock, [this] { return stopping || !queue.empty(); });
            if (queue.empty()) {
                return;
            }
            job = std::move(queue.front());
            queue.pop_front();
            ++running;
            drop = stopping;
        }
        job(static_cast<bool>(drop));
        {
            std::scoped_lock lock{mutex};
            --running;
        }
        done_cv.notify_all();
    }
}

} // namespace Vulkan
