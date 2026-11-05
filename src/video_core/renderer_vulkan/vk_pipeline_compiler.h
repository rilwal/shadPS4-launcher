// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "common/types.h"
#include "common/unique_function.h"

namespace Vulkan {

/// Worker threads that create pipelines in the driver, so the GPU thread doesn't wait for them.
class PipelineCompiler {
public:
    /// A job gets `true` when it's dropped at shutdown without running.
    using Job = Common::UniqueFunction<void, bool>;

    explicit PipelineCompiler(u32 num_threads);
    ~PipelineCompiler();

    PipelineCompiler(const PipelineCompiler&) = delete;
    PipelineCompiler& operator=(const PipelineCompiler&) = delete;

    void Submit(Job job);

    /// Waits until `done()` holds or `timeout` passes, checking each time a job finishes.
    /// Returns `done()`.
    template <typename Pred>
    bool WaitFor(Pred&& done, std::chrono::nanoseconds timeout) {
        std::unique_lock lock{mutex};
        return done_cv.wait_for(lock, timeout, std::forward<Pred>(done));
    }

    /// Waits until every submitted job has finished.
    void WaitIdle();

    /// Wakes WaitFor, for a job that has made its result visible before finishing.
    void Notify();

    [[nodiscard]] u32 NumThreads() const {
        return static_cast<u32>(threads.size());
    }

    /// Jobs submitted and not yet finished.
    [[nodiscard]] size_t Pending();

private:
    void Run(u32 index);

    std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable done_cv;
    std::deque<Job> queue;
    size_t running{};
    bool stopping{};
    std::vector<std::thread> threads;
};

/// How a pipeline gets created in the driver.
struct PipelineCompileOptions {
    /// Creates it on these workers instead of at once; draws skip it until it's ready.
    PipelineCompiler* compiler{};
    /// Also times an unoptimized creation of it on these workers, for comparison.
    PipelineCompiler* prober{};
    /// Called on the thread that created it, with the driver's time.
    std::function<void(u64)> on_created{};
    /// Called on a worker after a probe, with the optimized and unoptimized driver times.
    std::function<void(u64, u64)> on_probed{};
};

} // namespace Vulkan
