// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <coroutine>
#include <exception>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>
#include <queue>

#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/regs.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    u32 index_buffer_num_indices{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0; });
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    /// Whether a SubmitDone hasn't been handled by the command processor yet.
    bool IsSubmitDonePending() const {
        return submit_done;
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    /// Reads ahead from the draw or dispatch being run to the end of its submission, so the
    /// pipelines of the draws and dispatches still to come start compiling on the workers.
    /// Registers are restored after. Runs at most once per submission, only on the graphics
    /// queue and only with the compile_lookahead setting. Returns true if it ran.
    bool LookAhead();

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{64};

    std::thread::id GetGpuCommandProcessorThread() {
        return gpu_id;
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() {
        return gpu_tid;
    }
#endif

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb);
    /// One level of LookAhead: register writes, draws and dispatches in `dcb`, following
    /// indirect buffers. Returns false once the budget is spent or the reading has to stop.
    struct LookAheadBudget {
        u32 packets;
        u32 pipelines;
        /// The DE counter as the reading goes, to know which constant engine dumps exist yet.
        u32 de_count;
        /// Stopped at a draw whose descriptors the constant engine hasn't written yet.
        bool waits_for_ce;
    };
    bool ScanAhead(std::span<const u32> dcb, LookAheadBudget& budget, u32 depth);
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);

    /// Label writes of a fence signal, as dword address and value pairs.
    struct LabelWrite {
        VAddr address;
        u32 value;
        /// What the label should hold when a deferred signal writes it; anything else means
        /// it was written since, and the signal leaves it.
        u32 expected{};
    };
    using LabelWrites = boost::container::small_vector<LabelWrite, 2>;
    static void RecordLabelWrite(LabelWrites& writes, void* address, u64 data, u32 num_bytes);
    /// Writes a fence signal's label. Run deferred, it skips dwords that no longer hold the
    /// value expected when the signal was recorded.
    static void WriteFenceLabel(void* address, u64 data, u32 num_bytes);
    /// The label writes of the deferred signal running on this thread, with expected values.
    static thread_local const LabelWrites* running_label_writes;

    /// Hands a fence signal to the rasterizer to run later (fenced readbacks). Returns false
    /// if it must run now. `writes` are the label values the signal will write.
    bool DeferFenceSignal(Common::UniqueFunction<void>&& signal, bool throttle,
                          const LabelWrites& writes);

    /// Call before the command processor writes guest memory itself (WRITE_DATA, DMA_DATA,
    /// semaphores...). On hardware those writes happen in order after any wait that read a
    /// label, so after that label's write. Here a deferred label may still be on its way, so
    /// this waits for it if the write overlaps one, or follows a wait that read a deferred
    /// value: otherwise the late label write lands over this one (TLG's compute stall), or
    /// the guest learns of progress before the label it waited behind is written. Then drops
    /// the deferred values this write replaces.
    void OrderCpWrite(VAddr address, u64 size);

    /// Tests a WAIT_REG_MEM condition. A label whose signal is deferred reads as its new value.
    bool TestWait(const PM4CmdWaitRegMem* wait);

    /// Logs how often queue waits polled, every few seconds.
    void LogWaitStats();

    void ProcessCommands();
    /// Runs the constant engine's task, if any; returns whether there was one.
    bool ProcessCeQueue();
    void Process(std::stop_token stoken);
    void HandleGetLodStats(const PM4CmdGetLodStats* get_lod);

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    std::queue<Task::Handle> ce_submits;
    u32 num_mapped_queues{1u}; // GFX is always available

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};
    u32 vgt_strmout_buffer_filled_size[4]{};

    struct ConstantEngine {
        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    /// The command buffer of each graphics task level being run, innermost last, each at the
    /// packet it's on, for LookAhead.
    std::vector<std::span<const u32>*> gfx_frames;
    bool looked_ahead{};
    bool in_lookahead{};

    /// Labels whose fence signal is deferred until the data it guards is in guest memory.
    /// Only the guest CPU needs that delay: the command processor records everything in
    /// order on one Vulkan queue, so its own waits see the new value at once, as when signals
    /// aren't deferred. GPU thread only, apart from landed_label_seq.
    struct DeferredLabel {
        u32 value;
        u64 seq;
    };
    std::unordered_map<VAddr, DeferredLabel> deferred_labels;
    size_t deferred_labels_cleanup_at{256};
    u64 deferred_label_seq{};
    std::atomic<u64> landed_label_seq{};
    /// The newest deferred signal a queue wait passed on: the command processor's own memory
    /// writes after that wait must land after it.
    u64 cp_order_seq{};
    struct WaitStats {
        u64 failed_polls{};
        u64 deferred_label_hits{};
        u64 order_waits{};
        u64 order_wait_ns{};
        u64 last_log_ns{};
        u64 ab_switches{};
    } wait_stats;

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    const bool guest_markers_enabled;
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
#ifdef __linux__
    u32 gpu_tid;
#endif
    s32 curr_qid{-1};
};

} // namespace AmdGpu
