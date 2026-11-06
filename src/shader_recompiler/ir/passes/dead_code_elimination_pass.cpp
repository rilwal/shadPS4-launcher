// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>
#include <tsl/robin_set.h>
#include "shader_recompiler/ir/program.h"

namespace Shader::Optimization {

void DeadCodeEliminationPass(IR::Program& program) {
    // We iterate over the instructions in reverse order.
    // This is because removing an instruction reduces the number of uses for earlier instructions.
    // Open addressing and a vector stack: a node-based set and a deque allocated per insert, and
    // this pass runs twice per translation.
    thread_local tsl::robin_set<IR::Inst*> live;
    thread_local std::vector<IR::Inst*> worklist;
    live.clear();
    worklist.clear();

    for (IR::Block* block : program.post_order_blocks) {
        auto it{block->end()};
        while (it != block->begin()) {
            --it;
            const bool has_side_effects = it->MayHaveSideEffects();
            if (!it->HasUses() && !has_side_effects) {
                it->Invalidate();
                it = block->Instructions().erase(it);
            } else if (has_side_effects) {
                live.insert(&*it);
                worklist.push_back(&*it);
            }
        }
    }

    while (!worklist.empty()) {
        IR::Inst* inst = worklist.back();
        worklist.pop_back();

        for (size_t i = 0; i < inst->NumArgs(); i++) {
            if (inst->Arg(i).IsImmediate()) {
                continue;
            }
            IR::Inst* operand = inst->Arg(i).Inst();
            if (live.insert(operand).second) {
                worklist.push_back(operand);
            }
        }
    }

    for (IR::Block* block : program.post_order_blocks) {
        for (auto it = block->begin(); it != block->end();) {
            if (live.contains(&*it)) {
                ++it;
            } else {
                it->Invalidate();
                it = block->Instructions().erase(it);
            }
        }
    }
}

} // namespace Shader::Optimization
