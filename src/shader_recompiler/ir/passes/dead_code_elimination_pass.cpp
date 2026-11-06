// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <vector>
#include <tsl/robin_set.h>
#include "shader_recompiler/ir/program.h"

namespace Shader::Optimization {

void DeadCodeEliminationPass(IR::Program& program) {
    // Mark everything reachable from instructions with side effects, then sweep the rest.
    // Removing a dead instruction one at a time searches each operand's use list for it, which is
    // quadratic for values with many dead users, so the sweep drops dead instructions without
    // touching use lists and then prunes the use lists of the live operands they referenced once.
    // Open addressing and a vector stack: this pass runs twice per translation.
    thread_local tsl::robin_set<IR::Inst*> live;
    thread_local tsl::robin_set<IR::Inst*> dead;
    thread_local std::vector<IR::Inst*> worklist;
    thread_local std::vector<IR::Inst*> pruned;
    live.clear();
    dead.clear();
    worklist.clear();
    pruned.clear();

    for (IR::Block* block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.MayHaveSideEffects()) {
                live.insert(&inst);
                worklist.push_back(&inst);
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
                continue;
            }
            for (size_t i = 0; i < it->NumArgs(); i++) {
                IR::Inst* const operand = it->Arg(i).TryInst();
                if (operand && live.contains(operand)) {
                    pruned.push_back(operand);
                }
            }
            dead.insert(&*it);
            it->InvalidateUnlinked();
            it = block->Instructions().erase(it);
        }
    }

    std::ranges::sort(pruned);
    const auto [first, last] = std::ranges::unique(pruned);
    pruned.erase(first, last);
    for (IR::Inst* inst : pruned) {
        inst->EraseUsesIf([](const IR::Use& use) { return dead.contains(use.user); });
    }
}

} // namespace Shader::Optimization
