// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <utility>

#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>

#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/microinstruction.h"
#include "shader_recompiler/ir/opcodes.h"

namespace Shader::Optimization {

static bool IsAllowedInst(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::Ballot ||
           inst.GetOpcode() == IR::Opcode::UnpackUint2x32 ||
           inst.GetOpcode() == IR::Opcode::BitCastU32F32 ||
           inst.GetOpcode() == IR::Opcode::BitCastF32U32;
}

static bool IsIdenticalInst(const IR::Inst* a, const IR::Inst* b) {
    if (a->GetOpcode() != b->GetOpcode()) {
        return false;
    }
    for (size_t i = 0; i < a->NumArgs(); i++) {
        if (a->Arg(i) != b->Arg(i)) {
            return false;
        }
    }
    return true;
}

static void RunLocalCSE(IR::Block* block) {
    // Instructions on other instructions are found by their opcode and argument in a map, as a
    // block can hold thousands of bit casts; the few on immediates are compared one by one.
    struct KeyHash {
        size_t operator()(const std::pair<IR::Opcode, const IR::Inst*>& key) const {
            return std::hash<const void*>{}(key.second) ^ (size_t(key.first) * 0x9E3779B97F4A7C15);
        }
    };
    tsl::robin_map<std::pair<IR::Opcode, const IR::Inst*>, IR::Inst*, KeyHash> by_arg;
    boost::container::small_vector<IR::Inst*, 8> on_immediates;
    auto it = std::ranges::find_if_not(*block, IR::IsPhi);
    for (; it != block->end(); ++it) {
        IR::Inst& inst{*it};
        if (!IsAllowedInst(inst)) {
            continue;
        }
        if (inst.NumArgs() == 1 && !inst.Arg(0).IsImmediate()) {
            const auto [found, is_new] =
                by_arg.try_emplace(std::make_pair(inst.GetOpcode(), inst.Arg(0).Inst()), &inst);
            if (!is_new) {
                inst.ReplaceUsesWithAndRemove(IR::Value{found->second});
            }
            continue;
        }
        const auto it = std::ranges::find_if(
            on_immediates, [&](const IR::Inst* b) { return IsIdenticalInst(&inst, b); });
        if (it != on_immediates.end()) {
            inst.ReplaceUsesWithAndRemove(IR::Value{*it});
        } else {
            on_immediates.emplace_back(&inst);
        }
    }
}

} // namespace Shader::Optimization
