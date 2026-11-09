// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>
#include "common/logging/log.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"

namespace Shader::Optimization {

namespace {

using Op = ResourcePredicate::Op;

/// Turns the uniform parts of the conditions guarding each image use into predicates.
class PredicateBuilder {
public:
    explicit PredicateBuilder(ResourcePredicate& predicates_) : predicates{predicates_} {
        predicates.num_nodes = 0;
        predicates.max_flat = 0;
    }

    /// The clauses that must hold for the body of `if (cond)` to run, compiled to nodes. A clause
    /// that reads per-lane data can't be evaluated before the draw and is left out.
    void Clauses(const IR::Value& cond, bool negated, std::vector<u8>& out) {
        if (cond.IsImmediate()) {
            return;
        }
        IR::Inst* inst = cond.Inst();
        switch (inst->GetOpcode()) {
        case IR::Opcode::ConditionRef:
            return Clauses(inst->Arg(0), negated, out);
        case IR::Opcode::LogicalNot:
            return Clauses(inst->Arg(0), !negated, out);
        case IR::Opcode::LogicalAnd:
            if (!negated) {
                Clauses(inst->Arg(0), false, out);
                Clauses(inst->Arg(1), false, out);
                return;
            }
            break;
        case IR::Opcode::LogicalOr:
            if (negated) {
                Clauses(inst->Arg(0), true, out);
                Clauses(inst->Arg(1), true, out);
                return;
            }
            break;
        case IR::Opcode::SelectU1:
            // c ? x : false is c && x; c ? true : x is c || x.
            if (!negated && IsImmBool(inst->Arg(2), false)) {
                Clauses(inst->Arg(0), false, out);
                Clauses(inst->Arg(1), false, out);
                return;
            }
            if (negated && IsImmBool(inst->Arg(1), true)) {
                Clauses(inst->Arg(0), true, out);
                Clauses(inst->Arg(2), true, out);
                return;
            }
            break;
        default:
            break;
        }
        if (!IsUniform(cond)) {
            return;
        }
        if (const auto node = Compile(cond)) {
            if (const auto clause = negated ? Add(Op::LNot, *node) : node) {
                if (std::ranges::find(out, *clause) == out.end()) {
                    out.push_back(*clause);
                }
            }
        }
    }

    /// A node that holds when `base` (if any) and every clause do. Out of nodes, it leaves out the
    /// clauses that don't fit: fewer clauses still have to hold, so the result stays safe.
    std::optional<u8> AllOf(std::optional<u8> base, const std::vector<u8>& clauses) {
        std::optional<u8> result = base;
        for (const u8 clause : clauses) {
            const auto next = result ? Add(Op::LAnd, *result, clause) : clause;
            if (!next) {
                break;
            }
            result = next;
        }
        return result;
    }

    std::optional<u8> AnyOf(u8 a, u8 b) {
        return Add(Op::LOr, a, b);
    }

private:
    static bool IsImmBool(const IR::Value& value, bool imm) {
        return value.IsImmediate() && value.Type() == IR::Type::U1 && value.U1() == imm;
    }

    /// The node for (op, operands), reusing an equal one.
    std::optional<u8> Add(Op op, u8 a = 0, u8 b = 0, u8 c = 0, u32 imm = 0) {
        const u64 key = u64(op) | u64(a) << 8 | u64(b) << 16 | u64(c) << 24 | u64(imm) << 32;
        if (const auto it = nodes_by_key.find(key); it != nodes_by_key.end()) {
            return it->second;
        }
        if (predicates.num_nodes == ResourcePredicate::MaxNodes) {
            return std::nullopt;
        }
        predicates.nodes[predicates.num_nodes] = {op, a, b, c, imm};
        nodes_by_key.emplace(key, predicates.num_nodes);
        return predicates.num_nodes++;
    }

    /// Whether the value depends only on immediates and flattened user data, checked before
    /// compiling so a per-lane clause doesn't spend nodes on its uniform parts.
    bool IsUniform(const IR::Value& value) {
        if (value.IsImmediate()) {
            const auto type = value.Type();
            return type == IR::Type::U1 || type == IR::Type::U32 || type == IR::Type::F32;
        }
        IR::Inst* inst = value.Inst();
        if (const auto it = uniform.find(inst); it != uniform.end()) {
            return it->second;
        }
        bool result;
        switch (inst->GetOpcode()) {
        case IR::Opcode::GetUserData:
            result = true;
            break;
        case IR::Opcode::ReadConst:
            result = inst->Flags<u32>() != 0;
            break;
        case IR::Opcode::ReadConstBuffer:
            result = inst->Flags<IR::BufferInstInfo>().flatbuf_off_dw != 0;
            break;
        default:
            result = IsCompilableOp(inst->GetOpcode());
            for (u32 i = 0; result && i < inst->NumArgs(); ++i) {
                result = IsUniform(inst->Arg(i));
            }
            break;
        }
        uniform.emplace(inst, result);
        return result;
    }

    std::optional<u8> Flat(u32 offset) {
        if (offset == 0 || offset >= 0xFFFF) {
            return std::nullopt;
        }
        predicates.max_flat = std::max<u16>(predicates.max_flat, offset + 1);
        return Add(Op::Flat, 0, 0, 0, offset);
    }

    /// The value as nodes, if it depends only on immediates and the flattened user data.
    std::optional<u8> Compile(const IR::Value& value) {
        if (value.IsImmediate()) {
            switch (value.Type()) {
            case IR::Type::U1:
                return Add(Op::Imm, 0, 0, 0, value.U1() ? 1 : 0);
            case IR::Type::U32:
                return Add(Op::Imm, 0, 0, 0, value.U32());
            case IR::Type::F32:
                return Add(Op::Imm, 0, 0, 0, std::bit_cast<u32>(value.F32()));
            default:
                return std::nullopt;
            }
        }
        IR::Inst* inst = value.Inst();
        if (const auto it = compiled.find(inst); it != compiled.end()) {
            return it->second;
        }
        const auto result = CompileInst(inst);
        compiled.emplace(inst, result);
        return result;
    }

    std::optional<u8> CompileInst(IR::Inst* inst) {
        const auto unary = [&](Op op) -> std::optional<u8> {
            const auto a = Compile(inst->Arg(0));
            return a ? Add(op, *a) : std::nullopt;
        };
        const auto binary = [&](Op op) -> std::optional<u8> {
            const auto a = Compile(inst->Arg(0));
            const auto b = a ? Compile(inst->Arg(1)) : std::nullopt;
            return b ? Add(op, *a, *b) : std::nullopt;
        };
        const auto ternary = [&](Op op) -> std::optional<u8> {
            const auto a = Compile(inst->Arg(0));
            const auto b = a ? Compile(inst->Arg(1)) : std::nullopt;
            const auto c = b ? Compile(inst->Arg(2)) : std::nullopt;
            return c ? Add(op, *a, *b, *c) : std::nullopt;
        };
        switch (inst->GetOpcode()) {
        case IR::Opcode::GetUserData:
            return Flat(static_cast<u32>(inst->Arg(0).ScalarReg()));
        case IR::Opcode::ReadConst:
            // Zero when the read wasn't flattened (a dynamic offset).
            return Flat(inst->Flags<u32>());
        case IR::Opcode::ReadConstBuffer:
            return Flat(inst->Flags<IR::BufferInstInfo>().flatbuf_off_dw);
        case IR::Opcode::BitCastU32F32:
        case IR::Opcode::BitCastF32U32:
            return Compile(inst->Arg(0));
        case IR::Opcode::IAdd32:
            return binary(Op::IAdd);
        case IR::Opcode::ISub32:
            return binary(Op::ISub);
        case IR::Opcode::IMul32:
            return binary(Op::IMul);
        case IR::Opcode::BitwiseAnd32:
            return binary(Op::And);
        case IR::Opcode::BitwiseOr32:
            return binary(Op::Or);
        case IR::Opcode::BitwiseXor32:
            return binary(Op::Xor);
        case IR::Opcode::BitwiseNot32:
            return unary(Op::Not);
        case IR::Opcode::ShiftLeftLogical32:
            return binary(Op::Shl);
        case IR::Opcode::ShiftRightLogical32:
            return binary(Op::Shr);
        case IR::Opcode::ShiftRightArithmetic32:
            return binary(Op::Sar);
        case IR::Opcode::UMin32:
            return binary(Op::UMin);
        case IR::Opcode::UMax32:
            return binary(Op::UMax);
        case IR::Opcode::SMin32:
            return binary(Op::SMin);
        case IR::Opcode::SMax32:
            return binary(Op::SMax);
        case IR::Opcode::BitFieldUExtract:
            return ternary(Op::BfeU);
        case IR::Opcode::IEqual32:
            return binary(Op::IEq);
        case IR::Opcode::INotEqual32:
            return binary(Op::INe);
        case IR::Opcode::ULessThan32:
            return binary(Op::ULt);
        case IR::Opcode::ULessThanEqual32:
            return binary(Op::ULe);
        case IR::Opcode::UGreaterThan32:
            return binary(Op::UGt);
        case IR::Opcode::UGreaterThanEqual32:
            return binary(Op::UGe);
        case IR::Opcode::SLessThan32:
            return binary(Op::SLt);
        case IR::Opcode::SLessThanEqual32:
            return binary(Op::SLe);
        case IR::Opcode::SGreaterThan32:
            return binary(Op::SGt);
        case IR::Opcode::SGreaterThanEqual32:
            return binary(Op::SGe);
        case IR::Opcode::LogicalAnd:
            return binary(Op::LAnd);
        case IR::Opcode::LogicalOr:
            return binary(Op::LOr);
        case IR::Opcode::LogicalXor:
            return binary(Op::LXor);
        case IR::Opcode::LogicalNot:
            return unary(Op::LNot);
        case IR::Opcode::SelectU1:
        case IR::Opcode::SelectU32:
        case IR::Opcode::SelectF32:
            return ternary(Op::Select);
        case IR::Opcode::FPAdd32:
            return binary(Op::FAdd);
        case IR::Opcode::FPMul32:
            return binary(Op::FMul);
        case IR::Opcode::FPNeg32:
            return unary(Op::FNeg);
        case IR::Opcode::FPAbs32:
            return unary(Op::FAbs);
        case IR::Opcode::FPMin32:
            return binary(Op::FMin);
        case IR::Opcode::FPMax32:
            return binary(Op::FMax);
        case IR::Opcode::FPOrdEqual32:
            return binary(Op::FOrdEq);
        case IR::Opcode::FPOrdNotEqual32:
            return binary(Op::FOrdNe);
        case IR::Opcode::FPOrdLessThan32:
            return binary(Op::FOrdLt);
        case IR::Opcode::FPOrdLessThanEqual32:
            return binary(Op::FOrdLe);
        case IR::Opcode::FPOrdGreaterThan32:
            return binary(Op::FOrdGt);
        case IR::Opcode::FPOrdGreaterThanEqual32:
            return binary(Op::FOrdGe);
        case IR::Opcode::FPUnordEqual32:
            return binary(Op::FUnordEq);
        case IR::Opcode::FPUnordNotEqual32:
            return binary(Op::FUnordNe);
        case IR::Opcode::FPUnordLessThan32:
            return binary(Op::FUnordLt);
        case IR::Opcode::FPUnordLessThanEqual32:
            return binary(Op::FUnordLe);
        case IR::Opcode::FPUnordGreaterThan32:
            return binary(Op::FUnordGt);
        case IR::Opcode::FPUnordGreaterThanEqual32:
            return binary(Op::FUnordGe);
        default:
            return std::nullopt;
        }
    }

    static bool IsCompilableOp(IR::Opcode op) {
        switch (op) {
        case IR::Opcode::BitCastU32F32:
        case IR::Opcode::BitCastF32U32:
        case IR::Opcode::IAdd32:
        case IR::Opcode::ISub32:
        case IR::Opcode::IMul32:
        case IR::Opcode::BitwiseAnd32:
        case IR::Opcode::BitwiseOr32:
        case IR::Opcode::BitwiseXor32:
        case IR::Opcode::BitwiseNot32:
        case IR::Opcode::ShiftLeftLogical32:
        case IR::Opcode::ShiftRightLogical32:
        case IR::Opcode::ShiftRightArithmetic32:
        case IR::Opcode::UMin32:
        case IR::Opcode::UMax32:
        case IR::Opcode::SMin32:
        case IR::Opcode::SMax32:
        case IR::Opcode::BitFieldUExtract:
        case IR::Opcode::IEqual32:
        case IR::Opcode::INotEqual32:
        case IR::Opcode::ULessThan32:
        case IR::Opcode::ULessThanEqual32:
        case IR::Opcode::UGreaterThan32:
        case IR::Opcode::UGreaterThanEqual32:
        case IR::Opcode::SLessThan32:
        case IR::Opcode::SLessThanEqual32:
        case IR::Opcode::SGreaterThan32:
        case IR::Opcode::SGreaterThanEqual32:
        case IR::Opcode::LogicalAnd:
        case IR::Opcode::LogicalOr:
        case IR::Opcode::LogicalXor:
        case IR::Opcode::LogicalNot:
        case IR::Opcode::SelectU1:
        case IR::Opcode::SelectU32:
        case IR::Opcode::SelectF32:
        case IR::Opcode::FPAdd32:
        case IR::Opcode::FPMul32:
        case IR::Opcode::FPNeg32:
        case IR::Opcode::FPAbs32:
        case IR::Opcode::FPMin32:
        case IR::Opcode::FPMax32:
        case IR::Opcode::FPOrdEqual32:
        case IR::Opcode::FPOrdNotEqual32:
        case IR::Opcode::FPOrdLessThan32:
        case IR::Opcode::FPOrdLessThanEqual32:
        case IR::Opcode::FPOrdGreaterThan32:
        case IR::Opcode::FPOrdGreaterThanEqual32:
        case IR::Opcode::FPUnordEqual32:
        case IR::Opcode::FPUnordNotEqual32:
        case IR::Opcode::FPUnordLessThan32:
        case IR::Opcode::FPUnordLessThanEqual32:
        case IR::Opcode::FPUnordGreaterThan32:
        case IR::Opcode::FPUnordGreaterThanEqual32:
            return true;
        default:
            return false;
        }
    }

    ResourcePredicate& predicates;
    std::unordered_map<const IR::Inst*, std::optional<u8>> compiled;
    std::unordered_map<const IR::Inst*, bool> uniform;
    std::unordered_map<u64, u8> nodes_by_key;
};

/// The image an instruction reads or writes, if it's an image access.
std::optional<u32> ImageIndex(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::ImageSampleRaw:
    case IR::Opcode::ImageSampleImplicitLod:
    case IR::Opcode::ImageSampleExplicitLod:
    case IR::Opcode::ImageSampleDrefImplicitLod:
    case IR::Opcode::ImageSampleDrefExplicitLod:
    case IR::Opcode::ImageGather:
    case IR::Opcode::ImageGatherDref:
    case IR::Opcode::ImageQueryLod:
    case IR::Opcode::ImageGradient:
        // The sampler is in the upper half of the handle.
        return inst.Arg(0).IsImmediate() ? std::optional{inst.Arg(0).U32() & 0xFFFF} : std::nullopt;
    case IR::Opcode::ImageRead:
    case IR::Opcode::ImageWrite:
    case IR::Opcode::ImageQueryDimensions:
        return inst.Arg(0).IsImmediate() ? std::optional{inst.Arg(0).U32()} : std::nullopt;
    default:
        if (IsImageAtomicInstruction(inst) && inst.Arg(0).IsImmediate()) {
            return inst.Arg(0).U32();
        }
        return std::nullopt;
    }
}

} // Anonymous namespace

void ResourcePredicatePass(IR::Program& program) {
    Info& info = program.info;
    PredicateBuilder builder{info.predicates};
    for (auto& image : info.images) {
        image.predicate = 0;
    }

    // Per image: unset, the predicate of the uses seen so far, or always read.
    enum class State { Unseen, Predicated, Always };
    std::vector<State> states(info.images.size(), State::Unseen);
    std::vector<u8> roots(info.images.size());

    // One entry per enclosing if, innermost last, each the AND of its own clauses and the entry
    // before it. The body of an if runs only when its condition holds, and loops and breaks only
    // ever run blocks less often, so the top entry must hold for a block to run.
    std::vector<std::optional<u8>> scopes;
    std::vector<u8> clauses;
    for (const auto& node : program.syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::If: {
            const std::optional<u8> outer = scopes.empty() ? std::nullopt : scopes.back();
            clauses.clear();
            builder.Clauses(node.data.if_node.cond, false, clauses);
            scopes.push_back(builder.AllOf(outer, clauses));
            break;
        }
        case IR::AbstractSyntaxNode::Type::EndIf:
            if (!scopes.empty()) {
                scopes.pop_back();
            }
            break;
        case IR::AbstractSyntaxNode::Type::Block: {
            const std::optional<u8> block_predicate = scopes.empty() ? std::nullopt : scopes.back();
            for (const IR::Inst& inst : node.data.block->Instructions()) {
                const auto index = ImageIndex(inst);
                if (!index || *index >= states.size() || states[*index] == State::Always) {
                    continue;
                }
                if (!block_predicate) {
                    states[*index] = State::Always;
                } else if (states[*index] == State::Unseen) {
                    states[*index] = State::Predicated;
                    roots[*index] = *block_predicate;
                } else if (roots[*index] != *block_predicate) {
                    // Read in two places: read if either runs.
                    const auto either = builder.AnyOf(roots[*index], *block_predicate);
                    if (either) {
                        roots[*index] = *either;
                    } else {
                        states[*index] = State::Always;
                    }
                }
            }
            break;
        }
        default:
            break;
        }
    }

    u32 num_predicated = 0;
    for (u32 i = 0; i < info.images.size(); ++i) {
        if (states[i] == State::Predicated) {
            info.images[i].predicate = roots[i] + 1;
            ++num_predicated;
        }
    }
    if (num_predicated > 0) {
        LOG_DEBUG(Render_Recompiler,
                  "Shader {:#x}: {} of {} images read under uniform conditions ({} nodes)",
                  info.pgm_hash, num_predicated, info.images.size(), info.predicates.num_nodes);
    }
}

} // namespace Shader::Optimization
