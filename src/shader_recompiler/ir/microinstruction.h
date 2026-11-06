// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

#include <boost/container/list.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/intrusive/list.hpp>

#include "common/assert.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/ssa.h"
#include "shader_recompiler/ir/value.h"

namespace Shader::IR {

/// Allocates single list nodes from a free list per thread, carved from larger blocks. Use
/// lists gain and lose a node with every argument set, so the general allocator took about a
/// third of translation time. A node freed on another thread goes to that thread's list; blocks
/// stay with the process.
template <typename T>
struct NodeAllocator {
    using value_type = T;

    NodeAllocator() = default;
    template <typename U>
    NodeAllocator(const NodeAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        if (n != 1) {
            return std::allocator<T>{}.allocate(n);
        }
        void*& head = FreeList();
        if (head == nullptr) {
            Refill(head);
        }
        void* node = head;
        head = *static_cast<void**>(node);
        return static_cast<T*>(node);
    }

    void deallocate(T* p, size_t n) noexcept {
        if (n != 1) {
            std::allocator<T>{}.deallocate(p, n);
            return;
        }
        void*& head = FreeList();
        *reinterpret_cast<void**>(p) = head;
        head = p;
    }

    template <typename U>
    bool operator==(const NodeAllocator<U>&) const noexcept {
        return true;
    }

private:
    static constexpr size_t SlotSize =
        (std::max(sizeof(T), sizeof(void*)) + alignof(std::max_align_t) - 1) &
        ~(alignof(std::max_align_t) - 1);
    static constexpr size_t SlotsPerBlock = 256;

    static void*& FreeList() {
        thread_local void* head = nullptr;
        return head;
    }

    static void Refill(void*& head) {
        auto* block = static_cast<std::byte*>(::operator new(SlotSize * SlotsPerBlock));
        for (size_t i = SlotsPerBlock; i-- > 0;) {
            void* slot = block + i * SlotSize;
            *static_cast<void**>(slot) = head;
            head = slot;
        }
    }
};

struct Use {
    Inst* user;
    u32 operand;

    Use() = default;
    Use(Inst* user_, u32 operand_) : user(user_), operand(operand_) {}
    Use(const Use&) = default;
    bool operator==(const Use&) const noexcept = default;
};

class Inst : public boost::intrusive::list_base_hook<> {
public:
    explicit Inst(IR::Opcode op_, u32 flags_) noexcept;
    explicit Inst(const Inst& base);
    ~Inst();

    Inst& operator=(const Inst&) = delete;

    Inst& operator=(Inst&&) = delete;
    Inst(Inst&&) = delete;

    IR::Block* GetParent() const {
        ASSERT(parent);
        return parent;
    }
    void SetParent(IR::Block* block) {
        parent = block;
    }

    int UseCount() const noexcept {
        return uses.size();
    }

    bool HasUses() const noexcept {
        return uses.size() > 0;
    }

    IR::Opcode GetOpcode() const noexcept {
        return op;
    }

    size_t NumArgs() const {
        return op == IR::Opcode::Phi ? phi_args.size() : NumArgsOf(op);
    }

    Value Arg(size_t index) const noexcept {
        if (op == IR::Opcode::Phi) {
            return phi_args[index].second;
        } else {
            return args[index];
        }
    }

    void ReplaceUsesWithAndRemove(Value replacement) {
        ReplaceUsesWith(replacement, false);
    }

    void ReplaceUsesWith(Value replacement) {
        ReplaceUsesWith(replacement, true);
    }

    template <typename FlagsType>
        requires(sizeof(FlagsType) <= sizeof(u64) && std::is_trivially_copyable_v<FlagsType>)
    [[nodiscard]] FlagsType Flags() const noexcept {
        FlagsType ret;
        std::memcpy(reinterpret_cast<char*>(&ret), &flags, sizeof(ret));
        return ret;
    }

    template <typename FlagsType>
        requires(sizeof(FlagsType) <= sizeof(u64) && std::is_trivially_copyable_v<FlagsType>)
    void SetFlags(FlagsType value) noexcept {
        std::memcpy(&flags, &value, sizeof(value));
    }

    template <typename DefinitionType>
    void SetDefinition(DefinitionType def) {
        definition = std::bit_cast<u32>(def);
    }

    template <typename DefinitionType>
    [[nodiscard]] DefinitionType Definition() const noexcept {
        return std::bit_cast<DefinitionType>(definition);
    }

    const auto& Uses() const {
        return uses;
    }

    void SetRegTag(RegTag new_tag) {
        reg_tag = new_tag;
    }

    RegTag GetRegTag() const {
        return reg_tag;
    }

    bool MayHaveSideEffects() const noexcept;
    bool AreAllArgsImmediates() const;

    IR::Type Type() const;
    void ReplaceOpcode(IR::Opcode opcode);
    void SetArg(size_t index, Value value);

    Block* PhiBlock(size_t index) const;
    void AddPhiOperand(Block* predecessor, const Value& value);

    void Invalidate();
    void ClearArgs();

    u32 scc_index{};

private:
    struct NonTriviallyDummy {
        NonTriviallyDummy() noexcept {}
    };

    void Use(Inst* used, u32 operand);
    void UndoUse(Inst* used, u32 operand);
    void ReplaceUsesWith(Value replacement, bool preserve);

    IR::Opcode op{};
    u32 definition{};
    u64 flags{};
    IR::Block* parent{};
    RegTag reg_tag{};
    union {
        NonTriviallyDummy dummy{};
        boost::container::small_vector<std::pair<Block*, Value>, 2> phi_args;
        std::array<Value, 6> args;
    };

    boost::container::list<IR::Use, NodeAllocator<IR::Use>> uses;
};
static_assert(sizeof(Inst) <= 184, "Inst size unintentionally increased");

[[nodiscard]] inline bool IsPhi(const Inst& inst) {
    return inst.GetOpcode() == Opcode::Phi;
}

} // namespace Shader::IR
