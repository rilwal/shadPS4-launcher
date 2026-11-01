// A small JSON value, parser and writer: enough for the plugin protocol's messages, and no
// dependency for the emulators that build this library into theirs.
#ifndef LAUNCHER_CLIENT_JSON_H
#define LAUNCHER_CLIENT_JSON_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lc::json {

class Value {
public:
	enum class Kind { Null, Bool, Int, Uint, Float, String, Array, Object };
	using Member = std::pair<std::string, Value>;

	Value() = default;
	Value(std::nullptr_t) {}
	Value(bool b): m_kind(Kind::Bool), m_bool(b) {}
	Value(int v): m_kind(v < 0 ? Kind::Int : Kind::Uint), m_int(v), m_uint(static_cast<uint64_t>(v)) {}
	Value(int64_t v): m_kind(v < 0 ? Kind::Int : Kind::Uint), m_int(v), m_uint(static_cast<uint64_t>(v)) {}
	Value(uint32_t v): m_kind(Kind::Uint), m_int(v), m_uint(v) {}
	Value(uint64_t v): m_kind(Kind::Uint), m_int(static_cast<int64_t>(v)), m_uint(v) {}
	Value(double v): m_kind(Kind::Float), m_float(v) {}
	Value(const char* s): m_kind(Kind::String), m_string(s) {}
	Value(std::string_view s): m_kind(Kind::String), m_string(s) {}
	Value(std::string s): m_kind(Kind::String), m_string(std::move(s)) {}

	static Value array() {
		Value v;
		v.m_kind = Kind::Array;
		return v;
	}
	static Value object() {
		Value v;
		v.m_kind = Kind::Object;
		return v;
	}
	// A string, or null for nullptr
	static Value text_or_null(const char* s) { return s != nullptr ? Value(s) : Value(); }

	[[nodiscard]] Kind kind() const { return m_kind; }
	[[nodiscard]] bool is_null() const { return m_kind == Kind::Null; }
	[[nodiscard]] bool is_bool() const { return m_kind == Kind::Bool; }
	[[nodiscard]] bool is_string() const { return m_kind == Kind::String; }
	[[nodiscard]] bool is_array() const { return m_kind == Kind::Array; }
	[[nodiscard]] bool is_object() const { return m_kind == Kind::Object; }
	[[nodiscard]] bool is_number() const {
		return m_kind == Kind::Int || m_kind == Kind::Uint || m_kind == Kind::Float;
	}
	[[nodiscard]] bool is_integer() const { return m_kind == Kind::Int || m_kind == Kind::Uint; }
	[[nodiscard]] bool is_unsigned() const { return m_kind == Kind::Uint; }

	[[nodiscard]] bool               as_bool() const { return m_bool; }
	[[nodiscard]] int64_t            as_int() const { return m_int; }
	[[nodiscard]] uint64_t           as_uint() const { return m_uint; }
	[[nodiscard]] double             as_float() const;
	[[nodiscard]] const std::string& as_string() const { return m_string; }

	// Arrays
	[[nodiscard]] const std::vector<Value>& items() const { return m_items; }
	Value&                                  push(Value v) { return m_items.emplace_back(std::move(v)); }

	// Objects, in the order members were set
	[[nodiscard]] const std::vector<Member>& members() const { return m_members; }
	[[nodiscard]] const Value*               find(std::string_view key) const;
	Value&                                   set(std::string_view key, Value v);
	// The member if it's a string, else ""
	[[nodiscard]] std::string text(std::string_view key) const;

	[[nodiscard]] std::string dump() const;

private:
	void dump_to(std::string& out) const;

	Kind                m_kind  = Kind::Null;
	bool                m_bool  = false;
	int64_t             m_int   = 0;
	uint64_t            m_uint  = 0;
	double              m_float = 0;
	std::string         m_string;
	std::vector<Value>  m_items;
	std::vector<Member> m_members;
};

// Strict JSON, nested at most 64 deep. Nothing for anything else.
std::optional<Value> parse(std::string_view text);

} // namespace lc::json

#endif // LAUNCHER_CLIENT_JSON_H
