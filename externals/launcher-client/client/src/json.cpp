#include "json.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace lc::json {

double Value::as_float() const {
	switch (m_kind) {
		case Kind::Int: return static_cast<double>(m_int);
		case Kind::Uint: return static_cast<double>(m_uint);
		default: return m_float;
	}
}

const Value* Value::find(std::string_view key) const {
	for (const auto& [name, value]: m_members) {
		if (name == key) {
			return &value;
		}
	}
	return nullptr;
}

Value& Value::set(std::string_view key, Value v) {
	for (auto& [name, value]: m_members) {
		if (name == key) {
			value = std::move(v);
			return value;
		}
	}
	return m_members.emplace_back(std::string(key), std::move(v)).second;
}

std::string Value::text(std::string_view key) const {
	const Value* v = find(key);
	return v != nullptr && v->is_string() ? v->as_string() : std::string {};
}

// Invalid UTF-8 becomes U+FFFD, so a bad title from a game can't break the frame.
static void dump_string(std::string& out, const std::string& s) {
	out += '"';
	for (size_t i = 0; i < s.size();) {
		const auto c = static_cast<unsigned char>(s[i]);
		if (c < 0x80) {
			switch (c) {
				case '"': out += "\\\""; break;
				case '\\': out += "\\\\"; break;
				case '\n': out += "\\n"; break;
				case '\r': out += "\\r"; break;
				case '\t': out += "\\t"; break;
				default:
					if (c < 0x20) {
						char buf[8];
						std::snprintf(buf, sizeof(buf), "\\u%04x", c);
						out += buf;
					} else {
						out += static_cast<char>(c);
					}
			}
			i++;
			continue;
		}
		// The length a lead byte says, and the smallest code point that length may hold
		size_t   len = 0;
		uint32_t min = 0;
		if ((c & 0xE0) == 0xC0) {
			len = 2;
			min = 0x80;
		} else if ((c & 0xF0) == 0xE0) {
			len = 3;
			min = 0x800;
		} else if ((c & 0xF8) == 0xF0) {
			len = 4;
			min = 0x10000;
		}
		uint32_t cp = len == 2 ? c & 0x1F : len == 3 ? c & 0x0F : c & 0x07;
		bool     ok = len != 0 && i + len <= s.size();
		for (size_t k = 1; ok && k < len; k++) {
			const auto next = static_cast<unsigned char>(s[i + k]);
			ok              = (next & 0xC0) == 0x80;
			cp              = (cp << 6) | (next & 0x3F);
		}
		ok = ok && cp >= min && cp <= 0x10FFFF && (cp < 0xD800 || cp > 0xDFFF);
		if (ok) {
			out.append(s, i, len);
			i += len;
		} else {
			out += "\xEF\xBF\xBD";
			i++;
		}
	}
	out += '"';
}

void Value::dump_to(std::string& out) const {
	switch (m_kind) {
		case Kind::Null: out += "null"; break;
		case Kind::Bool: out += m_bool ? "true" : "false"; break;
		case Kind::Int: out += std::to_string(m_int); break;
		case Kind::Uint: out += std::to_string(m_uint); break;
		case Kind::Float: {
			if (!std::isfinite(m_float)) {
				out += "null";
				break;
			}
			char buf[32];
			std::snprintf(buf, sizeof(buf), "%.9g", m_float);
			out += buf;
			break;
		}
		case Kind::String: dump_string(out, m_string); break;
		case Kind::Array: {
			out += '[';
			for (size_t i = 0; i < m_items.size(); i++) {
				if (i != 0) {
					out += ',';
				}
				m_items[i].dump_to(out);
			}
			out += ']';
			break;
		}
		case Kind::Object: {
			out += '{';
			for (size_t i = 0; i < m_members.size(); i++) {
				if (i != 0) {
					out += ',';
				}
				dump_string(out, m_members[i].first);
				out += ':';
				m_members[i].second.dump_to(out);
			}
			out += '}';
			break;
		}
	}
}

std::string Value::dump() const {
	std::string out;
	dump_to(out);
	return out;
}

namespace {

class Parser {
public:
	explicit Parser(std::string_view text): m_text(text) {}

	std::optional<Value> document() {
		Value v;
		if (!value(v, 0)) {
			return std::nullopt;
		}
		skip_space();
		if (m_pos != m_text.size()) {
			return std::nullopt;
		}
		return v;
	}

private:
	static constexpr int MAX_DEPTH = 64;

	void skip_space() {
		while (m_pos < m_text.size() && (m_text[m_pos] == ' ' || m_text[m_pos] == '\t' ||
		                                 m_text[m_pos] == '\n' || m_text[m_pos] == '\r')) {
			m_pos++;
		}
	}

	bool literal(std::string_view word) {
		if (m_text.substr(m_pos, word.size()) != word) {
			return false;
		}
		m_pos += word.size();
		return true;
	}

	bool hex4(uint32_t& out) {
		if (m_pos + 4 > m_text.size()) {
			return false;
		}
		out = 0;
		for (int i = 0; i < 4; i++) {
			const char c = m_text[m_pos++];
			out <<= 4;
			if (c >= '0' && c <= '9') {
				out |= static_cast<uint32_t>(c - '0');
			} else if (c >= 'a' && c <= 'f') {
				out |= static_cast<uint32_t>(c - 'a' + 10);
			} else if (c >= 'A' && c <= 'F') {
				out |= static_cast<uint32_t>(c - 'A' + 10);
			} else {
				return false;
			}
		}
		return true;
	}

	static void put_utf8(std::string& out, uint32_t cp) {
		if (cp < 0x80) {
			out += static_cast<char>(cp);
		} else if (cp < 0x800) {
			out += static_cast<char>(0xC0 | (cp >> 6));
			out += static_cast<char>(0x80 | (cp & 0x3F));
		} else if (cp < 0x10000) {
			out += static_cast<char>(0xE0 | (cp >> 12));
			out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
			out += static_cast<char>(0x80 | (cp & 0x3F));
		} else {
			out += static_cast<char>(0xF0 | (cp >> 18));
			out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
			out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
			out += static_cast<char>(0x80 | (cp & 0x3F));
		}
	}

	bool string(std::string& out) {
		if (m_pos >= m_text.size() || m_text[m_pos] != '"') {
			return false;
		}
		m_pos++;
		while (m_pos < m_text.size()) {
			const char c = m_text[m_pos++];
			if (c == '"') {
				return true;
			}
			if (static_cast<unsigned char>(c) < 0x20) {
				return false;
			}
			if (c != '\\') {
				out += c;
				continue;
			}
			if (m_pos >= m_text.size()) {
				return false;
			}
			switch (m_text[m_pos++]) {
				case '"': out += '"'; break;
				case '\\': out += '\\'; break;
				case '/': out += '/'; break;
				case 'b': out += '\b'; break;
				case 'f': out += '\f'; break;
				case 'n': out += '\n'; break;
				case 'r': out += '\r'; break;
				case 't': out += '\t'; break;
				case 'u': {
					uint32_t cp = 0;
					if (!hex4(cp)) {
						return false;
					}
					if (cp >= 0xD800 && cp <= 0xDBFF) {
						uint32_t low = 0;
						if (!literal("\\u") || !hex4(low) || low < 0xDC00 || low > 0xDFFF) {
							return false;
						}
						cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
					} else if (cp >= 0xDC00 && cp <= 0xDFFF) {
						return false;
					}
					put_utf8(out, cp);
					break;
				}
				default: return false;
			}
		}
		return false;
	}

	bool number(Value& out) {
		const size_t start = m_pos;
		bool         integer = true;
		if (m_pos < m_text.size() && m_text[m_pos] == '-') {
			m_pos++;
		}
		const auto digits = [&]() {
			const size_t from = m_pos;
			while (m_pos < m_text.size() && m_text[m_pos] >= '0' && m_text[m_pos] <= '9') {
				m_pos++;
			}
			return m_pos > from;
		};
		if (!digits()) {
			return false;
		}
		if (m_pos < m_text.size() && m_text[m_pos] == '.') {
			integer = false;
			m_pos++;
			if (!digits()) {
				return false;
			}
		}
		if (m_pos < m_text.size() && (m_text[m_pos] == 'e' || m_text[m_pos] == 'E')) {
			integer = false;
			m_pos++;
			if (m_pos < m_text.size() && (m_text[m_pos] == '+' || m_text[m_pos] == '-')) {
				m_pos++;
			}
			if (!digits()) {
				return false;
			}
		}
		const std::string token(m_text.substr(start, m_pos - start));
		if (integer) {
			errno = 0;
			char* end = nullptr;
			if (token[0] == '-') {
				const long long v = std::strtoll(token.c_str(), &end, 10);
				if (errno == 0) {
					out = Value(static_cast<int64_t>(v));
					return true;
				}
			} else {
				const unsigned long long v = std::strtoull(token.c_str(), &end, 10);
				if (errno == 0) {
					out = Value(static_cast<uint64_t>(v));
					return true;
				}
			}
		}
		out = Value(std::strtod(token.c_str(), nullptr));
		return true;
	}

	bool value(Value& out, int depth) {
		if (depth > MAX_DEPTH) {
			return false;
		}
		skip_space();
		if (m_pos >= m_text.size()) {
			return false;
		}
		switch (m_text[m_pos]) {
			case 'n': out = Value(); return literal("null");
			case 't': out = Value(true); return literal("true");
			case 'f': out = Value(false); return literal("false");
			case '"': {
				std::string s;
				if (!string(s)) {
					return false;
				}
				out = Value(std::move(s));
				return true;
			}
			case '[': {
				m_pos++;
				out = Value::array();
				skip_space();
				if (m_pos < m_text.size() && m_text[m_pos] == ']') {
					m_pos++;
					return true;
				}
				for (;;) {
					Value item;
					if (!value(item, depth + 1)) {
						return false;
					}
					out.push(std::move(item));
					skip_space();
					if (m_pos >= m_text.size()) {
						return false;
					}
					const char c = m_text[m_pos++];
					if (c == ']') {
						return true;
					}
					if (c != ',') {
						return false;
					}
				}
			}
			case '{': {
				m_pos++;
				out = Value::object();
				skip_space();
				if (m_pos < m_text.size() && m_text[m_pos] == '}') {
					m_pos++;
					return true;
				}
				for (;;) {
					skip_space();
					std::string key;
					if (!string(key)) {
						return false;
					}
					skip_space();
					if (m_pos >= m_text.size() || m_text[m_pos++] != ':') {
						return false;
					}
					Value member;
					if (!value(member, depth + 1)) {
						return false;
					}
					out.set(key, std::move(member));
					skip_space();
					if (m_pos >= m_text.size()) {
						return false;
					}
					const char c = m_text[m_pos++];
					if (c == '}') {
						return true;
					}
					if (c != ',') {
						return false;
					}
				}
			}
			default: return number(out);
		}
	}

	std::string_view m_text;
	size_t           m_pos = 0;
};

} // namespace

std::optional<Value> parse(std::string_view text) {
	return Parser(text).document();
}

} // namespace lc::json
