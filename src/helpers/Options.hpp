#pragma once
// Advanced (key=value) escape hatch.
//
// Each object exposes a free-text "Advanced" line-edit; the config builders parse
// it here into a small key->value list and override the matching sherpa-onnx
// config fields. This exposes the long tail of options (hotwords, rule FSTs,
// endpoint rules, VAD timing, whisper language/task, per-model TTS scales, ...)
// without a dedicated port for each.
//
// Entries are separated by newlines or ';' (NOT ',', because sherpa values such
// as rule_fsts are themselves comma-separated lists). Each entry is `key = value`;
// keys are matched case-insensitively.
//
// Parsing runs on the worker thread inside the builders (std::string ops), never
// on the audio thread. The parsed strings are OWNED by the KeyValues object, so
// the pointers returned by find()/get_cstr() stay valid for its lifetime -- keep
// it in scope until after the Create* call that copies them.

#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sherpa::opts
{

class KeyValues
{
public:
  KeyValues() = default;
  explicit KeyValues(std::string_view text) { parse(text); }

  bool empty() const noexcept { return m_kv.empty(); }

  // Pointer to the owned value string for `key`, or nullptr if absent.
  const std::string* find(std::string_view key) const noexcept
  {
    for(const auto& [k, v] : m_kv)
      if(iequal(k, key))
        return &v;
    return nullptr;
  }
  bool has(std::string_view key) const noexcept { return find(key) != nullptr; }

  // c_str() for a string-valued option, or nullptr when absent -- lets callers do
  // `if(const char* s = adv.get_cstr("x")) cfg.x = s;`.
  const char* get_cstr(std::string_view key) const noexcept
  {
    const std::string* v = find(key);
    return v ? v->c_str() : nullptr;
  }

  // Override `target` in place, but only when the key is present and parses.
  void apply_float(std::string_view key, float& target) const noexcept
  {
    if(const std::string* v = find(key))
    {
      float out{};
      if(to_float(*v, out))
        target = out;
    }
  }
  void apply_int(std::string_view key, int& target) const noexcept
  {
    if(const std::string* v = find(key))
    {
      int out{};
      if(to_int(*v, out))
        target = out;
    }
  }
  // Set an int32 config flag from 0/1/true/false/on/off/yes/no.
  void apply_bool01(std::string_view key, int& target) const noexcept
  {
    if(const std::string* v = find(key))
      target = to_bool(*v) ? 1 : 0;
  }
  // Override a `const char*` config field from an owned value (kept alive by this).
  void apply_cstr(std::string_view key, const char*& target) const noexcept
  {
    if(const char* s = get_cstr(key))
      target = s;
  }

private:
  std::vector<std::pair<std::string, std::string>> m_kv;

  static char lower(char c) noexcept
  {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
  }
  static bool iequal(std::string_view a, std::string_view b) noexcept
  {
    if(a.size() != b.size())
      return false;
    for(std::size_t i = 0; i < a.size(); ++i)
      if(lower(a[i]) != lower(b[i]))
        return false;
    return true;
  }
  static std::string_view trim(std::string_view s) noexcept
  {
    constexpr std::string_view ws = " \t\r\n\f\v";
    const auto b = s.find_first_not_of(ws);
    if(b == std::string_view::npos)
      return {};
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
  }
  static bool to_float(const std::string& s, float& out) noexcept
  {
    // strtof rather than from_chars<float> (not on every libc++ we target).
    const char* b = s.c_str();
    char* end = nullptr;
    const float v = std::strtof(b, &end);
    if(end == b)
      return false;
    out = v;
    return true;
  }
  static bool to_int(const std::string& s, int& out) noexcept
  {
    const char* b = s.c_str();
    char* end = nullptr;
    const long v = std::strtol(b, &end, 10);
    if(end == b)
      return false;
    out = static_cast<int>(v);
    return true;
  }
  static bool to_bool(std::string_view s) noexcept
  {
    return iequal(s, "1") || iequal(s, "true") || iequal(s, "on")
           || iequal(s, "yes");
  }

  void parse(std::string_view text)
  {
    std::size_t i = 0;
    const std::size_t n = text.size();
    while(i < n)
    {
      std::size_t j = i;
      while(j < n && text[j] != '\n' && text[j] != ';')
        ++j;
      const std::string_view entry = trim(text.substr(i, j - i));
      i = j + 1;
      if(entry.empty())
        continue;
      const auto eq = entry.find('=');
      if(eq == std::string_view::npos)
        continue;
      const std::string_view k = trim(entry.substr(0, eq));
      const std::string_view v = trim(entry.substr(eq + 1));
      if(!k.empty())
        m_kv.emplace_back(std::string{k}, std::string{v});
    }
  }
};

}
