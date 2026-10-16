// Independently authored Strata diagnostic addition; repository LICENSE applies.
// Applied to official Intel PTI (MIT); preserve PTI's LICENSE and notices.
#ifndef STRATA_METRIC_QUERY_AUDIT_H
#define STRATA_METRIC_QUERY_AUDIT_H
#if defined(_WIN32) || !defined(__linux__)
#error "This separately built tiny query audit collector supports Linux only"
#endif
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <level_zero/ze_api.h>

namespace OwnMetricAudit {
struct State {
  std::mutex mutex;
  bool enabled = false;
  bool closed = false;
  uint64_t sequence = 0;
  uint64_t calls = 0;
  std::map<uint64_t, uint64_t> pending; // query -> instance, zero before submission
};
// Intentionally process-lifetime: no destructor ordering against provider threads.
inline State& state() { static State* value = new State; return *value; }
inline std::string quote(const char* p, size_t cap = 4096) {
  std::string out = "\"";
  const char* digits = "0123456789abcdef";
  size_t i = 0;
  for (; i < cap && p[i]; ++i) {
    const unsigned char c = p[i];
    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
    else if (c < 32 || c >= 127) { out += "\\u00"; out += digits[c >> 4]; out += digits[c & 15]; }
    else out += char(c);
  }
  if (i == cap) std::_Exit(125);
  return out + '"';
}
template<class T> inline uint64_t id(T* p) { return reinterpret_cast<uintptr_t>(p); }
template<class T> inline std::string field(const char* key, T n) {
  return "," + quote(key) + ":" + std::to_string(uint64_t(n));
}
inline std::string text(const char* key, const char* value) { return "," + quote(key) + ":" + quote(value); }
template<size_t N> inline std::string array_text(const char* key, const char (&value)[N]) {
  return "," + quote(key) + ":" + quote(value, N);
}
inline const char* result_name(ze_result_t result) {
  switch (result) {
#define OWN_RESULT(name) case name: return #name;
#include "metric_query_result_names.inc"
#undef OWN_RESULT
  default: return "ZE_RESULT_UNRECOGNIZED";
  }
}
// Caller holds the audit mutex. One <= PIPE_BUF write, no buffered stream or retry.
inline void emit(const std::string& body) {
  auto& s = state();
  std::string line = "OWN_METRIC_V2 {\"schema\":2,\"pid\":" + std::to_string(getpid()) +
      field("seq", ++s.sequence) + body + "}\n";
  if (line.size() > 4096 || write(STDERR_FILENO, line.data(), line.size()) != ssize_t(line.size()))
    std::_Exit(125); // No stack unwind, freeing, retry or continued qualification.
}
[[noreturn]] inline void stop(const char* reason) {
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  emit(",\"phase\":\"reject\"" + text("reason", reason));
  std::_Exit(125);
}
inline void require(bool condition, const char* reason) {
  if (state().enabled && !condition) stop(reason);
}
inline void enable(bool selected) {
  if (!selected) return;
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.enabled) std::_Exit(125);
  s.enabled = true;
  emit(",\"phase\":\"open\",\"route\":\"Linux_EVENT_BASED_ComputeBasic\"");
}
// A missing function has no API result. Record it as unavailable; never invent one.
// Entry inputs are captured before invocation. Output data is read only on success.
template<class F, class Post, class... Args>
inline ze_result_t call(const char* site, const char* api, F fn,
                       const std::string& inputs, Post post, Args... args) {
  auto& s = state();
  if (!s.enabled) return fn(args...);
  uint64_t call_id;
  try {
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      if (s.closed) std::_Exit(125);
      call_id = ++s.calls;
      emit(",\"phase\":\"entry\"" + field("call", call_id) + text("site", site) + text("api", api) +
           ",\"data\":{\"present\":true" + inputs + "}");
      if (!fn) {
        emit(",\"phase\":\"unavailable\"" + field("call", call_id) + text("site", site) + text("api", api) +
             ",\"reason\":\"missing_loader_symbol\"");
        std::_Exit(125);
      }
    }
    const ze_result_t result = fn(args...); // Original call, arguments and return on success.
    {
      std::lock_guard<std::mutex> lock(s.mutex);
      std::ostringstream hex; hex << "0x" << std::hex << uint32_t(result);
      emit(",\"phase\":\"result\"" + field("call", call_id) + text("site", site) + text("api", api) +
           field("code", uint32_t(result)) + text("hex", hex.str().c_str()) + text("name", result_name(result)) +
           ",\"data\":" + (result == ZE_RESULT_SUCCESS ? "{\"present\":true" + post() + "}" : "null"));
    }
    // NOT_READY is an explicitly recorded poll outcome, never a completed sample.
    if (result != ZE_RESULT_SUCCESS && !(std::string(site) == "query-ready" && result == ZE_RESULT_NOT_READY))
      stop("metric_api_non_success");
    return result;
  } catch (...) { std::_Exit(125); }
}
inline void begin(uint64_t query) {
  auto& s = state(); if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!query || s.pending.count(query)) { emit(",\"phase\":\"reject\",\"reason\":\"query_busy_at_begin\""); std::_Exit(125); }
  s.pending[query] = 0;
}
inline void submitted(uint64_t query, uint64_t instance) {
  auto& s = state(); if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!query || !instance || (s.pending.count(query) && s.pending[query] != 0)) {
    emit(",\"phase\":\"reject\",\"reason\":\"overlapping_query_submission\""); std::_Exit(125);
  }
  s.pending[query] = instance;
  emit(",\"phase\":\"submit\"" + field("query", query) + field("instance", instance));
}
inline void completed(uint64_t query, uint64_t instance) {
  auto& s = state(); if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.pending.count(query) || s.pending[query] != instance || !instance) {
    emit(",\"phase\":\"reject\",\"reason\":\"unmatched_query_completion\""); std::_Exit(125);
  }
  s.pending.erase(query);
  emit(",\"phase\":\"collected\"" + field("query", query) + field("instance", instance));
}
inline void safe_query(uint64_t query) {
  auto& s = state(); if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.pending.count(query)) { emit(",\"phase\":\"reject\",\"reason\":\"uncertain_query_owner\""); std::_Exit(125); }
}
inline void safe_finalize() {
  auto& s = state(); if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.pending.empty()) { emit(",\"phase\":\"reject\",\"reason\":\"pending_queries_at_finalize\""); std::_Exit(125); }
}
inline void close() {
  auto& s = state(); if (!s.enabled) return;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.closed || !s.pending.empty()) std::_Exit(125);
  emit(",\"phase\":\"close\"" + field("calls", s.calls));
  s.closed = true;
}
} // namespace OwnMetricAudit
#endif
