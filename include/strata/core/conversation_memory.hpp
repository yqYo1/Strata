// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace strata::core {

// The RAM a runtime allocation (parking a conversation, saving or restoring a session file) may still take.
//
// available_host_bytes() is the smallest of
//   * MemAvailable (Windows: the available physical memory);
//   * the room under every memory limit of this process's cgroup and of each ancestor visible under
//     /sys/fs/cgroup (cgroup v2: memory.max and memory.high, each minus usage; cgroup v1:
//     memory.limit_in_bytes minus memory.usage_in_bytes);
//   * the operator's own cap, when one is set (--memory-limit-mib / STRATA_MEMORY_LIMIT_MIB): that many MiB minus
//     what the container uses now: the top visible cgroup's memory.current, or MemTotal - MemAvailable when the
//     cgroup files cannot be read.  For a container whose real limit sits on a parent cgroup it cannot see.
//
// Usage is memory.current as reported by the kernel. Only a fully resolved non-root
// process cgroup with no child directories may credit its clean inactive file cache
// against its own visible max/high (v1 limit), less dirty/writeback and bounded by
// current. Ancestor, namespace-root and explicit operator-cap usage stays charged
// in full: descendant protection and hidden hierarchy cannot be inferred here.
// Missing/malformed stats, symlinks or an uncertain leaf grant no credit. A leaf's
// own memory.min/low is not deducted during targeted reclaim at that leaf.
// Sampling is not a reservation; cgroup membership/children may change afterwards.
//
// Unknown telemetry is deliberately distinct from a measured zero.  A limit file that exists but cannot be parsed
// with an explicit cap makes the sample unknown (fail closed); without a cap, unusable cgroup telemetry warns once
// and falls back to MemAvailable, as upstream. An absent file (no such controller, not mounted) is ignored.
// Not a reservation: other writers can take the room after the sample.

enum class MemorySource { meminfo, cgroup, flag };

const char* memory_source_name(MemorySource source);

/// One sample.  `available` is the smallest term; `source`, `limit` and `current` describe the term that was the
/// smallest (meminfo: MemTotal and MemTotal - MemAvailable; cgroup: that group's limit and memory.current; flag: the
/// operator's cap and the usage it was measured against).  `mem_available` is MemAvailable by itself (0 if unknown).
struct HostMemoryReading {
    uint64_t available = 0;
    MemorySource source = MemorySource::meminfo;
    uint64_t limit = 0;
    uint64_t current = 0;        ///< memory.current (meminfo: MemTotal - MemAvailable), before the credit
    uint64_t credit = 0;         ///< the clean inactive file cache counted as reclaimable (0 for meminfo)
    uint64_t mem_available = 0;
};

/// The files the sampler reads; the defaults are the real ones (tests point them at a fixture directory).
struct MemoryProbePaths {
    std::string meminfo = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string cgroup_root = "/sys/fs/cgroup";
};

/// The operator's cap in bytes for the container/cgroup this engine runs in (0 = none).  Set once at start.
void set_memory_limit_bytes(uint64_t bytes);
uint64_t memory_limit_bytes();

/// Linux: reads `paths`.  `explicit_limit` 0 = no operator cap.  nullopt when the RAM cannot be determined.
std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit);

/// The real files (Windows: GlobalMemoryStatusEx) and the cap set by set_memory_limit_bytes().
std::optional<HostMemoryReading> sample_host_memory();

std::optional<uint64_t> available_host_bytes();

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

} // namespace strata::core
