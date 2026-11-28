#pragma once

#include "strata/ngram/prefill_record_plan.hpp"
#include "strata/platform/direct_file.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>

namespace strata::ngram {

struct PrefillPleReadStats {
    uint64_t issued = 0;
    uint64_t requested_bytes = 0;
    uint64_t returned_bytes = 0;
};

#if defined(__linux__)

// Own the file's queue for this window. The caller keeps the file, plan and
// destination live on this thread until the call returns. A ready callback
// may publish a chunk because all writes to that chunk have completed.
template<class Ready, class Stop>
bool prefill_ple_execute_record_plan(
        platform::DirectFile& file, const PrefillPleRecordPlan& plan, std::span<uint8_t> packed,
        uint32_t row_bytes, uint32_t chunk_rows, uint32_t max_inflight, Ready ready, Stop stop,
        PrefillPleReadStats& stats, std::string& error, bool trace_first_submit = false) {
    const uint64_t chunk_bytes = uint64_t{row_bytes} * chunk_rows;
    if (!file.is_open() || chunk_bytes == 0 || max_inflight == 0 || max_inflight > prefill_ple_record_max_inflight ||
        plan.chunk_fragments.empty() || packed.empty() ||
        plan.chunk_fragments.size() != 1 + (packed.size() - 1) / chunk_bytes) {
        error = "prefill PLE: invalid record read window";
        return false;
    }

    using platform::DirectFile;
    const size_t slots = std::min<size_t>(max_inflight, plan.reads.size());
    std::unique_ptr<void, decltype(&DirectFile::free_aligned)> slab(
        slots == 0 ? nullptr : DirectFile::alloc_aligned(slots * prefill_ple_record_bytes),
        &DirectFile::free_aligned);
    if (slots != 0 && !slab) {
        error = "prefill PLE: cannot allocate record read slots";
        return false;
    }
    // Linux close joins the blocking pread workers, including queued reads.
    // Declare this owner after the slab so any accepted I/O drains first.
    struct PendingIo {
        DirectFile& file;
        size_t active = 0;
        ~PendingIo() {
            if (active != 0) file.close();
        }
    } pending{file};

    auto* buffer = static_cast<uint8_t*>(slab.get());
    auto slot_data = [&](size_t slot) { return buffer + slot * prefill_ple_record_bytes; };
    constexpr size_t no_read = std::numeric_limits<size_t>::max();
    std::vector<size_t> slot_reads(slots, no_read);
    std::vector<uint32_t> free_slots(slots);
    std::iota(free_slots.begin(), free_slots.end(), 0);
    auto remaining = plan.chunk_fragments;
    size_t next_read = 0, next_chunk = 0;
    auto publish_ready = [&] {
        while (next_chunk < remaining.size() && remaining[next_chunk] == 0) {
            if (stop() || !ready(next_chunk)) {
                error = "prefill PLE: record read window cancelled";
                return false;
            }
            ++next_chunk;
        }
        return true;
    };
    if (!publish_ready()) return false;
    while (next_read < plan.reads.size() || pending.active != 0) {
        if (stop()) {
            error = "prefill PLE: record read window cancelled";
            return false;
        }
        // Shared records also carry any requested bytes for the next chunk.
        // Leave next-chunk-only reads off the file queue until this chunk is ready.
        while (next_read < plan.reads.size() && !free_slots.empty() &&
               plan.reads[next_read].first_chunk == next_chunk) {
            const auto slot = free_slots.back();
            const auto& read = plan.reads[next_read];
            if (read.length == 0 || read.length > prefill_ple_record_bytes ||
                read.required_bytes == 0 || read.required_bytes > read.length ||
                read.offset >= file.size() || read.required_bytes > file.size() - read.offset ||
                read.copy_count == 0 || read.copy_begin > plan.copies.size() ||
                read.copy_count > plan.copies.size() - read.copy_begin) {
                error = "prefill PLE: invalid record read span";
                return false;
            }
            if (!file.submit(read.offset, slot_data(slot), read.length, slot, error)) return false;
            free_slots.pop_back();
            slot_reads[slot] = next_read++;
            ++pending.active;
            ++stats.issued;
            stats.requested_bytes += read.length;
            if (trace_first_submit && stats.issued == 1) {
                std::fputs("strata trace: prefill PLE first direct read submitted\n", stderr);
                std::fflush(stderr);
            }
        }

        if (pending.active == 0) {
            error = "prefill PLE: missing record reads for the next chunk";
            return false;
        }
        platform::Completion completions[64];
        const int count = file.wait(completions, 64, 100);
        for (int i = 0; i < count; ++i) {
            const auto& completion = completions[i];
            if (completion.tag >= slot_reads.size() || slot_reads[completion.tag] == no_read) {
                error = "prefill PLE: unexpected record completion";
                return false;
            }
            const size_t slot = static_cast<size_t>(completion.tag);
            const auto& read = plan.reads[slot_reads[slot]];
            slot_reads[slot] = no_read;
            --pending.active;
            if (!completion.ok || completion.bytes < read.required_bytes || completion.bytes > read.length) {
                error = "prefill PLE: record read ended before requested row data";
                return false;
            }
            stats.returned_bytes += completion.bytes;
            for (const auto& copy : std::span(plan.copies).subspan(read.copy_begin, read.copy_count)) {
                const uint64_t end = uint64_t{copy.destination} + copy.bytes;
                const auto chunk = copy.destination / chunk_bytes;
                if (copy.bytes == 0 || uint64_t{copy.source} + copy.bytes > completion.bytes ||
                    end > packed.size() || chunk != (end - 1) / chunk_bytes || chunk >= remaining.size() ||
                    chunk < next_chunk || remaining[chunk] == 0) {
                    error = "prefill PLE: invalid record scatter";
                    return false;
                }
                std::memcpy(packed.data() + copy.destination, slot_data(slot) + copy.source, copy.bytes);
                --remaining[chunk];
            }
            free_slots.push_back(static_cast<uint32_t>(slot));
            if (!publish_ready()) return false;
        }
    }
    if (next_chunk == remaining.size()) return true;
    error = "prefill PLE: incomplete record read window";
    return false;
}

#endif

}  // namespace strata::ngram
