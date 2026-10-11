// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Source-only bounded integration probe v2. Link production Xe kernels and Runtime.
#include "strata/core/runtime.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/prefill/publication.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace publication = strata::prefill::publication;

namespace {
constexpr int slots = 8192, blocks = slots + 64, cap = 512, queries = 4, resolves = 6;
constexpr size_t codes_per_page = size_t{2} * 4 * 64, scales_per_page = size_t{2} * 4;
constexpr size_t buffer_limit = size_t{64} * 1024 * 1024;

// Catch at the call boundary, before any caller-owned source/destination unwinds.
// Includes submission failures: a partial enqueue cannot be declared quiescent.
template<class Action> void checked(const char* operation, Action action) noexcept {
    try {
        action();
    } catch (const std::exception& error) {
        publication::cleanup_require(false, operation, [&] { return error.what(); });
        std::_Exit(74);
    } catch (...) {
        publication::cleanup_require(false, operation, [] { return "nonstandard exception"; });
        std::_Exit(74);
    }
}

struct Counts {
    uint64_t resets = 0, resolves = 0, uploads = 0, upload_bytes = 0;
    uint64_t readbacks = 0, readback_bytes = 0, drains = 0, released = 0;
    uint64_t map_entries = 0, selected_cells = 0, pool_compares = 0, oracle_bytes = 0;
    uint64_t lookups = 0, missed_pages = 0, production_copy_runs = 0, production_copy_bytes = 0;
};

void print_counts(const char* kind, const char* name, const Counts& c) {
    std::printf("%s format=%s resets=%llu resolves=%llu uploads=%llu upload_bytes=%llu "
                "readbacks=%llu readback_bytes=%llu drains=%llu released=%llu "
                "map_entries=%llu selected_cells=%llu pool_compares=%llu oracle_bytes=%llu "
                "lookups=%llu missed_pages=%llu production_copy_runs=%llu production_copy_bytes=%llu\n",
                kind, name, (unsigned long long) c.resets, (unsigned long long) c.resolves,
                (unsigned long long) c.uploads, (unsigned long long) c.upload_bytes,
                (unsigned long long) c.readbacks, (unsigned long long) c.readback_bytes,
                (unsigned long long) c.drains, (unsigned long long) c.released,
                (unsigned long long) c.map_entries, (unsigned long long) c.selected_cells,
                (unsigned long long) c.pool_compares, (unsigned long long) c.oracle_bytes,
                (unsigned long long) c.lookups, (unsigned long long) c.missed_pages,
                (unsigned long long) c.production_copy_runs, (unsigned long long) c.production_copy_bytes);
    std::fflush(stdout);
}

struct Run {
    unsigned char* host = nullptr;
    unsigned char* device = nullptr;
    size_t page_bytes = 0;
    std::vector<unsigned char> readback;
};

bool run_format(strata::core::Runtime& rt, int fmt, Counts& count) {
    const char* name = fmt == k::kKvF16 ? "fp16" : "int8";
    auto& queue = rt.compute();
    // All owning containers outlive the exception handler and its checked drain.
    std::vector<void*> allocations;
    std::array<Run, 4> runs{};
    std::vector<int32_t> pt, sb, refs, stamps, selection, steps;
    std::vector<bool> seen;
    std::array<int32_t, k::kKvCtlInts> ctl{};
    size_t usm_bytes = 0;
    const int nruns = fmt == k::kKvF16 ? 2 : 4;
    auto drain = [&] {
        checked("bounded terminal/operation drain", [&] { rt.finish(); });
        ++count.drains;
    };
    auto release = [&] {
        // Called only after a successful checked whole-Runtime terminal drain.
        for (void* p : allocations) {
            checked("bounded USM release after terminal drain", [&] { sycl::free(p, rt.context()); });
            ++count.released;
        }
        allocations.clear();
    };
    std::printf("FORMAT_BEGIN format=%s slots=%d blocks=%d nq=2,3,4,2,3,4 resolves=%d\n",
                name, slots, blocks, resolves);
    std::fflush(stdout);
    try {
        allocations.reserve(32); // No allocation-vector growth after a USM allocation succeeds.
        pt.assign(blocks, -1); sb.resize(slots); refs.assign(slots, 1); stamps.assign(slots, 0);
        selection.resize(static_cast<size_t>(queries) * cap); steps.resize(static_cast<size_t>(queries) * k::kStepCount); seen.resize(slots);
        auto alloc = [&](size_t bytes, bool host) {
            if (bytes > buffer_limit - usm_bytes) throw std::runtime_error("USM byte budget exceeded");
            void* p = host ? sycl::malloc_host(bytes, queue) : sycl::malloc_device(bytes, queue);
            if (!p) throw std::bad_alloc();
            allocations.push_back(p);
            usm_bytes += bytes;
            if (sycl::get_pointer_type(p, rt.context()) !=
                (host ? sycl::usm::alloc::host : sycl::usm::alloc::device))
                throw std::runtime_error("wrong USM context/type");
            if (strata::core::Runtime::owner(p) != &rt)
                throw std::runtime_error("wrong Runtime allocation owner");
            return p;
        };
        auto ints = [&](size_t n) { return static_cast<int32_t*>(alloc(n * sizeof(int32_t), false)); };
        k::KvStreamMap m{};
        m.n_blocks = blocks; m.n_slots = slots;
        m.page_table = ints(blocks); m.slot_block = ints(slots); m.slot_stamp = ints(slots);
        m.slot_ref = ints(slots); m.miss_block = ints(slots); m.miss_slot = ints(slots);
        m.ctl = ints(k::kKvCtlInts);
        auto* ids = ints(static_cast<size_t>(queries) * cap);
        auto* device_steps = ints(static_cast<size_t>(queries) * k::kStepCount);
        for (int a = 0; a < nruns; ++a) {
            auto& run = runs[a];
            run.page_bytes = fmt == k::kKvF16 ? codes_per_page * 2 :
                             a < 2 ? codes_per_page : scales_per_page * 2;
            run.host = static_cast<unsigned char*>(alloc(blocks * run.page_bytes, true));
            run.device = static_cast<unsigned char*>(alloc(slots * run.page_bytes, false));
            run.readback.resize(slots * run.page_bytes);
            // No command uses these pools yet: CPU exclusively owns host USM initialization.
            if (fmt == k::kKvInt8 && a < 2) {
                auto* values = reinterpret_cast<int8_t*>(run.host);
                for (size_t i = 0; i < blocks * run.page_bytes; ++i)
                    values[i] = static_cast<int8_t>(static_cast<int>((i * (17 + a * 12) +
                                      i / run.page_bytes + static_cast<size_t>(a) * 43) % 255) - 127);
            } else {
                auto* values = reinterpret_cast<uint16_t*>(run.host);
                for (size_t i = 0; i < blocks * run.page_bytes / 2; ++i)
                    values[i] = static_cast<uint16_t>(0x3000 +
                                (i * (17 + a * 12) + i / (run.page_bytes / 2) + static_cast<size_t>(a) * 113) % 0x1000);
                // Positive normal finite half values for FP16 data and INT8 K/V scales.
            }
        }
        size_t live_bytes = usm_bytes;
        for (int a = 0; a < nruns; ++a) live_bytes += runs[a].readback.capacity();
        live_bytes += (pt.capacity() + sb.capacity() + refs.capacity() + stamps.capacity() +
                       selection.capacity() + steps.capacity()) * sizeof(int32_t);
        live_bytes += seen.capacity() + allocations.capacity() * sizeof(void*) + sizeof(ctl);
        if (live_bytes >= buffer_limit) throw std::runtime_error("probe live-buffer budget exceeded");
        std::printf("FORMAT_ALLOCATED format=%s usm_bytes=%zu live_buffer_bytes_upper=%zu limit=%zu\n",
                    name, usm_bytes, live_bytes, buffer_limit);
        std::fflush(stdout);
        k::KvHostPools host{};
        k::QsaAttnPools pool{};
        if (fmt == k::kKvF16) {
            host.k_pool = reinterpret_cast<uint16_t*>(runs[0].host);
            host.v_pool = reinterpret_cast<uint16_t*>(runs[1].host);
            pool.k_pool = reinterpret_cast<uint16_t*>(runs[0].device);
            pool.v_pool = reinterpret_cast<uint16_t*>(runs[1].device);
        } else {
            host.k_q = reinterpret_cast<int8_t*>(runs[0].host);
            host.v_q = reinterpret_cast<int8_t*>(runs[1].host);
            host.k_scale = reinterpret_cast<uint16_t*>(runs[2].host);
            host.v_scale = reinterpret_cast<uint16_t*>(runs[3].host);
            pool.k_q = reinterpret_cast<int8_t*>(runs[0].device);
            pool.v_q = reinterpret_cast<int8_t*>(runs[1].device);
            pool.k_scale = reinterpret_cast<uint16_t*>(runs[2].device);
            pool.v_scale = reinterpret_cast<uint16_t*>(runs[3].device);
        }
        pool.page_table = m.page_table;
        k::QsaShapes s{};
        s.n_head = 2; s.n_head_kv = 2; s.head_dim = 64; s.page_size = 4;
        auto upload = [&](void* dst, const void* src, size_t bytes) {
            checked("bounded upload submit/wait", [&] {
                rt.wait(queue.memcpy(dst, src, bytes), "bounded upload");
            });
            ++count.uploads; count.upload_bytes += bytes;
        };
        auto read = [&](void* dst, const void* src, size_t bytes) {
            checked("bounded readback submit/wait", [&] {
                rt.wait(queue.memcpy(dst, src, bytes), "bounded readback");
            });
            ++count.readbacks; count.readback_bytes += bytes;
        };
        // Explicit nonnull production stream avoids hidden finish() inside these APIs.
        checked("bounded production reset submission", [&] { k::kv_stream_reset(m, &queue); });
        drain();
        ++count.resets;
        for (int i = 0; i < slots; ++i) { pt[i] = i; sb[i] = i; }
        upload(m.page_table, pt.data(), pt.size() * sizeof(int32_t));
        upload(m.slot_block, sb.data(), sb.size() * sizeof(int32_t));
        upload(m.slot_ref, refs.data(), refs.size() * sizeof(int32_t));
        upload(m.slot_stamp, stamps.data(), stamps.size() * sizeof(int32_t));
        for (int a = 0; a < nruns; ++a)
            upload(runs[a].device, runs[a].host, slots * runs[a].page_bytes);
        constexpr std::array<int, resolves> expected_misses{36, 8, 8, 0, 4, 8};
        for (int pass = 0; pass < resolves; ++pass) {
            const int nq = 2 + pass % 3;
            std::fill(selection.begin(), selection.end(), 0);
            std::fill(steps.begin(), steps.end(), 0);
            for (int q = 0; q < nq; ++q) {
                const int first = slots - 96 + pass * 4 + q * 4;
                for (int i = 0; i < cap; ++i) selection[q * cap + i] = first * 4 + i;
                steps[q * k::kStepCount + k::kStepPos] = blocks * 4 - 1;
                steps[q * k::kStepCount + k::kStepNKv] = blocks * 4;
                steps[q * k::kStepCount + k::kStepNBid] = blocks;
                steps[q * k::kStepCount + k::kStepWidth] = cap;
            }
            std::printf("RESOLVE_BEGIN format=%s pass=%d nq=%d expected_misses=%d\n",
                        name, pass + 1, nq, expected_misses[pass]);
            std::fflush(stdout);
            upload(ids, selection.data(), selection.size() * sizeof(int32_t));
            upload(device_steps, steps.data(), steps.size() * sizeof(int32_t));
            checked("bounded production resolve submission", [&] {
                k::kv_stream_resolve(m, pool, host, fmt, ids, device_steps, nq, cap, s, &queue);
            });
            drain(); // Whole Runtime completes before any CPU access to any host-USM pool.
            ++count.resolves;
            read(pt.data(), m.page_table, pt.size() * sizeof(int32_t));
            read(sb.data(), m.slot_block, sb.size() * sizeof(int32_t));
            read(ctl.data(), m.ctl, sizeof(ctl));
            for (int a = 0; a < nruns; ++a)
                read(runs[a].readback.data(), runs[a].device, runs[a].readback.size());
            std::fill(seen.begin(), seen.end(), false);
            for (int b = 0; b < blocks; ++b) {
                const int sl = pt[b];
                if (sl < -1 || sl >= slots) throw std::runtime_error("invalid page map");
                if (sl >= 0) {
                    if (seen[sl] || sb[sl] != b) throw std::runtime_error("nonunique/inconsistent page map");
                    seen[sl] = true;
                }
                ++count.map_entries;
            }
            for (int sl = 0; sl < slots; ++sl) {
                const int b = sb[sl];
                if (b < 0 || b >= blocks || pt[b] != sl)
                    throw std::runtime_error("invalid/full-slot inverse");
                ++count.map_entries;
                for (int a = 0; a < nruns; ++a) {
                    const auto& run = runs[a];
                    if (std::memcmp(run.readback.data() + sl * run.page_bytes,
                                    run.host + b * run.page_bytes, run.page_bytes) != 0)
                        throw std::runtime_error("resident pool differs from host byte oracle");
                    ++count.pool_compares; count.oracle_bytes += run.page_bytes;
                }
            }
            for (int q = 0; q < nq; ++q)
                for (int i = 0; i < cap; ++i) {
                    if (pt[selection[q * cap + i] / 4] < 0)
                        throw std::runtime_error("queried page not resident");
                    ++count.selected_cells;
                }
            count.lookups += static_cast<uint64_t>(nq) * (cap / 4);
            count.missed_pages += expected_misses[pass];
            uint64_t misses = 0, lookups = 0, calls = 0;
            std::memcpy(&misses, ctl.data() + 4, sizeof(misses));
            std::memcpy(&lookups, ctl.data() + 6, sizeof(lookups));
            std::memcpy(&calls, ctl.data() + 8, sizeof(calls));
            if (ctl[0] != pass + 1 || ctl[2] != expected_misses[pass] || ctl[3] != 0 ||
                misses != count.missed_pages || lookups != count.lookups || calls != count.resolves)
                throw std::runtime_error("epoch/miss/lookup/call/overflow counters differ");
            count.production_copy_runs += static_cast<uint64_t>(expected_misses[pass]) * nruns;
            count.production_copy_bytes += static_cast<uint64_t>(expected_misses[pass]) * k::kv_block_bytes(s, fmt);
            std::printf("RESOLVE_DONE format=%s pass=%d nq=%d misses=%d "
                        "map_entries=%d pool_compares=%d oracle_bytes=%zu selected_cells=%d\n",
                        name, pass + 1, nq, expected_misses[pass], blocks + slots, slots * nruns,
                        slots * static_cast<size_t>(k::kv_block_bytes(s, fmt)), nq * cap);
            std::fflush(stdout);
        }
        const uint64_t pool_bytes = fmt == k::kKvF16 ? 16777216 : 8650752;
        if (count.resets != 1 || count.resolves != 6 || count.uploads != 16 + static_cast<uint64_t>(nruns) ||
            count.upload_bytes != pool_bytes + 180864 ||
            count.readbacks != uint64_t{6} * (3 + nruns) ||
            count.readback_bytes != 6 * (pool_bytes + 65856) || count.drains != 7 ||
            count.map_entries != 98688 || count.selected_cells != 9216 ||
            count.pool_compares != uint64_t{6} * slots * nruns ||
            count.oracle_bytes != 6 * pool_bytes || count.lookups != 2304 || count.missed_pages != 64 ||
            count.production_copy_runs != uint64_t{64} * nruns ||
            count.production_copy_bytes != 64 * k::kv_block_bytes(s, fmt))
            throw std::runtime_error("finite receipt counts differ");
        drain(); // Terminal drain before every USM free and before the next format starts.
        release();
        if (count.drains != 8 || count.released != 9 + uint64_t{2} * nruns)
            throw std::runtime_error("terminal receipt counts differ");
        print_counts("FORMAT_PASS", name, count);
        return true;
    } catch (const std::exception& error) {
        // All buffers are still alive. One checked drain, no retry if it throws.
        drain();
        release();
        std::fprintf(stderr, "FORMAT_FAIL format=%s: %s\n", name, error.what());
        std::fflush(stderr);
        return false;
    } catch (...) {
        drain();
        release();
        std::fprintf(stderr, "FORMAT_FAIL format=%s: nonstandard exception\n", name);
        std::fflush(stderr);
        return false;
    }
}
} // namespace

int main(int argc, char**) {
    if (argc != 1) {
        std::fputs("FAIL: no CLI arguments accepted; Runtime not initialized\n", stderr);
        std::fflush(stderr);
        return 2;
    }
    strata::core::Runtime* runtime = nullptr;
    // Initialization may invoke production waits. Do not unwind the application
    // on an uncertain initialization exception, even before probe buffers exist.
    checked("bounded Runtime initialization", [&] { runtime = &strata::core::Runtime::get(); });
    try {
        auto& rt = *runtime;
        auto& queue = rt.compute();
        if (!rt.device().is_gpu() || !queue.has_property<sycl::property::queue::in_order>() ||
            queue.get_context() != rt.context() || queue.get_device() != rt.device() ||
            strata::core::Runtime::owner_of_stream(&queue) != &rt)
            throw std::runtime_error("GPU in-order owned compute queue required");
        if (!rt.device().has(sycl::aspect::usm_host_allocations) ||
            !rt.device().has(sycl::aspect::usm_device_allocations))
            throw std::runtime_error("host/device USM required");
        Counts fp16{}, int8{};
        if (!run_format(rt, k::kKvF16, fp16)) std::_Exit(1);
        if (!run_format(rt, k::kKvInt8, int8)) std::_Exit(1);
        print_counts("FINAL_COUNTS", "fp16", fp16);
        print_counts("FINAL_COUNTS", "int8", int8);
        std::puts("PASS formats=2 resets=2 resolves=12 uploads=38 readbacks=72 "
                  "direct_copies=110 direct_copy_bytes=179147776 drains=16 released=30 "
                  "map_entries=197376 selected_cells=18432 pool_compares=294912 "
                  "oracle_bytes=152567808 lookups=4608 missed_pages=128 "
                  "production_copy_runs=384 production_copy_bytes=198656");
        std::fflush(stdout);
        // Both format frames have already drained, freed every USM allocation,
        // and destroyed their ordinary host buffers. Skip Runtime's implicit
        // destructor waits so no unchecked wait occurs after the finite receipt.
        std::_Exit(0);
    } catch (const std::exception& error) {
        checked("bounded pre-format exception drain", [&] { runtime->finish(); });
        std::fprintf(stderr, "FAIL before format: %s\n", error.what());
        std::fflush(stderr);
        std::_Exit(1);
    } catch (...) {
        checked("bounded pre-format nonstandard exception drain", [&] { runtime->finish(); });
        std::fputs("FAIL before format: nonstandard exception\n", stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }
}
