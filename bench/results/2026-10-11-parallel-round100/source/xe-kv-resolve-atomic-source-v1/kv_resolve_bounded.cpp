// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Source-only integration probe: link the production Xe kernels and Runtime.
// Six real resolves, FP16 pool bytes only; no attention or RESTORE claim.
#include "strata/core/runtime.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;

int main() {
    try {
        auto& rt = strata::core::Runtime::get();
        auto& queue = rt.compute();
        if (!rt.device().is_gpu()) throw std::runtime_error("GPU required");
        if (!rt.device().has(sycl::aspect::usm_host_allocations) ||
            !rt.device().has(sycl::aspect::usm_device_allocations))
            throw std::runtime_error("host and device USM required");
        constexpr int slots = 8192, blocks = slots + 64, cap = 512, queries = 4;
        k::QsaShapes s{};
        s.n_head = 2; s.n_head_kv = 2; s.head_dim = 64; s.page_size = 4;
        constexpr size_t words = 2 * 4 * 64;
        std::vector<void*> allocations;
        auto alloc = [&](size_t bytes, bool host) {
            void* p = host ? sycl::malloc_host(bytes, queue) : sycl::malloc_device(bytes, queue);
            if (!p) throw std::bad_alloc();
            allocations.push_back(p);
            const auto type = sycl::get_pointer_type(p, rt.context());
            if (type != (host ? sycl::usm::alloc::host : sycl::usm::alloc::device))
                throw std::runtime_error("wrong allocation context/type");
            if (strata::core::Runtime::owner(p) != &rt)
                throw std::runtime_error("wrong Runtime owner");
            return p;
        };
        auto ints = [&](size_t count) { return static_cast<int32_t*>(alloc(count * sizeof(int32_t), false)); };
        k::KvStreamMap m{};
        m.n_blocks = blocks; m.n_slots = slots;
        m.page_table = ints(blocks); m.slot_block = ints(slots); m.slot_stamp = ints(slots);
        m.slot_ref = ints(slots); m.miss_block = ints(slots); m.miss_slot = ints(slots);
        m.ctl = ints(k::kKvCtlInts);
        auto* ids = ints(queries * cap);
        auto* steps = ints(queries * k::kStepCount);
        k::KvHostPools host{};
        host.k_pool = static_cast<uint16_t*>(alloc(blocks * words * 2, true));
        host.v_pool = static_cast<uint16_t*>(alloc(blocks * words * 2, true));
        k::QsaAttnPools pool{};
        auto* kp = static_cast<uint16_t*>(alloc(slots * words * 2, false));
        auto* vp = static_cast<uint16_t*>(alloc(slots * words * 2, false));
        pool.k_pool = kp; pool.v_pool = vp; pool.page_table = m.page_table;
        // No kernel has been submitted: CPU owns the host USM initialization.
        // Normal finite half encodings, with distinct deterministic K/V patterns.
        for (size_t i = 0; i < blocks * words; ++i) {
            host.k_pool[i] = static_cast<uint16_t>(0x3000 + (i * 17 + i / words) % 0x800);
            host.v_pool[i] = static_cast<uint16_t>(0x3800 + (i * 29 + i / words) % 0x800);
        }
        auto upload = [&](void* dst, const void* src, size_t bytes) {
            rt.wait(queue.memcpy(dst, src, bytes), "bounded probe upload");
        };
        auto read = [&](void* dst, const void* src, size_t bytes) {
            rt.wait(queue.memcpy(dst, src, bytes), "bounded probe read");
        };
        k::kv_stream_reset(m, nullptr);
        std::vector<int32_t> pt(blocks, -1), sb(slots), refs(slots, 1), stamps(slots, 0);
        for (int i = 0; i < slots; ++i) { pt[i] = i; sb[i] = i; }
        upload(m.page_table, pt.data(), pt.size() * 4);
        upload(m.slot_block, sb.data(), sb.size() * 4);
        upload(m.slot_ref, refs.data(), refs.size() * 4);
        upload(m.slot_stamp, stamps.data(), stamps.size() * 4);
        upload(kp, host.k_pool, slots * words * 2);
        upload(vp, host.v_pool, slots * words * 2);
        std::vector<uint16_t> kr(slots * words), vr(slots * words);
        for (int pass = 0; pass < 6; ++pass) {
            const int nq = 2 + pass % 3;
            std::vector<int32_t> selection(queries * cap, 0), step(queries * k::kStepCount, 0);
            for (int q = 0; q < nq; ++q) {
                // Overlapping pages occur at different lane positions in different queries.
                const int first = slots - 96 + pass * 4 + q * 4;
                for (int i = 0; i < cap; ++i) selection[q * cap + i] = first * 4 + i;
                step[q * k::kStepCount + k::kStepPos] = blocks * 4 - 1;
                step[q * k::kStepCount + k::kStepNKv] = blocks * 4;
                step[q * k::kStepCount + k::kStepNBid] = blocks;
                step[q * k::kStepCount + k::kStepWidth] = cap;
            }
            upload(ids, selection.data(), selection.size() * 4);
            upload(steps, step.data(), step.size() * 4);
            k::kv_stream_resolve(m, pool, host, k::kKvF16, ids, steps, nq, cap, s, nullptr);
            rt.finish(queue); // Host USM cannot be accessed while the production kernels use it.
            read(pt.data(), m.page_table, pt.size() * 4);
            read(sb.data(), m.slot_block, sb.size() * 4);
            read(kr.data(), kp, kr.size() * 2);
            read(vr.data(), vp, vr.size() * 2);
            std::vector<bool> seen(slots, false);
            for (int b = 0; b < blocks; ++b) {
                const int sl = pt[b];
                if (sl < -1 || sl >= slots) throw std::runtime_error("invalid page map");
                if (sl < 0) continue;
                if (seen[sl] || sb[sl] != b) throw std::runtime_error("nonunique/inconsistent page map");
                seen[sl] = true;
            }
            for (int sl = 0; sl < slots; ++sl) {
                const int b = sb[sl];
                if (b < -1 || b >= blocks || (b >= 0 && pt[b] != sl))
                    throw std::runtime_error("invalid slot inverse");
                if (b >= 0 && (std::memcmp(kr.data() + sl * words, host.k_pool + b * words, words * 2) ||
                               std::memcmp(vr.data() + sl * words, host.v_pool + b * words, words * 2)))
                    throw std::runtime_error("pool differs from host oracle");
            }
            for (int q = 0; q < nq; ++q)
                for (int i = 0; i < cap; ++i)
                    if (pt[selection[q * cap + i] / 4] < 0) throw std::runtime_error("selection not resident");
            const auto counters = k::kv_stream_counters(m);
            if (counters.overflow || counters.calls != static_cast<uint64_t>(pass + 1))
                throw std::runtime_error("resolve counter/overflow failure");
        }
        rt.finish();
        for (void* p : allocations) sycl::free(p, rt.context());
        std::puts("PASS: six production resolves, nq 2..4, inverse/unique map and exact FP16 pools");
        return 0;
    } catch (const std::exception& error) {
        // On exceptional completion do not free USM that may still be in device use.
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
