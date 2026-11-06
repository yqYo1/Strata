"""Compose an opt-in serving audit onto the exact selected native generator.

Main-authored; repository LICENSE. This pure source transformation neither
writes the active tree nor builds/runs anything. Its next consumer is the
post-cohort native diagnostic build, after the current collector's ELF consumers
close. Host assertion success alone does not qualify these native attachments.
"""
import hashlib
import textwrap

GENERATOR = 'src/program/generate.cpp'
GENERATOR_SHA = 'b4423da4a95f09544599ea20fc82fd830717397fec3cc6f8e28ebc7e83a3e17d'
HEADER = 'include/strata/core/adaptive_residency_audit.hpp'
HEADER_SOURCE = '0a47fe046f2828c111009edecbf95f5595f0000b'
HEADER_SHA = '299218f250c51d4f9cfe33d0a5aee07bfaf04bc02473db971079d7e38767bb9e'


def cpp(value, spaces=8):
    return textwrap.indent(textwrap.dedent(value).strip() + '\n', ' ' * spaces)


def one(value, before, after):
    if value.count(before) != 1:
        raise RuntimeError('adaptive attachment requires one exact native anchor: ' + before[:100])
    return value.replace(before, after, 1)


def generator(original):
    if hashlib.sha256(original).hexdigest() != GENERATOR_SHA:
        raise RuntimeError('recover the actual selected generator;do not refresh its source pin')
    value = original.decode('utf-8')
    anchor = '#ifdef STRATA_ENABLE_SYCL\n#include "strata/sycl/decode_service_profile.hpp"'
    value = one(value, anchor, anchor.replace('#include', '#include "strata/core/adaptive_residency_audit.hpp"\n#include', 1))

    anchor = '    if (o.serve) {\n        if (o.spec < 2 || o.mtp.empty() || o.prefill_chunk <= 0 ||'
    guard = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            const char* adapt_audit_value = std::getenv("STRATA_SYCL_ADAPT_AUDIT");
            const bool adapt_audit_on = adapt_audit_value && std::strcmp(adapt_audit_value, "1") == 0;
            if (adapt_audit_on && !o.serve) {
                std::fprintf(stderr, "strata: adaptive audit requires the serving entrypoint\n");
                return 2;
            }
        #endif
    ''', 0)
    value = one(value, anchor, guard + anchor)

    anchor = '        const bool trace_adapt = std::getenv("STRATA_TRACE_ADAPT") != nullptr;\n'
    setup = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            using AdaptiveAudit = strata::core::AdaptiveResidencyAudit;
            std::unique_ptr<AdaptiveAudit> adapt_audit;
            std::vector<int32_t> adapt_audit_device;
            std::vector<AdaptiveAudit::Swap> adapt_audit_changes;
            uint64_t adapt_audit_refill_generation = 0;
            if (adapt_audit_on) {
                // This diagnostic covers the immutable full-RAM arena only.
                // FileExpertSource RAM-complement exchange and split devices
                // require their own saved-victim/ownership contract.
                if (!native_pack || srcp != &arena_src || multi_gpu || peer.valid() ||
                    !xcache.valid() || d_res == nullptr || host_res.empty() ||
                    o.adapt_every <= 0 || o.adapt_swaps <= 0 ||
                    int64_t(o.adapt_swaps) > xcache.slots() ||
                    g.n_expert <= 0 || g.n_layers <= 0 ||
                    size_t(g.n_layers) > AdaptiveAudit::max_entries / size_t(g.n_expert)) {
                    std::fprintf(stderr, "strata serve: adaptive audit needs positive swaps,single-GPU native full-RAM arena and profiled cache\n");
                    return 2;
                }
                std::vector<size_t> bytes(size_t(g.n_layers));
                for (size_t l = 0; l < bytes.size(); ++l)
                    bytes[l] = size_t(strata::kernels::cpu::expert_layout().blob_bytes(int64_t(l)));
                adapt_audit = std::make_unique<AdaptiveAudit>(
                    std::move(bytes), size_t(g.n_expert), size_t(xcache.slots()), size_t(o.adapt_swaps));
                adapt_audit_device.resize(host_res.size());
                adapt_audit_changes.reserve(size_t(o.adapt_swaps));
            }
            auto adapt_audit_lookup = [&](size_t row) {
                return xcache.slot_of(int64_t(row / size_t(g.n_expert)),
                                      int64_t(row % size_t(g.n_expert)));
            };
            auto adapt_audit_read = [&](std::string& e) -> bool {
                if (cudaMemcpy(adapt_audit_device.data(), d_res,
                               adapt_audit_device.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    e = "adaptive audit: device residency readback failed";
                    return false;
                }
                return true;
            };
            auto adapt_audit_payload = [&](const AdaptiveAudit::Swap& s, bool incoming,
                                           size_t bytes, std::string& e) -> bool {
                const size_t row = incoming ? s.incoming : s.outgoing;
                return xcache.verify_slot(s.slot,
                    srcp->blob(int64_t(row / size_t(g.n_expert)), int64_t(row % size_t(g.n_expert))),
                    e, int64_t(bytes));
            };
        #endif
    ''')
    value = one(value, anchor, setup + anchor)

    anchor = '            if (peer.valid()) peer.apply_pending(wait);\n            if (pending.empty()) return true;'
    replacement = cpp(r'''
        if (peer.valid()) peer.apply_pending(wait);
        if (pending.empty()) {
        #ifdef STRATA_ENABLE_SYCL
            if (adapt_audit && !adapt_audit->idle()) {
                e = "adaptive audit: empty pending group with unfinished identity cycle";
                return false;
            }
        #endif
            return true;
        }
    ''', 12).rstrip()
    value = one(value, anchor, replacement)

    anchor = '            const size_t completed = pending.size();\n            src.commit_exchanges();'
    complete = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            if (adapt_audit) {
                if (adapt_audit_refill_generation != adapt_audit->generation() ||
                    pending.size() != adapt_audit->swaps().size()) {
                    e = "adaptive audit: refill generation/pending extent differs";
                    return false;
                }
                for (size_t i = 0; i < pending.size(); ++i) {
                    const auto& s = adapt_audit->swaps()[i];
                    if (pending[i].first < 0 || size_t(pending[i].first) != s.incoming || pending[i].second != s.slot) {
                        e = "adaptive audit: pending identity differs from submitted swap";
                        return false;
                    }
                }
                // The real adapt_ev wait/query above already returned success.
                if (!adapt_audit->complete([&] { return status == cudaSuccess; },
                        [&](const auto& s, size_t bytes, std::string& why) {
                            return adapt_audit_payload(s, true, bytes, why);
                        }, e)) return false;
            }
        #endif
    ''', 12)
    value = one(value, anchor, complete + anchor)

    anchor = '            const bool published = adaptive_res_upload("serve adaptive refill", e);\n'
    publish = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            if (published && adapt_audit) {
                if (!adapt_audit_read(e) ||
                    !adapt_audit->published(host_res, adapt_audit_device, adapt_audit_lookup, e)) return false;
                std::fprintf(stderr, "strata serve: ADAPT_AUDIT phase=published generation=%llu swaps=%zu identities=3 payload_match=1\n",
                    (unsigned long long)adapt_audit->completed(), completed);
            }
        #endif
    ''', 12)
    value = one(value, anchor, anchor + publish)

    anchor = '            if (!resident_stage_swaps(src, xcache, host_res, g.n_expert, swaps, adapt_stream)) return false;\n'
    prepare = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            if (adapt_audit && !swaps.empty()) {
                adapt_audit_changes.clear();
                for (const auto& s : swaps) {
                    const size_t in = size_t(s.layer) * size_t(g.n_expert) + size_t(s.in);
                    const size_t out = size_t(s.layer) * size_t(g.n_expert) + size_t(s.out);
                    adapt_audit_changes.push_back({in, out, host_res.at(out)});
                }
                std::string why;
                if (!adapt_audit_read(why) ||
                    !adapt_audit->prepare(host_res, adapt_audit_device, adapt_audit_changes, adapt_audit_lookup,
                        [&](const auto& s, size_t bytes, std::string& e) {
                            return adapt_audit_payload(s, false, bytes, e);
                        }, why)) {
                    std::fprintf(stderr, "strata serve: %s\n", why.c_str());
                    return false;
                }
                for (const auto& s : adapt_audit->swaps())
                    std::fprintf(stderr, "strata serve: ADAPT_AUDIT phase=prepared generation=%llu incoming=%zu outgoing=%zu slot=%d\n",
                        (unsigned long long)adapt_audit->generation(), s.incoming, s.outgoing, int(s.slot));
            }
        #endif
    ''', 12)
    value = one(value, anchor, prepare + anchor)

    anchor = ('                std::fprintf(stderr, "strata serve: %s\\n", mask_error.c_str()); return false;\n'
              '            }\n#endif\n            bool main_live = false;')
    mask = cpp(r'''
        if (adapt_audit && !swaps.empty()) {
            if (!adapt_audit_read(mask_error) ||
                !adapt_audit->masked(host_res, adapt_audit_device, adapt_audit_lookup, mask_error)) {
                std::fprintf(stderr, "strata serve: %s\n", mask_error.c_str());
                return false;
            }
            std::fprintf(stderr, "strata serve: ADAPT_AUDIT phase=masked generation=%llu swaps=%zu identities=3\n",
                (unsigned long long)adapt_audit->generation(), adapt_audit->swaps().size());
        }
    ''', 12)
    value = one(value, anchor, anchor.replace('\n#endif\n', '\n' + mask + '#endif\n', 1))

    anchor = ('                std::fprintf(stderr, "strata serve: cannot record the adaptive refill event\\n");\n'
              '                return false;\n            }\n            (void) main_live;')
    recorded = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            // Record this generation only after the real API accepted the event.
            // The legacy adapt_records counter advances only in injection mode.
            if (adapt_audit && !swaps.empty())
                adapt_audit_refill_generation = adapt_audit->generation();
        #endif
    ''', 12)
    value = one(value, anchor, anchor.replace('            (void) main_live;', recorded + '            (void) main_live;', 1))
    return dispatch_and_loan(value).encode('utf-8')


def dispatch_and_loan(value):
    """Inspect the real window plan/jobs after completion;no core/kernel edits."""
    anchor = '#include "strata/core/adaptive_residency_audit.hpp"'
    value = one(value, anchor, anchor + '\n#include <functional>')
    anchor = '    std::FILE* expert_inputs = nullptr;\n};'
    fields = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            std::function<void(const strata::core::ExpertDispatch&, const int32_t*, int64_t, int64_t,
                               const float*, int64_t, int64_t, int64_t, int64_t)> adaptive_dispatch_audit;
        #endif
    ''', 4)
    value = one(value, anchor, anchor.replace('\n};', '\n' + fields + '};'))
    anchor = '    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);\n'
    before = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            int64_t audit_before[3] = {};
            if (t->adaptive_dispatch_audit) {
                audit_before[0] = t->d.cache_hits;
                audit_before[1] = t->d.multi_entries;
                audit_before[2] = t->d.multi_misses;
            }
        #endif
    ''', 4)
    after = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            if (t->adaptive_dispatch_audit && !t->d.failed)
                t->adaptive_dispatch_audit(t->d, ids, n_tok, k, out, layer,
                    t->d.cache_hits - audit_before[0], t->d.multi_entries - audit_before[1],
                    t->d.multi_misses - audit_before[2]);
        #endif
    ''', 4)
    value = one(value, anchor, before + anchor + after)
    anchor = '            std::vector<AdaptiveAudit::Swap> adapt_audit_changes;\n'
    value = one(value, anchor, anchor + cpp(r'''
        std::vector<int8_t> adapt_audit_role;
        uint64_t adapt_audit_dispatch_calls = 0, adapt_audit_incoming_hits = 0;
        uint64_t adapt_audit_outgoing_misses = 0, adapt_audit_post_swap_refills = 0;
    ''', 12))
    anchor = '                    o.adapt_every <= 0 || o.adapt_swaps <= 0 ||\n'
    value = one(value, anchor, anchor +
        '                    !loan_audit || !cache_scratch_lease_enabled || adapt_nowait() ||\n'
        '                    o.pcie_frac != 0 || o.spec < 2 ||\n')
    anchor = '                adapt_audit_device.resize(host_res.size());\n'
    value = one(value, anchor, anchor + '                adapt_audit_role.assign(host_res.size(), 0);\n')
    anchor = '            auto adapt_audit_payload = [&](const AdaptiveAudit::Swap& s, bool incoming,\n'
    callback = cpp(r'''
        if (adapt_audit) {
            // The real pool has completed and reset its job_of table. The next
            // adapt worker has not started;the previous one was joined and its
            // checked refill was published before this verifier window.
            drive.adaptive_dispatch_audit = [&](const strata::core::ExpertDispatch& D,
                const int32_t* ids, int64_t nt, int64_t k, const float* out, int64_t layer,
                int64_t hits, int64_t cpu_entries, int64_t cpu_jobs) {
                auto need = [](bool ok, const char* why) {
                    if (!ok) {
                        std::fprintf(stderr, "strata serve: ADAPT_AUDIT_FAILURE %s\n", why);
                        std::fflush(stderr);
                        std::_Exit(1); // Do not unwind live graph/USM owners.
                    }
                };
                need(adapt_audit->idle() && layer >= 0 && layer < g.n_layers &&
                     D.layers == layer + 1 && D.cache == &xcache && D.src == srcp &&
                     D.host_res == host_res.data() && D.plan && D.pcie_num == 0 &&
                     D.remote_count == 0 && !D.peer && nt > 0 && nt <= 8 && k > 0 && k <= 64,
                     "unsupported dispatch ownership/geometry");
                const int64_t n = nt * k;
                const auto& P = *D.plan;
                need(n <= 512 && P.cap >= n && P.counts && P.start && P.dst && P.tok && P.ptr,
                     "missing window plan buffers");
                const int groups = P.counts[0], entries = P.counts[1];
                need(groups >= 0 && groups <= n && entries >= 0 && entries <= n &&
                     P.counts[2] == 0 && P.start[0] == 0 && P.start[groups] == entries,
                     "window plan group/entry extent");
                bool seen[512] = {};
                for (int q = 0; q < groups; ++q) {
                    need(P.start[q] >= 0 && P.start[q] < P.start[q + 1] && P.start[q + 1] <= entries,
                         "window plan group range");
                    int32_t expert = -1;
                    for (int j = P.start[q]; j < P.start[q + 1]; ++j) {
                        const int i = P.dst[j];
                        need(i >= 0 && i < n && !seen[i] && P.tok[j] == i / k,
                             "window plan duplicate or wrong destination");
                        const int32_t e = ids[i];
                        need(e >= 0 && e < g.n_expert && (expert < 0 || e == expert),
                             "window plan expert identity");
                        expert = e;
                        const int32_t slot = host_res[size_t(layer * g.n_expert + e)];
                        need(slot >= 0 && slot < xcache.slots() && xcache.slot_of(layer, e) == slot &&
                             P.ptr[q] == (unsigned long long)xcache.device_slot(slot),
                             "actual GPU plan names a stale slot/payload");
                        seen[i] = true;
                    }
                }
                int64_t expected_hits = 0, expected_cpu = 0, distinct = 0;
                int32_t misses[512];
                for (int64_t i = 0; i < n; ++i) {
                    const int32_t e = ids[i];
                    need(e >= 0 && e < g.n_expert, "routed expert bounds");
                    const size_t row = size_t(layer * g.n_expert + e);
                    const int32_t slot = host_res[row];
                    need(xcache.slot_of(layer, e) == slot && seen[i] == (slot >= 0),
                         "actual hit/miss decision differs from residency");
                    if (slot >= 0) {
                        ++expected_hits;
                        if (adapt_audit_role[row] == 1) ++adapt_audit_incoming_hits;
                        for (int x = 0; x < strata::kernels::cpu::H; ++x)
                            need(out[size_t(i) * strata::kernels::cpu::H + size_t(x)] == 0.0f,
                                 "GPU hit also has a CPU result");
                    } else {
                        ++expected_cpu;
                        if (adapt_audit_role[row] == -1) ++adapt_audit_outgoing_misses;
                        if (std::find(misses, misses + distinct, e) == misses + distinct)
                            misses[distinct++] = e;
                    }
                }
                need(hits == expected_hits && cpu_entries == expected_cpu && cpu_jobs == distinct &&
                     entries == expected_hits && size_t(distinct) <= D.jobs_multi.size(),
                     "actual GPU/CPU work counters do not conserve routed entries");
                for (int64_t j = 0; j < distinct; ++j) {
                    const auto& job = D.jobs_multi[size_t(j)];
                    const int32_t e = misses[j];
                    need(job.blob == srcp->blob(layer, e), "actual CPU job names the wrong payload");
                    int count = 0;
                    for (int64_t i = 0; i < n; ++i) if (ids[i] == e) {
                        need(count < job.nt && count < strata::kernels::cpu::MAXT &&
                             job.out[count] == out + size_t(i) * strata::kernels::cpu::H &&
                             job.nact[count] == D.nact_multi.data() + size_t(i / k) * strata::kernels::cpu::kNativeActBytes,
                             "actual CPU job token/destination differs");
                        ++count;
                    }
                    need(count == job.nt, "actual CPU job entry extent");
                }
                ++adapt_audit_dispatch_calls;
            };
        }
    ''', 12)
    value = one(value, anchor, callback + anchor)
    anchor = '                std::fprintf(stderr, "strata serve: ADAPT_AUDIT phase=published generation=%llu swaps=%zu identities=3 payload_match=1\\n",\n'
    roles = cpp(r'''
        for (const auto& s : adapt_audit->swaps()) {
            adapt_audit_role[s.outgoing] = -1;
            adapt_audit_role[s.incoming] = 1;
        }
    ''', 16)
    value = one(value, anchor, roles + anchor)
    anchor = '                    p.lent.clear(); p.lent_chunk = 0; p.lease = {};\n'
    loan = cpp(r'''
        if (adapt_audit) {
            if (!adapt_audit->idle() || !p.cache->decode_ready() || !adapt_audit_read(e) ||
                adapt_audit_device != host_res) {
                e = "adaptive audit: loan/refill did not restore the published map"; return false;
            }
            for (size_t row = 0; row < host_res.size(); ++row)
                if (adapt_audit_lookup(row) != host_res[row]) {
                    e = "adaptive audit: loan/refill private identity differs"; return false;
                }
            if (adapt_audit->completed() > 0) {
                ++adapt_audit_post_swap_refills;
                std::fprintf(stderr, "strata serve: ADAPT_AUDIT phase=loan-after-positive completed=%llu lent=%zu identities=3 payload_match=1\n",
                    (unsigned long long)adapt_audit->completed(), p.lent.size());
            }
        }
    ''', 20)
    value = one(value, anchor, loan + anchor)
    anchor = ('            if (!ver.wait_commit(err)) {\n'
              '                std::printf("ERR %s\\n", err.c_str());\n'
              '                return 1;\n'
              '            }\n#ifdef STRATA_ENABLE_SYCL\n            decode_service.finish')
    footer = cpp(r'''
        #ifdef STRATA_ENABLE_SYCL
            if (adapt_audit) {
                if (!apply_pending(true, err) || !adapt_audit->idle()) {
                    std::fprintf(stderr, "strata serve: ADAPT_AUDIT_FAILURE final refill %s\n", err.c_str());
                    std::fflush(stderr);std::_Exit(1);
                }
                std::fprintf(stderr, "strata serve: ADAPT_AUDIT phase=dispatch request=%llu published=%llu calls=%llu incoming_vram_entries=%llu outgoing_cpu_entries=%llu post_swap_refills=%llu\n",
                    (unsigned long long)census_request,(unsigned long long)adapt_audit->completed(),
                    (unsigned long long)adapt_audit_dispatch_calls,(unsigned long long)adapt_audit_incoming_hits,
                    (unsigned long long)adapt_audit_outgoing_misses,(unsigned long long)adapt_audit_post_swap_refills);
            }
        #endif
    ''', 12)
    value = one(value, anchor, anchor.replace('\n#ifdef STRATA_ENABLE_SYCL\n', '\n' + footer + '#ifdef STRATA_ENABLE_SYCL\n', 1))
    return value


def projection(selected, header):
    if (len(selected) != 567 or HEADER in selected or
            hashlib.sha256(header).hexdigest() != HEADER_SHA):
        raise RuntimeError('recover the selected567-file source and exact tested assertion header')
    values = dict(selected)
    values[GENERATOR] = generator(selected[GENERATOR])
    values[HEADER] = header
    if len(values) != 568 or {n for n in values if values[n] != selected.get(n)} != {GENERATOR, HEADER}:
        raise RuntimeError('unplanned adaptive diagnostic source change')
    return values
