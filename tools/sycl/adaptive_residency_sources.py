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
    return value.encode('utf-8')


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
