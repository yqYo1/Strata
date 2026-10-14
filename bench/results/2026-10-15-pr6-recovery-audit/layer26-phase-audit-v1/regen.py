#!/usr/bin/env python3
"""Regenerate the finite proposal only; never edits the product tree or applies a patch."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'expected one anchor, got {source.count(old)}: {old[:100]}')
    return source.replace(old, new, 1)


def sources():
    p = 'src/prefill/prefill.cpp'
    source = (ROOT / p).read_text()
    original = source
    source = replace_once(source, '#include "strata/prefill/layer_major_boundary_audit.hpp"',
        '#include "strata/prefill/layer_major_boundary_audit.hpp"\n#include "strata/prefill/layer26_phase_audit.hpp"')
    source = replace_once(source, '    detail::LayerMajorBoundaryAudit* layer_major_audit = nullptr;',
        '    detail::Layer26PhaseAudit* layer26_phase_audit = nullptr;\n    detail::LayerMajorBoundaryAudit* layer_major_audit = nullptr;')
    source = replace_once(source,
        '                    // ======================= GDN =======================',
        '''                    if (m.layer26_phase_audit && !m.layer26_phase_audit->capture(
                            detail::Layer26PhaseAudit::gdn_input, l, c0, T, m.mixed_h, m.layer_major_audit_copy)) {
                        err = "prefill phase audit: GDN input copy/drain failed"; return false;
                    }
                    // ======================= GDN =======================''')
    source = replace_once(source,
        '                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;',
        '''                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    if (m.layer26_phase_audit && !m.layer26_phase_audit->capture(
                            detail::Layer26PhaseAudit::gdn_out, l, c0, T, m.bo, m.layer_major_audit_copy)) {
                        err = "prefill phase audit: GDN output copy/drain failed"; return false;
                    }''')
    source = replace_once(source,
        '                    // ======================= MoE =======================',
        '''                    if (m.layer26_phase_audit) {
                        auto* a = m.layer26_phase_audit;
                        using A = detail::Layer26PhaseAudit;
                        if (!a->capture(A::ffn_input_h, l, c0, T, m.mixed_h, m.layer_major_audit_copy) ||
                            !a->capture(A::ffn_input_bf, l, c0, T, m.mixed_bf, m.layer_major_audit_copy) ||
                            (m.mixed_bf_lo && !a->capture(A::ffn_input_low, l, c0, T, m.mixed_bf_lo, m.layer_major_audit_copy))) {
                            err = "prefill phase audit: FFN mixed input copy/drain failed"; return false;
                        }
                    }
                    // ======================= MoE =======================''')
    source = replace_once(source,
        '                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);',
        '''                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    if (m.layer26_phase_audit) {
                        auto* a = m.layer26_phase_audit;
                        using A = detail::Layer26PhaseAudit;
                        if (!a->capture(A::router_logits, l, c0, T, m.logits, m.layer_major_audit_copy) ||
                            !a->capture(A::router_ids, l, c0, T, m.ids, m.layer_major_audit_copy) ||
                            !a->capture(A::router_weights, l, c0, T, m.w, m.layer_major_audit_copy)) {
                            err = "prefill phase audit: router copy/drain failed"; return false;
                        }
                    }''')
    source = replace_once(source, '                        n_order = order.size();',
        '''                        n_order = order.size();
                        if (m.layer26_phase_audit && !m.layer26_phase_audit->capture_grouping(
                                l, c0, T, ids_h, slot_h, src_h, m.cnt, m.off, order)) {
                            err = "prefill phase audit: grouping capture failed"; return false;
                        }''')
    source = replace_once(source,
        '                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);',
        '''                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                    if (m.layer26_phase_audit && !m.layer26_phase_audit->capture(
                            detail::Layer26PhaseAudit::moe_out, l, c0, T, m.bo, m.layer_major_audit_copy)) {
                        err = "prefill phase audit: MoE output copy/drain failed"; return false;
                    }''')
    source = replace_once(source,
        '                    strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);',
        '''                    strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
                if (m.layer26_phase_audit && !m.layer26_phase_audit->capture(
                        half == 0 ? detail::Layer26PhaseAudit::attn_R : detail::Layer26PhaseAudit::final_R,
                        l, c0, T, m.R, m.layer_major_audit_copy)) {
                    err = "prefill phase audit: residual write copy/drain failed"; return false;
                }''')
    yield p, original, source

    p = 'src/prefill/layer_major.inc'
    source = (ROOT / p).read_text()
    original = source
    source = replace_once(source,
        '    // Reuse only pays across chunks.',
        '''    const char* phase_switch = std::getenv("STRATA_PREFILL_PHASE_AUDIT");
    if (phase_switch && std::strcmp(phase_switch, "1") == 0 &&
        (!requested || std::strcmp(requested, "1") || !impl_ || n <= impl_->T)) {
        err = "prefill phase audit: requires the long-prompt layer-major path"; return false;
    }
    // Reuse only pays across chunks.''')
    source = replace_once(source,
        '    const char* mmq = std::getenv("STRATA_PREFILL_MMQ");',
        '''    const char* phase_switch = std::getenv("STRATA_PREFILL_PHASE_AUDIT");
    const bool phase_requested = phase_switch && std::strcmp(phase_switch, "1") == 0;
    if (phase_requested && residual_dtype != detail::LayerMajorResidualDtype::fp32) {
        err = "prefill phase audit: FP16 saved residual storage is outside this diagnostic"; return false;
    }
    const char* mmq = std::getenv("STRATA_PREFILL_MMQ");''')
    source = replace_once(source,
        '    int64_t logical_chunks = 0, layer = 0;',
        '''    // This owner precedes Restore and the checked lease, just like boundary audit.
    // Reserve BOTH optional owners cumulatively; no file/raw bytes are held in RAM.
    std::unique_ptr<detail::Layer26PhaseAudit> phase_audit;
    if (phase_requested) {
        uint64_t need = detail::layer_major_add(host_need, audit ? 4ull << 20 : 0);
        need = detail::layer_major_add(need, detail::Layer26PhaseAudit::host_reservation);
        if (!core::conversation_memory_admit(core::conversation_available_memory(), need, host_headroom)) {
            err = "prefill phase audit: additional diagnostic storage exceeds RAM headroom"; return false;
        }
        phase_audit.reset(new detail::Layer26PhaseAudit(n, pos0, m.T, m.mixed_bf_lo != nullptr));
    }
    int64_t logical_chunks = 0, layer = 0;''')
    source = replace_once(source, '        detail::LayerMajorBoundaryAudit* audit;',
        '        detail::Layer26PhaseAudit* phase;\n        detail::LayerMajorBoundaryAudit* audit;')
    source = replace_once(source,
        '            m.layer_major_audit = audit; m.layer_major_audit_copy = std::move(audit_copy);',
        '            m.layer26_phase_audit = phase;\n            m.layer_major_audit = audit; m.layer_major_audit_copy = std::move(audit_copy);')
    source = replace_once(source,
        '              m.layer_major_audit, m.layer_major_audit_copy,',
        '              m.layer26_phase_audit, m.layer_major_audit, m.layer_major_audit_copy,')
    source = replace_once(source,
        '    if (audit) {\n        m.layer_major_audit = audit.get();',
        '    if (audit || phase_audit) {\n        m.layer26_phase_audit = phase_audit.get();\n        m.layer_major_audit = audit.get();')
    source = replace_once(source, '    if (audit) audit->complete();',
        '    if (audit) audit->complete();\n    if (phase_audit) phase_audit->complete();')
    yield p, original, source
    yield 'include/strata/prefill/layer26_phase_audit.hpp', '', (HERE / 'layer26_phase_audit.hpp.in').read_text()


def main():
    pins = json.loads((HERE / 'pins.json').read_text())
    for path, expected in pins['source_before'].items():
        if sha((ROOT / path).read_bytes()) != expected:
            raise RuntimeError('source pin mismatch; Main must review/rebase explicitly: ' + path)
    patch = []
    after = {}
    for path, before, proposed in sources():
        patch.extend(difflib.unified_diff(before.splitlines(True), proposed.splitlines(True),
                     fromfile='a/'+path if before else '/dev/null', tofile='b/'+path, n=3))
        after[path] = sha(proposed.encode())
    data = ''.join(patch).encode()
    (HERE / 'source.diff').write_bytes(data)
    (HERE / 'regen-receipt.json').write_text(json.dumps({
        'status': 'proposal_only_not_applied', 'compile': 'not_run', 'tests': 'not_run', 'GPU': 'not_run',
        'source_diff_sha256': sha(data), 'source_after': after,
        'raw_max_bytes': 8388608, 'text_max_bytes': 1048576, 'new_host_reservation_bytes': 1048576,
        'regen': 'python3 -B ' + str(HERE / 'regen.py'),
    }, indent=2)+'\n')
    print(sha(data))


if __name__ == '__main__':
    main()
