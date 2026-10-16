#!/usr/bin/env python3
"""Capacity delta tied to the actual production alias allocation gate.

No GPU launch. Preserve the original metadata and failed fixture receipts.
"""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
import measure_prefill_capacity_v1 as v1


def main():
    with (v1.B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_path = v1.A/'prefill-capacity-metadata-v2.json'
        probe_path = v1.A/'prefill-scratch-allocation-v2.json'
        build_path = v1.A/'prefill-scratch-allocation-build-v2.json'
        original = json.loads(original_path.read_text())
        probe = json.loads(probe_path.read_text())
        build = json.loads(build_path.read_text())
        need = v1.require
        need(original['passed'] and original['complete'] and not original['active'], 'metadata gate')
        need(probe['passed'] and probe['complete'] and not probe['active'] and
             not probe['fresh_faults'] and probe['boot_unchanged'] and
             not probe['driver_dump_exists'] and probe['boot_id'] == v1.boot_id(), 'closed actual GPU gate')
        owner = v1.load_owner(v1.P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        need(all(owner.closed(c) for c in probe['commands']), 'closed GPU commands')
        for c in probe['commands']:
            for old in c['owners']:
                now = owner.identity(old['pid'])
                need(now is None or now['start_ticks'] != old['start_ticks'], 'retired GPU owner')
        need(all(v1.sha(v1.P/n) == h for n,h in build['source_pins'].items()), 'exact production source')
        source = (v1.P/'src/prefill/prefill.cpp').read_text()
        need('constexpr size_t GEMM_WS = 0;' in source and
             'return o.used + (8u << 20);' in source, 'workspace and reservation slack')
        stage_bytes = 276824064+4*256
        cases = []
        for old in original['cases']:
            t, fmt = old['chunk'], old['format']
            for alias in (False, True):
                alias_saved = (3*v1.alloc(t*2560*4)) if alias else 0
                no_ws = v1.alloc(32*v1.MIB)
                saved = no_ws+alias_saved
                work_bytes = old['workspace']['carve_accounted_bytes']-saved
                actual_carve = work_bytes+stage_bytes
                if alias:
                    matched = [line for line in probe['carve_lines'] if
                        f'chunk={t},' in line and f'accounted={actual_carve},' in line]
                    need(len(matched) >= (2 if t == 8192 else 1), 'actual streamed carve conservation')
                budget = old['available_for_KV_hidden_bytes']+saved
                def used(c):
                    return v1.qsa_state(c, fmt)+20480*c
                max_c = min(262144, ((budget-43200)//(34956 if fmt == 'K8V8' else 32076))//4*4)
                h = min(262144, max(0, (budget-v1.qsa_state(262144, fmt))//20480))
                need(used(max_c) <= budget and (max_c == 262144 or used(max_c+4) > budget), 'maximum full page')
                need(v1.qsa_state(262144, fmt)+h*20480 <= budget and
                     (h == 262144 or v1.qsa_state(262144, fmt)+(h+1)*20480 > budget), 'maximum hidden row')
                # Explicitly expose a 1GiB allowance sensitivity, not a new measured runtime maximum.
                conservative_budget = budget-512*v1.MIB
                h1 = max(0, (conservative_budget-v1.qsa_state(262144, fmt))//20480)
                c1 = min(262144, ((conservative_budget-43200)//(34956 if fmt == 'K8V8' else 32076))//4*4)
                cases.append(dict(layout=old['layout'], chunk=t, format=fmt, scratch_alias=alias,
                    active_layer_reservation_bytes=old['active_layer_reservation_bytes'],
                    full_resident_workspace_bytes=work_bytes,
                    streamed_workspace_bytes=actual_carve,
                    borrowed_reservation_slack_bytes=8*v1.MIB,
                    alias_saved_bytes=alias_saved, removed_unused_workspace_bytes=no_ws,
                    separate_token_buffer_bytes=old['workspace']['separate_token_buffer_bytes'],
                    recurrent_GDN_bytes=old['recurrent_GDN_bytes'], PLE_history_bytes=old['PLE_history_bytes'],
                    small_owner_allowance_bytes=old['small_owner_allowance_bytes'],
                    runtime_headroom_bytes=old['runtime_headroom_bytes'],
                    available_for_KV_hidden_bytes=budget,
                    max_context_with_all_hidden_rows=max_c,
                    full_262144_QSA_bytes=v1.qsa_state(262144, fmt),
                    hidden_rows_with_full_KV=h,
                    full_context_shortfall_bytes=max(0, used(262144)-budget),
                    one_active_QSA_remaining_bytes=budget-v1.qsa_state(262144, fmt, layers=1)-262144*20480,
                    sensitivity_1GiB_runtime_allowance=dict(available_for_KV_hidden_bytes=conservative_budget,
                        max_context_with_all_hidden_rows=c1, hidden_rows_with_full_KV=h1)))
        result = dict(schema='prefill-capacity-no-workspace-alias-delta-v1', passed=True, complete=True,
            active=False, GPU_executed=False, upstream_check_performed=False,
            linked_GPU_allocation_probe_passed=True, physical_model_262144_qualified=False,
            quality_qualified=False, performance_qualified=False,
            recipe_sha256=v1.sha(Path(__file__)),
            references={p.name:v1.sha(p) for p in (original_path, probe_path, build_path)},
            production_source_pins=build['source_pins'], observed_device_total_bytes=original['observed_device_total_bytes'],
            largest_layer={k:original['largest_layer'][k] for k in
                ('layer','kind','canonical_bytes','native_bytes','packed_experts_bytes','total_bytes')},
            saved_hidden_bytes_per_token=20480, cases=cases,
            limits=[
                'One combined layer weight bank is hypothetical. NativeDense proposal alone keeps canonical weights resident and does not implement it.',
                'All12 QSA state/indexers and shared RoPE are resident in the main calculation; no separate streamed stage is added.',
                'The actual allocation gate used streamed262144 metadata and no inference/state payload. Subtracting its explicitly observed stage is a source-based resident calculation.',
                'FP16/BF16 boundary storage uses2B but numeric/task quality is unqualified. K8V4 long-prefill is hypothetical and currently rejected by the configured layer-major path.',
                '512MiB runtime and16MiB small-owner allowances are assumptions, not measured worst-case bounds. Allocation-pool/cache free counters do not establish a resident maximum.',
                'The GPU fixture after retiring the owned slab retained830853120B relative to initial free-memory query; cause/pool versus other owner is not isolated, not a proven leak. Do not subtract that number as an identified runtime constant.',
                'The8MiB slack is a conservative borrowed reservation, not a separately carved live buffer; exact-carve and one-byte-short admission were tested.'
            ], finished_utc=owner.utc())
        target = v1.A/'prefill-capacity-alias-delta-v1.json'
        with target.open('x') as out:
            json.dump(result, out, indent=2); out.write('\n')
        print(json.dumps([c for c in cases if c['layout']=='one_layer_bank']))


if __name__ == '__main__':
    main()
