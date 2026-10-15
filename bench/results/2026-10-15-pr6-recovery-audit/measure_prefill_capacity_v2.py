#!/usr/bin/env python3
"""Correct PLE history and evaluate both one-bank and separate-bank layouts.

Preserves the original v1 receipt: its PLE-history assumption was 2 rows,
where the actual NG_HIST=(4-1)*3 is9. No GPU execution.
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
        old_path = v1.A/'prefill-capacity-metadata-v1.json'
        old = json.loads(old_path.read_text())
        d = v1.calculate()
        require = v1.require
        ngram = (v1.P/'include/strata/kernels/ngram.hpp').read_text()
        require(re.search(r'PLE_CONV_KERNEL\s*=\s*4\s*;', ngram) and
                re.search(r'NGRAM_SIZE\s*=\s*3\s*;', ngram) and
                'NG_HIST = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE' in ngram, 'actual PLE history geometry')
        ple_bytes = 9*10240*4
        original_cases = list(d['cases'])
        cases = []
        layouts = dict(
            one_layer_bank=d['largest_layer']['total_bytes'],
            separate_max_expert_dense_banks=d['layer_reservation_bytes'])
        for layout, layer_bytes in layouts.items():
            for old_case in original_cases:
                t, fmt = old_case['chunk'], old_case['format']
                work = old_case['workspace']
                fixed = (layer_bytes+work['carve_accounted_bytes']+work['separate_token_buffer_bytes']+
                    old_case['recurrent_GDN_bytes']+ple_bytes+old_case['small_owner_allowance_bytes']+
                    old_case['runtime_headroom_bytes'])
                budget = d['observed_device_total_bytes']-fixed
                def needed(c):
                    return v1.qsa_state(c, fmt)+c*20480
                lo, hi = 0, 262144//4
                while lo < hi:
                    mid = (lo+hi+1)//2
                    if needed(mid*4) <= budget:
                        lo = mid
                    else:
                        hi = mid-1
                full = v1.qsa_state(262144, fmt)
                hidden = max(0, budget-full)
                c = dict(old_case, layout=layout, active_layer_reservation_bytes=layer_bytes,
                    PLE_history_bytes=ple_bytes, fixed_before_KV_hidden_bytes=fixed,
                    available_for_KV_hidden_bytes=budget, max_context_with_all_hidden_rows=lo*4,
                    bytes_at_max_context=needed(lo*4), next_page_bytes=needed((lo+1)*4),
                    hidden_bytes_with_full_KV=hidden,
                    hidden_rows_with_full_KV=min(262144, hidden//20480),
                    full_context_shortfall_bytes=max(0, needed(262144)-budget),
                    one_active_QSA_remaining_bytes=budget-v1.qsa_state(262144, fmt, layers=1)-262144*20480)
                c['all_hidden_with_one_active_QSA_fits'] = c['one_active_QSA_remaining_bytes'] >= 0
                cases.append(c)
        # Confirm exact ceil/page and source-derived total arithmetic separately
        # from the observed streamed C4 workspace conservation gate in v1.
        require(v1.qsa_state(262144, 'K8V8') == 3794839744 and
                v1.qsa_state(262144, 'K8V4') == 3039865024, 'full QSA arithmetic')
        for case in cases:
            c, fmt = case['max_context_with_all_hidden_rows'], case['format']
            b = case['available_for_KV_hidden_bytes']
            require(v1.qsa_state(c, fmt)+c*20480 <= b and
                    (c == 262144 or v1.qsa_state(c+4, fmt)+(c+4)*20480 > b), 'maximum complete page')
            h = case['hidden_rows_with_full_KV']
            require(case['full_262144_QSA_bytes']+h*20480 <= b and
                    (h == 262144 or case['full_262144_QSA_bytes']+(h+1)*20480 > b), 'maximum hidden row')
        d.update(schema='prefill-capacity-metadata-and-source-v2', cases=cases,
            supersedes=dict(path=old_path.name, sha256=v1.sha(old_path),
                correction='PLE history is9 rows=368640B, not2=81920B; preserve v1 unchanged. Add exact largest combined layer bank as main planning scenario; separate max-expert/max-dense banks remain conservative alternative.'),
            recipe_sha256=v1.sha(Path(__file__)), layer_layouts=layouts,
            checks=dict(observed_streamed_C4096_allocation_reproduced=True,
                        QSA_full_context_byte_formulas_crosschecked=True,
                        maximum_complete_pages_and_hidden_rows_checked=True))
        d['source_pins'][str(Path(__file__))] = v1.sha(Path(__file__))
        d['source_pins'][str(v1.P/'include/strata/kernels/ngram.hpp')] = v1.sha(v1.P/'include/strata/kernels/ngram.hpp')
        d['assumptions'] += [
            'one_layer_bank is a proposed reusable combined packed-expert+dense bank sized to maximum actual layer sum; requires a new loader/owner layout.',
            'separate_max_expert_dense_banks retains separate maximum expert and dense capacities, whose maxima occur on different layers.',
            'max_context_with_all_hidden_rows sizes KV to that same C; configured262144 preallocated KV uses hidden_rows_with_full_KV instead.'
        ]
        target = v1.A/'prefill-capacity-metadata-v2.json'
        with target.open('x') as stream:
            json.dump(d, stream, indent=2)
            stream.write('\n')
        require(v1.sha(old_path) == d['supersedes']['sha256'], 'historical receipt preserved')
        print(json.dumps(dict(passed=True, observed_device_GiB=d['observed_device_total_bytes']/v1.GIB,
            actual_largest_layer=d['largest_layer']['layer'], layer_layouts=layouts,
            cases=[{k:c[k] for k in ('layout','chunk','format','available_for_KV_hidden_bytes',
                'max_context_with_all_hidden_rows','hidden_rows_with_full_KV',
                'full_context_shortfall_bytes','one_active_QSA_remaining_bytes')} for c in cases])), flush=True)


if __name__ == '__main__':
    main()
