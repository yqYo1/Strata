#!/usr/bin/env python3
"""Exact selected layer-boundary comparison of closed alias processes."""
import array
import fcntl
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(sys.byteorder == 'little', 'FP32 capture byte order')
        paths = [A/f'prefill-scratch-alias-model-{arm}-v1.json' for arm in (0,1)]
        arms = [json.loads(p.read_text()) for p in paths]
        owner = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for i,r in enumerate(arms):
            require(r['alias_arm'] == i and r['passed'] and r['complete'] and not r['active'] and
                not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'], 'closed healthy arm')
            for c in r['commands']:
                require(owner.closed(c), 'closed command')
                for old in c['owners']:
                    now = owner.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'retired native owner')
            require(r['boundary']['same_run_copy_equal'], 'same-run stored/reloaded rows')
            for name in ('text','raw'):
                require(sha(Path(r['boundary'][name+'_path'])) == r['boundary'][name+'_sha256'], 'capture identity')
        off,on = arms
        require(all(off[k] == on[k] for k in ('binary_sha256','source_pins','linked_inputs',
            'engine_args','benchmark_options','benchmark_source','pack_metadata_sha256',
            'MTP_metadata_sha256','model_identity')), 'same binary/source/model/workload')
        exceptions = {'XDG_CACHE_HOME','STRATA_MEASUREMENT_ENGINE_START','STRATA_MEASUREMENT_ENGINE_EXIT',
            'STRATA_PREFILL_BOUNDARY_PATH','STRATA_SYCL_PREFILL_SCRATCH_ALIAS'}
        require({k:v for k,v in off['environment'].items() if k not in exceptions} ==
                {k:v for k,v in on['environment'].items() if k not in exceptions}, 'same effective experiment settings')
        requests = [r['benchmark_result']['results']['prefill_only']['requests'][0] for r in arms]
        require(requests[0]['request_sha256'] == requests[1]['request_sha256'], 'exact request body')
        require(all(r['actual_prompt_tokens'] == r['actual_read_tokens'] == 78453 and
            r['actual_reused_tokens'] == 0 and r['thinking_requested'] and r['stream_complete']
            for r in requests), 'same actual work')
        require(off['boundary']['output_order'] == on['boundary']['output_order'] and
            off['boundary']['shape'] == on['boundary']['shape'], 'exact row order and shape')
        summary = [dict(layer=l, rows=0, equal_rows=0, differing_float_values=0,
            nonfinite_values=0, max_abs=0.) for l in range(48)]
        first = None
        raw = [Path(r['boundary']['raw_path']) for r in arms]
        with raw[0].open('rb') as left, raw[1].open('rb') as right:
            for layer,row in off['boundary']['output_order']:
                x,y = left.read(40960),right.read(40960)
                require(len(x) == len(y) == 40960, 'complete row bytes')
                a,b = array.array('f'),array.array('f');a.frombytes(x);b.frombytes(y)
                s = summary[layer];s['rows'] += 1;s['equal_rows'] += int(x == y)
                finite = sum(not math.isfinite(v) for v in a)+sum(not math.isfinite(v) for v in b)
                s['nonfinite_values'] += finite
                if x != y:
                    count = sum(u != v for u,v in zip(a,b))
                    maximum = max((abs(float(u)-float(v)) for u,v in zip(a,b)
                        if math.isfinite(u) and math.isfinite(v)), default=0.)
                    s['differing_float_values'] += count;s['max_abs'] = max(s['max_abs'],maximum)
                    if first is None:
                        byte = next(i for i,(u,v) in enumerate(zip(x,y)) if u != v)
                        first = dict(layer=layer,row=row,first_byte=byte,float_index=byte//4,
                            off_value=a[byte//4],on_value=b[byte//4],row_max_abs=maximum)
            require(left.read(1) == right.read(1) == b'', 'no extra/missing rows')
        require(all(s['rows'] == 16 for s in summary), 'all48 layers and16 rows')
        passed = first is None and not any(s['nonfinite_values'] for s in summary)
        result = dict(passed=passed,complete=True,active=False,GPU_executed=False,
            recipe_sha256=sha(Path(__file__)), references={p.name:sha(p) for p in paths},
            binary_sha256=off['binary_sha256'],request_sha256=requests[0]['request_sha256'],
            same_run_copy_equal=True,sampled_layer_residuals_bitwise_equal=first is None,
            first_difference=first,layers=summary,raw_bytes_each=off['boundary']['raw_bytes'],
            raw_sha256=[sha(p) for p in raw],
            output_fingerprints=[{k:r[k] for k in ('output_sha256','reasoning_sha256','generated_tokens')}
                for r in requests],
            scope='16 selected rows x48 layer outputs, exact bytes and finite checks with same-run stored/input equality. Does not establish unselected rows, unsynchronized behavior, task quality, session RESTORE, physical262144 or speed.',
            synchronization_may_mask_races=True,quality_qualified=False,performance_qualified=False,
            restore_qualified=False,physical_262144_qualified=False,finished_utc=owner.utc())
        target = A/'prefill-scratch-alias-model-comparison-v1.json'
        with target.open('x') as out:
            json.dump(result,out,indent=2);out.write('\n')
        print(json.dumps(dict(passed=passed,selected_rows=768,first_difference=first)))
        require(passed, 'sampled alias residual equivalence')


if __name__ == '__main__':
    main()
