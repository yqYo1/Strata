#!/usr/bin/env python3
"""Closed same-request exact selection candidate FP32 comparison; diagnostic only."""
import array
import datetime
import fcntl
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main(arm):
    require(arm in (0,1,2), "declared selection arm")
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = A/f'own-qsa-select-model-a{arm}-comparison-v1.json'
        require(not target.exists(), 'fresh selected-row comparison')
        paths = [A/'qsa-subprofile-model-v2.json', A/f'own-qsa-select-model-a{arm}-v1.json']
        records = [json.loads(p.read_text()) for p in paths]
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        jobs = []
        for r in records:
            require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults']
                and r['boot_unchanged'] and not r['driver_dump_exists'], 'closed healthy diagnostic')
            for c in r['commands']:
                require(module.closed(c), 'owned command closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owned identity retired')
            actual = [c for c in r['commands'] if c['label'] == 'prefill-only']
            require(len(actual) == 1, 'one diagnostic native/server process')
            jobs.append(actual[0])
        require(datetime.datetime.fromisoformat(jobs[0]['finished_utc']) <=
            datetime.datetime.fromisoformat(jobs[1]['started_utc']), 'serial independent processes')
        left,right = records
        require(all(left[k] == right[k] for k in ('engine_args',
            'benchmark_options','benchmark_source','model_identity','pack_metadata_sha256','MTP_metadata_sha256','boot_id')),
            'same requested model/work')
        build_path = A/'own-qsa-select-build-v1.json'
        build = json.loads(build_path.read_text())
        require(build['passed'] and build['complete'] and not build['active'] and
            build['binary_sha256'] == right['binary_sha256'] and build['source_pins'] == right['source_pins'], 'actual selection build')
        allowed={'src/kernels/sycl/prompt_attention.cpp','src/kernels/sycl/qsa_select.cpp',
                 'cmake/sycl_backend.cmake','tests/sycl/prompt_k_cellmajor.cpp','tests/sycl/qsa_select_radix.cpp'}
        transforms={}
        for path in sorted(set(left['source_pins']) | set(right['source_pins'])):
            old,new=left['source_pins'].get(path),right['source_pins'].get(path)
            if old!=new:
                require(path in allowed,'only reviewed exact selector/prompt fixture changes '+path)
                transforms[path]=dict(base_sha256=old,proposed_sha256=new)
        require('src/kernels/sycl/qsa_select.cpp' in transforms,'actual changed selector')
        component_path=A/'own-qsa-select-bench-comparison-v1.json'
        component=json.loads(component_path.read_text())
        require(component['passed'] and component['complete'],'actual exact positive component closure')
        if arm:require(component['component_positive'][str(arm)],'selected component positive')
        require(all(left['source_pins'][n] == right['source_pins'][n] == h for n,h in right['unchanged_math_and_runtime'].items()),
            'unchanged arithmetic/runtime/MTP/session sources')
        ignored = {'STRATA_SYCL_SELECT_RADIX4','STRATA_SYCL_SELECT_CYCLIC','STRATA_SYCL_PROMPT_K_CELL_MAJOR','STRATA_SYCL_PROMPT_NATIVE_KV_MIN','XDG_CACHE_HOME','STRATA_MEASUREMENT_ENGINE_START',
            'STRATA_MEASUREMENT_ENGINE_EXIT','STRATA_PREFILL_BOUNDARY_PATH'}
        require({k:v for k,v in left['environment'].items() if k not in ignored} ==
            {k:v for k,v in right['environment'].items() if k not in ignored}, 'same effective diagnostic environment except explicit selector flags, default0prompt flags and artifact/cache paths')
        require(left['environment']['STRATA_SYCL_PROMPT_DIRECT'] == '1' and
            right['environment']['STRATA_SYCL_PROMPT_DIRECT'] == '1' and
            left['environment']['STRATA_PREFILL_LAYER_MAJOR_R_GPU'] == right['environment']['STRATA_PREFILL_LAYER_MAJOR_R_GPU'] == '32768', 'actual unchanged Direct1/GPU32768 settings')
        require(right['actual_experiment_environment']['STRATA_PREFILL_CANONICAL_DENSE_BANK']==right['actual_experiment_environment']['STRATA_PREFILL_LAYER_MAJOR_PROFILE']=='1','actual selected new route')
        require(left['environment']['STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE']==right['actual_experiment_environment']['STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE']=='1','fine profile enabled in both')
        require(left['environment'].get('STRATA_SYCL_SELECT_RADIX4','0')==left['environment'].get('STRATA_SYCL_SELECT_CYCLIC','0')=='0' and right['actual_experiment_environment']['STRATA_SYCL_SELECT_RADIX4']==str(int(arm==1)) and right['actual_experiment_environment']['STRATA_SYCL_SELECT_CYCLIC']==str(int(arm==2)),'actual selector arms')
        require(all(r['environment'].get('STRATA_SYCL_PROMPT_K_CELL_MAJOR','0')==r['environment'].get('STRATA_SYCL_PROMPT_NATIVE_KV_MIN','0')=='0' for r in records),'prompt default0 in both')
        require(left['actual_experiment_environment']['STRATA_PREFILL_CANONICAL_DENSE_BANK']==left['actual_experiment_environment']['STRATA_PREFILL_LAYER_MAJOR_PROFILE']=='1','baseline canonical/profile1')
        requests = [r['benchmark_result']['results']['prefill_only']['requests'][0] for r in records]
        require(requests[0]['request_sha256'] == requests[1]['request_sha256'] and
            all(q['actual_read_tokens'] == q['actual_prompt_tokens'] == 78453 and q['actual_reused_tokens'] == 0
                and q['thinking_requested'] and q['generated_tokens'] == 24 for q in requests), 'identical actual input/thinking work')
        require(all(int(phase[7])==0 for r in records for phase in r['dense_lifecycle']['phases']), 'no captured SessionGraphs in this explicit no-token-graph workload')
        wrapper_ms=float(right['coarse_profile']['header']['wrapper_ms'])
        require(math.isfinite(wrapper_ms) and wrapper_ms>0,'finite positive measured wrapper duration')
        boundaries = [r['boundary'] for r in records]
        require(all(b['same_run_copy_equal'] and b['shape'] == boundaries[0]['shape'] and
            b['rows'] == boundaries[0]['rows'] and b['output_order'] == boundaries[0]['output_order']
            and b['raw_bytes'] == 31457280 for b in boundaries), 'identical selected-row geometry and copy checks')
        raw = [Path(b['raw_path']) for b in boundaries]
        require(all(p.stat().st_size == b['raw_bytes'] and sha(p) == b['raw_sha256'] for p,b in zip(raw,boundaries)),
            'actual original capture identity')
        require(sys.byteorder == 'little' and array.array('f').itemsize == 4, 'actual FP32 capture encoding')
        layers = []
        first = None
        finite = True
        with raw[0].open('rb') as f0,raw[1].open('rb') as f1:
            for layer in range(48):
                b0,b1 = f0.read(16*10240*4),f1.read(16*10240*4)
                a0,a1 = array.array('f'),array.array('f')
                a0.frombytes(b0);a1.frombytes(b1)
                require(len(a0) == len(a1) == 16*10240, 'complete layer extent')
                is_finite = all(math.isfinite(x) and math.isfinite(y) for x,y in zip(a0,a1))
                finite = finite and is_finite
                equal = b0 == b1
                maximum = max(max(abs(x) for x in a0),max(abs(x) for x in a1)) if is_finite else None
                difference = 0.0 if is_finite else None
                relative = 0.0 if is_finite else None
                if not equal and is_finite:
                    difference = max(abs(x-y) for x,y in zip(a0,a1))
                    delta2 = math.fsum((x-y)*(x-y) for x,y in zip(a0,a1))
                    base2 = math.fsum(x*x for x in a0)
                    relative = math.sqrt(delta2/base2) if base2 else (0.0 if not delta2 else None)
                if not equal and first is None:
                    at = next(i for i in range(len(a0)) if b0[i*4:i*4+4] != b1[i*4:i*4+4])
                    first = dict(layer=layer, relative_row=boundaries[0]['rows'][at//10240],
                        hidden_column=at%10240,
                        left=a0[at] if math.isfinite(a0[at]) else repr(a0[at]),
                        right=a1[at] if math.isfinite(a1[at]) else repr(a1[at]))
                layers.append(dict(layer=layer, selected_rows=16, bytes_equal=equal, finite=is_finite,
                    maximum_absolute_value=maximum, max_absolute_difference=difference,
                    relative_L2=relative))
            require(not f0.read(1) and not f1.read(1), 'no extra capture data')
        all_equal = all(x['bytes_equal'] for x in layers)
        result = dict(passed=finite and all_equal, complete=True, active=False, GPU_executed=False,
            recipe_sha256=sha(Path(__file__)), references={p.name:sha(p) for p in paths+[build_path,component_path]},
            binary_sha256_before=left['binary_sha256'], binary_sha256_after=right['binary_sha256'], source_transformations=transforms, selection_arm=arm, queue_selection_ms=[r['qsa_profile']['rows'][-1]['queue_ms'][4] for r in records], queue_attention_ms=[r['qsa_profile']['rows'][-1]['queue_ms'][5] for r in records], request_sha256=requests[0]['request_sha256'],
            sampled_FP32_values_equal=all_equal, sampled_FP32_values_finite=finite,
            first_difference=first, layers=layers, same_run_store_input_equal=True, session_recapture_observed_zero=True, captured_session_graph_restore_qualified=False,
            individual_diagnostic_PP=[q['prompt_tps'] for q in requests],
            output_fingerprints=[{k:q[k] for k in ('output_sha256','reasoning_sha256')} for q in requests],
            quality_qualified=False, performance_qualified=False, physical_262144_qualified=False,
            restore_qualified=False, finished_utc=module.utc(),
            scope='Original selector versus own exact selector candidate; unchanged other arithmetic/runtime/MTP/session and canonical/coarse/fine/Direct1/GPU32768 settings. Prompt default0, separately qualified exact full component. Historical source/binary plus experiment differ; do not attribute diagnostic PP causally. Compare selected16rows*48layers and their stored/reloaded copies. Exact-copy diagnostic is not a generated-token equality requirement or all-state/task-quality proof. Instrumented n1/arm PP is not clean performance.')
        with target.open('x') as output:
            json.dump(result, output, indent=2, allow_nan=False); output.write('\n')
        print(json.dumps(dict(passed=result['passed'], selected_rows=768,
            first_difference=first, diagnostic_PP=result['individual_diagnostic_PP'])), flush=True)


if __name__ == '__main__':
    require(len(sys.argv)==2,'selection arm0|1|2')
    main(int(sys.argv[1]))
