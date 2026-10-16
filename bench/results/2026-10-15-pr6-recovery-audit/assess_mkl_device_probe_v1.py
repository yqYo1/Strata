#!/usr/bin/env python3
"""Match actual installed oneMKL verbose calls to closed probe cases."""
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import require, sha


def main():
    source = A / 'own-prefill-host-profile-build-v2.json'
    r = json.loads(source.read_text())
    require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults'], 'closed healthy probe')
    result = dict(passed=True, complete=True, GPU_executed=False, source_receipt_sha256=sha(source),
        recipe_sha256=sha(Path(__file__)), cases={},
        scope='Installed oneMKL reported execution target for 16 synthetic calls, using actual production Gemm wrapper; padded ld=N+3, not every model shape/stride or absence of host setup. MKL_VERBOSE=1, not performance evidence.')
    for label, q in r['GEMM_probe'].items():
        current = None; cases = []
        for line in q['stdout'].splitlines():
            m = re.match(r'BEGIN type=(\w+) role=(\w+) rows=(\d+) N=(\d+) K=(\d+) ld=(\d+)$', line)
            if m:
                require(current is None, 'previous case closed')
                dtype, role, rows, n, k, ld = m.groups()
                current = dict(dtype=dtype, role=role, rows=int(rows), N=int(n), K=int(k), ld=int(ld), calls=[])
            elif line.startswith('MKL_VERBOSE oneapi::mkl::blas::row_major::gemm['):
                require(current is not None, 'call inside labeled case')
                call = re.fullmatch(r'MKL_VERBOSE oneapi::mkl::blas::row_major::gemm\[([^]]+)\]\((.*)\) (.*)', line)
                require(call is not None, 'actual installed call format')
                types, args, suffix = call.groups(); args = args.split(',')
                expected = 'half,half,float' if current['dtype'] == 'fp16' else 'bfloat16,bfloat16,float'
                require(types == expected and args[1:3] == ['NonTranspose','Transpose'], 'actual input types/layout')
                require([int(args[i]) for i in (3,4,5,8,10,13)] == [current['rows'],current['N'],current['K'],current['K'],current['K'],current['ld']], 'actual exact dimensions/strides')
                gpu = re.search(r'\bGPU(\d+)$', suffix)
                current['calls'].append(dict(line=line, GPU_index=int(gpu[1]) if gpu else None))
            elif line.startswith('PASS type='):
                require(current is not None and len(current['calls']) == 1 and 'analytic=FULL_EXACT guards=PASS inputs=PASS' in line, 'closed case and one matching call')
                current.update(correctness_line=line, GPU_reported=current['calls'][0]['GPU_index'] is not None)
                cases.append(current); current = None
        require(current is None and len(cases) == q['cases'], 'all expected calls observed')
        result['cases'][label] = cases
    result['all_observed_calls_GPU_reported'] = all(v['GPU_reported'] for group in result['cases'].values() for v in group)
    result['whole_model_fallback_excluded'] = False
    target = A / 'mkl-device-probe-assessment-v1.json'
    with target.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(cases=sum(map(len,result['cases'].values())),all_observed_calls_GPU_reported=result['all_observed_calls_GPU_reported'])), flush=True)


if __name__ == '__main__': main()
