#!/usr/bin/env python3
"""Serial independent-process component comparison; no model throughput claim."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    dst = A/'qsa-bf16-prefix-probability-cohort-v1.json'
    assert not dst.exists()
    child = A/'run_qsa_bf16_prefix_probability_bench_v1.py'
    digest = sha(child)
    order = [('base',0),('parallel',0),('parallel',1),('base',1),('base',2),('parallel',2)]
    rec = dict(schema='strata-prefix-probability-component-cohort-v1', passed=False, complete=False,
               order=order, child=dict(path=str(child),sha256=digest), completed=[], active_child_pid=None,
               scope='Component8192 queries/width2051/physical32768 fully initialized. Each process2warmups/10samples; six processes, same binary/inputs/allocations, prefix pack inside every call. Host time, not whole model PP or hardware utilization; no numerical/full-model256K/RESTORE adoption.')
    def persist():
        dst.write_text(json.dumps(rec,indent=2)+'\n')
    try:
        for arm,n in order:
            assert sha(child) == digest
            mode = f'{arm}-32768-{n}'
            path = A/f'own-qsa-bf16-prefix-probability-bench-{mode}-v1.json'
            assert not path.exists()
            print(json.dumps(dict(start=mode)),flush=True)
            proc = subprocess.Popen([sys.executable,'-B',str(child),mode],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            rec['active_child_pid'] = proc.pid
            persist()
            output,_ = proc.communicate(timeout=300)
            rec['active_child_pid'] = None
            assert proc.returncode == 0, output[-4000:]
            value = json.loads(path.read_text())
            assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
            assert value['fresh_faults'] == [] and not value['driver_dump_entries'] and value['boot_unchanged']
            assert not value['traced'] and value['nq'] == 8192 and value['cells'] == 32768
            assert 'UR_ENABLE_LAYERS' not in value['native_environment'] and len(value['samples']) == 10
            samples = [float(x['whole_wall_ms']) for x in value['samples']]
            assert all(math.isfinite(x) and x > 0 for x in samples)
            case, = value['cases']
            evidence = dict(x.split('=',1) for x in case.split()[1:])
            assert all(evidence[k] == 'PASS' for k in ['status','guards','inputs','full_finite','repeat_bits'])
            rec['completed'].append(dict(arm=arm,repetition=n,path=str(path),sha256=sha(path),
                 binary_sha256=value['binary']['sha256'],frame=value['frame'],case=evidence,
                 whole_wall_ms=samples,median_ms=statistics.median(samples)))
            persist()
            print(json.dumps(dict(completed=mode,median_ms=statistics.median(samples))),flush=True)
        assert len({x['binary_sha256'] for x in rec['completed']}) == 1
        assert len({x['case']['output_FNV1a64'] for x in rec['completed']}) == 1
        for k in ['q','k','v','ks','vs','table','ids','steps','device_bytes','packed_scratch_payload_bytes','pack_cost']:
            assert len({x['frame'][k] for x in rec['completed']}) == 1, k
        assert all(x['frame']['pack_cost'] == 'IN_TIMED_CALL' for x in rec['completed'])
        stats = {}
        for arm in ['base','parallel']:
            medians = [x['median_ms'] for x in rec['completed'] if x['arm'] == arm]
            stats[arm] = dict(independent_processes=3,process_medians_ms=medians,
                             median_ms=statistics.median(medians),mean_ms=statistics.mean(medians),
                             sample_sd_ms=statistics.stdev(medians),min_ms=min(medians),max_ms=max(medians))
        rec.update(statistics=stats,parallel_to_base_latency_ratio=stats['parallel']['median_ms']/stats['base']['median_ms'],
                   passed=True,complete=True,source_recipe_sha256=sha(__file__),
                   equality_scope='Complete output FNV1a64 fingerprints match and sampled independent oracle passes. No saved full-frame byte comparator; hash equality is not represented as a direct byte comparison.')
    except BaseException as error:
        rec['error'] = repr(error)
        raise
    finally:
        persist()
    print(json.dumps(dict(passed=True,statistics=stats,latency_ratio=rec['parallel_to_base_latency_ratio'])),flush=True)


if __name__ == '__main__':
    main()
