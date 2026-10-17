#!/usr/bin/env python3
"""Sequential independent-process component comparison; never retry a failed GPU job."""
import datetime, hashlib, json, statistics, subprocess, sys
from pathlib import Path
A = Path(__file__).resolve().parent
P = A.parents[2]

def pin(p):
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def main():
    out = A / 'qsa-score-gemm-cohort-v2.json'
    assert not out.exists()
    runner = A / 'run_qsa_score_gemm_probe_v2.py'
    build = A / 'qsa-score-gemm-probe-build-v2.json'
    b = json.loads(build.read_text())
    assert b['passed'] and b['complete'] and b['ownership_closed']
    planned = []
    for mode in ('ctx32768', 'ctx78453', 'full'):
        for rep in range(3):
            for arm in (('gemm', 'head') if rep % 2 == 0 else ('head', 'gemm')):
                planned.append(dict(mode=mode, rep=rep, backend=arm))
    # Signed-zero/tie and physical-limit partial-query cases are correctness,
    # not extra independent timing samples in the main cohort.
    for mode in ('zero', 'tail'):
        for arm in ('gemm', 'head'):
            planned.append(dict(mode=mode, rep=0, backend=arm))
    rec = dict(schema=2, passed=False, complete=False, active=False,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               runner=pin(runner), controller=pin(Path(__file__)), build=pin(build),
               binary=b['binary'], planned=planned, runs=[],
               scope='Same-binary private GEMM versus public FP32 query8/head1 control. Three independent processes per arm/context, alternating order by repetition; each context uses its complete declared sample set. Zero/tail cases are separate. All GPU and resource closure checks remain in the individually locked Main runner. No model-quality or model-PP qualification.',
               production_selected=False, model_quality_qualified=False,
               hardware_utilization_qualified=False)
    def save():
        out.write_text(json.dumps(rec, indent=2) + '\n')
    save()
    try:
        for trial in planned:
            assert pin(runner) == rec['runner'] and pin(build) == rec['build']
            arm, mode, rep = trial['backend'], trial['mode'], trial['rep']
            receipt = A / f'qsa-score-gemm-probe-{arm}-{mode}-r{rep}-v2.json'
            assert not receipt.exists(), 'Never restart an existing attempt'
            argv = [sys.executable, str(runner), mode, str(rep), arm]
            rec.update(active=True, current=dict(trial, argv=argv))
            save()
            # Native process deadlines and owner cleanup are enforced inside
            # the runner. Do not kill its supervisor with a shorter timeout.
            result = subprocess.run(argv, cwd=P, check=False)
            rec['active'] = False
            assert receipt.exists(), 'Missing authoritative native receipt'
            r = json.loads(receipt.read_text())
            row = dict(trial, argv=argv, runner_exit=result.returncode,
                       receipt=pin(receipt), passed=r['passed'], complete=r['complete'],
                       ownership_closed=r['ownership_closed'], fresh_faults=r.get('fresh_faults'),
                       samples=r.get('samples'), quality=r.get('quality'), cases=r.get('cases'))
            row['stage_medians_ms'] = {
                stage: statistics.median(float(x['total_wall_ms']) for x in r['samples'] if x['stage'] == stage)
                for stage in ('score', 'topk', 'pair')
            } if r.get('samples') else {}
            rec['runs'].append(row)
            save()
            print(json.dumps({**trial, 'passed':r['passed'], 'stage_medians_ms':row['stage_medians_ms']}), flush=True)
            assert result.returncode == 0 and r['passed'] and r['complete'] and r['ownership_closed'] and not r.get('active') and r.get('fresh_faults') == [] and not r['driver_dump_entries']
        rec.update(passed=True, complete=True)
    except BaseException as error:
        import traceback
        rec.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        rec['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    return 0 if rec['passed'] else 1

if __name__ == '__main__':
    sys.exit(main())
