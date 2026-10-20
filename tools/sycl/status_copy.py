#!/usr/bin/env python3
"""Main-owned completed-event queue/observer diagnostic; no model run.

Commit source, prepare/commit the generated recipe, build/commit binary identity,
then run two traced smoke checks and twelve fixed measurement processes once.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import query_profile as q
import ur_validation as validation

gen, guards, supervision = q.gen, q.guards, q.supervision
P, B, A = q.P, q.B, q.A
ROOT = B / 'status-query-observer-v3'
RESULT = A / 'status-copy-current.json'
FIXTURE = 'bench/sycl/status_copy.cpp'
CELLS = [('profiled', 'trace'), ('plain', 'trace'), ('plain', 'quiet'), ('profiled', 'quiet')]
ORDERS = [CELLS, list(reversed(CELLS)), CELLS[2:] + CELLS[:2]]


def digest(path):
    return supervision.sha(path)


def inputs(source):
    pins = gen.benchmark_pins(source)
    raw = gen.git('show', source + ':' + FIXTURE)
    guards.require(raw == (P / FIXTURE).read_bytes(), 'commit exact fixture input')
    pins[FIXTURE] = hashlib.sha256(raw).hexdigest()
    return pins


def runtime_check():
    settings = gen.committed(gen.SETTINGS)
    guards.require(set(q.DELTA) <= set(settings['runtime_elfs']), 'unknown qualified runtime delta')
    for path, item in settings['runtime_elfs'].items():
        expected = item['sha256']
        if path in q.DELTA:
            original, expected = q.DELTA[path]
            guards.require(original == item['sha256'], 'original baseline differs')
        guards.require(digest(path) == expected, 'runtime changed: ' + path)
    return settings


def admission(record):
    full, count = q.current_build()
    guards.require(full['source_commit'] == record['native_source_commit'] and
        digest(q.BUILD_RECORD) == record['native_build_record_sha256'] and
        count == record['native_inputs_recovered'], 'native source/build changed')
    pins = inputs(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and
        q.assets.fingerprint(pins) == record['source_fingerprint'] and
        digest(full['compiler']['path']) == record['compiler']['sha256'], 'fixture/controller/compiler changed')
    runtime_check()
    guards.require(digest(gen.SETTINGS) == record['settings_sha256'] and
        validation.preload_qualification() == record['validation_probe'], 'runtime/helper proof changed')
    guards.require(record['native_environment'].get('STRATA_UR_PROBE_EXPECTED_LOADER') ==
        record['validation_probe']['loader']['path'] and
        record['native_environment'].get('LD_PRELOAD') == record['validation_probe']['library']['path'],
        'configure exact native-child validation provider before execution')


def parse_output(text, profiling, mode):
    rows = [json.loads(line) for line in text.splitlines()]
    rounds, queries = (1, 16) if mode == 'smoke' else (4, 4096)
    guards.require(len(rows) == rounds + 2, 'missing/extra protocol rows')
    begin = rows[0]
    expected = dict(type='begin', schema=3, profiling=profiling == 'profiled', mode=mode,
        rounds=rounds, queries_per_batch=queries, completed_before_batch=True)
    guards.require(set(begin) == set(expected) | {'clock_pair_cpu_s','clock_pair_wall_s'} and
        all(type(begin[k]) is type(v) and begin[k] == v for k,v in expected.items()), 'begin protocol differs')
    def duration(value):
        return type(value) in (int,float) and math.isfinite(value) and value >= 0
    guards.require(all(duration(begin[k]) for k in ('clock_pair_cpu_s','clock_pair_wall_s')), 'invalid clock reference')
    samples = rows[1:-1]
    for rep, row in enumerate(samples):
        expected = dict(type='sample', rep=rep, queries=queries, completed=queries, payload_and_guards_passed=True)
        guards.require(set(row) == set(expected) | {'cpu_s','wall_s','timing_eligible'} and
            all(type(row[k]) is type(v) and row[k] == v for k,v in expected.items()) and
            duration(row['cpu_s']) and duration(row['wall_s']), 'invalid sample/count/clock/guards')
        eligible = bool(mode == 'measure' and row['wall_s'] >= 20e-6 and
            row['wall_s'] >= 100 * begin['clock_pair_wall_s'] and
            row['cpu_s'] >= 100 * begin['clock_pair_cpu_s'])
        guards.require(type(row['timing_eligible']) is bool and row['timing_eligible'] == eligible,
            'timer eligibility mismatch')
    guards.require(rows[-1] == dict(type='end', passed=True, owners_closed=True) and
        type(rows[-1]['passed']) is bool and type(rows[-1]['owners_closed']) is bool,
        'incomplete native closure')
    return begin, samples


def prepare():
    guards.require(not ROOT.exists(), 'inspect prior job; no retry')
    previous = gen.committed(RESULT)
    guards.require(previous['stage']=='closed' and previous['complete'] and previous['passed'] and
        previous['ownership_closed'] and previous['fresh_faults']==[] and not previous['active'] and
        previous['pending_comparison_qualified'] is False, 'preserve the closed ineligible v2 component')
    previous_raw = gen.git('show', 'HEAD:' + str(RESULT.relative_to(P)))
    previous_ref = dict(commit=gen.git('rev-parse','HEAD').decode().strip(), path=str(RESULT.relative_to(P)),
        sha256=hashlib.sha256(previous_raw).hexdigest(), pending_comparison_qualified=False,
        outcome='v2 execution passes; both pending cells0/12; no pending comparison, model bottleneck or speedup qualified.')
    full, count = q.current_build()
    settings = runtime_check()
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = inputs(source)
    compiler = full['compiler']
    guards.require(digest(compiler['path']) == compiler['sha256'], 'compiler differs')
    env = dict(full['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    native = dict(settings['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'),
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
        CUDA_MODULE_LOADING='EAGER', STRATA_SYCL_COMMAND_PROFILE='0')
    probe = validation.preload_qualification()
    native.update(LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
    executable = ROOT / 'status_copy'
    command = [compiler['path'], '-O3', '-DNDEBUG', '-std=c++20', '-fsycl',
        '-fsycl-id-queries-range=size_t', '-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(P/'include'), '-I'+str(P/'include/strata/sycl/compat'), str(P/FIXTURE),
        *[str(P/('src/sycl/'+name+'.cpp')) for name in ('runtime','compat','command_profile')], '-o',str(executable)]
    providers = ('libsycl.so','libur_loader.so','libur_adapter_level_zero_v2.so','libze_loader.so','libze_intel_gpu.so')
    loaded_paths = [str(Path(p).resolve()) for p in settings['runtime_elfs']
        if any(Path(p).name.startswith(name) for name in providers)]
    guards.require(len(loaded_paths)==len(set(loaded_paths))==5,'ambiguous expected providers')
    record = dict(schema='strata-status-query-observer-v3', previous_component=previous_ref, stage='prepared',
        source_commit=source, source_count=len(pins), source_fingerprint=q.assets.fingerprint(pins),
        native_source_commit=full['source_commit'], native_inputs_recovered=count,
        native_build_record_sha256=digest(q.BUILD_RECORD), compiler=compiler,
        settings_sha256=digest(gen.SETTINGS), compile_argv=command, compile_environment=env,
        native_environment=native, validation_probe=probe, expected_loaded_paths=sorted(loaded_paths),
        smoke_cells=[['profiled','trace'],['plain','trace']], process_orders=ORDERS,
        gates='Two traced smoke processes first, then three fixed four-cell cohorts. Each measured process:4 rounds x4096 queries of a finite event AFTER checked queue completion. Exact complete counts and guarded payload required. Each batch must exceed20us wall and100x median128 clock-pair CPU/wall overhead. Below-floor batches remain ineligible, no adaptive enlargement/retry. Execution and timing eligibility are separate. Same-context in-order Runtime::make_queue profiling true/false, no production default change. Trace present/absent in separate immutable child environments; parameter validation/safety fixed. Exact provider maps, first-init helper, normal exit/owned closure/source/fault checks required.',
        budgets=dict(compile_wall_s=240, native_process_wall_s=45, text_bytes_per_command=16<<20,
            total_raw_bytes=64<<20, device_bytes=12, measured_processes=12, samples=48, queries_per_sample=4096),
        limits='Completed-event host API overhead only. No pending-progress gate, model engine rebuild, speedup adoption or physical traffic/saturation claim. The cached/runtime path may differ from v2 fresh finite producer and the model copy enqueue path; record trace call counts rather than assume UR dispatch for every query. Separate further qualification required for queue-default/elapsed-time changes.',
        retained_generations='396 fastest fallback and3b status engine unchanged; this standalone fixture replaces closed v2 after source/recipe/consumer retirement. No third engine generation.',
        retention='Keep48 timing samples, eligibility, exact provider/API counts, source/recipe and owner/fault decision. Retire successful raw traces and standalone build after useful evidence and exact input recovery.',
        complete=False, passed=False, active=False, GPU_executed=False)
    RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(prepared=True,source=source,source_count=len(pins),GPU_executed=False)))


def execute(build):
    record = gen.committed(RESULT)
    guards.require(record['stage'] == ('prepared' if build else 'built'), 'closed/mismatched stage; no retry')
    admission(record)
    if build:
        guards.require(not ROOT.exists(), 'build root exists; inspect handle')
        ROOT.mkdir(mode=0o700)
        for name in ('tmp','cache'): (ROOT/name).mkdir()
    else:
        guards.require(digest(record['binary']['path'])==record['binary']['sha256'], 'executable differs')
        guards.require(not (ROOT/'run.json').exists(), 'run attempted; no retry')
    owner = None
    output = ROOT / ('build.json' if build else 'run.json')
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        output.write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner = supervision.Owner(ROOT/('owned-build' if build else 'owned-run'), persist)
    record.update(complete=False,passed=False)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor = None
    persist()
    try:
        if build:
            entry,out,err = owner.run('compile',record['compile_argv'],record['compile_environment'],P,
                wall=240,cpu=240,rss_cap=4<<30,text_cap=16<<20,file_cap=128<<20,total_cap=1<<30)
            guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                'compile failed: '+err.read_text()[-4096:])
            executable = ROOT/'status_copy'
            record.update(stage='built',binary=dict(path=str(executable),bytes=executable.stat().st_size,sha256=digest(executable)))
        else:
            guards.driver_guard()
            journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
            cursor = next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
            record.update(GPU_executed=True,cells_results=[],smoke_results=[])
            planned = [('smoke',-1,i,cell) for i,cell in enumerate(record['smoke_cells'])] + [
                ('measure',cohort,index,cell) for cohort,order in enumerate(record['process_orders']) for index,cell in enumerate(order)]
            for mode,cohort,index,cell in planned:
                runtime_check(); guards.driver_guard()
                env = dict(record['native_environment'],UR_ENABLE_LAYERS=(
                    'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if cell[1]=='trace' else 'UR_LAYER_PARAMETER_VALIDATION'))
                entry,out,err = owner.run(f'{mode}{cohort}-{index}',[record['binary']['path'],cell[0],mode],env,P,
                    wall=45,cpu=45,rss_cap=2<<30,text_cap=16<<20,file_cap=16<<20,total_cap=64<<20)
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,'cell failed: '+err.read_text()[-2048:])
                begin,samples = parse_output(out.read_text(),cell[0],mode)
                log = err.read_text(); proof = validation.preload_evidence(log,entry['pid'])
                guards.require(proof['loader']==record['validation_probe']['loader']['path'] and
                    'strata UR validation failure:' not in log,'validation provider/error')
                paths = re.findall(r'^status query loaded: (.+)$',log,re.M)
                guards.require(sorted(paths)==record['expected_loaded_paths'],'actual runtime provider mismatch')
                api_counts = {name:len(re.findall(r'^\s*---> '+name+r'\s*$',log,re.M)) for name in
                    ('urEventGetInfo','urDeviceGetGlobalTimestamps','urEventRelease','urQueueFinish',
                     'urEnqueueUSMFill','urEnqueueUSMMemcpy')}
                guards.require((api_counts['urEnqueueUSMFill']>0)==(cell[1]=='trace'),
                    'trace presence/absence mismatch')
                result = dict(cell=cell,cohort=cohort,order_index=index,clock_reference=begin,samples=samples,
                    timing_qualified=all(s['timing_eligible'] for s in samples),api_counts=api_counts,
                    native_pid=entry['pid'],validation_activation=proof,loaded_paths=paths,native_exit=0,ownership_closed=True)
                if mode=='measure':
                    result['median_cpu_us_per_query']=statistics.median(s['cpu_s']/4096*1e6 for s in samples)
                    result['median_wall_us_per_query']=statistics.median(s['wall_s']/4096*1e6 for s in samples)
                    record['cells_results'].append(result)
                else: record['smoke_results'].append(result)
                persist()
            record.update(stage='closed',diagnostic_timing_qualified=all(r['timing_qualified'] for r in record['cells_results']))
        admission(record)
        record.update(complete=True,passed=True)
    except BaseException as error: record['error']=str(error)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['passed']=bool(record['passed'] and record['ownership_closed'])
        if not build and cursor is not None:
            record.update(fresh_faults=None,boot_unchanged=False)
            try:
                text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                record['fresh_faults']=[x for x in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
                record['driver_after']=guards.driver_guard()
                record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            except Exception as error: record['final_probe_error']=str(error)
            record['passed']=bool(record['passed'] and record['fresh_faults']==[] and record['boot_unchanged'] and 'final_probe_error' not in record)
        persist(); RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('stage','complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','build','run'))
    action=parser.parse_args().action
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        if action=='prepare': return prepare()
        return execute(action=='build')


if __name__=='__main__': sys.exit(main())
