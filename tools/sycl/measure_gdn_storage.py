#!/usr/bin/env python3
"""Own the oneMKL output-storage diagnostic and resident comparison; MIT.

Independent Main implementation, reusing the qualified owner/provider observer.
No model dispatch change. Default compute mode is a deliberate diagnostic arm.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_gdn_storage as built
import proc_observer
import runtime_probe as runtime
import ur_validation as validation

base, P, B, RESULT = built.base, built.P, built.B, built.RESULT
require, digest, write = base.require, base.digest, base.write
CONTROLLER = Path(__file__).resolve()
CHECK_KEY, SPEED_KEY = 'official_gdn_storage_device_v1', 'official_gdn_storage_speed_v1'
ARMS = ('oneMKL32', 'oneMKL16', 'oneMKL32_convert16')


def paths(speed):
    return (SPEED_KEY if speed else CHECK_KEY), built.ROOT / ('speed-v1' if speed else 'device-v1')


def save(root, key, row):
    write(root / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[key] = base.compact(row); write(RESULT, data)


def controls(row):
    require(base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'frozen measurement controller changed')
    require(all(digest(P / n) == sha for n, sha in row['native_sources'].items()), 'fixture changed')


def cases(speed):
    if speed:
        return [dict(name='host-canary', args=[], clean=False, trace=False),
                dict(name='clean-smoke', args=['--smoke'], clean=True, trace=False)] + [
            dict(name='resident-' + str(i), args=['--bench', str(i)], clean=True, trace=False)
            for i in range(3)]
    return [dict(name='host-canary', args=[], clean=False, trace=False),
            dict(name='storage-smoke', args=['--smoke'], clean=False, trace=True),
            dict(name='storage-general', args=['--validate'], clean=False, trace=False)]


def prerequisites(speed):
    build = built.verify()
    refs = dict(build=dict(path=str(built.ROOT / 'record.json'), sha256=digest(built.ROOT / 'record.json')))
    if speed:
        key, root = paths(False); row = json.loads((root / 'record.json').read_text())
        require(base.committed(RESULT)[key] == base.compact(row) and row['passed'] and row['complete']
                and not row['active'] and row['ownership_closed'] and row['fresh_faults'] == []
                and row['binary'] == build['binary'], 'closed same-build general gate required')
        controls(row)
        refs['device'] = dict(path=str(root / 'record.json'), sha256=digest(root / 'record.json'))
    gate = base.committed(RESULT)['official_prefill_strategies_device_v2']
    providers = built.runtime.inventory(gate['runtime_manifest'])
    return build, refs, gate, providers


def prepare(speed):
    key, root = paths(speed)
    require(key not in base.committed(RESULT) and not root.exists(), 'fresh frozen cohort only')
    build, refs, gate, _ = prerequisites(speed)
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    root.mkdir(mode=0o700)
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=CONTROLLER),
        native_sources={built.FIXTURE: digest(P / built.FIXTURE)}, prerequisite_records=refs,
        native_source_commit=build['source_commit'], binary=build['binary'],
        **{k: gate[k] for k in ('environment', 'canary', 'native_validation_probe', 'runtime_manifest', 'boot', 'driver')},
        cases=cases(speed), processes=[], commands_executed=[], active=False, ownership_closed=True,
        passed=False, complete=False, GPU_executed=False, model_executed=False, inference_adopted=False,
        budgets=dict(wall=480, cpu=1500, rss_cap=6 << 30, text_cap=8 << 20,
                     file_cap=64 << 20, total_cap=96 << 20), whole_wall_seconds=1800, whole_bytes=96 << 20,
        plan=dict(scope='Resident half-input oneMKL output-storage component only;not model PP,decode,PCIe,dequantization or physical262144.',
            compute_mode='Intentionally preserve production-style default compute_mode::unset in all three arms. FP16 C has exact half1/0 scalars;FP32 C has exact float1/0. This tests actual vendor default dispatch,not explicit-standard behavior or a guaranteed accumulator type. A separate prospective standard-mode variant is conditional on evidence,not required to observe this default API.',
            numerical='18general shapes plus4cancel,2zero,2onehot. All outputs finite,inputs/padding/guards unchanged. Bounded normal half inputs certify absolute dot sum<65504. Compare direct half to FP32 within2gamma32*Sbar plus half-round bound;64FP64 spots/case include final half rounding. This diagnostic is not model quality,subnormal/NaN/Inf coverage or a vendor-accumulator proof.',
            consumers='Only typed fixture output readers run. Production FP32 conv/z consumers,BF16 producer alternatives and GDN3/decode handoff remain outside this isolated comparison and mandatory before model adoption.',
            safety='Original in-order runtime and checked drain;copy offload/direct submission disabled;at most8GEMMs outstanding;no environment mutation after exec.',
            timing='Three fresh resident processes,start orders0/1/2;four4096/8192x6144/10240,K2560 shapes. First per-shape calls separate,six warmups,five rotated3-arm128-call batches. Copies,allocation,setup and validation outside timing. Checked host interval includes submission,8-call drains and final wait;no returned-MKL-event-as-whole-service assumption.',
            eligibility='Every steady batch>=100ms;median of5empty controls<=5%of every batch. No resizing or retry. Per-shape16bit follow-up requires all3independent median ratios<1 against each control;no arbitrary minimum gain. A component pass never directly authorizes a model trial.',
            providers='Use original84provider inventory. Clean mode removes observers before exec but requires actual libsycl,V2adapter,NEO maps;validated mode also requires preload/external URloader. Poll100ms with bounded /proc exit grace and positive pidfd retirement.',
            tracing='Pre-main validation canary,small traced smoke,warning-only general cohort. Record successful kernel creations/launch counts;names are evidence of dispatched kernels,not accumulator/range guarantees.',
            retention='Preserve individual speed samples,numeric maxima,source/runtime/closure and decisions. Retire successful full traces and closed raw logs after compact evidence verification.'))
    save(root, key, row)
    print('Prepared ' + key + ';commit before GPU execution.', flush=True)


def launch_census(text):
    names, launches, unmapped = {}, {}, 0
    for line in text.splitlines():
        if 'UR_RESULT_SUCCESS' not in line: continue
        if '<--- urKernelCreate(' in line:
            v = re.search(r'\.pKernelName = .*?\(([^()]+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
            if v: names[v[2]] = v[1]
        if not re.search(r'<--- urEnqueueKernelLaunch(?:WithArgsExp|CustomExp)?\(', line): continue
        v = re.search(r'\.hKernel = (0x[0-9a-f]+)', line)
        if not v or v[1] not in names: unmapped += 1; continue
        name = names[v[1]]; launches[name] = launches.get(name, 0) + 1
    require(launches and unmapped == 0, 'complete successful GPU kernel census required')
    return dict(launches=sum(launches.values()), per_kernel=launches, unmapped=unmapped)


def analyze(case, stdout, folder, pid):
    if case['name'] == 'host-canary':
        require(json.loads(stdout) == dict(pid=pid, initialized=0, checked=46, expected=46,
            version=(1 << 32)-1, closed=0), 'host canary differs')
        return dict(host_only=True)
    require(stdout.count('fixture_status=PASS;resident storage diagnostic only') == 1, 'fixture incomplete')
    rows = [json.loads(v) for v in stdout.splitlines() if v.startswith('{')]
    actual = [v for v in rows if v['kind'] == 'validation']
    if case['name'] in ('storage-smoke', 'clean-smoke'): expected = [(1,6144,'general'), (129,10240,'general')]
    elif case['name'] == 'storage-general':
        expected = [(t,n,'general') for t in (1,127,128,129,255,256,257,8191,8192) for n in (6144,10240)] + \
            [(t,n,'cancel') for t in (1,129) for n in (6144,10240)] + \
            [(1,6144,'zero'),(257,10240,'zero'),(129,6144,'onehot'),(8191,10240,'onehot')]
    else: expected = [(t,n,'general') for t in (4096,8192) for n in (6144,10240) for _ in range(2)]
    require([(v['T'],v['N'],v['pattern']) for v in actual] == expected and all(
        v['passed'] and v['guards_inputs'] and v['FP64_spots'] == 64 and v['outputs'] == v['T']*v['N'] and
        v['ldy'] == v['N']+16 and 0 <= v['max_comparator_bound_fraction'] <= 1 and
        0 <= v['max_FP64_bound_fraction'] <= 1 for v in actual), 'numerical coverage differs')
    if case['name'].startswith('resident-'):
        timing = [v for v in rows if v['kind'] == 'timing']; require(len(timing) == 92, 'timing rows missing')
        order = int(case['args'][1])
        for t in (4096,8192):
            for n in (6144,10240):
                group = [v for v in timing if v['T'] == t and v['N'] == n]
                plan = [('first_shape_call', ARMS[(i+order)%3], 0, 1) for i in range(3)] + \
                    [('timer_control','empty',i,128) for i in range(5)] + \
                    [('steady',ARMS[(i+j+order)%3],j,128) for j in range(5) for i in range(3)]
                require([(v['phase'],v['arm'],v['sample'],v['calls']) for v in group] == plan and all(
                    math.isfinite(v['host_ms']) and v['host_ms'] > 0 for v in group), 'frozen timing plan differs')
    result = dict(rows=rows)
    if case['trace']: result['kernel_census'] = launch_census((folder / 'ur.trace').read_text())
    return result


def decisions(row):
    outcomes = []
    for t in (4096,8192):
        for n in (6144,10240):
            medians = []
            for p in row['processes']:
                if not p['name'].startswith('resident-'): continue
                group = [v for v in p['details']['rows'] if v['kind'] == 'timing' and v['T'] == t and v['N'] == n]
                empty = statistics.median(v['host_ms'] for v in group if v['phase'] == 'timer_control')
                steady = [v for v in group if v['phase'] == 'steady']
                med = {arm: statistics.median(v['host_ms']/v['calls'] for v in steady if v['arm'] == arm) for arm in ARMS}
                medians.append(dict(process=p['name'], milliseconds=med, empty_control_ms=empty,
                    batches_eligible=all(v['host_ms'] >= 100 and empty <= .05*v['host_ms'] for v in steady),
                    ratios={control: med['oneMKL16']/med[control] for control in ('oneMKL32','oneMKL32_convert16')}))
            require(len(medians) == 3, 'three complete independent processes required')
            eligible = all(v['batches_eligible'] for v in medians)
            ratios = {control: math.exp(statistics.mean(math.log(v['ratios'][control]) for v in medians))
                      for control in ('oneMKL32','oneMKL32_convert16')}
            outcomes.append(dict(T=t,N=n,process_medians=medians,all_batches_eligible=eligible,
                paired_geomean_time_ratios=ratios,follow_up_eligible=eligible and
                all(r < 1 for v in medians for r in v['ratios'].values())))
    return outcomes


def run(speed):
    key, root = paths(speed); row = json.loads((root / 'record.json').read_text())
    require(base.committed(RESULT)[key] == base.compact(row) and row['stage'] == 'prepared' and
            not (root / 'owned').exists(), 'one frozen fresh cohort only')
    controls(row); build, refs, gate, providers = prerequisites(speed)
    require(row['binary'] == build['binary'] and row['prerequisite_records'] == refs and
            row['cases'] == cases(speed) and all(row[k] == gate[k] for k in
            ('environment','canary','native_validation_probe','runtime_manifest','boot','driver')), 'cohort envelope changed')
    frozen, owner, active, mapped_all = copy.deepcopy(row), None, None, set()
    cursor = next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    deadline = time.monotonic()+row['whole_wall_seconds']
    def persist():
        write(root / 'execution.json', dict(active_case=active, commands=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    try:
        owner = base.supervision.Owner(root / 'owned', persist)
        for case in row['cases']:
            active = case['name']; folder = root / active; folder.mkdir(mode=0o700)
            for part in ('tmp','cache'): (folder / part).mkdir()
            env = dict(row['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),OMP_NUM_THREADS='1')
            if case['clean']:
                for name in tuple(env):
                    if name.startswith('UR_LOG_') or name in ('UR_ENABLE_LAYERS','LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER'): del env[name]
            else:
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                    UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'ur.trace'),
                    UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
            asset = row['canary'] if active == 'host-canary' else row['binary']
            mapped, env_seen, observer = set(), False, proc_observer.ProcObserver()
            def observe(pid):
                nonlocal env_seen
                raw = Path('/proc',str(pid),'environ').read_bytes()
                if not raw: raise ProcessLookupError('empty retiring environment')
                actual = dict(v.split(b'=',1) for v in raw.split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode()) == v.encode() for k,v in env.items()) and
                        set(actual)-{k.encode() for k in env} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'native environment differs')
                seen = runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                if not seen: raise ProcessLookupError('empty retiring maps')
                require(seen <= set(providers) | {str(Path(asset['path']).resolve())},
                        'unplanned executable provider: '+repr(seen-set(providers)))
                mapped.update(seen); env_seen = True
            def monitor(io, entry):
                io.proc.stdin.close(); next_sample = 0.
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll(); require(time.monotonic() < deadline, 'whole cohort deadline')
                    if time.monotonic() >= next_sample:
                        observer.sample(lambda: observe(io.proc.pid),lambda: bool(io.exit_watch.poll(0)))
                        require(base.supervision.artifact_bytes(root) <= row['whole_bytes'], 'whole cohort byte budget')
                        next_sample = time.monotonic()+.1
                    if select.select([io.proc.stdout],[],[],.01)[0]: io.read();io.pending.clear()
                io.exit(1)
            device = active != 'host-canary'
            row['GPU_execution_requested'] = row.get('GPU_execution_requested',False) or device
            entry,out,err = owner.run(active,[asset['path'],*case['args']],env,P,interaction=monitor,**row['budgets'])
            stdout,stderr = out.read_text(errors='replace'),err.read_text(errors='replace')
            require(base.supervision.closed(entry) and entry['exit_code'] == 0, 'native process failed '+active+': '+stderr[-3000:])
            if not case['clean']:
                activation = validation.preload_evidence(stderr,entry['pid'])
                require(activation['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'], 'wrong pre-main loader')
                require(not (folder/'validation.log').exists() or (folder/'validation.log').stat().st_size == 0,
                        'parameter validation not quiet')
            details = analyze(case,stdout,folder,entry['pid'])
            if device:
                required = {env['UR_ADAPTERS_FORCE_LOAD'],env['ZE_ENABLE_ALT_DRIVERS'],
                            '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'}
                if not case['clean']: required.update((env['LD_PRELOAD'],env['STRATA_UR_PROBE_EXPECTED_LOADER']))
                if not env_seen or not required <= mapped:
                    row['failed_process_observation'] = dict(name=active,environment_observed=env_seen,
                        mapped=sorted(mapped),missing=sorted(required-mapped),exit_code=entry['exit_code'])
                require(env_seen and required <= mapped, 'actual provider observation incomplete: '+repr(required-mapped))
                row['GPU_executed'] = True
            check = dict(host_probe_accepted=True); validation.finish(check,owner,cursor,row['boot'])
            require(check['passed'] and check['driver_after'] == row['driver'], 'fault/ownership gate failed')
            row['processes'].append(dict(name=active,passed=True,exit_code=0,clean_speed=case['clean'],
                pre_main_parameter_validation=not case['clean'],environment_observed=env_seen,
                proc_observation=observer.summary(),ownership_closed=True,details=details,
                environment_sha256=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()))
            mapped_all.update(mapped);persist();print('Passed '+active,flush=True)
        controls(frozen); prerequisites(speed)
        require(base.committed(RESULT)[key] == base.compact(frozen), 'frozen comparison plan changed')
        if speed: row['component_decisions'] = decisions(row)
        row.update(stage='resident-speed-closed' if speed else 'general-check-passed',complete=True,host_probe_accepted=True)
    except BaseException as error:
        row.update(stage='failed',complete=False,host_probe_accepted=False,failed_case=active,error=repr(error))
    finally:
        if owner: validation.finish(row,owner,cursor,row['boot'])
        if row.get('driver_after') != row['driver']: row['passed'] = False
        if not row.get('passed'): row['stage'] = 'failed'
        if row.get('GPU_execution_requested') and not row['GPU_executed']: row['GPU_executed'] = None
        row.update(active=bool(owner and owner.active),observed_provider_count=len(mapped_all),
            ordinary_runtime_teardown_qualified=False,commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None, 'GPU process ownership unresolved')
        save(root,key,row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','GPU_executed','fresh_faults','error','component_decisions')}),flush=True)
    require(row['passed'], 'GDN storage cohort failed')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare-check','check','prepare-speed','speed'),
            'use prepare-check|check|prepare-speed|speed')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1].startswith('prepare-'): prepare(sys.argv[1] == 'prepare-speed')
        else: run(sys.argv[1] == 'speed')
