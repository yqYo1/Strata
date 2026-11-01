#!/usr/bin/env python3
"""Own the general-input GDN oracle and matched resident GEMM comparison. MIT.

Original exact fixture/qualification inputs stay immutable. This independent
fixture links the same admitted native AOT objects and one unchanged Runtime.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import check_gdn_projection_device as qualified

image, base, runtime = qualified.image, qualified.base, qualified.runtime
require, digest, write = base.require, base.digest, base.write
RESULT = base.RESULT
ROOT = image.ROOT / 'resident-bench-v1'
BUILD_KEY = 'official_gdn_fp32_resident_fixture_build_v1'
CHECK_KEY = 'official_gdn_fp32_general_check_v1'
SPEED_KEY = 'official_gdn_fp32_resident_speed_v1'
FIXTURE = 'bench/sycl/gdn_projection_fp32_bench.cpp'
CONTROLLER = Path(__file__).resolve()
qualified.configure('tails-v2')


def save(root, key, row):
    write(root / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[key] = base.compact(row); write(RESULT, data)


def prior():
    row = json.loads((qualified.ROOT / 'record.json').read_text())
    require(row['passed'] and row['complete'] and row['ownership_closed'] and
            not row['active'] and row['fresh_faults'] == [], 'closed original exact-shape gate required')
    return row, qualified.current(row)


def source_controls(row):
    require(base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'measurement controller inputs changed')
    for name, sha in row['native_sources'].items():
        require(digest(base.P / name) == sha, 'measurement fixture changed')


def allowed_inputs():
    proof = json.loads(image.RECORD.read_text())
    original = image.admission(proof)
    allowed = image.build.admit(original)
    full, _ = image.build.source.sources()
    for group in full['dependencies']:
        for name, sha in group['source_sha256'].items():
            path = str((Path(group['root']) / name).resolve())
            require(digest(path) == sha, 'pinned dependency changed')
            allowed[path] = sha
    allowed[str(base.P / FIXTURE)] = digest(base.P / FIXTURE)
    for name in ('image.o', 'host.o'):
        item = proof['outputs'][name]
        require(digest(item['path']) == item['sha256'], 'original native object changed')
        allowed[item['path']] = item['sha256']
    return allowed


def prepare_build():
    data = base.committed(RESULT)
    require(BUILD_KEY not in data and not ROOT.exists(), 'new resident fixture build only')
    old, _ = prior()
    original = json.loads((qualified.built.ROOT / 'record.json').read_text())
    revision = base.basis.build.git(base.P, 'rev-parse', 'HEAD').decode().strip()
    sha = hashlib.sha256(base.basis.build.git(base.P, 'show', revision + ':' + FIXTURE)).hexdigest()
    require(digest(base.P / FIXTURE) == sha, 'commit exact benchmark fixture')
    commands = copy.deepcopy(original['commands'])
    for _, argv in commands:
        for i, arg in enumerate(argv):
            argv[i] = arg.replace(str(qualified.built.ROOT), str(ROOT))
        if '-c' in argv and argv[argv.index('-c') + 1].endswith('/gdn_projection_fp32_device.cpp'):
            argv[argv.index('-c') + 1] = str(base.P / FIXTURE)
    commands[-1][1][-1] = str(ROOT / 'gdn-fp32-bench')
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache', 'build'): (ROOT / part).mkdir()
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=base.controls(revision, controller=CONTROLLER), native_sources={FIXTURE:sha},
        original_fixture_recipe=dict(path=str(qualified.built.ROOT / 'record.json'),
            sha256=digest(qualified.built.ROOT / 'record.json')),
        original_native_proof=dict(path=str(image.RECORD), sha256=digest(image.RECORD)),
        original_device_proof=dict(path=str(qualified.ROOT / 'record.json'), sha256=digest(qualified.ROOT / 'record.json')),
        native_implementation_commit=original['native_implementation_commit'], commands=commands,
        environment=dict(original['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache')),
        budgets=dict(wall=300, cpu=900, rss_cap=5 << 30, text_cap=16 << 20, file_cap=256 << 20, total_cap=1 << 30),
        plan=dict(scope='CPU compile/link only;independent oracle and timing fixture,same AOT generation.',
            admission='Exact production fixture/Runtime flags,actual compiler and host linker inputs pinned;no GPU execution.',
            retained='Selected eight-ELF engine remains fallback;current GDN objects/exact fixture remain qualification inputs. This fixture is the imminent same-generation comparison consumer.'),
        active=False, ownership_closed=True, passed=False, GPU_executed=False, commands_executed=[])
    require(old['binary'] == original['binary'], 'original exact fixture identity changed')
    save(ROOT, BUILD_KEY, row)
    print('Prepared CPU build;commit the generated recipe before build.', flush=True)


def build():
    row = json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[BUILD_KEY] == base.compact(row) and row['stage'] == 'prepared'
            and not (ROOT/'owned-build').exists(), 'one committed fresh build required')
    source_controls(row); prior()
    for field in ('original_fixture_recipe', 'original_native_proof', 'original_device_proof'):
        asset = row[field]; require(digest(asset['path']) == asset['sha256'], 'build prerequisite changed')
    allowed = allowed_inputs(); owner = None
    def persist():
        write(ROOT/'record.json', dict(row, commands_executed=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    def monitor(io, entry):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(base.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'build byte budget')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        row['stage'] = 'building'; owner = base.supervision.Owner(ROOT/'owned-build', persist)
        for label, argv in row['commands']:
            entry, out, err = owner.run(label, argv, row['environment'], ROOT, interaction=monitor, **row['budgets'])
            require(base.supervision.closed(entry) and entry['exit_code'] == 0,
                    label+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-4000:])
        found = set()
        for label in ('fixture-object', 'runtime-object'):
            for line in (ROOT/(label+'.d')).read_text().replace('\\\n', '').splitlines():
                if not line.strip(): continue
                left, colon, right = line.partition(':'); require(colon and left.strip(), 'invalid compiler depfile')
                found.update(str((ROOT/p).resolve()) for p in shlex.split(right))
        require(all(allowed.get(p) == digest(p) for p in found), 'uncovered actual compiler input')
        write(ROOT/'compiler-inputs.json', {p:allowed[p] for p in sorted(found)})
        row['actual_compiler_inputs'] = dict(path=str(ROOT/'compiler-inputs.json'),
            sha256=digest(ROOT/'compiler-inputs.json'), files=len(found), uncovered=0)
        row['actual_link_inputs'] = base.traced_link_inputs(allowed, ROOT)
        binary = ROOT/'gdn-fp32-bench'
        row.update(stage='built-not-executed', passed=True,
            binary=dict(path=str(binary), sha256=digest(binary), bytes=binary.stat().st_size))
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active),
            ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)),
            commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None, 'build ownership unresolved'); save(ROOT, BUILD_KEY, row)
    print(json.dumps({k:row.get(k) for k in ('stage', 'passed', 'error')}), flush=True)
    require(row['passed'], 'resident fixture build failed')


def admitted_build():
    row = json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[BUILD_KEY] == base.compact(row) and row['passed'] and
            row['stage'] == 'built-not-executed' and row['ownership_closed'] and not row['active'], 'closed committed build required')
    source_controls(row); allowed = allowed_inputs()
    for field in ('original_fixture_recipe', 'original_native_proof', 'original_device_proof'):
        asset = row[field]; require(digest(asset['path']) == asset['sha256'], 'original proof changed')
    asset = row['actual_compiler_inputs']; require(digest(asset['path']) == asset['sha256'], 'compiler manifest changed')
    actual = json.loads(Path(asset['path']).read_text())
    require(len(actual) == asset['files'] and all(allowed.get(p) == sha == digest(p) for p, sha in actual.items()),
            'actual compiler inputs changed')
    require(row['actual_link_inputs']['uncovered_external_inputs'] == 0 and all(
        allowed.get(p) == sha == digest(p) for p, sha in row['actual_link_inputs']['frozen_inputs'].items()), 'link inputs changed')
    asset = row['binary']; require(digest(asset['path']) == asset['sha256'] and
        Path(asset['path']).stat().st_size == asset['bytes'], 'actual benchmark executable changed')
    require(all(base.supervision.closed(v) for v in row['commands_executed']), 'build process ownership incomplete')
    return row


def cases(speed):
    if speed:
        return [dict(name='host-canary', args=[], trace=False, clean=False)] + [
            dict(name='resident-'+str(i), args=['--bench', str(i%2)], trace=False, clean=True) for i in range(3)]
    return [dict(name='host-canary', args=[], trace=False, clean=False),
        dict(name='rejections', args=['--guards'], trace=True, clean=False),
        dict(name='general-smoke', args=['--smoke'], trace=True, clean=False),
        dict(name='general-tails', args=['--validate'], trace=False, clean=False)]


def prepare(speed):
    key = SPEED_KEY if speed else CHECK_KEY; root = ROOT/('speed-v1' if speed else 'general-v1')
    require(key not in base.committed(RESULT) and not root.exists(), 'one fresh committed cohort only')
    old, _ = prior(); built = admitted_build()
    refs = dict(build=dict(path=str(ROOT/'record.json'), sha256=digest(ROOT/'record.json')))
    if speed:
        path = ROOT/'general-v1/record.json'; check = json.loads(path.read_text())
        require(base.committed(RESULT)[CHECK_KEY] == base.compact(check) and check['passed'] and check['complete'] and
                check['ownership_closed'] and check['fresh_faults'] == [], 'general/guard gate required before timing')
        refs['general'] = dict(path=str(path), sha256=digest(path))
    revision = base.basis.build.git(base.P, 'rev-parse', 'HEAD').decode().strip()
    root.mkdir(mode=0o700)
    row = dict(stage='prepared', source_commit=revision, controller_inputs=base.controls(revision, controller=CONTROLLER),
        native_sources={}, prerequisite_records=refs, binary=built['binary'],
        **{k:old[k] for k in ('environment', 'canary', 'native_validation_probe', 'runtime_manifest', 'boot', 'driver')},
        cases=cases(speed), processes=[], commands_executed=[], active=False, ownership_closed=True,
        passed=False, complete=False, GPU_executed=False, model_executed=False, inference_adopted=False,
        budgets=dict(wall=480, cpu=1500, rss_cap=6 << 30, text_cap=8 << 20, file_cap=64 << 20, total_cap=96 << 20),
        whole_wall_seconds=1800, whole_bytes=96 << 20,
        plan=dict(scope='Resident FP16/FP32 GDN component only;not model PP,PCIe,dequantization,decode or physical262144.',
            safety='Same qualified in-order compute queue;copy offload disabled;direct submission disabled;at most eight outstanding GEMMs;checked drain before releasing owners.',
            numerical='General18shapes,cancel4,zero2,onehot2. Finite normal half operands;subnormals/NaNs/infinities excluded. All outputs compare within2*gamma2559(2^-24)*upward(min(maxX*sumW,sumX*maxW));64 FP64 spots per case within(2*gamma32+gamma64)*upward(sumabs/(1-gamma64)). oneMKL is a comparator,not proof of its internal accumulation. Zero/onehot are exact.',
            rejection='Sixteen synchronous invalid-geometry/null/misalignment/overlap rejections;unchanged canaries;zero traced new-kernel launches. Short-span/wrong-context/exception-injection not qualified.',
            timing='Three fresh processes,arm-order0/1/0;four T4096/8192,N6144/10240 shapes. First shape calls separate,two warmups per arm,five alternating paired64-call batches. All setup/copies/numerical checks outside timing. Host interval includes submission,at-most-eight drains and ending barrier. Queue boundary gap may include host starvation;not physical busy. Native event sum separate;no oneMKL event-as-whole-service claim.',
            eligibility='Every steady batch>=50ms;median of five empty-control host times<=5% of every steady batch. Preserve ineligible samples;no batch resizing/retries. General correctness and final guards required. Per-shape follow-up eligible when every independent native/reference median host ratio<1 and all batches eligible;no minimum improvement floor.',
            observers='Correctness:pre-main UR validation,traced small cases,warning-only tails. Timing:remove UR tracing/validation/preload observers before exec,retain all safety settings;parent checks actual maps/environment at100ms intervals.',
            optional_queue_timestamps='Record null if barrier events report SYCL errc::invalid;checked host timing remains authoritative. Other errors fail. Native returned-event profiling must succeed.',
            timer_sources=['https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_enqueue_barrier.asciidoc',
                'https://github.khronos.org/SYCL_Reference/iface/event.html']),
        retention='Keep individual samples,compact numerical maxima,source/runtime/closure and decisions;retire successful raw traces and build logs after evidence commit. Existing exact qualification files stay unchanged.')
    save(root, key, row); print('Prepared '+key+';commit before execution.', flush=True)


def analyze(case, stdout, folder, pid):
    if case['name'] == 'host-canary':
        require(json.loads(stdout) == dict(pid=pid, initialized=0, checked=46, expected=46,
            version=(1 << 32)-1, closed=0), 'host canary differs'); return dict(host_only=True)
    require(stdout.count('fixture_status=PASS;resident component only') == 1, 'fixture did not finish')
    rows = [json.loads(v) for v in stdout.splitlines() if v.startswith('{')]
    if case['name'] == 'rejections':
        require(rows == [dict(kind='rejections', count=16, canaries_unchanged=True, passed=True)], 'rejection controls differ')
    else:
        validation = [v for v in rows if v['kind'] == 'validation']
        if case['name'] == 'general-smoke': expected = [(1,6144,'general'),(129,10240,'general')]
        elif case['name'] == 'general-tails':
            expected = [(t,n,'general') for t in (1,127,128,129,255,256,257,8191,8192) for n in (6144,10240)] + \
                [(t,n,'cancel') for t in (1,129) for n in (6144,10240)] + \
                [(1,6144,'zero'),(257,10240,'zero'),(129,6144,'onehot'),(8191,10240,'onehot')]
        else: expected = [(t,n,'general') for t in (4096,8192) for n in (6144,10240) for _ in range(2)]
        require([(v['T'],v['N'],v['pattern']) for v in validation] == expected and all(
            v['passed'] and v['guards_inputs'] and v['FP64_spots'] == 64 and
            v['outputs'] == v['T']*v['N'] and 0 <= v['max_comparator_bound_fraction'] <= 1 and
            0 <= v['max_FP64_bound_fraction'] <= 1 for v in validation), 'numerical coverage differs')
        if case['clean']:
            timing = [v for v in rows if v['kind'] == 'timing']
            require(len(timing) == 68, 'timing rows missing')
            for t in (4096,8192):
                for n in (6144,10240):
                    group = [v for v in timing if v['T'] == t and v['N'] == n]
                    require(len(group) == 17 and len([v for v in group if v['phase']=='first_shape_call' and v['calls']==1])==2 and
                        len([v for v in group if v['phase']=='timer_control' and v['arm']=='empty' and v['calls']==64])==5,
                        'first-use/timer controls differ')
                    for arm in ('native','oneMKL'):
                        selected = [v for v in group if v['phase']=='steady' and v['arm']==arm]
                        require([v['sample'] for v in selected] == list(range(5)) and all(v['calls']==64 and v['host_ms']>0 for v in selected), 'steady samples differ')
    details = dict(rows=rows)
    if case['trace']:
        census = qualified.launch_census((folder/'ur.trace').read_text())
        require(census['native_FP32_launches'] == (0 if case['name']=='rejections' else 2), 'actual native dispatch differs')
        details['kernel_census'] = census
    return details


def speed_decision(row):
    import math
    outcomes = []
    for t in (4096,8192):
        for n in (6144,10240):
            medians, eligible = [], True
            for p in row['processes']:
                if not p['name'].startswith('resident-'): continue
                group = [v for v in p['details']['rows'] if v['kind']=='timing' and v['T']==t and v['N']==n]
                overhead = statistics.median(v['host_ms'] for v in group if v['phase']=='timer_control')
                steady = [v for v in group if v['phase']=='steady']
                this_eligible = all(v['host_ms'] >= 50 and overhead <= .05*v['host_ms'] for v in steady)
                a,b = (statistics.median(v['host_ms']/v['calls'] for v in steady if v['arm']==arm) for arm in ('native','oneMKL'))
                medians.append(dict(process=p['name'], native_ms=a, oneMKL_ms=b, ratio=a/b,
                    empty_control_ms=overhead, batches_eligible=this_eligible))
                eligible = eligible and this_eligible
            ratios = [v['ratio'] for v in medians]
            outcomes.append(dict(T=t,N=n,process_medians=medians,all_batches_eligible=eligible,
                geomean_host_time_ratio=math.exp(statistics.mean(math.log(v) for v in ratios)),
                follow_up_eligible=eligible and all(v<1 for v in ratios)))
    return outcomes


def run(speed):
    key = SPEED_KEY if speed else CHECK_KEY; root = ROOT/('speed-v1' if speed else 'general-v1')
    row = json.loads((root/'record.json').read_text())
    require(base.committed(RESULT)[key] == base.compact(row) and row['stage']=='prepared' and not (root/'owned').exists(), 'one frozen cohort only')
    source_controls(row); old, providers = prior(); buildrow = admitted_build()
    require(row['binary']==buildrow['binary'] and row['cases']==cases(speed) and all(row[k]==old[k]
        for k in ('environment','canary','native_validation_probe','runtime_manifest','boot','driver')), 'cohort envelope changed')
    for asset in row['prerequisite_records'].values(): require(digest(asset['path'])==asset['sha256'], 'cohort prerequisite changed')
    frozen = copy.deepcopy(row); owner=None; observed=set(); active=None
    cursor = next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    deadline = time.monotonic()+row['whole_wall_seconds']
    def persist(): write(root/'execution.json',dict(active_case=active,commands=owner.commands if owner else [],active=bool(owner and owner.active)))
    try:
        owner = base.supervision.Owner(root/'owned',persist)
        for case in row['cases']:
            active=case['name']; folder=root/active; folder.mkdir(mode=0o700)
            for part in ('tmp','cache'): (folder/part).mkdir()
            env=dict(row['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),OMP_NUM_THREADS='1')
            if case['clean']:
                for name in tuple(env):
                    if name.startswith('UR_LOG_') or name in ('UR_ENABLE_LAYERS','LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER'): del env[name]
            else:
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                    UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'ur.trace'),
                    UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
            asset=row['canary'] if active=='host-canary' else row['binary']; mapped=set(); env_seen=False
            observer=qualified.ProcObserver()
            def observe(pid):
                nonlocal env_seen
                raw=Path('/proc',str(pid),'environ').read_bytes()
                if not raw: raise ProcessLookupError('empty retiring environment')
                actual=dict(v.split(b'=',1) for v in raw.split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode())==v.encode() for k,v in env.items()) and
                    set(actual)-{k.encode() for k in env}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment differs')
                seen=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                if not seen: raise ProcessLookupError('empty retiring maps')
                require(seen<=set(providers)|{str(Path(asset['path']).resolve())},'unplanned executable provider: '+repr(seen-set(providers)))
                mapped.update(seen); env_seen=True
            def monitor(io,entry):
                io.proc.stdin.close(); next_observation=0.
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll(); require(time.monotonic()<deadline,'whole cohort wall budget')
                    if time.monotonic()>=next_observation:
                        observer.sample(lambda:observe(io.proc.pid),lambda:bool(io.exit_watch.poll(0)))
                        require(base.supervision.artifact_bytes(root)<=row['whole_bytes'],'whole cohort byte budget')
                        next_observation=time.monotonic()+.1
                    if select.select([io.proc.stdout],[],[],.01)[0]: io.read();io.pending.clear()
                io.exit(1)
            device=active!='host-canary';row['GPU_execution_requested']=row.get('GPU_execution_requested',False) or device
            entry,out,err=owner.run(active,[asset['path'],*case['args']],env,base.P,interaction=monitor,**row['budgets'])
            stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
            require(base.supervision.closed(entry) and entry['exit_code']==0,'native process failed '+active+': '+stderr[-3000:])
            if not case['clean']:
                activation=runtime.validation.preload_evidence(stderr,entry['pid'])
                require(activation['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'wrong pre-main loader')
                require(not(folder/'validation.log').exists() or (folder/'validation.log').stat().st_size==0,'parameter validation not quiet')
            details=analyze(case,stdout,folder,entry['pid'])
            if device:
                required={row['environment']['STRATA_UR_PROBE_EXPECTED_LOADER'],env['ZE_ENABLE_ALT_DRIVERS'],
                    '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'}
                if not case['clean']:required.add(env['LD_PRELOAD'])
                require(env_seen and required<=mapped,'actual provider observation incomplete');row['GPU_executed']=True
            check=dict(host_probe_accepted=True);runtime.validation.finish(check,owner,cursor,row['boot'])
            require(check['passed'] and check['driver_after']==row['driver'],'fault/ownership gate failed')
            row['processes'].append(dict(name=active,passed=True,exit_code=0,clean_speed=case['clean'],
                pre_main_parameter_validation=not case['clean'],environment_observed=env_seen,
                proc_observation=observer.summary(),ownership_closed=True,details=details,
                environment_sha256=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()))
            observed.update(mapped);persist();print('Passed '+active,flush=True)
        source_controls(frozen);prior();admitted_build()
        require(base.committed(RESULT)[key]==base.compact(frozen),'frozen comparison plan changed')
        if speed: row['component_decisions']=speed_decision(row)
        row.update(stage='resident-speed-closed' if speed else 'general-check-passed',complete=True,host_probe_accepted=True)
    except BaseException as error:
        row.update(stage='failed',complete=False,host_probe_accepted=False,failed_case=active,error=repr(error))
    finally:
        if owner:runtime.validation.finish(row,owner,cursor,row['boot'])
        if row.get('driver_after')!=row['driver']:row['passed']=False
        if not row.get('passed'):row['stage']='failed'
        if row.get('GPU_execution_requested') and not row['GPU_executed']:row['GPU_executed']=None
        row.update(active=bool(owner and owner.active),observed_provider_count=len(observed),ordinary_runtime_teardown_qualified=False)
        require(owner is None or owner.active is None,'GPU process ownership unresolved');save(root,key,row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','GPU_executed','fresh_faults','error','component_decisions')}),flush=True)
    require(row['passed'],'resident GDN cohort failed')


if __name__=='__main__':
    require(sys.argv[1:] and sys.argv[1] in ('prepare-build','build','prepare-check','check','prepare-speed','speed'),
        'use prepare-build|build|prepare-check|check|prepare-speed|speed')
    with (base.B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare-build':prepare_build()
        elif action=='build':build()
        elif action.startswith('prepare-'):prepare(action=='prepare-speed')
        else:run(action=='speed')
