#!/usr/bin/env python3
"""Main-owned packed GDN3 producer/consumer comparison; repository LICENSE.

CPU build and finite GPU cohorts are distinct committed stages. Production
dispatch remains unchanged. Reuse qualified supervision/provider code, not old
prototype executable bytes or implicit current-HEAD source labels.
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
import retained_prefill_candidate as retained
import measure_gdn_storage as prior
import proc_observer
import runtime_probe as runtime
import ur_validation as validation

base, P, B, RESULT = retained.base, retained.P, retained.B, retained.RESULT
require, digest, write = base.require, base.digest, base.write
ROOT = B / 'gdn16-pipeline-v1'
BUILD, CHECK, SPEED = 'gdn16_pipeline_build_v1', 'gdn16_pipeline_device_v1', 'gdn16_pipeline_speed_v1'
FIXTURE, KERNEL, HEADER = ('bench/sycl/gdn_storage_pipeline.cpp',
    'src/kernels/sycl/fused_gdn.cpp', 'include/strata/prefill/kernels.hpp')
CONTROLLER = Path(__file__).resolve()


def save(root, key, row):
    write(root / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[key] = base.compact(row); write(RESULT, data)


def controls(row):
    require(base.controls(row['source_commit'], controller=CONTROLLER) == row['controller_inputs'],
            'frozen pipeline controller changed')
    for name, sha in row['native_sources'].items():
        raw = base.basis.build.git(P, 'show', row['source_commit'] + ':' + name)
        require(hashlib.sha256(raw).hexdigest() == sha == digest(P / name), 'native input changed: ' + name)


def allowed_inputs(revision):
    original, providers, _ = retained.verify()
    allowed = base.sdk()
    allowed.update(retained.runtime.inventory(original['external_manifest'])); allowed.update(providers)
    asset = original['compiled_inputs']; require(digest(asset['path']) == asset['sha256'], 'selected input map changed')
    allowed.update(json.loads(Path(asset['path']).read_text()))
    for name, sha in original['native_sources'].items():
        path = (retained.ROOT / 'source' / name).resolve()
        require(digest(path) == sha, 'selected native source changed'); allowed[str(path)] = sha
    names = base.basis.build.git(P, 'ls-tree', '-rz', '--name-only', revision, '--', 'include', FIXTURE, KERNEL).decode().split('\0')
    for name, raw in base.basis.build.blobs(P, revision, [n for n in names if n]):
        path = P / name; sha = hashlib.sha256(raw).hexdigest()
        require(digest(path) == sha, 'commit exact project compiler input: ' + name); allowed[str(path.resolve())] = sha
    return original, allowed


def prepare_build():
    require(BUILD not in base.committed(RESULT) and not ROOT.exists(), 'one fresh pipeline build')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    original, _ = allowed_inputs(revision)
    previous = base.committed(RESULT)['official_gdn_storage_speed_v1']
    require(previous['passed'] and previous['complete'] and previous['ownership_closed'] and
            all(v['follow_up_eligible'] for v in previous['component_decisions']), 'direct-half follow-up not eligible')
    old = base.basis.build.git(P, 'show', retained.NATIVE + ':' + KERNEL)
    new = (P / KERNEL).read_bytes(); seam = b'// Main-authored scheduling variant'
    require(old[:old.index(seam)] == new[:new.index(seam)].replace(b'#include <type_traits>\n', b''),
            'decode conv/recurrence arithmetic changed')
    begin, end = b'} // namespace\nvoid fused_gdn_conv_l2', b'} // namespace strata::kernels'
    require(old[old.index(begin):old.index(end)] == new[new.index(begin):new.index(end)], 'decode wrappers changed')
    begin, end = b'void gdn_conv(float', b'namespace {\n'
    require(old[old.index(begin):old.index(end, old.index(begin))] ==
            new[new.index(begin):new.index('namespace {\ntemplate <class Input>', new.index(begin))],
            'original prefill conv/fused recurrence changed')
    commands = json.loads((retained.ROOT / 'build/compile_commands.json').read_text()); build = ROOT / 'build'
    steps = []
    for label, name in [('runtime', 'src/sycl/runtime.cpp'), ('profile', 'src/sycl/command_profile.cpp'),
                         ('kernel', KERNEL), ('fixture', FIXTURE)]:
        basis_name = KERNEL if label in ('kernel', 'fixture') else name
        matches = [v for v in commands if v['file'] == str(retained.ROOT / 'source' / basis_name)]
        command = shlex.split(matches[0]['command']); require(command[0] == base.COMPILER, 'original compiler changed')
        flags = command[1:command.index('-o')]
        source = retained.ROOT / 'source' / name if label in ('runtime', 'profile') else P / name
        if label in ('kernel', 'fixture'): flags = ['-I' + str(P / 'include'), *flags]
        if label == 'fixture': flags += ['-isystem', '/opt/intel/oneapi/mkl/2026.1/include', '-DMKL_ILP64']
        steps.append([label, [base.COMPILER, *flags, '-MD', '-MF', str(build / (label + '.d')),
            '-o', str(build / (label + '.o')), '-c', str(source)]])
    mkl = Path('/opt/intel/oneapi/mkl/2026.1/lib')
    libraries = [str(mkl / n) for n in ('libmkl_sycl_blas.so', 'libmkl_intel_ilp64.so', 'libmkl_sequential.so', 'libmkl_core.so')]
    steps.append(['compile', [base.COMPILER, '-O3', '-fsycl', '-fsycl-id-queries-range=size_t',
        '-fsycl-allow-device-image-dependencies', '-Wl,-t', *[str(build / (n + '.o')) for n in ('fixture','kernel','runtime','profile')],
        '-Wl,-rpath=' + str(mkl), *libraries, '-lm', '-ldl', '-lpthread', '-o', str(build / 'gdn-pipeline')]])
    ROOT.mkdir(mode=0o700)
    for name in ('build', 'tmp', 'cache'): (ROOT / name).mkdir()
    row = dict(stage='prepared', source_commit=revision, controller_inputs=base.controls(revision,controller=CONTROLLER),
        native_sources={n:digest(P/n) for n in (FIXTURE,KERNEL,HEADER)}, selected_native_source=retained.NATIVE,
        original_compile_database=dict(path=str(retained.ROOT/'build/compile_commands.json'),sha256=digest(retained.ROOT/'build/compile_commands.json')),
        commands=steps,environment=dict(original['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        budgets=dict(wall=300,cpu=1000,rss_cap=6<<30,text_cap=16<<20,file_cap=256<<20,total_cap=1<<30),
        plan=dict(scope='Packed resident QKV/z oneMKL projections,causal conv,three-barrier recurrence and RMS/gating. No dequantization,PCIe,output projection,model or decode execution.',
            native_isolation='Original decode arithmetic/wrappers and original FP32 prefill conv/fused recurrence verified byte-identical;new typed APIs not selected by model dispatch.',
            baseline='Retain selected555-source/eight-ELF f6cea2engine. Retire closed old storage fixture before this new generation;both projection/storage controls are built into the new fixture.',
            retention='Current comparison source/recipe and individual useful evidence only;retire closed raw output and extra generation at comparison closure.'),
        active=False,ownership_closed=True,passed=False,GPU_executed=False,commands_executed=[])
    controls(row); save(ROOT,BUILD,row); print('Prepared CPU-only pipeline recipe;commit before build.',flush=True)


def build():
    row = json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[BUILD] == base.compact(row) and row['stage']=='prepared' and
            not (ROOT/'owned-build').exists(), 'one frozen fresh build')
    controls(row); _,allowed = allowed_inputs(row['source_commit']); owner=None
    require(digest(row['original_compile_database']['path']) == row['original_compile_database']['sha256'], 'recipe changed')
    require(not Path(base.committed(RESULT)['official_gdn_storage_fixture_build_v1']['binary']['path']).exists(),
            'retire closed old component generation before another build')
    def monitor(io,entry):
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(base.supervision.artifact_bytes(ROOT)<=row['budgets']['total_cap'],'build byte budget')
            if select.select([io.proc.stdout],[],[],.02)[0]: io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=base.supervision.Owner(ROOT/'owned-build')
        for label,argv in row['commands']:
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,interaction=monitor,**row['budgets'])
            require(base.supervision.closed(entry) and entry['exit_code']==0,
                    label+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-6000:])
        found=set()
        for label in ('runtime','profile','kernel','fixture'):
            for line in (ROOT/'build'/(label+'.d')).read_text().replace('\\\n','').splitlines():
                if not line.strip(): continue
                left,colon,right=line.partition(':'); require(colon and left.strip(),'invalid depfile')
                found.update(str((ROOT/p).resolve()) for p in shlex.split(right))
        require(all(allowed.get(p)==digest(p) for p in found),'unfrozen actual compiler input')
        write(ROOT/'compiler-inputs.json',{p:allowed[p] for p in sorted(found)})
        row['actual_compiler_inputs']=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(found),uncovered=0)
        row['actual_link_inputs']=base.traced_link_inputs(allowed,ROOT)
        binary=ROOT/'build/gdn-pipeline'
        row.update(stage='built-not-executed',passed=True,binary=dict(path=str(binary),sha256=digest(binary),bytes=binary.stat().st_size))
        retained.verify()
    except BaseException as error: row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active),commands_executed=owner.commands if owner else [],
                   ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)))
        require(owner is None or owner.active is None,'build ownership unresolved'); save(ROOT,BUILD,row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','error')}),flush=True); require(row['passed'],'pipeline build failed')


def verify():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[BUILD]==base.compact(row) and row['passed'] and row['stage']=='built-not-executed' and
            row['ownership_closed'] and not row['active'] and all(base.supervision.closed(v) for v in row['commands_executed']), 'closed build required')
    controls(row); _,allowed=allowed_inputs(row['source_commit']); asset=row['actual_compiler_inputs']
    require(digest(asset['path'])==asset['sha256'],'actual input map changed')
    found=json.loads(Path(asset['path']).read_text())
    require(len(found)==asset['files'] and all(allowed.get(p)==sha==digest(p) for p,sha in found.items()),'actual compiler inputs changed')
    require(row['actual_link_inputs']['uncovered_external_inputs']==0 and all(allowed.get(p)==sha==digest(p)
        for p,sha in row['actual_link_inputs']['frozen_inputs'].items()),'link inputs changed')
    asset=row['binary']; require(digest(asset['path'])==asset['sha256'] and Path(asset['path']).stat().st_size==asset['bytes'],'binary changed')
    return row


def cases(speed):
    return [dict(name='host-canary',args=[],clean=False,trace=False),
            dict(name='clean-smoke' if speed else 'pipeline-smoke',args=['--smoke'],clean=speed,trace=not speed)] + (
        [dict(name='resident-'+str(i),args=['--bench',str(i)],clean=True,trace=False) for i in range(3)] if speed else
        [dict(name='pipeline-general',args=['--validate'],clean=False,trace=False)])


def prerequisites(speed):
    built=verify(); data=base.committed(RESULT); refs=dict(build=dict(path=str(ROOT/'record.json'),sha256=digest(ROOT/'record.json')))
    if speed:
        device=json.loads((ROOT/'device-v1/record.json').read_text())
        require(data[CHECK]==base.compact(device) and device['passed'] and device['complete'] and
                device['ownership_closed'] and not device['active'] and not device['fresh_faults'] and
                device['binary']==built['binary'],'same-build closed device cohort required')
        controls(device); refs['device']=dict(path=str(ROOT/'device-v1/record.json'),sha256=digest(ROOT/'device-v1/record.json'))
    gate=data['official_prefill_strategies_device_v2']; return built,refs,gate,retained.runtime.inventory(gate['runtime_manifest'])


def prepare_run(speed):
    key,root=(SPEED,ROOT/'speed-v1') if speed else (CHECK,ROOT/'device-v1')
    require(key not in base.committed(RESULT) and not root.exists(),'one fresh cohort')
    built,refs,gate,_=prerequisites(speed); root.mkdir(mode=0o700)
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    row=dict(stage='prepared',source_commit=revision,controller_inputs=base.controls(revision,controller=CONTROLLER),
        native_sources=built['native_sources'],native_source_commit=built['source_commit'],binary=built['binary'],
        prerequisite_records=refs,**{k:gate[k] for k in ('environment','canary','native_validation_probe','runtime_manifest','boot','driver')},
        cases=cases(speed),processes=[],active=False,ownership_closed=True,passed=False,complete=False,
        GPU_executed=False,model_executed=False,inference_adopted=False,commands_executed=[],
        budgets=dict(wall=600,cpu=2400,rss_cap=8<<30,text_cap=8<<20,file_cap=64<<20,total_cap=96<<20),whole_wall_seconds=2400,whole_bytes=96<<20,
        plan=dict(scope='Packed resident two-projection+conv+GDN3+RMS/gating component;K2560,C10240,V6144. No model PP,TG,dequantization,PCIe,output projection or physical262144model qualification.',
            precision='Default oneMKL compute mode unset in both arms. Half C uses half1/0 scalars;FP32 C uses float1/0. All storage conversions RNE;FP32 persistent state/history,gates,reductions/accumulation. Passing diagnostics does not establish vendor accumulator type or real-model quality/range.',
            numerical='Whole typed conv matches FP32 conv followed by host RNE. Typed recurrence state/history/final FP16 output match rounded-storage FP32 reference exactly,including1/2/tail splits. 64FP64 packed-producer spots/case bounded by2gamma32*absdot plus final-half-rounding. All finite outputs,guards,immutable inputs. Original FP32 fused-vs-columns smoke bit-exact. 18synchronous no-write geometry/epsilon/null/alias/alignment/host-pointer refusals. FP32-vs-FP16 drift is recorded diagnostically,not an extra-precision adoption criterion.',
            timing='Three fresh processes;two4096/8192-row cases. Alternating/reversed arm order by process/round;first shape calls separate,four warmups,five paired64call batches. State/history resets,allocation,copies,CPU oracles outside timing. Host interval includes complete twoGEMM+conv/history+recurrence/RMS submissions,drain every4calls and final checked wait;no returned-MKL-event whole-operation assumption.',
            eligibility='Each steady batch>=100ms;median of5empty controls<=5%of each batch. No resize/retry. Follow-up requires all3paired independent-process median ratios<1 at both lengths;no minimum percentage gain. Component success alone does not authorize a model trial.',
            safety='Original in-order Runtime;copy offload/direct submission disabled and preexec environment immutable. At most4compound calls before drain. Canary,small traced smoke,warning-only general shapes before separate clean speed processes.',
            retention='Keep useful samples,numerical maxima and decisions. Retire closed raw logs/traces after extracting names/provenance. Keep selected engine;new component generation has only this imminent comparison consumer.'))
    save(root,key,row); print('Prepared '+key+';commit before GPU execution.',flush=True)


def analyze(case,stdout,folder,pid):
    if case['name']=='host-canary':
        require(json.loads(stdout)==dict(pid=pid,initialized=0,checked=46,expected=46,version=(1<<32)-1,closed=0),'host canary differs')
        return dict(host_only=True)
    require(stdout.count('fixture_status=PASS;packed resident GDN3 only')==1,'fixture incomplete')
    rows=[json.loads(s) for s in stdout.splitlines() if s.startswith('{')]
    require(rows and all(v.get('passed') is True for v in rows),'invalid fixture row')
    bench=case['name'].startswith('resident-'); smoke=case['args']==['--smoke']
    expected=([(4096,'general'),(8192,'general')] if bench else [(1,'general'),(17,'general')] if smoke else
        [(t,'general') for t in (1,2,3,127,129,8191,8192)]+[(129,p) for p in ('zero','cancel','onehot')])
    canonical=[v for v in rows if v['kind']=='canonical']
    require([(v['T'],v['pattern'],v['repeats']) for v in canonical]==
        [(t,p,r) for t,p in expected for r in ((1,64) if bench else (1,))] and all(
            v['exact_rounded_storage_state_history_output'] and v['guards'] and
            v['split_1_2_tail']==(v['repeats']==1 and v['T']>=4) for v in canonical),'canonical coverage differs')
    producer=[v for v in rows if v['kind']=='packed_producers']
    require([v['T'] for v in producer]==[t for t,_ in expected] and all(v['FP64_spots']==64 and
            0<=v['max_bound_fraction']<=1 for v in producer),'packed producer coverage differs')
    require(len(rows)==(48 if bench else 8 if smoke else 31),'unexpected result rows')
    result=dict(canonical=canonical,packed_producers=producer,
        FP32_difference=[v for v in rows if v['kind']=='FP32_difference'])
    if not bench:
        require([v['cases'] for v in rows if v['kind']=='rejection']==[18],'no-write refusal coverage differs')
        require(len([v for v in rows if v['kind']=='legacy'])==(1 if smoke else 0),'legacy check differs')
    else:
        timing=[v for v in rows if v['kind']=='timing']; require(len(timing)==38,'timing coverage differs')
        for t in (4096,8192):
            selected=[v for v in timing if v['T']==t]; order=int(case['args'][1])
            require([v['arm'] for v in selected if v['phase']=='first_shape_call']==
                    [('FP32','FP16')[(i+order)%2] for i in range(2)],'first-call order differs')
            steady=[v for v in selected if v['phase']=='steady']
            require([(v['round'],v['arm'],v['calls']) for v in steady]==
                [(r,('FP32','FP16')[(i+r+order)%2],64) for r in range(5) for i in range(2)],'steady order differs')
            empty=[v['host_ms'] for v in selected if v['phase']=='timer_control']
            require(len(empty)==5 and all(v>=0 for v in empty),'empty controls missing')
            for v in steady: v['eligible']=v['host_ms']>=100 and statistics.median(empty)<=.05*v['host_ms']
        result['timing']=timing
    if case['trace']:
        result['trace']=dict(path=str(folder/'ur.trace'),sha256=digest(folder/'ur.trace'),bytes=(folder/'ur.trace').stat().st_size,
            **prior.launch_census((folder/'ur.trace').read_text()))
    return result


def decisions(row):
    out=[]
    for t in (4096,8192):
        pairs=[]
        for proc in row['processes']:
            if not proc['name'].startswith('resident-'): continue
            timing=[v for v in proc['details']['timing'] if v['T']==t and v['phase']=='steady']
            medians={arm:statistics.median(v['host_ms']/v['calls'] for v in timing if v['arm']==arm) for arm in ('FP32','FP16')}
            pairs.append(dict(process=proc['name'],milliseconds=medians,ratio=medians['FP16']/medians['FP32'],eligible=all(v['eligible'] for v in timing)))
        require(len(pairs)==3,'three fresh process pairs required')
        import math
        out.append(dict(T=t,process_medians=pairs,paired_geomean_time_ratio=math.exp(sum(math.log(p['ratio']) for p in pairs)/3),
            follow_up_eligible=all(p['eligible'] and p['ratio']<1 for p in pairs)))
    return out


def run(speed):
    key,root=(SPEED,ROOT/'speed-v1') if speed else (CHECK,ROOT/'device-v1'); row=json.loads((root/'record.json').read_text())
    require(base.committed(RESULT)[key]==base.compact(row) and row['stage']=='prepared' and not (root/'owned').exists(),'one frozen fresh cohort')
    controls(row);built,refs,gate,providers=prerequisites(speed)
    require(row['binary']==built['binary'] and row['prerequisite_records']==refs and row['cases']==cases(speed) and
            all(row[k]==gate[k] for k in ('environment','canary','native_validation_probe','runtime_manifest','boot','driver')),'cohort envelope changed')
    frozen,owner,active,mapped_all=copy.deepcopy(row),None,None,set()
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    deadline=time.monotonic()+row['whole_wall_seconds']
    def persist(): write(root/'execution.json',dict(active_case=active,commands=owner.commands if owner else [],active=bool(owner and owner.active)))
    try:
        owner=base.supervision.Owner(root/'owned',persist)
        for case in row['cases']:
            active=case['name']; folder=root/active;folder.mkdir(mode=0o700)
            for part in ('tmp','cache'): (folder/part).mkdir()
            env=dict(row['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),OMP_NUM_THREADS='1')
            if case['clean']:
                for name in tuple(env):
                    if name.startswith('UR_LOG_') or name in ('UR_ENABLE_LAYERS','LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER'): del env[name]
            else: env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'ur.trace'),UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
            asset=row['canary'] if active=='host-canary' else row['binary'];mapped,env_seen,observer=set(),False,proc_observer.ProcObserver()
            def observe(pid):
                nonlocal env_seen
                raw=Path('/proc',str(pid),'environ').read_bytes()
                if not raw: raise ProcessLookupError('empty retiring environment')
                actual=dict(v.split(b'=',1) for v in raw.split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode())==v.encode() for k,v in env.items()) and set(actual)-{k.encode() for k in env}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment differs')
                seen=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                if not seen: raise ProcessLookupError('empty retiring maps')
                require(seen<=set(providers)|{str(Path(asset['path']).resolve())},'unplanned provider: '+repr(seen-set(providers)))
                mapped.update(seen);env_seen=True
            def monitor(io,entry):
                io.proc.stdin.close();next_sample=0.
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll();require(time.monotonic()<deadline,'whole cohort deadline')
                    if time.monotonic()>=next_sample:
                        observer.sample(lambda:observe(io.proc.pid),lambda:bool(io.exit_watch.poll(0)))
                        require(base.supervision.artifact_bytes(root)<=row['whole_bytes'],'whole cohort byte budget');next_sample=time.monotonic()+.1
                    if select.select([io.proc.stdout],[],[],.01)[0]:io.read();io.pending.clear()
                io.exit(1)
            row['GPU_execution_requested']=row.get('GPU_execution_requested',False) or active!='host-canary'
            entry,out,err=owner.run(active,[asset['path'],*case['args']],env,P,interaction=monitor,**row['budgets'])
            stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
            require(base.supervision.closed(entry) and entry['exit_code']==0,'native process failed '+active+': '+stderr[-3000:])
            if not case['clean']:
                activation=validation.preload_evidence(stderr,entry['pid']);require(activation['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'pre-main loader differs')
                require(not (folder/'validation.log').exists() or (folder/'validation.log').stat().st_size==0,'parameter validation not quiet')
            details=analyze(case,stdout,folder,entry['pid'])
            if active!='host-canary':
                required={env['UR_ADAPTERS_FORCE_LOAD'],env['ZE_ENABLE_ALT_DRIVERS'],'/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'}
                if not case['clean']:required.update((env['LD_PRELOAD'],env['STRATA_UR_PROBE_EXPECTED_LOADER']))
                require(env_seen and required<=mapped,'actual provider observation incomplete: '+repr(required-mapped));row['GPU_executed']=True
            check=dict(host_probe_accepted=True);validation.finish(check,owner,cursor,row['boot'])
            require(check['passed'] and check['driver_after']==row['driver'],'fault/ownership gate failed')
            row['processes'].append(dict(name=active,passed=True,exit_code=0,clean_speed=case['clean'],pre_main_parameter_validation=not case['clean'],
                environment_observed=env_seen,proc_observation=observer.summary(),ownership_closed=True,details=details,
                environment_sha256=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest()))
            mapped_all.update(mapped);persist();print('Passed '+active,flush=True)
        controls(frozen);prerequisites(speed);require(base.committed(RESULT)[key]==base.compact(frozen),'frozen plan changed')
        if speed:row['component_decisions']=decisions(row)
        row.update(stage='resident-speed-closed' if speed else 'general-check-passed',complete=True,host_probe_accepted=True)
    except BaseException as error:row.update(stage='failed',complete=False,host_probe_accepted=False,failed_case=active,error=repr(error))
    finally:
        if owner:validation.finish(row,owner,cursor,row['boot'])
        if row.get('driver_after')!=row['driver']:row['passed']=False
        if not row.get('passed'):row['stage']='failed'
        if row.get('GPU_execution_requested') and not row['GPU_executed']:row['GPU_executed']=None
        row.update(active=bool(owner and owner.active),observed_provider_count=len(mapped_all),ordinary_runtime_teardown_qualified=False,commands_executed=owner.commands if owner else [])
        require(owner is None or owner.active is None,'GPU ownership unresolved');save(root,key,row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','GPU_executed','fresh_faults','error','component_decisions')}),flush=True)
    require(row['passed'],'pipeline cohort failed')


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare-build','build','verify','prepare-check','check','prepare-speed','speed'),'use prepare-build|build|verify|prepare-check|check|prepare-speed|speed')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        mode=sys.argv[1]
        if mode=='prepare-build':prepare_build()
        elif mode=='build':build()
        elif mode=='verify':print(json.dumps(dict(binary=verify()['binary'],GPU_executed=False)))
        elif mode.startswith('prepare-'):prepare_run(mode=='prepare-speed')
        else:run(mode=='speed')
