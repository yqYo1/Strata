#!/usr/bin/env python3
"""Main-owned real-GGUF NativeHead lifecycle gate;repository LICENSE.
Build/census helpers recovered from Main f9789357. Not a model benchmark,
production Verifier lease proof,physical-resource ceiling or default adoption.
"""
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys
import tarfile
sys.dont_write_bytecode=True
import benchmark_runtime as python_runtime
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation
import build_compat_profile as selected
from q6_dpas_component import embedded_dpas
P=Path(__file__).resolve().parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
ROOT=B/'q6-head-owner-v1'
KEY='q6_head_owner_v1'
TARGET='sycl_q6_head_owner_test'
CPP='tests/sycl/native_head_dpas.cpp'
UNITS=(CPP,'src/core/native_head.cpp','src/kernels/sycl/q6_dpas_component.cpp',
 'src/kernels/sycl/native_mmvq.cpp','src/kernels/sycl/native_q5_head.cpp',
 'src/kernels/sycl/native_q3_esimd.cpp','src/kernels/sycl/native_q4_esimd.cpp',
 'src/kernels/sycl/native_q2_esimd.cpp','src/kernels/sycl/native_iq4xs_esimd.cpp',
 'src/kernels/sycl/native_iq_esimd.cpp','src/kernels/sycl/native_mmq_xmx.cpp',
 'src/kernels/sycl/iq.cpp','src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp')
RECIPE_COMMIT='a7c6309c56724f3b5eea5f5f4f8b2b37baa46970'
RUNTIME=B/'decode-iq3-audit-speed-v2/runtime-inputs.json'
RUNTIME_SHA='43de9b527585addcc6da1f8fdc6f1e8266e2d982ba4c752d909578ed8c7bde21'
require=guards.require
sha=owner.sha

def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)

def signature(pins):
    return dict(files=len(pins), aggregate_sha256=hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode()).hexdigest())

def dependency_pins(roots):
    pending, visited, pins, total, entries = list(map(Path, roots)), set(), {}, 0, 0
    while pending:
        root = pending.pop().resolve()
        if root in visited: continue
        require(root.is_dir(), 'dependency include/config directory missing')
        visited.add(root)
        for path in root.rglob('*'):
            entries += 1
            require(entries <= 32768, 'dependency traversal budget')
            if path.is_symlink() and path.is_dir():
                pending.append(path.resolve())
            elif path.is_file():
                name = str(path.resolve())
                if name in pins: continue
                total += path.stat().st_size
                require(len(pins) < 16384 and total <= 256 << 20, 'finite dependency inventory')
                pins[name] = sha(path)
    return pins

def census(row, dependencies, link_text):
    """Reject actual compiler/link files outside the precommitted inventories."""
    allowed = {str((ROOT / 'source' / p).resolve()): h for p, h in row['sources'].items()}
    allowed.update({str(Path(p).resolve()): h for p, h in row['dependency_pins'].items()})
    allowed.update({str(Path(p).resolve()): h for p, h in
        json.loads(Path(row['external_inputs']['path']).read_text()).items()})
    allowed.update({v['realpath']: v['sha256'] for v in row['tools'].values()})
    compiled, generated, links, owned = {}, {}, {}, set()
    for line in dependencies.splitlines():
        if not line.startswith('    ') or not line.strip(): continue
        p = Path(line.strip()); p = (p if p.is_absolute() else ROOT / 'build' / p).resolve()
        value = sha(p)
        if p.is_relative_to(ROOT / 'build'): generated[str(p)] = value
        else:
            require(allowed.get(str(p)) == value, 'unfrozen actual compiler input: ' + str(p))
            compiled[str(p)] = value
    expected = {str((ROOT / 'source' / p).resolve()) for p in UNITS}
    observed = {p for p in compiled if Path(p).suffix in ('.cpp', '.c', '.cc', '.cu')}
    require(observed == expected and len(compiled) > 100 and not generated,
        'exact component-unit compiler census with no uncovered/generated input required')
    for line in link_text.splitlines():
        name = line.strip(); p = Path(name)
        if not name or any(c.isspace() for c in name) or not (
            p.is_absolute() or p.suffix in ('.o', '.a') or '.so' in p.name): continue
        p = (p if p.is_absolute() else ROOT / 'build' / p).resolve()
        if any(p.is_relative_to(ROOT / n) for n in ('build', 'tmp')):
            owned.add(str(p)); continue
        require(p.is_file() and allowed.get(str(p)) == sha(p), 'unfrozen actual linker input: ' + str(p))
        links[str(p)] = sha(p)
    require(len(links) > 8, 'actual external link census incomplete')
    path = ROOT / 'actual-inputs.json'
    path.write_text(json.dumps(dict(compiled=compiled, generated=generated, links=links,
        owned_link_products=sorted(owned)), indent=2) + '\n')
    return dict(path=str(path), sha256=sha(path), compiler_files=len(compiled),
        generated_files=len(generated), external_link_files=len(links), uncovered=0,
        translation_units=UNITS, source_commit=row['source_commit'])

def compact(row):
    out={k:v for k,v in row.items() if k not in ('sources','helpers','runtime_pins','dependency_pins','commands')}
    for field in ('sources','helpers','runtime_pins','dependency_pins'):
        out[field+'_identity']=signature(row[field])
    out['commands']=[dict(label=c['label'],exit_code=c.get('exit_code'),ownership_closed=owner.closed(c),
        error=c.get('error'),errors=c.get('errors',[])) for c in row['commands']]
    return out

def save(row):
    (ROOT/'record.json').write_text(json.dumps(row,indent=2)+'\n')
    data=json.loads(RESULT.read_text());data[KEY]=compact(row)
    RESULT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def model_assets():
    root=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
    paths=sorted(root.glob('*.gguf'))
    require(len(paths)==2 and all(p.is_file() and p.stat().st_size>1<<30 for p in paths),
            'two existing real model shards required')
    return [dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]

def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT),'one fresh owner build')
    reference=validation.committed(RESULT,('q6_dpas_component_v7',))
    require(reference['decision']['head_component_gain'] and not reference['decision']['cells']['640']['component_gain_qualified']
        and not reference['decision']['model_speed_qualified'],'preserve component decisions')
    proposal=reference['next_model_proposal']
    require(proposal['cpu_policy_result']['passed'] and proposal['cpu_policy_result']['complete']
        and not proposal['cpu_policy_result']['GPU_executed'],'scoped CPU phase-policy prerequisite')
    revision=git('rev-parse','HEAD').decode().strip()
    names=git('ls-files','-z','.gitattributes','CMakeLists.txt','LICENSE','cmake','src','include','tests','bench','tools','third_party').decode().split('\0')
    names=sorted(n for n in names if n and not n.startswith('bench/results/') and n!='bench/performance-records.csv')
    require(CPP in names and str(Path(__file__).relative_to(P)) in names,'commit exact fixture/controller first')
    require(subprocess.run(['git','diff','--quiet',revision,'--',*names],cwd=P).returncode==0,'native inputs dirty')
    archive=git('archive','--format=tar',revision,*names)
    require(len(names)<=4096 and len(archive)<=64<<20,'bounded native source archive')
    sources={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        for item in t.getmembers():
            if item.isdir():continue
            require(item.isfile() and item.name in names,'named regular source only')
            sources[item.name]=hashlib.sha256(t.extractfile(item).read()).hexdigest()
    require(set(sources)==set(names),'complete exact source archive')
    helpers={n:sha(P/n) for n in names if n.startswith('tools/sycl/')}
    require(all((P/n).read_bytes()==git('show',revision+':'+n) for n in helpers),'committed live Python helpers')
    raw=json.loads(git('show',RECIPE_COMMIT+':'+str(RESULT.relative_to(P))))['q6_dpas_component_v7']
    require(sha(RUNTIME)==RUNTIME_SHA,'qualified runtime identity manifest')
    runtime=json.loads(RUNTIME.read_text());require(len(runtime)==84,'same84 qualified runtime files')
    require(all(sha(p)==v for p,v in runtime.items()),'actual runtime files changed')
    external=raw['external_inputs'];require(sha(external['path'])==external['sha256'],'external compiler census changed')
    require(all(sha(p)==v for p,v in json.loads(Path(external['path']).read_text()).items()),'external compiler/link bytes changed')
    require(all(sha(t['path'])==t['sha256'] and str(Path(t['path']).resolve())==t['realpath'] for t in raw['tools'].values()),'toolchain changed')
    packages=[v.split()[0] for v in raw['system_packages']]
    actual=subprocess.check_output(['/usr/bin/dpkg-query','-W','-f='+chr(36)+'{binary:Package} '+chr(36)+'{Version}\n',*packages],text=True,timeout=15).splitlines()
    require(actual==raw['system_packages'],'system compiler/SDK packages changed')
    _,identity=selected.basis()
    assets=model_assets();guards.no_model_owners([p['path'] for p in assets])
    old_root=str(B/'q6-dpas-component-v7')
    env={k:v.replace(old_root,str(ROOT)) for k,v in raw['environment'].items()}
    env.update(STRATA_SYCL_DECODE_Q6_COLUMNS='0',STRATA_SYCL_DECODE_Q6_DPAS_HEAD='0',STRATA_PREFILL_RELEASE_HEADS='1')
    require(not env.get('LD_PRELOAD'),'validation preload added only to native child before exec')
    row=dict(stage='prepared',source_commit=revision,controller_commit=revision,sources=sources,helpers=helpers,
        source_generation=dict(command='git archive --format=tar <source_commit> <named paths>',attributes='.gitattributes'),
        sources_provenance='Main f2a9fb447 NativeHead/selector plus current fixture;C4 MIT GGML34ca79122 reference and Main packing;no XeStrata.',
        selected_native=identity,tools=raw['tools'],tool_versions=raw['tool_versions'],
        system_packages=actual,external_inputs=external,dependency_roots=raw['dependency_roots'],
        dependency_pins=dependency_pins(raw['dependency_roots']),runtime_pins=runtime,
        runtime_manifest=dict(path=str(RUNTIME),sha256=RUNTIME_SHA),native_validation_probe=raw['native_validation_probe'],
        environment=env,configure=[v.replace(old_root,str(ROOT)) for v in raw['configure']],
        build=[v.replace(old_root,str(ROOT)).replace('sycl_q6_dpas_component',TARGET) for v in raw['build']],
        build_budget=raw['build_budget'],host_budget=raw['host_budget'],native_budget=raw['native_budget'],
        host_command=[raw['tools']['python']['path'],'-B','-m','unittest','-v','tools.sycl.test_q6_dpas_component'],
        assets=assets,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),driver=guards.driver_guard(),
        plan=dict(order='13CPU reader/parser controls;15-unit fresh fixture build and linked DPAS check;five native CPU-only owner/switch cases;two fresh real-head processes reference0 then packed1.Commit each stage;stop first failure,no retries.',
            shape=dict(K=2560,N=248320,columns=4,C1_reference=True),
            owner_cases='Per process:cancel1;two all-graph drain/retire/detach/Device-overwrite/restore/publish/recapture cycles;four independent543-row/all-output-envelope checks;canonical full-byte immutability;all buffer guards.',
            admission='Attest preload/loader before Runtime;forced adapter/driver after Runtime but before fixture allocations/copies/kernels. Immutable environment.',
            correctness='Independent scalar double Q6/Q8 dot on543 distributed/head-tail rows and L1 gamma bound;all outputs finite/bounded against reference;C1 and guards. Not task quality.',
            timing='Eleven whole-head-API-wrapper wall/thread CPU intervals;diagnostic only,not clean performance,physical bandwidth or whole model startup.',
            limits='No production Verifier lease,full model,PP/TTFT/TG,agent/task,physical262144/RESTORE or default adoption. Provider retained until _Exit;ordinary teardown unqualified.',
            retention='One fresh fixture through owner closure and next model-plan source admission;retire ended raw/build after exact source/recipe recovery. Preserve90f/552b/e1b guards and separate56legacy holds.'),
        results=[],commands=[],active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('source','tmp','cache'):(ROOT/name).mkdir()
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as t:
            for item in t.getmembers():
                if item.isdir():continue
                p=ROOT/'source'/item.name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(t.extractfile(item).read());p.chmod(item.mode&0o777)
        save(row)
    except BaseException as e:
        row.update(stage='failed',preparation_error=repr(e));save(row);raise
    print('Prepared fresh source/owner plan;commit before execution.',flush=True)

def current(row):
    require(validation.committed(RESULT,(KEY,))==compact(row),'commit unchanged owner gate')
    require(all(sha(P/n)==v and (P/n).read_bytes()==git('show',row['controller_commit']+':'+n) for n,v in row['helpers'].items()),'control drift')
    require(subprocess.run(['git','diff','--quiet',row['source_commit'],'--',*row['sources']],cwd=P).returncode==0,'live native inputs changed')
    require(all(sha(ROOT/'source'/n)==v for n,v in row['sources'].items()),'private source drift')
    _,identity=selected.basis();require(identity==row['selected_native'],'selected native provenance changed')
    e=row['external_inputs'];require(sha(e['path'])==e['sha256'] and
        all(sha(p)==v for p,v in json.loads(Path(e['path']).read_text()).items()),'external compiler/link drift')
    require(dependency_pins(row['dependency_roots'])==row['dependency_pins'],'SDK/header drift')
    require(all(sha(p)==v for p,v in row['runtime_pins'].items()),'runtime drift')
    require(all(sha(t['path'])==t['sha256'] and str(Path(t['path']).resolve())==t['realpath'] for t in row['tools'].values()),'tool drift')
    require(guards.driver_guard()==row['driver'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'driver/boot changed')
    if 'binary' in row:
        require(sha(row['binary']['path'])==row['binary']['sha256'],'actual fixture ELF drift')
        require(embedded_dpas(Path(row['binary']['path']).read_bytes())==row['codegen'],'actual C4 image drift')
        inv=row['actual_inputs'];require(sha(inv['path'])==inv['sha256'],'actual input inventory drift')
        inputs=json.loads(Path(inv['path']).read_text())
        require(all(sha(p)==v for field in ('compiled','links','generated') for p,v in inputs[field].items()),'actual compiler/link input drift')
        require(sha(row['compile_database']['path'])==row['compile_database']['sha256'],'compile database drift')
    guards.no_model_owners([a['path'] for a in row['assets']])


def parse_native(text,packed):
    require(type(packed) is int and packed in (0,1),'explicit owner arm')
    lines=text.splitlines()
    captures=f'CAPTURE graphs=3 nodes_each=2 packed_decode_calls={packed} packed_prompt_calls=0 packed_C1_calls=0'
    require(lines.count(captures)==3 and sum(v.startswith('CAPTURE ') for v in lines)==3,'all graph families recaptured')
    end=f'OWNER_CHECK packed={packed} numeric_cases=4 blocked_cases=4 restore_cycles=2 cancel_cases=1 graph_replays=12 canonical_immutable=PASS packed_immutable='+('PASS' if packed else 'absent')+' model=0 Verifier_owner=0'
    require(lines.count(end)==1 and lines.count('HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified')==1
        and lines[-1]=='HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified','complete owner/exit proof')
    numeric=[]
    for line in lines:
        if not line.startswith('NUMERIC '):continue
        m=re.fullmatch(r'NUMERIC cycle=(-?\d+) seed=(\d+) outputs=993280 oracle_rows=543 max_abs_pair_error=([0-9.eE+-]+) C1=PASS prompt=reference guards=PASS',line)
        require(m is not None,'numeric row schema')
        cycle,seed,value=int(m[1]),int(m[2]),float(m[3])
        require(math.isfinite(value) and value>=0,'finite numerical summary')
        numeric.append(dict(cycle=cycle,seed=seed,max_abs_pair_error=value,outputs=993280,oracle_rows=543))
    require([(v['cycle'],v['seed']) for v in numeric]==[(-1,7),(-1,19),(0,31),(1,32)],'complete fresh-data numerics')
    expected_blocked=['BLOCKED phase='+p+' canonical=PASS packed=PASS' for _ in range(2)
        for p in ('detached','restored_not_published')]
    require([v for v in lines if v.startswith('BLOCKED ')]==expected_blocked,'publication/blocked owner proof')
    reclaimed=[f'RECLAIM cycle={c} overwritten_device_bytes={521472000*(1+packed)} tail=PASS' for c in range(2)]
    require([v for v in lines if v.startswith('RECLAIM ')]==reclaimed,'temporary reuse/write extent')
    retired=[f'RETIREMENT cycle={c} canonical_span=retired packed_span='+('retired' if packed else 'absent') for c in range(2)]
    require([v for v in lines if v.startswith('RETIREMENT ')]==retired,'registered pointer retirement')
    admission=[v for v in lines if v.startswith('HEAD admitted=')]
    require(len(admission)==1,'one actual admission')
    match=re.fullmatch(r'HEAD admitted=1 packed=([01]) canonical_bytes=521472000 packed_device_bytes=(\d+) packed_host_bytes=(\d+) max_alloc_bytes=(\d+)',admission[0])
    require(match and [int(match[i]) for i in (1,2,3)]==[packed,521472000*packed,521472000*packed]
        and int(match[4])>=521472000*(1+packed),'admitted head bytes/single-allocation bound')
    timings=[]
    for line in lines:
        if not line.startswith('TRANSITION '):continue
        m=re.fullmatch(r'TRANSITION phase=(\w+) cycle=(-?\d+) wall_ns=(\d+) thread_cpu_ns=(\d+) scope=whole_wrapper',line)
        require(m is not None,'transition clock schema')
        timings.append(dict(phase=m[1],cycle=int(m[2]),wall_ns=int(m[3]),thread_cpu_ns=int(m[4])))
    require([(v['phase'],v['cycle']) for v in timings]==[('load',-1)]+[(p,c) for c in range(2)
        for p in ('prepare','retire_readers','detach','restore','resume')],'all head transition clocks')
    return dict(packed=packed,numeric=numeric,transitions=timings,restore_cycles=2,blocked_cases=4,
        graph_replays=12,graphs_each_capture=3,nodes_each_graph=2,canonical_bytes=521472000,
        packed_device_bytes=521472000*packed,packed_host_bytes=521472000*packed,max_alloc_bytes=int(match[4]),
        canonical_immutable=True,packed_immutable=True if packed else None,
        scope='Actual NativeHead and selector with fixture graphs;not production Verifier/model ownership or speed.')

def run_gpu(row,scope,mode,outcome):
    packed=int(mode=='packed')
    require(model_assets()==row['assets'],'full pre-exec real-shard identities')
    env=dict(row['environment'],STRATA_SYCL_DECODE_Q6_DPAS_HEAD=str(packed),STRATA_Q6_HEAD_OWNER_GATE='1',
        LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/(mode+'-validation.log')))
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    outcome['native_execution_attempted']=True
    try:
        def interact(io,command):
            keys=('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER','UR_ADAPTERS_FORCE_LOAD','ZE_ENABLE_ALT_DRIVERS')
            all_required={str(Path(env[k]).resolve()) for k in keys}
            for phase,tag in enumerate(('STRATA_Q6_HEAD_OWNER_GATE ','STRATA_Q6_HEAD_OWNER_DEVICE_GATE ')):
                while True:
                    line=io.line(120)
                    if line.startswith(tag):break
                    require(not line.startswith('STRATA_Q6_HEAD_OWNER_'),'native gate order')
                identity=owner.identity(command['pid'])
                require(identity and identity['pid']==identity['sid']==command['pid']
                    and all(identity[k]==command['direct_identity'][k] for k in ('pid','start_ticks','sid'))
                    and line==tag+'pid='+str(command['pid']),'actual native owner at both gates')
                require(str(Path('/proc',str(command['pid']),'exe').resolve())==row['binary']['path'],'actual native ELF')
                actual=dict(v.split('=',1) for v in Path('/proc',str(command['pid']),'environ').read_text().split('\0') if '=' in v)
                mapped=set(python_runtime.code_maps(command['pid']))
                required=all_required if phase else {str(Path(env[k]).resolve()) for k in keys[:2]}
                missing=sorted(required-mapped);extra=sorted(mapped-(set(row['runtime_pins'])|{row['binary']['path']}))
                outcome.setdefault('provider_admission',[]).append(dict(phase='post-device-pre-fixture' if phase else 'pre-main',
                    mapped_files=len(mapped),mapped_path_sha256=hashlib.sha256('\0'.join(sorted(mapped)).encode()).hexdigest(),
                    missing_required=missing,unexpected_mapped_paths=extra,
                    not_yet_mapped_forced_providers=sorted(all_required-mapped),environment_immutable=actual==env))
                require(actual==env and not missing and not extra,'immutable environment and phase-appropriate providers')
                outcome['native_identity']=identity
                io.send(b'continue\n')
                if phase:row['GPU_executed']=True
            io.proc.stdin.close()
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
            io.exit(1)
        e,out,err=scope.run('native-'+mode,[row['binary']['path'],'--check',*[v['path'] for v in row['assets']]],
            env,ROOT,interaction=interact,**row['native_budget'])
        require(owner.closed(e) and e['exit_code']==0,'owned NativeHead process failed')
        text,diagnostic=out.read_text(),err.read_text()
        outcome['native_exit_code']=e['exit_code']
        outcome['validation']=validation.preload_evidence(diagnostic,e['pid'])
        require(outcome['validation']['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'actual validation loader')
        log=ROOT/(mode+'-validation.log')
        require(not log.exists() or log.stat().st_size==0,'parameter validation error')
        outcome['owner']=parse_native(text,packed)
        if packed:require('Q6DpasC4' in diagnostic and 'urEnqueueCommandBufferExp' in diagnostic,'traced packed graph path')
        require(model_assets()==row['assets'],'full post-run real-shard identities')
        outcome['assets_pre_post_match']=True
        outcome['passed']=True
    except BaseException as e:
        outcome.update(passed=False,error=repr(e))
    finally:
        validation.finish(outcome,scope,cursor,row['boot'])
        if not outcome['passed'] or outcome.get('driver_after')!=row['driver'] or 'error' in outcome:
            outcome['passed']=False
    require(outcome['passed'],outcome.get('error','native owner/fault/closure gate failed'))

def execute(mode):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    states=dict(controls='prepared',build='cpu-controlled',host='built',reference='host-checked',packed='reference-checked')
    require(mode in states and row['stage']==states[mode],'one ordered owner stage')
    scope=None;outcome=dict(mode=mode,passed=False,ownership_closed=False)
    def persist():
        (ROOT/'execution.json').write_text(json.dumps(dict(mode=mode,active=bool(scope and scope.active),
            commands=scope.commands if scope else []),indent=2)+'\n')
    try:
        scope=owner.Owner(ROOT/('owned-'+mode),persist)
        if mode=='controls':
            e,out,err=scope.run('reader-parser-controls',row['host_command'],row['environment'],P,**row['host_budget'])
            text=out.read_text()+err.read_text()
            require(owner.closed(e) and e['exit_code']==0 and 'Ran 13 tests' in text and '\nOK\n' in text
                and 'skipped' not in text,'all13 reader/parser controls')
            outcome.update(tests=13,GPU_executed=False);row['stage']='cpu-controlled'
        elif mode=='build':
            outputs={}
            for name,command in [('configure',row['configure']),('build',row['build'])]:
                e,out,err=scope.run(name,command,row['environment'],ROOT,**row['build_budget'])
                require(owner.closed(e) and e['exit_code']==0,name+' failed:'+(out.read_text()+err.read_text())[-2500:])
                outputs[name]=out
            e,out,_=scope.run('dependencies',[row['tools']['ninja']['path'],'-C',str(ROOT/'build'),'-t','deps'],
                row['environment'],ROOT,**row['build_budget'])
            require(owner.closed(e) and e['exit_code']==0,'actual native dependencies')
            row['actual_inputs']=census(row,out.read_text(),outputs['build'].read_text())
            binary=ROOT/'build'/TARGET
            row['binary']=dict(path=str(binary),sha256=sha(binary),bytes=binary.stat().st_size)
            database_path=ROOT/'build/compile_commands.json';database=json.loads(database_path.read_text())
            commands=[v for v in database if TARGET+'.dir' in v['command'] or 'strata_sycl_runtime.dir' in v['command']]
            require(len(commands)==15 and {str(Path(v['file']).relative_to(ROOT/'source')) for v in commands}==set(UNITS)
                and all('-fsycl' in shlex.split(v['command']) and '-fsycl-id-queries-range=size_t' in shlex.split(v['command']) for v in commands)
                and all('-fno-fast-math' in shlex.split(v['command']) and '-ffp-contract=off' in shlex.split(v['command'])
                    for v in commands if TARGET+'.dir' in v['command']),'exact15-unit build/flags')
            row['compile_database']=dict(path=str(database_path),sha256=sha(database_path),translation_units=15)
            row['codegen']=embedded_dpas(binary.read_bytes())
            outcome.update(GPU_executed=False,translation_units=15,compiler_files=row['actual_inputs']['compiler_files'],
                external_link_files=row['actual_inputs']['external_link_files'],dpas_static_calls=row['codegen']['dpas_static_calls'])
            row['stage']='built'
        elif mode=='host':
            cases=[('empty-owner',['--host'],{},'HOST owner_negative_cases=3 GPU=0'),
                ('bad-switch',['--reject-switch','must be 0 or 1'],{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'2'},'REJECT_SWITCH passed=1 GPU=0'),
                ('empty-switch',['--reject-switch','must be 0 or 1'],{'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':''},'REJECT_SWITCH passed=1 GPU=0'),
                ('release-off',['--reject-switch','requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'1','STRATA_PREFILL_RELEASE_HEADS':'0'},'REJECT_SWITCH passed=1 GPU=0'),
                ('release-absent',['--reject-switch','requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'1','STRATA_PREFILL_RELEASE_HEADS':None},'REJECT_SWITCH passed=1 GPU=0')]
            for name,args,changes,expected in cases:
                env=dict(row['environment']);env.update(changes);env={k:v for k,v in env.items() if v is not None}
                e,out,err=scope.run(name,[row['binary']['path'],*args],env,ROOT,**row['host_budget'])
                require(owner.closed(e) and e['exit_code']==0 and out.read_text().splitlines()==[expected],name+' host rejection')
            outcome.update(cases=5,GPU_executed=False);row['stage']='host-checked'
        else:
            run_gpu(row,scope,mode,outcome)
            row['stage']='reference-checked' if mode=='reference' else 'closed'
            if mode=='packed':row['complete']=True
        current(json.loads((ROOT/'record.json').read_text()))
        outcome.update(passed=True,ownership_closed=True)
    except BaseException as e:
        outcome['error']=repr(e);row.update(stage='failed',passed=False,complete=False)
    finally:
        if scope:
            require(not scope.active,'owned process still active')
            row['commands'].extend(scope.commands)
            outcome['ownership_closed']=all(owner.closed(v) for v in scope.commands)
        row['results'].append(outcome);row['active']=False;row['ownership_closed']=outcome['ownership_closed']
        row['passed']=bool(row['complete'] and outcome['passed'] and row['ownership_closed'])
        persist();save(row);print(json.dumps(dict(mode=mode,stage=row['stage'],passed=outcome['passed'],
            ownership_closed=outcome['ownership_closed'],error=outcome.get('error'))),flush=True)
    require(outcome['passed'],outcome.get('error','owner stage failed'))

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(len(sys.argv)==2,'use prepare|controls|build|host|reference|packed')
        if sys.argv[1]=='prepare':prepare()
        else:execute(sys.argv[1])
if __name__=='__main__':main()
