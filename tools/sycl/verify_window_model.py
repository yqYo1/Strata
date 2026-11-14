#!/usr/bin/env python3
"""Compose and check the explicit decode Q6 route; repository LICENSE.

Keep selected479a and both prefill implementations. This file prepares source
and host/syntax checks only; it cannot bypass the unresolved legacy-build
retirement gate or launch a model. Main owns all compilation and testing.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import cache_map_model_entry as selected
import measurement_owner as supervision

P, B, RESULT = selected.P, selected.B, selected.RESULT
base = selected.base
require, digest, write = base.require, base.digest, base.write
ROOT = B / 'verify-window-policy-host-v3'
KEY = 'verify_window_policy_host_v3'
SELF = Path(__file__).resolve()
BEFORE = '0ec7f128ea973658e9653076a93f534658b2dfd6'
Q6 = '225e13889f5ace66fb26fa054952ae588d87d159'
POLICY = 'include/strata/core/verify_window_policy.hpp'
FIXTURE = 'tests/core/verify_window_policy_test.cpp'
Q6_FILES = ('src/kernels/sycl/native_mmvq.cpp', 'include/strata/kernels/native_mmvq.hpp')
PHASE_FILES = ('include/strata/core/verify.hpp', 'src/core/verify.cpp',
               'src/program/generate.cpp', 'tests/sycl/verifier_retirement.cpp')


def projection(revision):
    _, before = selected.model.projection(worker=True)
    record = json.loads((selected.model.ROOT / 'record.json').read_text())
    require(record['source_commit'] == '479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54'
            and {n:hashlib.sha256(b).hexdigest() for n,b in before.items()} == record['native_sources'],
            'recover the exact selected native source first')
    values = dict(before)
    for name in Q6_FILES:
        require(base.basis.build.git(P, 'show', '79e240181:' + name) == before[name],
                'original Q6 reference source differs: ' + name)
        values[name] = base.basis.build.git(P, 'show', Q6 + ':' + name)
        require((P / name).read_bytes() == values[name], 'freeze the component-qualified Q6 source')
    for name in PHASE_FILES:
        require(name in before, 'phase patch needs an original selected source: ' + name)
        values[name] = base.basis.merge_delta(before[name], name, BEFORE, revision)
        require(base.basis.merge_delta(values[name], name, revision, BEFORE) == before[name],
                'phase patch is not reversible: ' + name)
    for name in (POLICY, FIXTURE):
        require(name not in before, 'new policy/host fixture unexpectedly exists in selected source')
        values[name] = base.basis.build.git(P, 'show', revision + ':' + name)
        require(values[name] == (P / name).read_bytes(), 'commit exact new policy/fixture')
    changed = {n for n,b in values.items() if before.get(n) != b}
    require(changed == set((*Q6_FILES, *PHASE_FILES, POLICY, FIXTURE)),
            'unrelated observer,math,placement or build changes entered the candidate')
    return record, values


def source_current(row):
    for name, sha in row['controller_inputs'].items():
        raw = base.basis.build.git(P, 'show', row['source_commit'] + ':' + name)
        require(hashlib.sha256(raw).hexdigest() == sha and digest(P/name) == sha,
                'frozen source/controller changed: ' + name)
    native, values = projection(row['source_commit'])
    require({n:hashlib.sha256(b).hexdigest() for n,b in values.items()} == row['native_sources'],
            'composed phase source changed')
    allowed = selected.model.inputs(native, reuse_revision=row['source_commit'],
                                    controller_changes=selected.NATIVE_CONTROLS)
    for name, raw in values.items():
        path = ROOT/'source'/name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'prepared phase source changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    require(digest(row['compiler']['path']) == row['compiler']['sha256'], 'compiler changed')
    return allowed


def compact(row):
    out = dict(row)
    pins = out.pop('native_sources')
    out['native_sources'] = dict(files=len(pins),
        aggregate_sha256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),
        changed={n:pins[n] for n in row['changed_files']},
        projection='tools/sycl/verify_window_model.py:projection')
    environment=out.pop('environment')
    out['environment'] = dict(inherit_selected_native=row['selected_native_source'],
        overrides={k:environment[k] for k in ('TMPDIR','XDG_CACHE_HOME')},
        sha256=hashlib.sha256(json.dumps(environment,sort_keys=True).encode()).hexdigest())
    return out


def save(row):
    write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def prepare():
    require(KEY not in base.committed(RESULT) and not ROOT.exists(), 'one fresh preparation only')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    require(base.basis.build.git(P, 'show', revision+':'+str(SELF.relative_to(P))) == SELF.read_bytes(),
            'commit the exact projection/check controller first')
    native, values = projection(revision)
    data = base.committed(RESULT)
    previous=data['verify_window_policy_host_v2']
    require(previous['stage']=='failed' and previous['ownership_closed'] and
            'TimeoutExpired' in previous['error'] and 'dpkg-query' in previous['error'] and
            len(previous['results'])==6 and all(x['exit_code']==0 for x in previous['results']),
            'preserve the completed-six-command scope with failed post-check')
    inventory=data['host_link_inventory_batch_v1']
    require(inventory['passed'] and inventory['complete'] and inventory['ownership_closed'] and
            inventory['actual_sha256']==inventory['expected_sha256'] and
            inventory['sources']['tools/sycl/host_link_inventory.py']==digest(P/'tools/sycl/host_link_inventory.py'),
            'qualify the same provider inventory before allowing batched-query control reuse')
    component = data['q6_byte_load_component_v2']
    require(component['source_commit'] == Q6 and component['passed'] and component['complete']
            and component['ownership_closed'] and component['decision']['qualified_shapes'] == 30,
            'closed repeated Q6 component qualification required')
    environment = dict(native['environment'])
    environment.update(TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    exe, obj = ROOT/'fixture', ROOT/'fixture.o'
    compiler = native['compiler']
    common = [compiler['path'], '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
              '-fno-fast-math', '-ffp-contract=off', '-I'+str(ROOT/'source/include')]
    commands = [('compile-host', [*common, '-MD', '-MF', str(ROOT/'fixture.d'), '-c',
                 str(ROOT/'source'/FIXTURE), '-o', str(obj)]),
                ('link-host', [compiler['path'], str(obj), '-Wl,-t', '-o', str(exe)]),
                ('run-host', [str(exe)])]
    # Read the selected recipe; remove object/dependency outputs and substitute
    # only the source tree. Generated selected headers remain read-only inputs.
    compdb = json.loads(subprocess.check_output(['/usr/bin/ninja', '-C',
        str(selected.model.ROOT/'build'), '-t', 'compdb'], env=environment, timeout=15))
    for name, label in zip(PHASE_FILES[1:], ('syntax-verify', 'syntax-program', 'syntax-retirement')):
        hits = [x for x in compdb if x['file'] == str(selected.model.ROOT/'source'/name)]
        require(len(hits) == 1, 'one selected compiler command required: ' + name)
        original = shlex.split(hits[0]['command']); args=[]; skip=False
        for arg in original:
            if skip: skip=False; continue
            if arg in ('-o', '-MF', '-MT', '-MQ'): skip=True; continue
            if arg in ('-MD', '-MMD', '-MP', '-c'): continue
            args.append(arg.replace(str(selected.model.ROOT/'source'), str(ROOT/'source')))
        require(not skip and args[0] == compiler['path'], 'unexpected selected compiler recipe')
        commands.append((label, [*args, '-fsyntax-only', '-MD', '-MF', str(ROOT/(label+'.d'))]))
    controls = {str(SELF.relative_to(P)), *PHASE_FILES, POLICY, FIXTURE,
                'tools/sycl/measurement_owner.py', 'tools/sycl/host_link_inventory.py',
                'tools/sycl/cache_map_model_entry.py'}
    pins = {n:hashlib.sha256(base.basis.build.git(P,'show',revision+':'+n)).hexdigest() for n in controls}
    row = dict(stage='prepared', source_commit=revision, selected_native_source=native['source_commit'],
        component_source=Q6, controller_inputs=pins, compiler=compiler,
        native_sources={n:hashlib.sha256(b).hexdigest() for n,b in values.items()},
        commands=commands, environment=environment, binary_path=str(exe),
        budgets=dict(wall=180,cpu=180,rss_cap=3<<30,text_cap=1<<20,file_cap=16<<20,total_cap=256<<20),
        changed_files=sorted((*Q6_FILES,*PHASE_FILES,POLICY,FIXTURE)),
        active=False,ownership_closed=True,passed=False,complete=False,
        GPU_executed=False,model_executed=False,inference_adopted=False,
        retention=dict(consumer='Immediate host policy/phase-cache and actual SYCL translation-unit checks before the decode-only model comparison',
            review_point='Retire successful compiler output/temporary host ELF after recording results and recovering source;selected nine ELFs remain unchanged',
            legacy_gate='retained_builds_review_20261103 remains partial;no new engine build or CPU-observer gate bypass'))
    ROOT.mkdir(mode=0o700)
    for directory in ('source','tmp','cache'): (ROOT/directory).mkdir()
    for name, raw in values.items():
        path=ROOT/'source'/name; path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    try: source_current(row)
    except BaseException as error:
        row.update(stage='failed-admission',error=repr(error));save(row);raise
    save(row)
    print('Prepared exact phase composition and host/syntax plan;commit before execution.',flush=True)


def check():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY] == compact(row) and row['stage']=='prepared'
            and not (ROOT/'commands').exists(), 'one committed fresh host/syntax check')
    allowed=source_current(row);owner=supervision.Owner(ROOT/'commands')
    owner.persist=lambda:write(ROOT/'execution.json',dict(commands=owner.commands))
    try:
        results=[]
        for label,argv in row['commands']:
            if label=='run-host':
                row['binary_before_run']=dict(path=row['binary_path'],sha256=digest(row['binary_path']))
                write(ROOT/'record.json',row)
            print(label,flush=True)
            command,out,err=owner.run(label,argv,row['environment'],selected.model.ROOT/'build',**row['budgets'])
            require(supervision.closed(command) and command['exit_code']==0,
                    label+' failed: '+err.read_text()[-4000:])
            if label=='run-host':
                require(digest(row['binary_path'])==row['binary_before_run']['sha256'], 'host executable changed')
                require(not err.read_text(), label+' runtime diagnostics')
                require(out.read_text()=='verify-window-policy: passed selections=30 phase-roundtrip=1 invalid-keys=4 retirement=1\n',
                        'host contract output differs')
            results.append(dict(label=label,exit_code=command['exit_code'],ownership_closed=True))
        actual={}
        for dep in sorted(ROOT.glob('*.d')):
            for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
                path=str(Path(name).resolve())
                require(path in allowed and digest(path)==allowed[path], 'uncovered compiler input: '+path)
                actual[path]=allowed[path]
        for line in (ROOT/'commands/link-host.stdout').read_text().splitlines():
            if not line.startswith('/'): continue
            path=Path(line).resolve()
            if path==ROOT/'fixture.o':continue
            require(str(path) in allowed and digest(path)==allowed[str(path)], 'uncovered host link input: '+str(path))
            actual[str(path)]=allowed[str(path)]
        require(len(actual)>100, 'compiler input closure incomplete')
        write(ROOT/'actual-inputs.json',actual);source_current(row)
        row.update(stage='closed',passed=True,complete=True,results=results,
            binary=dict(path=row['binary_path'],bytes=Path(row['binary_path']).stat().st_size,sha256=digest(row['binary_path'])),
            actual_inputs=dict(path=str(ROOT/'actual-inputs.json'),sha256=digest(ROOT/'actual-inputs.json'),files=len(actual)),
            decision='Host phase cache/dispatch/retirement and actual selected SYCL syntax pass only;GPU lifecycle/model speed/prefill preservation remain unqualified.')
    except BaseException as error:
        row.update(stage='failed',error=repr(error));raise
    finally:
        row['ownership_closed']=owner.active is None and all(supervision.closed(x) for x in owner.commands)
        row['results']=results if 'results' in locals() else []
        save(row)


# The closed legacy composition above remains available to its historical
# consumers. These actions check the new policy directly;they do not reconstruct
# or execute a model,or inherit its GPU qualification.
HALF_ROOT = B / 'q6-native-half-policy-host-v1'
HALF_KEY = 'q6_native_half_policy_host_v1'


def half_compact(row):
    out = {k:v for k,v in row.items() if k != 'sources'}
    out['source_inputs'] = dict(files=len(row['sources']),aggregate_sha256=hashlib.sha256(
        json.dumps(row['sources'],sort_keys=True).encode()).hexdigest())
    return out


def half_save(row):
    write(HALF_ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[HALF_KEY]=half_compact(row);write(RESULT,data)


def half_committed(key):
    original=json.loads(base.basis.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))))
    current=json.loads(RESULT.read_text())
    require(original[key]==current[key],'commit exact consumed result: '+key)
    return current[key]


def half_prepare():
    import device_fp64_probe as compiler_image
    require(not HALF_ROOT.exists() and HALF_KEY not in json.loads(RESULT.read_text()),'one fresh host-policy attempt')
    component=half_committed('q6_native_half_component_v3')
    require(component['passed'] and component['complete'] and component['ownership_closed'] and
        not component['active'] and component['decision']['qualified_shapes']==27 and
        len(component['processes'])==3,'closed native-half entry evidence required')
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    tracked=base.basis.build.git(P,'ls-files','-z','include','tools/sycl').decode().split('\0')
    names=sorted({n for n in tracked if n and (n.startswith('include/') or n.endswith('.py'))} |
        {POLICY,FIXTURE,'src/core/verify.cpp','src/program/generate.cpp','src/kernels/sycl/native_mmvq.cpp',
         'AGENTS.md','docs/ARTIFACT_RETENTION.md'})
    sources={n:hashlib.sha256(base.basis.build.git(P,'show',revision+':'+n)).hexdigest() for n in names}
    require(all(digest(P/n)==h for n,h in sources.items()),'commit exact policy,engine and controller inputs')
    image=compiler_image.toolchain();encoded=(json.dumps(image,indent=2)+'\n').encode()
    compiler='/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    exe=HALF_ROOT/'fixture'
    environment=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),LANG='C.UTF-8',LC_ALL='C.UTF-8',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib',TMPDIR=str(HALF_ROOT/'tmp'),
        XDG_CACHE_HOME=str(HALF_ROOT/'cache'),PYTHONDONTWRITEBYTECODE='1')
    flags=['--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13','-std=c++20','-O2','-DNDEBUG',
        '-Wall','-Wextra','-Werror','-ffp-contract=off','-fsanitize=undefined','-fno-sanitize-recover=all',
        '-I'+str(HALF_ROOT/'source/include')]
    commands=[['compile',[compiler,*flags,'-MD','-MF',str(HALF_ROOT/'fixture.d'),
        str(HALF_ROOT/'source'/FIXTURE),'-Wl,-t','-o',str(exe)]],['run',[str(exe)]]]
    row=dict(stage='prepared',source_commit=revision,sources=sources,
        component_evidence=dict(commit='3e6b836a438b419dcd43e2eab68ba3cc3f7ea798',key='q6_native_half_component_v3'),
        compiler=dict(path=compiler,sha256=digest(compiler),version=subprocess.check_output(
            [compiler,'--version'],text=True,env=environment,timeout=15).splitlines()[0]),
        toolchain=dict(path=str(HALF_ROOT/'toolchain.json'),sha256=hashlib.sha256(encoded).hexdigest(),
            files=len(image['files']),packages=image['packages']),environment=environment,commands=commands,
        budgets=dict(wall=120,cpu=120,rss_cap=3<<30,text_cap=1<<20,file_cap=32<<20,total_cap=64<<20),
        expected='verify-window-policy: passed selections=30 centering=18 native-half=26 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64\n',
        plan='Compile and run once with UBSan/no-recover. Check the exact independently measured26-cell incumbent intersection;prompt/disabled/nonexact/type/geometry refusals;actual phase/T cache and route metadata isolation;interrupted clear,recapture,bounded route table and invalid-route refusal. Stop on first error;no retries. This does not compile the full Verifier/program or run their GPU/capture/replay paths.',
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,model_executed=False,
        retention='Immediate host-policy check only. Retire the auxiliary ELF,source copies,compiler inventory and normal logs after recording outcomes and exact-source recovery;keep14inference ELFs untouched.')
    HALF_ROOT.mkdir(mode=0o700)
    for directory in ['tmp','cache','source']:(HALF_ROOT/directory).mkdir()
    (HALF_ROOT/'toolchain.json').write_bytes(encoded)
    for name in [POLICY,FIXTURE]:
        path=HALF_ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((P/name).read_bytes())
    half_save(row)
    print('Prepared UBSan host-policy check;commit plan before execution.',flush=True)


def half_check():
    import device_fp64_probe as compiler_image
    row=json.loads((HALF_ROOT/'record.json').read_text())
    require(half_committed(HALF_KEY)==half_compact(row) and row['stage']=='prepared' and
        not (HALF_ROOT/'commands').exists() and not (HALF_ROOT/'check-attempt.json').exists(),
        'one committed host-policy check')
    def inputs():
        require(all(digest(P/n)==h and hashlib.sha256(base.basis.build.git(P,'show',row['source_commit']+':'+n)).hexdigest()==h
            for n,h in row['sources'].items()),'host-policy inputs changed')
        require(digest(row['compiler']['path'])==row['compiler']['sha256'] and
            digest(row['toolchain']['path'])==row['toolchain']['sha256'] and
            compiler_image.toolchain()==json.loads(Path(row['toolchain']['path']).read_text()),'host compiler/package image changed')
        for n in [POLICY,FIXTURE]:require(digest(HALF_ROOT/'source'/n)==row['sources'][n],'private host input changed')
    # Consume the attempt before admission/Owner setup:an early failure must
    # remain terminal instead of silently permitting the same test to restart.
    with (HALF_ROOT/'check-attempt.json').open('x') as receipt:
        json.dump(dict(source_commit=row['source_commit'],operation='native-half-check'),receipt)
    owner=None
    row.update(stage='running',active=True,ownership_closed=False);half_save(row)
    try:
        inputs();owner=supervision.Owner(HALF_ROOT/'commands')
        owner.persist=lambda:write(HALF_ROOT/'execution.json',dict(commands=owner.commands))
        for label,argv in row['commands']:
            if label=='run':row['binary']=dict(sha256=digest(argv[0]),bytes=Path(argv[0]).stat().st_size)
            entry,out,err=owner.run(label,argv,row['environment'],HALF_ROOT,**row['budgets'])
            require(supervision.closed(entry) and entry['exit_code']==0,label+' failed: '+err.read_text()[-4000:])
            if label=='run':require(out.read_text()==row['expected'] and not err.read_text() and
                digest(argv[0])==row['binary']['sha256'],'host policy/UBSan output or executable mismatch')
        image=json.loads(Path(row['toolchain']['path']).read_text())
        allowed={v['realpath']:v['sha256'] for v in image['files'].values() if v['kind']=='file'}
        allowed.update({str((HALF_ROOT/'source'/n).resolve()):row['sources'][n] for n in [POLICY,FIXTURE]})
        names=shlex.split((HALF_ROOT/'fixture.d').read_text().replace('\\\n',' ').split(':',1)[1])
        names += [line.strip() for line in (HALF_ROOT/'commands/compile.stdout').read_text().splitlines() if line.startswith('/')]
        used={}
        for name in names:
            path=Path(name).resolve()
            if path.is_relative_to(HALF_ROOT/'tmp'):continue
            require(str(path) in allowed and digest(path)==allowed[str(path)],'uncovered host compiler/link input: '+str(path))
            used[str(path)]=allowed[str(path)]
        require(len(used)>100,'missing actual host compiler/link inputs')
        inputs();row.update(stage='closed',passed=True,complete=True,actual_inputs=dict(files=len(used),
            sha256=hashlib.sha256(json.dumps(used,sort_keys=True).encode()).hexdigest()),
            results=[dict(label=e['label'],pid=e['pid'],start_ticks=e['direct_identity']['start_ticks'],exit_code=e['exit_code'],
                ownership_closed=supervision.closed(e)) for e in owner.commands],
            scope='UBSan host policy/cache/route-table checks only;full engine syntax/build,GPU routing/model speed and adoption remain unqualified.')
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,error=repr(error));raise
    finally:
        row.update(active=owner is not None and owner.active is not None,
            ownership_closed=owner is None or (owner.active is None and
                all(supervision.closed(e) for e in owner.commands)))
        half_save(row)
    print('Host native-half policy/cache check passed;full engine/GPU/model qualification remains pending.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('projection','prepare','check','native-half-prepare','native-half-check'),
        'use projection|prepare|check|native-half-prepare|native-half-check')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1]=='native-half-prepare':half_prepare()
        elif sys.argv[1]=='native-half-check':half_check()
        elif sys.argv[1]=='projection':
            _,values=projection(base.basis.build.git(P,'rev-parse','HEAD').decode().strip())
            print(json.dumps(dict(source_files=len(values),changed_files=[*Q6_FILES,*PHASE_FILES,POLICY,FIXTURE],GPU_executed=False)))
        else: {'prepare':prepare,'check':check}[sys.argv[1]]()
