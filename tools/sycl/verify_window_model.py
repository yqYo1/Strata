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
ROOT = B / 'verify-window-policy-host-v1'
KEY = 'verify_window_policy_host_v1'
SELF = Path(__file__).resolve()
BEFORE = '0ec7f128ea973658e9653076a93f534658b2dfd6'
Q6 = '664a280d5aab4e272fef37538634207d38704b4c'
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
    data = base.committed(RESULT); component = data['q6_columns_component_v4']
    require(component['source_commit'] == Q6 and component['passed'] and component['complete']
            and component['ownership_closed'] and component['decision']['qualified_shapes'] == 28,
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
                'tools/sycl/measurement_owner.py'}
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
            print(label,flush=True)
            command,out,err=owner.run(label,argv,row['environment'],selected.model.ROOT/'build',**row['budgets'])
            require(supervision.closed(command) and command['exit_code']==0,
                    label+' failed: '+err.read_text()[-4000:])
            if label=='run-host':
                require(not err.read_text(), label+' runtime diagnostics')
                require(out.read_text()=='verify-window-policy: passed selections=28 phase-roundtrip=1 invalid-keys=4 retirement=1\n',
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


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('projection','prepare','check'), 'use projection|prepare|check')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1]=='projection':
            _,values=projection(base.basis.build.git(P,'rev-parse','HEAD').decode().strip())
            print(json.dumps(dict(source_files=len(values),changed_files=[*Q6_FILES,*PHASE_FILES,POLICY,FIXTURE],GPU_executed=False)))
        else: {'prepare':prepare,'check':check}[sys.argv[1]]()
