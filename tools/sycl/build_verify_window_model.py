#!/usr/bin/env python3
"""Prepare one decode-Q6 comparison from selected479a; repository LICENSE.

Only the eight phase/Q6 source changes enter the engine. Main owns all builds
and tests. Syntax checks may precede legacy retention reconciliation; creating
the engine generation may not. No GPU or model execution in this controller.
"""
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import verify_window_model as phase
import measurement_owner as supervision

selected, base = phase.selected, phase.base
P, B, RESULT = phase.P, phase.B, phase.RESULT
ROOT = B / 'verify-window-model-v3'
KEY = 'verify_window_model_build_v3'
PREVIOUS = 'verify_window_model_build_v2'
HOST_ROOT = B / 'verify-window-build-monitor-host-v1'
HOST_KEY = 'verify_window_build_monitor_host_v1'
SELF = Path(__file__).resolve()
require, digest, write = phase.require, phase.digest, phase.write
TARGETS = ['strata', 'sycl_submission_test', 'sycl_verifier_retirement_test']


def compact(row):
    out = base.compact(row)
    recipe = out.pop('recipe')
    out['recipe'] = dict(inherit_selected_native='479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54',
        sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        targets=TARGETS, changes='Only generation paths;compiler,optimization,dependencies unchanged')
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def inputs(row, *, syntax=False):
    require(base.controls(row['source_commit'], controller=SELF) == row['controller_inputs'],
            'commit and freeze all transitive build controls')
    native, values = phase.projection(row['source_commit'])
    require(digest(selected.model.ROOT / 'record.json') == row['selected_record_sha256'] and
            native['binaries'] == row['selected_binaries'] and
            {n:hashlib.sha256(b).hexdigest() for n,b in values.items()} == row['native_sources'],
            'selected source/ELFs or phase composition changed')
    allowed = selected.model.inputs(native, reuse_revision=row['source_commit'],
                                    controller_changes=selected.NATIVE_CONTROLS)
    if not syntax:
        allowed = {n:h for n,h in allowed.items() if not Path(n).is_relative_to(selected.model.ROOT)}
    for name, raw in values.items():
        path = ROOT / 'source' / name
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                'prepared native source changed: ' + name)
        allowed[str(path.resolve())] = row['native_sources'][name]
    allowed.update(row['generated_expected'])
    return allowed


def retention_gate():
    review = base.committed(RESULT)['retained_builds_review_20261103']
    require(review['stage'] == 'reconciled' and review['passed'] is True and
            review['selected_root'] == str(selected.model.ROOT) and
            review['next_comparison_root'] == str(ROOT) and
            review.get('original_inputs_recovered_or_failure_recorded') is True,
            'reconcile the original-source/consumer inventory before creating this engine generation')
    manifest = review['retirement_manifest']
    require(digest(manifest['path']) == manifest['sha256'], 'retirement manifest changed')
    retired = json.loads(Path(manifest['path']).read_text())['files']
    require(retired and all(not Path(x['path']).exists() for x in retired),
            'recorded retirement is not physically complete')
    retained = review['retained_project_binaries']
    require(retained and all(x.get('consumer') and Path(x['path']).is_file() and
            digest(x['path']) == x['sha256'] for x in retained), 'retained consumers changed')
    native = json.loads((selected.model.ROOT / 'record.json').read_text())
    require({x['path'] for x in native['binaries'].values()} <= {x['path'] for x in retained},
            'protect all nine selected fallback ELFs')


def monitor(root, deadline, io, entry, total_cap=4 << 30):
    next_size = 0.
    while not io.exit_watch.poll(0) or not io.eof:
        io.poll()
        require(time.monotonic() < deadline, 'whole generation deadline')
        if time.monotonic() >= next_size:
            require(supervision.artifact_bytes(root) <= total_cap, 'generation byte budget')
            next_size = time.monotonic() + 1
        if select.select([io.proc.stdout], [], [], .02)[0]:
            io.read()
            io.pending.clear()
    io.exit(1)


def check_monitor():
    data = base.committed(RESULT)
    require(HOST_KEY not in data and not HOST_ROOT.exists(), 'one fresh CPU monitor control')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    controls = base.controls(revision, controller=SELF)
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C', LC_ALL='C',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    size = 2 << 20
    argv = ['/usr/bin/python3', '-c',
            'import sys;sys.stdout.buffer.write(b"x"*(2<<20));sys.stdout.buffer.flush()']
    HOST_ROOT.mkdir(mode=0o700)
    owner = supervision.Owner(HOST_ROOT / 'owned')
    row = dict(stage='prepared', source_commit=revision, controller_inputs=controls,
        command=argv, interpreter=dict(path=argv[0], sha256=digest(argv[0])), environment=env,
        budgets=dict(wall=15, cpu=10, rss_cap=128 << 20, text_cap=3 << 20,
                     file_cap=3 << 20, total_cap=4 << 20),
        expected_stdout_bytes=size, expected_stdout_sha256=hashlib.sha256(b'x' * size).hexdigest(),
        plan='Real child writes2MiB without a newline, exceeding pipe and pending-buffer capacity. Require exact bytes/hash,EOF,exit0 and closed owners through the production build monitor.',
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False)
    try:
        deadline = time.monotonic() + row['budgets']['wall']
        entry, out, err = owner.run('pipe-eof', argv, env, HOST_ROOT,
            interaction=lambda io, e: monitor(HOST_ROOT, deadline, io, e, 4 << 20),
            **row['budgets'])
        require(supervision.closed(entry) and entry['exit_code'] == 0 and
                not err.stat().st_size and out.stat().st_size == size and
                digest(out) == row['expected_stdout_sha256'], 'stdout/EOF monitor control failed')
        row.update(stage='closed', passed=True, complete=True, stdout_bytes=out.stat().st_size,
                   stdout_sha256=digest(out), native_exit=entry['exit_code'])
    except BaseException as error:
        row.update(stage='failed', error=repr(error)); raise
    finally:
        row.update(ownership_closed=owner.active is None and all(supervision.closed(e) for e in owner.commands),
                   commands_executed=owner.commands,
                   retention='Keep compact CPU outcome;retire successful2MiB stdout after first candidate build closes.')
        write(HOST_ROOT / 'record.json', row)
        out = {k:v for k,v in row.items() if k not in ('controller_inputs', 'commands_executed')}
        out['controller_inputs'] = dict(files=len(controls), aggregate_sha256=hashlib.sha256(
            json.dumps(controls, sort_keys=True).encode()).hexdigest())
        data[HOST_KEY] = out; write(RESULT, data)
    print('Production build monitor passes real2MiB pipe/backpressure/EOF control;no GPU.', flush=True)


def monitor_qualification(data):
    host=data[HOST_KEY]
    original=json.loads((HOST_ROOT/'record.json').read_text())
    require(host['passed'] and host['complete'] and host['ownership_closed'] and
            not host['GPU_executed'], 'closed real pipe/EOF control required')
    old=base.basis.build.git(P,'show',original['source_commit']+':'+str(SELF.relative_to(P)))
    def body(raw):
        return ast.dump(next(node for node in ast.parse(raw).body
                             if isinstance(node,ast.FunctionDef) and node.name=='monitor'))
    require(body(old)==body(SELF.read_bytes()), 'changed monitor needs a fresh CPU qualification')
    names=('tools/sycl/measurement_owner.py','tools/sycl/measurement_guards.py')
    require(all(digest(P/name)==original['controller_inputs'][name] for name in names) and
            digest(host['interpreter']['path'])==host['interpreter']['sha256'],
            'monitor owner/guard/interpreter changed')
    return dict(original_source_commit=original['source_commit'],evidence_key=HOST_KEY,
        monitor_AST_sha256=hashlib.sha256(body(old).encode()).hexdigest(),
        unchanged_helpers={name:original['controller_inputs'][name] for name in names},
        scope='Reuse only the identical monitor function,owner,guard and interpreter;not the new preparation or metadata-query code.')


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'one fresh prospective phase/Q6 generation')
    previous = data[PREVIOUS]
    require(previous['stage']=='built-not-executed' and previous['passed'] and
            previous['complete'] and previous['ownership_closed'] and
            not (B/'verify-window-model-v2').exists(),
            'recover and retire the superseded pre-fix generation before new preparation')
    monitor_basis=monitor_qualification(data)
    host = data[phase.KEY]
    require(host['passed'] and host['complete'] and host['ownership_closed'] and
            not host['GPU_executed'], 'closed phase-policy host/syntax evidence required')
    component = data['q6_byte_load_component_v2']
    require(component['passed'] and component['complete'] and component['ownership_closed'] and
            component['source_commit']==phase.Q6 and
            component['decision']['qualified_shapes'] == 30, 'closed fixed-source 30-shape Q6 evidence required')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    native, values = phase.projection(revision)
    host_record=json.loads((phase.ROOT/'record.json').read_text())
    phase.source_current(host_record)
    require(phase.compact(host_record)==host and
            host_record['native_sources']=={n:hashlib.sha256(raw).hexdigest() for n,raw in values.items()} and
            digest(host['binary']['path'])==host['binary']['sha256'],
            'same-source current30 host fixture and composition required')
    # The closed component's record/map remain as provenance,not its retired
    # standalone ELF. Bind its actual tested native fields and build recipe.
    component_root=B/'q6-byte-load-component-v2'
    component_record=json.loads((component_root/'record.json').read_text())
    for name in phase.Q6_FILES:
        require(hashlib.sha256(values[name]).hexdigest()==component_record['project_inputs']['files'][name],
                'model Q6 source differs from the numerical component')
    require(component_record['build_input_pins']==component['build_input_pins'] and
            digest(component['actual_inputs']['path'])==component['actual_inputs']['sha256'],
            'component source/compile evidence changed')
    old = str(selected.model.ROOT)
    recipe = dict(configure=[x.replace(old, str(ROOT)) for x in native['configure_argv']],
        environment={k:v.replace(old,str(ROOT)) for k,v in native['environment'].items()})
    compdb = json.loads(subprocess.check_output(['/usr/bin/ninja','-C',old+'/build','-t','compdb'],
        env=native['environment'],timeout=15))
    name = phase.PHASE_FILES[-1]
    matches = [x for x in compdb if x['file'] == old+'/source/'+name]
    require(len(matches) == 1, 'one actual selected retirement compile recipe required')
    args, skip = [], False
    for arg in shlex.split(matches[0]['command']):
        if skip: skip=False; continue
        if arg in ('-o','-MF','-MT','-MQ'): skip=True; continue
        if arg in ('-MD','-MMD','-MP','-c'): continue
        args.append(arg.replace(old+'/source',str(ROOT/'source')))
    require(not skip and args[0] == native['compiler']['path'], 'unexpected syntax compiler')
    row = dict(stage='prepared',source_commit=revision,controller_inputs=base.controls(revision,controller=SELF),
        selected_record_sha256=digest(selected.model.ROOT/'record.json'),selected_binaries=native['binaries'],
        selected_native_source=native['source_commit'],q6_source=phase.Q6,
        monitor_qualification=monitor_basis,
        native_sources={n:hashlib.sha256(b).hexdigest() for n,b in values.items()},
        changed_files=sorted((*phase.Q6_FILES,*phase.PHASE_FILES,phase.POLICY,phase.FIXTURE)),
        recipe=recipe,generated_expected={n.replace(old,str(ROOT)):h for n,h in native['generated_expected'].items()},
        syntax_command=[*args,'-fsyntax-only','-MD','-MF',str(ROOT/'syntax.d')],
        commands=[['configure',recipe['configure']],['compile',['/usr/bin/cmake','--build',str(ROOT/'build'),
            '--target',*TARGETS,'-j','4']],['dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps']]],
        budgets=dict(wall=1800,cpu=7200,rss_cap=12<<30,text_cap=16<<20,file_cap=512<<20,total_cap=4<<30),
        whole_wall_seconds=3600,commands_executed=[],active=False,ownership_closed=True,
        passed=False,complete=False,GPU_executed=False,model_executed=False,inference_adopted=False,
        GPU_plan=dict(first_case=['sycl_verifier_retirement_test','--phase-cache'],
            contract='16 callback windows;prompt/decode T1,4,8;12 lazy captures;drained clear/recapture;48 real graph owners;guards and copy counts;normal destructor',
            limits='Synthetic payload kernels. Does not qualify record_window/run,Q6 arithmetic,dense/head resume,physical262144 or model performance.',
            prerequisites='Committed bounded validation/log/fault/owner plan after this build;no GPU execution here'),
        retention=dict(consumer='One imminent decode-only Q6 model comparison against selected479a',
            review_point='At comparison closure select the validated generation, recover source/recipe and retire the extra',
            legacy_gate='Preparation/syntax may precede inventory closure;engine compilation must pass retention_gate'))
    ROOT.mkdir(mode=0o700)
    for folder in ('source','tmp','cache'): (ROOT/folder).mkdir()
    for name, raw in values.items():
        path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    try: inputs(row,syntax=True)
    except BaseException as error:
        row.update(stage='failed-admission',error=repr(error));save(row);raise
    save(row)
    print('Prepared phase/Q6 engine and direct, no-fork GPU fixture;commit before syntax/build.',flush=True)


def syntax():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY] == compact(row) and row['stage']=='prepared' and
            not (ROOT/'syntax-check').exists(), 'one committed syntax check')
    allowed=inputs(row,syntax=True);owner=supervision.Owner(ROOT/'syntax-check')
    try:
        command,out,err=owner.run('retirement',row['syntax_command'],row['recipe']['environment'],
            selected.model.ROOT/'build',wall=180,cpu=180,rss_cap=3<<30,text_cap=1<<20,
            file_cap=16<<20,total_cap=256<<20)
        require(supervision.closed(command) and command['exit_code']==0,
                'SYCL fixture syntax failed: '+err.read_text()[-3000:])
        paths={str(Path(x).resolve()) for x in shlex.split((ROOT/'syntax.d').read_text().replace('\\\n',' ').split(':',1)[1])}
        require(len(paths)>100 and all(allowed.get(p)==digest(p) for p in paths), 'uncovered syntax input')
        row.update(stage='syntax-checked',syntax=dict(passed=True,compiler_inputs=len(paths),exit_code=0))
    except BaseException as error:
        row.update(stage='failed-syntax',error=repr(error));raise
    finally:
        row.update(ownership_closed=owner.active is None and all(supervision.closed(x) for x in owner.commands))
        save(row)
    print('Phase-cache GPU fixture syntax passes;no engine or GPU execution.',flush=True)


def build():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY] == compact(row) and row['stage']=='syntax-checked' and
            not (ROOT/'owned-build').exists(), 'one frozen new-engine build after syntax')
    retention_gate();allowed=inputs(row);frozen=copy.deepcopy(row)
    owner=supervision.Owner(ROOT/'owned-build');deadline=time.monotonic()+row['whole_wall_seconds']
    owner.persist=lambda:write(ROOT/'execution.json',dict(commands=owner.commands))
    try:
        for label,argv in row['commands']:
            print('Phase build:',label,flush=True)
            entry,out,err=owner.run(label,argv,row['recipe']['environment'],ROOT,
                interaction=lambda io, e: monitor(ROOT, deadline, io, e),
                **dict(row['budgets'],wall=min(row['budgets']['wall'],deadline-time.monotonic())))
            require(supervision.closed(entry) and entry['exit_code']==0,
                    label+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-4000:])
        inputs(frozen)
        paths={str((Path(x.strip()) if Path(x.strip()).is_absolute() else ROOT/'build'/x.strip()).resolve())
               for x in out.read_text().splitlines() if x.startswith('    ') and x.strip()}
        required={str(ROOT/'source'/n) for n in (phase.Q6_FILES[0],*phase.PHASE_FILES[1:],phase.POLICY)}
        require(required<=paths and len(paths)>600 and all(allowed.get(p)==digest(p) for p in paths),
                'uncovered or incomplete compiled native inputs')
        write(ROOT/'compiler-inputs.json',{p:digest(p) for p in sorted(paths)})
        row.update(stage='built-not-executed',passed=True,complete=True,
            compiled_inputs=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(paths)),
            compile_database=dict(path=str(ROOT/'build/compile_commands.json'),sha256=digest(ROOT/'build/compile_commands.json')),
            actual_link_inputs=base.traced_link_inputs(allowed,ROOT),
            binaries={n:dict(path=str(ROOT/'build'/n),sha256=digest(ROOT/'build'/n),bytes=(ROOT/'build'/n).stat().st_size) for n in TARGETS})
    except BaseException as error:
        row.update(stage='failed-build',passed=False,complete=False,error=repr(error));raise
    finally:
        row.update(commands_executed=owner.commands,ownership_closed=owner.active is None and all(supervision.closed(x) for x in owner.commands))
        save(row)
    print('Build closed;new engine remains unselected and unqualified for GPU/model use.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','syntax','build','check-monitor'),
            'use prepare|syntax|build|check-monitor')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        {'prepare':prepare,'syntax':syntax,'build':build,'check-monitor':check_monitor}[sys.argv[1]]()
