#!/usr/bin/env python3
"""Main-owned actual-ID cell64 benchmark build; repository LICENSE applies.

Reuse the fixed component kernel objects in its one live generation. Original
native/controller source identities remain distinct from later qualification.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_qsa_cell64 as b
import check_qsa_cell64 as c

P, B, ROOT, RESULT = b.P, b.B, b.ROOT, b.RESULT
CPP = 'bench/sycl/qsa_cell64.cpp'


def inputs(revision):
    pins = c.inputs(revision)
    for name in (str(Path(__file__).resolve().relative_to(P)), CPP):
        raw = b.gen.git('show', revision + ':' + name)
        b.guards.require(raw == (P/name).read_bytes(), 'commit exact actual-ID input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def admit(record, plan, executable=False):
    b.admission(record, True)
    # Recover the original build controller from Git. Later controller-only
    # additions may change this script,not the native inputs/recipe below.
    for name, sha in plan['input_sha256'].items():
        raw = b.gen.git('show', plan['source_commit'] + ':' + name)
        b.guards.require(hashlib.sha256(raw).hexdigest() == sha, 'original harness build source unrecoverable')
    b.guards.require(b.digest(P/CPP) == plan['input_sha256'][CPP], 'actual-ID native harness changed')
    for asset in plan['common_objects']:
        b.guards.require(b.digest(asset['path']) == asset['sha256'], 'fixed common native object changed')
    b.guards.require(b.compiled_inputs(record) == record['actual_compiled_inputs'],
                     'original component dependency closure changed')
    if executable:
        b.guards.require(plan['stage'] == 'built' and plan['passed'] and plan['ownership_closed'] and
            b.digest(plan['binary']['path']) == plan['binary']['sha256'], 'closed actual-ID executable required')


def prepare():
    record = b.gen.committed(RESULT)
    b.guards.require('actual_id_build' not in record and not (ROOT/'actual-id-build').exists(),
                     'fresh actual-ID build only')
    passed = [v for v in record.get('oracle_checks', []) if v['mode'] == 'oracle-component' and v['passed']]
    b.guards.require(len(passed) == 1 and passed[0]['complete'] and passed[0]['ownership_closed'] and
        passed[0]['fresh_faults'] == [], 'complete extended numerical/local/physical-address component required')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    b.admission(record, True)
    objects = [ROOT/f'input-{i}.o' for i in range(7)]
    common = [dict(path=str(p), bytes=p.stat().st_size, sha256=b.digest(p)) for p in objects]
    command = record['commands'][0]
    flags = command[:command.index('-MD')]
    root = ROOT/'actual-id-build'
    obj = root/'harness.o'; binary = ROOT/'actual-id-bench'
    plan = dict(stage='prepared', source_commit=revision, input_sha256=pins,
        native_kernel_source=record['source_commit'], original_kernel_recipe_commit='d320c1ff7',
        common_objects=common, component_result=dict(commit=revision,
            path=str(RESULT.relative_to(P)), mode='oracle-component',
            scope='One closed passed extended cohort;original observer failure stays failed.'),
        source_origin='OwnMIT208c diagnostic with independent scalar32/64spot oracle at0a3e79e98;no XeStrata application source.',
        commands=[[*flags,'-MD','-MF',str(root/'harness.d'),'-c',str(P/CPP),'-o',str(obj)],
                  [*flags,*map(str,objects),str(obj),'-pthread','-ldl','-o',str(binary)]],
        environment=dict(record['build_environment'], TMPDIR=str(root/'tmp')),
        budgets=dict(wall=420, cpu=600, rss_cap=8 << 30, text_cap=8 << 20,
                     file_cap=256 << 20, total_cap=512 << 20),
        output_binary=str(binary),
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False,
        scope='Two compile/link commands only. The seven kernel/runtime objects and original oracle/property fixtures remain unchanged. This is an extra fixture of the same live component generation,not an engine replacement. No benchmark execution follows merely from linking.',
        retention='Immediate actual-ID numerical/event smoke and repeated512/6491comparison after independent harness review. Retire this extra fixture/object and success logs when the candidate closes;original13known-good artifacts remain.')
    admit(record, plan)
    root.mkdir(mode=0o700);(root/'tmp').mkdir()
    b.write(root/'plan.json', plan);record['actual_id_build'] = plan;b.write(RESULT, record)
    print('Prepared fixed-object actual-ID harness;commit recipe before compile.', flush=True)


def build():
    record = b.gen.committed(RESULT);plan = record['actual_id_build'];root = ROOT/'actual-id-build'
    b.guards.require(plan['stage'] == 'prepared' and plan == json.loads((root/'plan.json').read_text()) and
        inputs(plan['source_commit']) == plan['input_sha256'] and not (root/'owned.json').exists(),
        'fresh committed actual-ID build required')
    admit(record, plan);owner = None
    def persist():
        plan['active'] = bool(owner and owner.active is not None)
        b.write(root/'owned.json', dict(plan, commands=owner.commands if owner else []))
    owner = b.supervision.Owner(root/'owned', persist)
    try:
        for index, command in enumerate(plan['commands']):
            print('Actual-ID harness compile/link ' + str(index+1) + '/2', flush=True)
            entry, _, _ = owner.run(str(index), command, plan['environment'], P, **plan['budgets'])
            b.guards.require(b.supervision.closed(entry) and entry['exit_code'] == 0, 'actual-ID harness build failed')
        admit(record, plan)
        allowed = {Path(v['path']).resolve():v['sha256'] for v in json.loads(Path(record['compiler_inputs']['path']).read_text())}
        allowed.update({(P/n).resolve():s for n,s in record['input_sha256'].items()})
        allowed.update({Path(n).resolve():s for n,s in record['generated'].items()})
        allowed[(P/CPP).resolve()] = plan['input_sha256'][CPP]
        target, deptext = (root/'harness.d').read_text().replace('\\\n',' ').split(':',1)
        b.guards.require(Path(target).resolve() == (root/'harness.o').resolve(), 'unexpected harness dependency target')
        actual = {}
        for name in shlex.split(deptext):
            path = Path(name).resolve();sha = b.digest(path)
            b.guards.require(allowed.get(path) == sha, 'unfrozen harness compiler dependency: ' + str(path))
            actual[str(path)] = sha
        b.guards.require(len(actual) > 100, 'incomplete harness dependency closure')
        binary = Path(plan['output_binary'])
        plan.update(stage='built', passed=True, complete=True,
            binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=b.digest(binary)),
            actual_compiled_inputs=dict(files=len(actual), source_sha256=actual))
    except BaseException as error:
        plan.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        plan['ownership_closed'] = owner.active is None and all(b.supervision.closed(v) for v in owner.commands)
        if not plan['ownership_closed']:
            plan.update(stage='failed', passed=False, complete=False, error='harness build ownership open')
        persist();b.write(RESULT, record)
    print(json.dumps({k:plan.get(k) for k in ('stage','passed','complete','ownership_closed','error')}), flush=True)
    return int(not plan['passed'])


if __name__ == '__main__':
    b.guards.require(sys.argv[1:] in (['prepare-build'],['build']), 'prepare-build|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.guards.no_model_owners([]);b.guards.driver_guard()
        raise SystemExit(prepare() if sys.argv[1] == 'prepare-build' else build())
