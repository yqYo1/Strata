#!/usr/bin/env python3
"""Check the diagnostic admission/terminal bounds on the CPU; never run a model."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import build_engine as build
import measurement_guards as guards
import measurement_owner as supervision

P, B = build.P, build.B
ROOT = B / 'full-context-source-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/full-context-current.json'
REFERENCE = B / 'main-native-gdn-engine/record.json'
REFERENCE_SHA = 'a38f477670fafcd1a931be2e20d90d6d9718df75b61954b421f361af9f6881b2'
INPUTS = ['tools/sycl/check_full_context_source.py', 'tools/sycl/build_engine.py',
          'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py',
          'tools/sycl/measurement_environment.json', 'src/program/generate.cpp',
          'include/strata/program/full_context_diagnostic.hpp',
          'tests/core/full_context_diagnostic_test.cpp']


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def inputs(revision):
    found = {}
    for name, data in build.blobs(P, revision, INPUTS):
        guards.require(data == (P / name).read_bytes(), 'commit exact input: ' + name)
        found[name] = hashlib.sha256(data).hexdigest()
    return found


def baseline():
    guards.require(build.digest(REFERENCE) == REFERENCE_SHA, 'selected original source receipt changed')
    full = json.loads(REFERENCE.read_text())
    guards.require(full['source_commit'] == '6e40afc1beddd059ee518617f6c1efce1da14aca' and
                   full['passed'] and full['complete'] and full['ownership_closed'] and not full['active'],
                   'selected original build is not closed/qualified')
    count = 0
    for source in [full['project'], *full['dependencies']]:
        recovered = {name: hashlib.sha256(data).hexdigest() for name, data in
                     build.blobs(Path(source['root']), source['revision'], source['source_sha256'])}
        guards.require(recovered == source['source_sha256'], 'original source unrecoverable')
        count += len(recovered)
    guards.require(count == 9853, 'original source count')
    recipe = full['recipe_reference']
    original_recipe = build.git(Path(full['project']['root']), 'show',
                                full['project']['revision'] + ':' + recipe['path'])
    guards.require(hashlib.sha256(original_recipe).hexdigest() == recipe['sha256'],
                   'original recipe unrecoverable')
    for item in [full['binary'], *full['fixtures'].values()]:
        guards.require(build.digest(item['path']) == item['sha256'], 'selected asset changed')
    guards.require(build.digest(full['compiler']['path']) == full['compiler']['sha256'], 'compiler changed')
    return full, dict(source_commit=full['source_commit'], inputs_recovered=count, assets_verified=13,
                      receipt=dict(path=str(REFERENCE), sha256=REFERENCE_SHA),
                      limits='Original build remains selected; this source-only diagnostic is not adoption.')


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'single-use source gate; inspect existing ownership')
    revision = build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    full, reference = baseline()
    ROOT.mkdir(mode=0o700)
    compiler = '/usr/bin/g++'
    record = dict(stage='prepared', passed=False, complete=False, active=False, GPU_executed=False,
        native_engine_built=False, native_source_changed=True, production_selected=False,
        model_adoption_qualified=False, physical262144_qualified=False,
        source_commit=revision, input_sha256=pins, baseline=reference,
        compiler=dict(path=compiler, sha256=build.digest(compiler)),
        command=[compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
                 '-I', str(P / 'include'), str(P / INPUTS[-1]), '-o', str(ROOT / 'host_bounds')],
        environment=full['environment'],
        criteria='Exactly134 admission/terminal boundary cases,including integer extremes,final-slot/width/token/output refusal and capture-before-clearing-first-window. Normal owned compile/fixture exits; no GPU/model execution.',
        next_gate='Native diagnostic syntax/build,actual262144 allocation ledger/admission,bounded validation,full-fill/near-cap/RESTORE/failure-owner checks. No model claim from this host fixture.',
        retention='Keep source/recipe and compact case/outcome. Host executable and normal compile/test logs are disposable after source-gate closure. Original selected13assets remain intact.')
    write(ROOT / 'plan.json', record)
    write(RESULT, {k:v for k,v in record.items() if k != 'environment'})
    print('Prepared source-only gate; commit plan before checking.', flush=True)


def check():
    compact = json.loads(RESULT.read_text())
    guards.require(RESULT.read_bytes() == build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))),
                   'commit exact plan first')
    record = json.loads((ROOT / 'plan.json').read_text())
    guards.require(compact == {k:v for k,v in record.items() if k != 'environment'} and
                   record['stage'] == 'prepared' and inputs(record['source_commit']) == record['input_sha256'],
                   'frozen source/plan differs')
    full, reference = baseline()
    guards.require(reference == record['baseline'] and full['environment'] == record['environment'] and
                   build.digest(record['compiler']['path']) == record['compiler']['sha256'], 'build inputs changed')
    def persist():
        record['commands'] = owner.commands
        write(ROOT / 'record.json', record)
    owner = supervision.Owner(ROOT / 'owned', persist)
    record.update(stage='running', active=True)
    persist()
    try:
        limits = dict(wall=30, cpu=30, rss_cap=1<<30, text_cap=1<<20, total_cap=4<<20, file_cap=4<<20)
        entry, _, _ = owner.run('compile', record['command'], record['environment'], P, **limits)
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'host fixture compile failed')
        fixture = ROOT / 'host_bounds'
        record['fixture'] = dict(path=str(fixture), sha256=build.digest(fixture), bytes=fixture.stat().st_size)
        entry, out, _ = owner.run('check', [str(fixture)], record['environment'], ROOT, **limits)
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0 and
                       out.read_text() == 'cases=134 full_context_request_and_terminal_bounds=PASS\n',
                       'host boundary fixture failed')
        guards.require(baseline()[1] == reference, 'selected baseline changed during CPU check')
        record.update(stage='closed', passed=True, complete=True, cases=134)
    except BaseException as error:
        record.update(stage='failed', error=repr(error))
    finally:
        record['active'] = owner.active is not None
        record['ownership_closed'] = not record['active'] and all(supervision.closed(c) for c in owner.commands)
        record['passed'] = bool(record['passed'] and record['ownership_closed'])
        persist()
        write(RESULT, {k:v for k,v in record.items() if k not in ('environment', 'commands')})
    print(json.dumps({k:record.get(k) for k in ['stage','passed','complete','cases','ownership_closed','error']}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    guards.require(sys.argv[1:] in (['prepare'], ['check']), 'prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else sys.exit(check())
