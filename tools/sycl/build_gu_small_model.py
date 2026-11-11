#!/usr/bin/env python3
"""Relink the qualified compact-GU archive without rebuilding unrelated math.

Main-authored from this fork's qualified build records; repository LICENSE
applies. The baseline relink must reproduce its original ELF before advancing.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import build_gu_compact as component

P, B, RESULT = component.P, component.B, component.RESULT
ROOT, KEY = B/'gu-small-model-native-v2', 'gu_small_model_native_v2'
THIS = 'tools/sycl/build_gu_small_model.py'
CHECK = 'tools/sycl/check_gu_small_model.py'
TARGETS = component.device.build.TARGETS
require, sha = component.require, component.sha
original = component.device.original


def compact(row):
    return {k: v for k, v in row.items() if k != 'execution'}


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row, indent=2)+'\n')
    public = json.loads(RESULT.read_text()); public[KEY] = compact(row)
    RESULT.write_text(json.dumps(public, indent=2)+'\n')


def qualified_component(observers=()):
    extras = {THIS, CHECK, *observers}
    n = component.native(extra_observers=extras)
    row = json.loads((component.ROOT/'record.json').read_text())
    require(original.validation.committed(RESULT, (component.KEY,)) == component.compact(row)
        and row['stage'] == 'built' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'], 'closed committed GU build required')
    proof = copy.deepcopy(n['observer_admission'])
    for name in extras:
        require(name in proof['added_observers'], 'declared observer missing')
        del proof['added_observers'][name]
    proof['observer_source_files'] -= len(extras)
    require(proof == row['native_admission'] and n['binaries']['strata'] == row['engine'],
        'only named new observers may extend the original GU admission')
    for name, h in row['source_files'].items():
        require(hashlib.sha256(original.common.source.git(P, 'show', row['source_commit']+':'+name)).hexdigest()
            == h == sha(P/name), 'qualified GU source changed')
    require(sha(row['compile_seed']['path']) == row['compile_seed']['sha256']
        and all(sha(p) == h for p, h in row['link_inputs'].items())
        and sha(row['archive_tool']['path']) == row['archive_tool']['sha256']
        and sha(row['actual_headers']['path']) == row['actual_headers']['sha256']
        and all(sha(p) == h for p, h in json.loads(Path(row['actual_headers']['path']).read_text()).items())
        and sha(row['patched_archive']['path']) == row['patched_archive']['sha256']
        and sha(row['binary']['path']) == row['binary']['sha256'],
        'qualified archive, fixture, SDK or link input changed')
    device = original.validation.committed(RESULT, ('gu_compact_device_v4',))
    require(device['stage'] == 'closed' and device['passed'] and device['complete']
        and device['ownership_closed'] and not device['active']
        and device['binary'] == row['binary']
        and [r['mode'] for r in device['results']] == ['smoke', 'numerics', 'perf0', 'perf1', 'perf2']
        and all(r['passed'] and r['complete'] for r in device['results'])
        and device['decision']['eligible_rows'] == [1, 16, 32, 64, 65, 95, 96],
        'actual production numerical and repeated component gain gates required')
    return n, row


def abi(n):
    old = original.common.source.git(P, 'show', n['native_source_commit']+':'+component.HEADER).decode()
    new = (P/component.HEADER).read_text()
    def normalize(text):
        return re.sub(r'\s+', '', re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S))
    before, after = normalize(old), normalize(new)
    overload = ('booltry_gu_swiglu16(constuint16_t*X,constuint16_t*W,'
        'uint16_t*hidden,int64_tT,boolallow_compact);')
    require(after.count(overload) == 1 and after.replace(overload, '') == before,
        'Gemm class layout and original declarations must remain exact')
    return dict(original_header_commit=n['native_source_commit'],
        original_header_sha256=hashlib.sha256(old.encode()).hexdigest(),
        candidate_header_sha256=sha(P/component.HEADER),
        only_added_declaration=overload, original_class_layout_and_declarations_exact=True,
        link_existing_callers_required=True)


def current(row, *, observers=()):
    require(original.validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact relink plan/result')
    n, gu = qualified_component(observers)
    require(n['native_source_commit'] == row['native_source_commit']
        and gu['source_commit'] == row['wrapper_source_commit']
        and gu['patched_archive'] == row['patched_archive']
        and abi(n) == row['header_ABI'], 'declared relink basis changed')
    for name, h in row['source_files'].items():
        require(hashlib.sha256(original.common.source.git(P, 'show', row['source_commit']+':'+name)).hexdigest()
            == h == sha(P/name), 'relink controller source changed')
    require(all(sha(p) == h for p, h in row['link_inputs'].items()), 'frozen relink input changed')
    if row['stage'] == 'built':
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'candidate engine changed')
    return n


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh relink only')
    n, gu = qualified_component()
    build = component.NATIVE/'build'
    command = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', 'strata'], text=True).splitlines()[-1]
    require(command.startswith(': && ') and command.endswith(' && :'), 'known original engine link rule required')
    link = shlex.split(command[5:-5])
    require(link.count('libstrata_prefill.a') == 1 and link[link.index('-o')+1] == 'strata',
        'exact one-archive engine relink required')
    inputs = {str((build/p).resolve()): sha((build/p).resolve()) for p in link
        if (p.endswith('.a') or p.endswith('.o') or p.endswith('.so')) and (build/p).is_file()}
    inputs[str(Path(link[0]).resolve())] = sha(link[0])
    require(sum(p.endswith('.a') for p in inputs) == 9
        and sum('/CMakeFiles/strata.dir/' in p for p in inputs) == 2,
        'nine archives and two original program objects required')
    fixture_links = {}
    for target in TARGETS:
        if target == 'strata': continue
        rule = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', target], text=True).splitlines()[-1]
        require('libstrata_prefill.a' not in shlex.split(rule),
            'a fixture consuming the changed archive must be relinked: '+target)
        fixture_links[target] = hashlib.sha256(rule.encode()).hexdigest()
    for p, h in gu['link_inputs'].items():
        require(inputs[p] == h, 'original shared link input differs from qualified component')
    inputs[gu['patched_archive']['path']] = gu['patched_archive']['sha256']
    baseline = list(link); baseline[baseline.index('-o')+1] = str(ROOT/'baseline')
    candidate = list(link); candidate[candidate.index('-o')+1] = str(ROOT/'strata')
    candidate[candidate.index('libstrata_prefill.a')] = gu['patched_archive']['path']
    revision = original.common.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    sources = {name: sha(P/name) for name in (THIS, CHECK)}
    require(all(hashlib.sha256(original.common.source.git(P, 'show', revision+':'+name)).hexdigest() == h
        for name, h in sources.items()), 'commit both controllers before preparing')
    row = dict(stage='prepared', source_commit=revision, source_files=sources,
        native_source_commit=n['native_source_commit'], wrapper_source_commit=gu['source_commit'],
        baseline=n['binaries']['strata'], original_binaries=n['binaries'],
        unchanged_fixture_link_rules=fixture_links,
        patched_archive=gu['patched_archive'], header_ABI=abi(n), link_inputs=inputs,
        recipe=dict(cwd=str(build), baseline=baseline, candidate=candidate,
            environment=dict(n['recipe']['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))),
        budget=dict(wall=300, cpu=900, rss_cap=8<<30, text_cap=8<<20, file_cap=128<<20, total_cap=320<<20),
        plan='First relink the original objects/archives and require byte-identical original97fda ELF. Stop on any mismatch. Then replace only the qualified GUv5 prefill archive. Reuse eight exact original fixtures:their original link rules contain no changed prefill archive. Fresh native/phase checks and long/short/long model handoff precede clean matched32K/64K and independent sustained decode. No model result inherited.',
        retention='Temporary candidate alongside97fda native fallback,56cad9 imminent prefill and a347 independent decode references. Baseline relink is an identical proof copy,retire after committed hash evidence. Keep candidate only through fresh qualification/comparison;select the validated generation then retire superseded outputs. Old9af/552b already retired;legacy holds remain separate.',
        previous_failure=dict(key='gu_small_model_native_v1', source_commit='3e535fd9e39a4927fb029d4094ade9e482268c37',
            evidence_commit='2985c1153', stage='failed', GPU_executed=False, ownership_closed=True,
            reason='Both links exit0 and original-object baseline is byte-identical97fda,but the controller compared its mutated outcome row with the committed preparation. V2checks the frozen original row instead;all source,ABI,ELF and stop criteria remain unchanged. V1is not relabeled.'),
        commands=[], passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT/name).mkdir(mode=0o700)
    save(row)
    print('Prepared exact baseline control and one-archive model relink;commit before link.', flush=True)


def build():
    row = json.loads((ROOT/'record.json').read_text()); current(row)
    frozen = copy.deepcopy(row)
    require(row['stage'] == 'prepared' and row['ownership_closed'] and not row['active'], 'one prospective link')
    scope = None
    def persist():
        (ROOT/'execution.json').write_text(json.dumps(dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []), indent=2)+'\n')
    try:
        scope = component.owner.Owner(ROOT/'owned-link', persist)
        for label in ('baseline', 'candidate'):
            e, out, err = scope.run(label, row['recipe'][label], row['recipe']['environment'],
                Path(row['recipe']['cwd']), **row['budget'])
            require(component.owner.closed(e) and e['exit_code'] == 0, 'relink failed: '+label)
            path = ROOT/('baseline' if label == 'baseline' else 'strata')
            record = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
            row['commands'].append(dict(label=label, exit_code=0, ownership_closed=True,
                stdout_sha256=sha(out), stderr_sha256=sha(err), output=record))
            if label == 'baseline':
                require(record['sha256'] == row['baseline']['sha256'] and record['bytes'] == row['baseline']['bytes'],
                    'original-object control must reproduce byte-identical qualified baseline')
                row['original_object_relink_proof'] = dict(record, byte_identical_qualified_engine=True)
        current(frozen)
        row['binary'] = row['commands'][-1]['output']
        row['binaries'] = dict(row['original_binaries'], strata=row['binary'])
        require(set(row['binaries']) == set(TARGETS), 'all nine current output identities required')
        row.update(stage='built', passed=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', error=repr(error)); raise
    finally:
        row['active'] = bool(scope and scope.active)
        row['ownership_closed'] = bool(scope and not scope.active and all(component.owner.closed(e) for e in scope.commands))
        save(row)
    print('Relinked compact-GU model;fresh model/device gates remain open.', flush=True)


def native(expected=None, *, observers=()):
    row = json.loads((ROOT/'record.json').read_text()); n = current(row, observers=observers)
    require(row['stage'] == 'built' and row['passed'] and row['complete']
        and row['ownership_closed'] and not row['active'], 'closed committed model relink required')
    proof = dict(native_source_commit=row['native_source_commit'], wrapper_source_commit=row['wrapper_source_commit'],
        declared_inputs=component.native(extra_observers={THIS, CHECK, *observers})['observer_admission'],
        baseline_relink=row['original_object_relink_proof'], patched_archive=row['patched_archive'],
        header_ABI=row['header_ABI'], candidate_engine=row['binary'], eight_fixtures_reused_exactly=True,
        new_model_or_GPU_result_inherited=False)
    require(expected is None or expected == proof, 'native relink admission changed')
    return dict(n, binaries=row['binaries'], observer_admission=proof)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build'), 'prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); original.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else build()
