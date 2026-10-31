#!/usr/bin/env python3
"""Bounded official-SDK compatibility smoke before public GU calibration.

Main-authored, repository LICENSE. No model execution or engine selection.
Recover the native build's original inputs independently of this later controller.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_native16_control as build
import generation_engine as gen

r, P, B, RESULT = build.r, build.P, build.B, build.RESULT
ROOT = B / 'native16-compatibility-smoke-v2'
KEY = 'native16_compatibility_smoke'
require, digest, write = r.require, r.digest, r.write


def compact(row):
    return {k: v for k, v in row.items() if k not in ('controller_inputs', 'runtime_files')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    build.scope(desired)[KEY] = compact(row)
    literal = json.dumps(KEY) + ':'
    value = json.dumps(compact(row), indent=2).replace('\n', '\n      ')
    if KEY in build.scope(json.loads(raw)):
        require(raw.count(literal) == 1, 'ambiguous compatibility result key')
        first = raw.index(literal) + len(literal)
        while raw[first].isspace(): first += 1
        _, count = json.JSONDecoder().raw_decode(raw[first:])
        raw = raw[:first] + value + raw[first + count:]
    else:
        anchor = '"fused_command_profile": {'
        require(raw.count(anchor) == 1, 'ambiguous fused-profile container')
        raw = raw.replace(anchor, anchor + '\n      ' + literal + ' ' + value + ',', 1)
    require(json.loads(raw) == desired, 'scoped update changed other evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def native(root=None, key=None):
    # A closed fixture keeps its original source identity when later fixture
    # work edits this worktree. Verify compiled inputs against recovered Git
    # blobs, not today's files; SDK/vendor/generated inputs still require their
    # actual pinned bytes. This does not qualify a changed native executable.
    root = Path(root) if root is not None else build.ROOT
    key = key if key is not None else build.KEY
    record = root / 'record.json'; row = json.loads(record.read_text())
    require(build.scope(r.validation.committed(RESULT))[key] == build.compact(row) and
        row['stage'] == 'built' and row['complete'] and row['passed'] and row['ownership_closed'] and
        not row['active'] and not row['GPU_executed'] and not row['engine_built'], 'closed exact native build required')
    count = 0; original_inputs = {}
    for group in [row['project'], *row['dependencies']]:
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
            build.shared.build.blobs(Path(group['root']), group['revision'], group['source_sha256'])}
        require(recovered == group['source_sha256'] and len(recovered) == group['count'], 'native inputs unrecoverable')
        require(hashlib.sha256(json.dumps(recovered, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            == group['aggregate_sha256'], 'original source aggregate differs')
        original_inputs.update({str((Path(group['root']) / name).resolve()): sha for name, sha in recovered.items()})
        count += len(recovered)
    require(count == sum(group['count'] for group in [row['project'], *row['dependencies']]),
        'native source count differs')
    closure = row['compiled_inputs']; require(digest(closure['path']) == closure['sha256'], 'compiler manifest changed')
    files = json.loads(Path(closure['path']).read_text())
    sdk = build.sdk_admit(row['sdk'])
    expected_inputs = dict(sdk, **row['external_inputs'], **row['generated_expected'], **original_inputs)
    require(len(files) == closure['files'] >= 100 and closure['uncovered'] == 0 and
        all(expected_inputs.get(p) == sha and (p in original_inputs or digest(p) == sha)
            for p, sha in files.items()), 'original actual compile closure changed or uncovered')
    for tool in row['tools'].values(): require(digest(tool['path']) == tool['sha256'], 'native tool changed')
    for artifact in row['fixtures'].values():
        require(digest(artifact['path']) == artifact['sha256'] and Path(artifact['path']).stat().st_size == artifact['bytes'],
            'native fixture changed')
    # Check actual imported link files too; compiler headers alone are insufficient.
    expected = dict(sdk, **row['external_inputs']); imported = {}; generated = {}
    ninja = (root / 'build/build.ninja').read_text()
    for name in row['targets']:
        stanza = ninja[ninja.index('build ' + name + ':'):].split('\n\n', 1)[0]
        line = next(v.split(' = ', 1)[1] for v in stanza.splitlines() if v.startswith('  LINK_LIBRARIES = '))
        for value in shlex.split(line):
            if not value.startswith('/'): continue
            path = Path(value).resolve(); sha = digest(path)
            if path.is_relative_to(root / 'build'):
                generated[str(path)] = sha
            else:
                require(str(path) in expected and sha == expected[str(path)], 'unplanned imported link file: ' + str(path))
                require('/MistVVK/' not in str(path), 'historical SDK linked into official fixture')
                imported[str(path)] = sha
    proof = dict(record=dict(path=str(record), sha256=digest(record)), source_commit=row['source_commit'],
        recovered_inputs=count, compiled_inputs=closure, fixtures=row['fixtures'],
        imported_link_files=imported, generated_link_files=generated, independent_SDK=row['sdk'])
    return row, sdk, proof


def runtime_inputs(native_row, sdk, rt):
    old_lib = Path(next(p for p in rt['runtime_files'] if '/libsycl.so.' in p)).parent
    files = {p: h for p, h in rt['runtime_files'].items() if not Path(p).is_relative_to(old_lib)}
    for p, sha in sdk.items():
        path = Path(p)
        if (path.is_relative_to(build.OFFICIAL) or path.is_relative_to(build.UMF)) and '.so' in path.name:
            with path.open('rb') as stream: elf = stream.read(4) == b'\x7fELF'
            if elf: files[p] = sha
    seeds = [Path(v['path']) for v in native_row['fixtures'].values()]
    seeds += [build.OFFICIAL / 'lib' / name for name in
        ('libsycl.so.9.0.0', 'libur_loader.so.0.12.0', 'libur_adapter_level_zero_v2.so.0.12.0')]
    seeds.append(build.UMF / 'lib/libumf.so.1.2.0')
    for seed in seeds:
        output = subprocess.check_output(['/usr/bin/ldd', str(seed)], env=native_row['environment'], text=True, timeout=15)
        require('not found' not in output, 'unresolved prospective runtime dependency')
        for line in output.splitlines():
            path = next((v for v in line.split() if v.startswith('/')), None)
            if path:
                path = str(Path(path).resolve())
                require('/MistVVK/' not in path and not re.search('nvidia|cuda|rocm', path, re.I), 'unexpected runtime dependency')
                sha = digest(path)
                require(path not in files or files[path] == sha, 'qualified provider changed')
                files[path] = sha
    return files


def prepare():
    require(not ROOT.exists(), 'single-use compatibility check')
    previous = build.scope(r.validation.committed(RESULT)).get(KEY)
    if previous is not None:
        require(previous['stage'] == 'failed' and not previous['GPU_process_launched'] and
            previous['failure_analysis']['stage'] == 'host_only_observation' and
            previous['failure_analysis']['physical_closure']['direct_child_reaped'] and
            previous['failure_analysis']['physical_closure']['session_empty_observed_before_reap'],
            'only the closed host-observation failure precedes this corrected plan')
    built, sdk, proof = native(); rt = r.validation.committed(r.RESULT); r.current(rt)
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    loader = str((build.OFFICIAL / 'lib/libur_loader.so.0').resolve())
    env = dict(built['environment'], STRATA_SYCL_PREFILL_NATIVE16='bf16',
        STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm', STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0', STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='0',
        ZE_ENABLE_ALT_DRIVERS='/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758',
        LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=loader,
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION')
    files = runtime_inputs(built, sdk, rt)
    cases = [dict(name='host-canary', target=None, args=[]),
        dict(name='submission', target='sycl_submission_test', args=['--submission']),
        dict(name='native16-smoke', target='sycl_vllm_native16_test', args=['--device', '0', '--suite', 'smoke']),
        dict(name='native16-guards', target='sycl_vllm_native16_test', args=['--device', '0', '--suite', 'guards']),
        dict(name='GU-smoke', target='sycl_moe_gu_swiglu_bench', args=['--smoke']),
        *[dict(name='GU-public' + str(arm), target='sycl_moe_gu_swiglu_bench',
            args=['--smoke-public', str(arm)], gu_arm=arm) for arm in (0, 1)],
        dict(name='Down-smoke', target='sycl_moe_down_scatter_bench', args=['--smoke']),
        *[dict(name='Down-public' + str(arm), target='sycl_moe_down_scatter_bench',
            args=['--route-smoke', str(arm)], down_arm=arm) for arm in (0, 1)]]
    row = dict(stage='prepared', source_commit=revision, controller_inputs=gen.benchmark_pins(revision),
        native_build=proof, runtime_sha256=digest(r.RESULT), runtime_files=files,
        boot=rt['boot'], driver=rt['driver'], environment=env, canary=rt['preload']['canary'],
        cases=cases, processes=[], active=False, ownership_closed=True, complete=False, passed=False,
        GPU_executed=False, model_executed=False, model_adoption_qualified=False,
        full_numerical_contract_qualified=False, physical262144_qualified=False,
        criteria='Ten ordered fresh owned processes;stop first unexpected status,source,provider,environment,oracle,'
            'guard,dispatch,validation,budget or fault failure. Pre-main native-only parameter validation and finite '
            'full API tracing.32normal dyadic smoke cases,26pre-enqueue guards,GU direct independent3-row dot '
            'oracle and both public arms at3/160/2333/2334;Down direct guards/invalid maps and public threshold '
            'routes. No runtime environment mutation or graph replay. Existing strict gradual-underflow edges,'
            'large full/chain matrix tests,model/oneMKL ABI and262144 are separate unqualified gates. '
            'Passing authorizes only finite normal-input public-GU calibration,not a model route or hardware ceiling.',
        budgets=dict(wall=300, cpu=1200, rss_cap=4 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        whole_bytes=512 << 20,
        retention='Keep source/recipe,compact individual case counts/dispatch/provider/validation/fault decisions. '
            'Retire successful full trace/JIT and process logs after verification. Candidate4fixtures and '
            '1427input manifest remain for imminent public-GU calibration. Original13assets unchanged.')
    if previous is not None:
        row['previous_failed_attempt'] = dict(result_commit='99c26a585',
            source_commit=previous['source_commit'], stage=previous['stage'],
            original_observation_failed=True, physical_process_closed=True,
            GPU_process_launched=False, failure_analysis=previous['failure_analysis'])
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', files)
    row['runtime_manifest'] = dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files))
    save(row); print('Prepared;commit the frozen compatibility plan before execution.', flush=True)


def current(row):
    require(build.scope(r.validation.committed(RESULT))[KEY] == compact(row), 'commit exact compatibility plan')
    # The completed cohort is not re-executed by this reader. Recover its
    # immutable controller source; a new caller must pin its own current code.
    closed = row['complete'] and row['passed'] and row['ownership_closed'] and not row['active']
    require(gen.benchmark_pins(row['source_commit'], match_worktree=not closed) == row['controller_inputs'],
        'controller input changed or unrecoverable')
    require(native()[2] == row['native_build'] and digest(r.RESULT) == row['runtime_sha256'], 'native/runtime proof changed')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(rt['boot'] == row['boot'] and rt['driver'] == row['driver'], 'boot/driver changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'] and
        all(digest(p) == sha for p, sha in row['runtime_files'].items()), 'runtime provider changed')


def analyze(case, text, pid):
    name = case['name']
    if name == 'host-canary':
        require(json.loads(text) == dict(pid=pid, initialized=0, checked=46, expected=46, version=(1 << 32)-1, closed=0),
            'host-only canary differs'); return dict(host_only=True)
    if name == 'submission':
        require('fixture_status=PASS mode=submission' in text, 'submission checks incomplete')
        return dict(submission_ordering_and_guards=True)
    if name == 'native16-smoke':
        require(text.count('dyadic PASS ') == 32 and 'COMPONENT_SUITE_PASS only;' in text, '32dyadic smoke cases required')
        return dict(dyadic_exact_RNE_cases=32, storage='allFP16/BF16input/output pairs,both tiles')
    if name == 'native16-guards':
        require('guards PASS rejected=26 output/inputs immutable' in text, '26admission guards required')
        return dict(pre_enqueue_rejections=26, inputs_outputs_immutable=True)
    require('fixture_status=PASS' in text, 'fixture completion missing')
    values = [json.loads(v) for v in text.splitlines() if v.startswith('{')]
    require(values and all(v.get('status') == 'PASS' for v in values), 'fixture case failed')
    if name.startswith('GU-public'):
        cases = [v for v in values if v.get('kind') == 'profiled_engine_gu']
        require([v['rows'] for v in cases] == [3,160,2333,2334] and
            all(bool(v['selected']) == bool(case['gu_arm'] and v['rows'] >= 2334) and
                v['guards'] and v['inputs_immutable'] and v['max_half_ulp'] <= 1 for v in cases), 'public GU route/numerics differ')
        return dict(cases=cases)
    if name == 'GU-smoke':
        require(sum(v.get('kind') == 'correctness' for v in values) == 4 and
            any(v.get('kind') == 'admissions' and v.get('rejected') == 13 for v in values), 'GU smoke incomplete')
    if name == 'Down-smoke':
        require(sum(v.get('kind') == 'invalid_map' for v in values) == 21, 'Down invalid-map coverage differs')
    return dict(case_kinds={kind: sum(v['kind'] == kind for v in values) for kind in sorted({v['kind'] for v in values})})


def run(row):
    current(row); require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use compatibility cohort')
    initial = copy.deepcopy(row); owner = None; seen = set()
    cursor = next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist():
        write(ROOT / 'execution.json', dict(processes=row['processes'], active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    outcome = dict(host_probe_accepted=False)
    try:
        owner = r.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            root = ROOT / case['name']; root.mkdir()
            for sub in ('tmp','cache'): (root / sub).mkdir()
            env = dict(row['environment'], TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'),
                STRATA_SYCL_PREFILL_GU_SWIGLU='large' if case.get('gu_arm',1) else '0',
                STRATA_SYCL_PREFILL_DOWN_SCATTER='large' if case.get('down_arm',1) else '0',
                UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
            artifact = row['native_build']['fixtures'][case['target']] if case['target'] else row['canary']
            allowed = dict(row['runtime_files']); allowed[str(Path(artifact['path']).resolve())] = artifact['sha256']
            case_seen = set(); env_seen = False
            def monitor(io, entry):
                nonlocal env_seen
                next_size = 0
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    try:
                        mapped = r.executable_mappings(Path('/proc',str(io.proc.pid),'maps').read_text())
                        actual = dict(v.split(b'=',1) for v in Path('/proc',str(io.proc.pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                        require(all(actual.get(k.encode()) == v.encode() for k,v in env.items()) and
                            set(actual)-{k.encode() for k in env} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'}, 'actual native environment differs')
                        env_seen = True
                    except (FileNotFoundError, ProcessLookupError): mapped = set()
                    except PermissionError:
                        # /proc can deny mm access while a short child exits.
                        # Require positive pidfd exit evidence; never reap here
                        # or waive access denial for a still-live GPU process.
                        require(io.exit_watch.poll(20), 'live native /proc access denied')
                        mapped = set()
                    require(mapped <= set(allowed), 'unplanned executable map: '+repr(mapped-set(allowed)))
                    case_seen.update(mapped)
                    if time.monotonic() >= next_size:
                        require(r.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole compatibility byte budget')
                        next_size = time.monotonic()+.5
                    if select.select([io.proc.stdout],[],[],.01)[0]: io.read();io.pending.clear()
                io.exit(1)
            entry, out, err = owner.run(case['name'], [artifact['path'],*case['args']], env, P,
                interaction=monitor, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'native case failed: '+case['name'])
            activation = r.validation.preload_evidence(err.read_text(), entry['pid'])
            require(activation['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'], 'wrong validation loader')
            require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size, 'parameter warning')
            details = analyze(case, out.read_text(), entry['pid'])
            if case['target']:
                trace = (root/'ur.trace').read_text()
                require(env_seen and 'urEnqueue' in trace and
                    {p for p in case_seen if '/libze_intel_gpu.so.' in p} == {env['ZE_ENABLE_ALT_DRIVERS']} and
                    env['STRATA_UR_PROBE_EXPECTED_LOADER'] in case_seen and
                    str(build.OFFICIAL/'lib/libsycl.so.9.0.0') in case_seen, 'actual GPU/provider evidence missing')
            item = dict(name=case['name'], passed=True, ownership_closed=True,
                native_pid=entry['pid'], pre_main_parameter_validation=True, details=details)
            r.validation.finish(dict_out := dict(host_probe_accepted=True), owner, cursor, row['boot'])
            require(dict_out['passed'] and dict_out['driver_after'] == row['driver'], 'fault/ownership check failed')
            row['processes'].append(item); seen.update(case_seen); persist()
            print('Passed '+case['name'], flush=True)
        current(initial); outcome['host_probe_accepted'] = True
    except BaseException as error:
        outcome['error'] = repr(error)
        if owner and owner.commands:
            outcome['last_case'] = owner.commands[-1]['label']
    finally:
        if owner: r.validation.finish(outcome, owner, cursor, row['boot'])
        device_started = bool(owner and any(v['label'] != 'host-canary' for v in owner.commands))
        row.update(stage='qualified' if outcome.get('passed') and len(row['processes']) == 10 else 'failed',
            complete=bool(outcome.get('passed') and len(row['processes']) == 10), passed=bool(outcome.get('passed')),
            active=False, ownership_closed=outcome.get('ownership_closed',True), outcome=outcome,
            observed_runtime_files={p: row['runtime_files'][p] for p in sorted(seen & set(row['runtime_files']))},
            GPU_process_launched=device_started,
            GPU_executed=True if len(row['processes']) > 1 else None if device_started else False)
        persist();save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','complete','passed','ownership_closed','outcome')}),flush=True)
    return int(row['stage'] != 'qualified')


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare();return 0
        if sys.argv[1:] == ['run']: return run(json.loads((ROOT/'record.json').read_text()))
        raise RuntimeError('use prepare or run')


if __name__ == '__main__': sys.exit(main())
