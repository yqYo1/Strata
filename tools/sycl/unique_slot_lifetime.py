#!/usr/bin/env python3
"""Source-pinned unique-slot lifetime checks; Main owns every native process.

Main-authored from the owned fixture controllers; repository LICENSE.
This is safety/ordering validation, not a speed or model qualification.
"""
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode = True
import prefill_weight_pipeline as previous

base = previous.base
P, B, RESULT = previous.P, previous.B, previous.RESULT
KEY = 'prefill_unique_slot_lifetime_v1'
ROOT = B / 'prefill-unique-slot-lifetime-v1'
VIEW = ROOT / 'source'
RECORD = ROOT / 'record.json'
SOURCE = 'tests/sycl/prefill_unique_slot_copies.cpp'
HEADER = 'include/strata/prefill/sycl_unique_slot_copies.hpp'
TEST = 'tests/core/test_unique_slot_lifetime.py'
PREFIXES = [*previous.PREFIXES, SOURCE, HEADER, TEST]
ORIGINAL = '14508acbf1b70eb994fdf37f97fb6d0f5cff7d43'
BUILD_EVIDENCE = 'ee8a41fd6'
FULL_EVIDENCE = '1496a4902076814e4747e051d0ed201d0f3021b3'
require, sha, fingerprint = previous.require, previous.sha, previous.fingerprint
git, write = previous.git, previous.write
BUILD = dict(wall=900, cpu=1800, rss_cap=8 << 30, text_cap=32 << 20,
             file_cap=256 << 20, total_cap=512 << 20)
RUN = dict(wall=120, cpu=180, rss_cap=2 << 30, text_cap=32 << 20,
           file_cap=48 << 20, total_cap=128 << 20)
PHASE_REFERENCES = {
    'prefill': dict(path=str(B / 'coherent-prefill-native-v1/build/strata'),
        source=ORIGINAL, sha256='dcb8efd4698d4c9c6de913cb5f92e2229e945b242f24bd1fca2998e4cf89e6d1'),
    'decode': dict(path=str(B / 'phase-kv-naming-native-v1/build/strata'),
        source='9849e35803f011aaafb2753c1e0f31411e40308e',
        sha256='6d2b4f7d6a79424c3da5ac6e35d7e1336016604e2f7bdb68b84031d2879903de')}


def compact(row):
    result = {k: v for k, v in row.items() if k not in
              ['project', 'allowed', 'actual_inputs', 'commands']}
    result['project'] = {k: v for k, v in row['project'].items() if k != 'source_sha256'}
    result['allowed'] = dict(files=len(row['allowed']), sha256=fingerprint(row['allowed']))
    if 'actual_inputs' in row:
        result['actual_inputs'] = {k: dict(files=len(v), sha256=fingerprint(v))
                                  for k, v in row['actual_inputs'].items()}
    return result


def save(row):
    write(RECORD, row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def counts(mode):
    if mode in ['host', 'metadata']: return 0, 0, 0
    if mode == 'throw': return 0, 2, 0
    if mode == 'terminal': return 0, 1, 0
    require(mode in ['ordered', 'partial129', 'full512'], 'finite unique-slot mode')
    n = 512 if mode == 'full512' else 129 if mode == 'partial129' else 128
    return n, n + 1, n


def case_gate(case):
    mode, placement, width = case['mode'], case['placement'], case['width']
    kernels, copies, ready = counts(mode)
    expected = [str(ROOT / 'fixture'), 'host'] if mode == 'host' else [
        str(ROOT / 'fixture'), mode, placement, str(width)]
    label = mode + '-' + placement + '-w' + str(width)
    require(case['argv'] == expected and case['label'] == label
            and case['root'] == str(ROOT / label) and placement in ['compute', 'separate']
            and type(width) is int and width in [1, 64], 'frozen unique-slot argv')
    require(fingerprint([case['expected_kernels'], case['expected_payload_copies'],
                         case['expected_ready_barriers']]) ==
            fingerprint([kernels, copies, (ready + width - 1) // width]),
            'frozen unique-slot counts')
    required = previous.SAFETY | previous.BACKEND | dict(
        CUDA_MODULE_LOADING='EAGER', ZE_ENABLE_VALIDATION_LAYER='1',
        ZE_ENABLE_PARAMETER_VALIDATION='1', ZE_ENABLE_THREADING_VALIDATION='1',
        ZE_ENABLE_HANDLE_LIFETIME='0',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION')
    require(all(case['environment'].get(k) == v for k, v in required.items()),
            'qualified runtime safety settings differ')


def descriptors():
    cases = [('host', 'compute', 1)]
    for placement in ['compute', 'separate']:
        cases += [('metadata', placement, 1), ('ordered', placement, 1),
                  ('ordered', placement, 64), ('partial129', placement, 64),
                  ('full512', placement, 64), ('terminal', placement, 1), ('throw', placement, 1)]
    return cases


def plan_gate(row):
    require([(c['mode'], c['placement'], c['width']) for c in row['cases']] == descriptors(),
            'complete ordered finite case plan required')
    for case in row['cases']: case_gate(case)
    require(row['model_trial_allowed'] is False and row['performance_qualified'] is False,
            'small lifetime fixture cannot admit a model or qualify speed')


def check(row, committed=True):
    if committed:
        require(fingerprint(compact(row)) == fingerprint(base.ur_validation.committed(RESULT, (KEY,))),
                'commit exact unique-slot plan/outcome first')
    current = base.original.coherent.build.common.source.pin(P, PREFIXES)
    require(current['source_sha256'] == row['project']['source_sha256'], 'frozen fixture inputs changed')
    plan_gate(row)
    require(base.guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
            'boot/driver changed')
    require(all(sha(p) == h for p, h in row['recipe_tools'].items()), 'compiler/tool changed')
    require(sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'],
            'runtime manifest changed')
    providers = json.loads(Path(row['runtime_manifest']['path']).read_text())
    require(all(sha(p) == h for p, h in providers.items()), 'qualified runtime provider changed')
    require(row['phase_references'] == PHASE_REFERENCES and
            all(sha(v['path']) == v['sha256'] for v in PHASE_REFERENCES.values()),
            'preserve both selected phase references')
    if 'binary' in row:
        # The completed compiler census determines actual inputs. Unused SDK
        # headers in the admission allowlist are not inputs to this executable.
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'fixture executable changed')
        required = {p: h for group in row['actual_inputs'].values() for p, h in group.items()}
    else:
        required = row['allowed']
    require(all(sha(p) == h for p, h in required.items()), 'actual build input changed')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh unique-slot root;no retry')
    old = base.ur_validation.committed(RESULT, (previous.KEY,))
    require(old['complete'] and old['ownership_closed'] and not old['active']
            and old['retention']['stage'] == 'complete'
            and not Path(old['retention']['root']).exists(), 'retire ended third generation first')
    native, proof = base.original.verify_original()
    seed = previous.seed_recipe()
    original_build = json.loads(git(P, 'show', BUILD_EVIDENCE + ':' + str(RESULT.relative_to(P))))[previous.KEY]
    full = json.loads(git(P, 'show', FULL_EVIDENCE + ':' + str(RESULT.relative_to(P))))[previous.KEY]
    project = base.original.coherent.build.common.source.pin(P, PREFIXES)
    ROOT.mkdir(mode=0o700)
    for name in ['tmp', 'cache', 'owned']: (ROOT / name).mkdir(mode=0o700)
    generated = {}
    names = [p for p in native['project']['source_sha256'] if p.startswith(('include/', 'src/'))]
    for name, raw in base.original.coherent.build.common.source.blobs(P, ORIGINAL, names):
        path = VIEW / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        generated[str(path)] = sha(path)
    allowed = generated | json.loads(Path(native['external_inputs']['path']).read_text())
    allowed.update(json.loads(Path(seed['runtime_manifest']['path']).read_text()))
    for name in [SOURCE, HEADER, previous.LANE]: allowed[str(P / name)] = sha(P / name)
    for image in [seed['validation_provider'], *seed['native_validation_probe'].values()]:
        allowed[str(Path(image['path']).resolve())] = image['sha256']
    before = str(Path(original_build['binary']['path']).parent)
    recipe = []
    for label in ['runtime', 'commands', 'compat', 'lifetime', 'link-lifetime']:
        entry = next(x for x in original_build['recipe'] if x['label'] == label)
        entry = {k: ([x.replace(before, str(ROOT)) for x in v] if isinstance(v, list) else
                     v.replace(before, str(ROOT)) if isinstance(v, str) else v) for k, v in entry.items()}
        if label == 'lifetime': entry['argv'][entry['argv'].index('-c') + 1] = str(P / SOURCE)
        if label == 'link-lifetime':
            entry['argv'][entry['argv'].index('-o') + 1] = str(ROOT / 'fixture')
            entry['outputs'] = [str(ROOT / 'fixture')]
        recipe.append(entry)
    template = next(c for c in full['cases'] if c['family'] == 'lifetime' and c['mode'] == 'metadata')
    cases = []
    for mode, placement, width in descriptors():
        label = mode + '-' + placement + '-w' + str(width); root = ROOT / label
        env = {k: v.replace(template['root'], str(root)) for k, v in template['environment'].items()}
        kernels, copies, ready = counts(mode)
        case = dict(label=label, root=str(root), mode=mode, placement=placement, width=width,
            argv=[str(ROOT / 'fixture'), 'host'] if mode == 'host' else
                 [str(ROOT / 'fixture'), mode, placement, str(width)], environment=env,
            expected_kernels=kernels, expected_payload_copies=copies,
            expected_ready_barriers=(ready + width - 1) // width)
        case_gate(case); cases.append(case)
    row = dict(stage='prepared', passed=False, complete=False, active=False, ownership_closed=True,
        project=project, native_source_commit=project['revision'], original_runtime_source=ORIGINAL,
        original_build_evidence=BUILD_EVIDENCE, foundation_sha256=fingerprint(proof),
        source_view=dict(files=len(generated), sha256=fingerprint(generated)), allowed=allowed,
        recipe=recipe, recipe_tools=original_build['recipe_tools'],
        runtime_manifest=seed['runtime_manifest'], native_validation_probe=seed['native_validation_probe'],
        validation_provider=seed['validation_provider'],
        build_environment={k: v.replace(before, str(ROOT)) for k, v in original_build['build_environment'].items()},
        driver=base.guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        phase_references=PHASE_REFERENCES,
        cases=cases, results=[], commands=[], model_trial_allowed=False, performance_qualified=False,
        plan='Host controller negative-admission controls first,then four fresh translation units/four objects plus link;no archive/object reuse. Then15independent traced processes in declared order:host first,compute placement,then separate. Exact source/recipe/actual dependency census/provider/ELF/environment,normal0,quiet validation,positive owner closure and no fresh fault. Verify every accepted GPU consumer byte,unique512source patterns,partial readiness tails,capacity/order/alias/foreign/stale/refusal/terminal and accepted-copy host-exception retirement. Stop first failure;no retry or changed criteria. No timing/model gate is passed by this fixture.',
        coherent_unit='Enforced sequential single-write destinations plus omitted overwrite-reader markers are inseparable;omitting markers without prohibiting reuse is unsafe. This fixture varies only queue placement/readiness width for ordering coverage,not a performance comparison. A new committed full-math plan must compare the owner against matched layout,then independently compare placement and readiness width before model adoption.',
        limits='No injected asynchronous/submission/driver fault,active command observer,model/cache publication/borrowed-lease handoff,quality,agent,physical262144/RESTORE,ordinary provider teardown or speed qualification.',
        retention='Named third generation for these small checks only;preserve both phase references. Commit useful results/recover exact inputs,then retire this fixture before a distinct full-math/model build.')
    plan_gate(row); save(row)


def execute(action):
    row = json.loads(RECORD.read_text())
    require(row['stage'] in (['prepared'] if action == 'build' else ['built', 'testing']), 'wrong/closed stage')
    owner, cursor = None, None
    try:
        check(row)
        label = 'build' if action == 'build' else 'case-' + str(len(row['results']))
        # Setup failure is terminal too. This receipt is never overwritten or
        # used to refresh hashes after a rejected launch.
        with (ROOT / (label + '.started')).open('x') as stream:
            json.dump(dict(action=action, source=row['native_source_commit']), stream)
        owner = base.owner_api.Owner(ROOT / 'owned', lambda: write(ROOT / 'ownership.json', owner.commands))
        if action == 'build':
            row.update(stage='building', active=True, ownership_closed=False); save(row)
            command, out, err = owner.run('host-controls',
                ['/usr/bin/python3', '-B', '-m', 'unittest', 'discover', '-s', 'tests/core',
                 '-p', 'test_unique_slot_lifetime.py', '-v'], row['build_environment'], P, **BUILD)
            require(base.owner_api.closed(command) and command['exit_code'] == 0
                    and '\nOK\n' in err.read_text(), 'host admission controls failed:' + str(err))
            row['host_admission_controls'] = dict(passed=True, GPU_executed=False)
            allowed = dict(row['allowed']); actual = dict(compiled={}, links={}, objects={})
            for entry in row['recipe']:
                command, out, err = owner.run(entry['label'], entry['argv'], row['build_environment'], ROOT, **BUILD)
                require(base.owner_api.closed(command) and command['exit_code'] == 0, 'fixture build failed:' + str(err))
                if 'dep' in entry:
                    for name in shlex.split(Path(entry['dep']).read_text().replace('\\\n', ' ').split(':', 1)[1]):
                        p = str(Path(name).resolve()); h = sha(p)
                        require(allowed.get(p) == h, 'unfrozen compiler input:' + p); actual['compiled'][p] = h
                if entry.get('link'):
                    for line in out.read_text().splitlines():
                        p = Path(line.strip())
                        if p.is_absolute() and p.is_file():
                            key = str(p.resolve()); h = sha(p)
                            require(allowed.get(key) == h, 'unfrozen link input:' + key); actual['links'][key] = h
                for name in entry['outputs']:
                    allowed[str(Path(name).resolve())] = sha(name)
                    if name.endswith('.o'): actual['objects'][name] = sha(name)
            require(len(actual['compiled']) > 400 and len(actual['objects']) == 4 and len(actual['links']) > 8,
                    'complete fresh compiler/link/object census required')
            row.update(stage='built', passed=True, actual_inputs=actual,
                       binary=dict(path=str(ROOT / 'fixture'), sha256=sha(ROOT / 'fixture')))
        else:
            case = row['cases'][len(row['results'])]; root = Path(case['root'])
            require(not root.exists(), 'existing case:inspect owner,no retry')
            root.mkdir(mode=0o700)
            for name in ['tmp', 'cache']: (root / name).mkdir()
            owner = base.owner_api.Owner(root / 'owned', lambda: write(ROOT / 'ownership.json', owner.commands))
            cursor = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
                                             text=True, timeout=15).split('-- cursor: ', 1)[1].strip()
            row.update(stage='testing', active=True, ownership_closed=False); save(row)
            expected = ['host-ready', 'host-exit'] if case['mode'] == 'host' else ['before-runtime', 'runtime-ready', 'before-exit']
            gates, records, maps = [], [], {}
            environment = base.engine_once.native_environment(row, case['environment'])
            def interact(channel, _entry):
                while True:
                    item = json.loads(channel.line(45))
                    if item['kind'] == 'gate':
                        require(len(gates) < len(expected) and item['name'] == expected[len(gates)], 'gate order')
                        actual = dict(x.split('=', 1) for x in Path('/proc', str(channel.proc.pid), 'environ').read_text().strip('\0').split('\0'))
                        require(all(actual.get(k) == v for k, v in environment.items()) and
                                set(actual) - set(environment) <= {'PWD', 'OLDPWD', 'SHLVL', '_'}, 'native environment differs')
                        require(Path('/proc', str(channel.proc.pid), 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0') == case['argv'], 'native argv differs')
                        maps[item['name']] = base.event.mappings(channel.proc.pid, row)
                        require(str(Path(row['binary']['path']).resolve()) in maps[item['name']],
                                'actual fixture mapping missing')
                        if item['name'] in ['runtime-ready', 'before-exit']:
                            required = [base.event.LOADER, base.event.VALIDATION,
                                environment['ZE_ENABLE_ALT_DRIVERS'], environment['UR_ADAPTERS_FORCE_LOAD'],
                                '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so']
                            require(all(str(Path(p).resolve()) in maps[item['name']] for p in required),
                                    'initialized fixture provider missing')
                        gates.append(item['name']); channel.send(b'continue\n')
                    else:
                        records.append(item); channel.exit(30); break
            command, out, err = owner.run(case['label'], case['argv'], environment, P,
                                         interaction=interact, **RUN)
            require(base.owner_api.closed(command) and command['exit_code'] == 0, 'native fixture failed:' + str(err))
            validation = base.ur_validation.preload_evidence(err.read_text(), command['pid'])
            require(gates == expected and len(records) == 1 and records[0].get('passed') is True, 'incomplete fixture')
            if case['mode'] == 'host': require(records[0] == dict(kind='host', passed=True, capacity_refusals=2), 'host result')
            else:
                count = 512 if case['mode'] == 'full512' else 129 if case['mode'] == 'partial129' else 128
                require(records[0] == dict(kind='done', passed=True, mode=case['mode'], slots=count,
                                          ready_width=case['width']), 'fixture mode/extent differs')
            require(not (root / 'validation.log').exists() or (root / 'validation.log').stat().st_size == 0,
                    'UR validation output')
            require(not re.search(r'\[(?:error|critical)\]|prefill unique-slot fixture failure:|Strata SYCL cleanup:',
                                  err.read_text(), re.I), 'native validation/cleanup error')
            trace = (root / 'ur.trace').read_text() if (root / 'ur.trace').exists() else ''
            kernels = len(re.findall(r'---> urEnqueueKernelLaunch\w*', trace))
            copies = sum('SUCCESS (ZE_RESULT_SUCCESS) in zeCommandListAppendMemoryCopy(' in line and
                         'size=256,' in line for line in err.read_text().splitlines())
            barriers = trace.count('---> urEnqueueEventsWaitWithBarrierExt')
            require((kernels, copies, barriers) == (case['expected_kernels'], case['expected_payload_copies'],
                                                  case['expected_ready_barriers']), 'actual copy/consumer/readiness counts differ')
            check(row, False)
            row['results'].append(dict(label=case['label'], passed=True, native_exit_code=0, ownership_closed=True,
                kernels=kernels, payload_copies=copies, readiness_barriers=barriers,
                validation=validation, providers_sha256=fingerprint(maps), outcome=records[0]))
            row.update(active=False, ownership_closed=True, passed=True)
            if len(row['results']) == len(row['cases']): row.update(stage='closed', complete=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error)); raise
    finally:
        if owner is not None: row['commands'].extend(owner.commands)
        row['active'] = owner is not None and owner.active is not None
        row['ownership_closed'] = not row['active'] and all(base.owner_api.closed(c) for c in row['commands'])
        try:
            row['boot_unchanged'] = row['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            row['driver_after'] = base.guards.driver_guard()
            if not row['boot_unchanged'] or row['driver_after'] != row['driver']:
                row.update(stage='failed', passed=False, complete=False)
        except BaseException as error:
            row.update(stage='failed', passed=False, complete=False, health_error=repr(error))
        if cursor:
            try:
                lines = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True, timeout=15)
                row['fresh_faults'] = [line for line in lines.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)', line, re.I)]
                if row['fresh_faults']:
                    row.update(stage='failed', passed=False, complete=False)
            except BaseException as error: row.update(stage='failed', passed=False, complete=False, health_error=repr(error))
        save(row)
    print(json.dumps({k: row.get(k) for k in ['stage', 'passed', 'complete', 'active', 'ownership_closed']}), flush=True)
    return 0 if row['passed'] else 1


if __name__ == '__main__':
    require(sys.argv[1:] in [['prepare'], ['build'], ['next']], 'prepare|build|next')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: raise SystemExit(execute(sys.argv[1]))
