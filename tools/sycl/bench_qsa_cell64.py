#!/usr/bin/env python3
"""Main-owned actual-ID cell64 benchmark build; repository LICENSE applies.

Reuse the fixed component kernel objects in its one live generation. Original
native/controller source identities remain distinct from later qualification.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

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


def compiled_dependencies(record, plan):
    """Verify the harness closure; retain only its delta from fixed objects."""
    root = ROOT/'actual-id-build'
    target, text = (root/'harness.d').read_text().replace('\\\n', ' ').split(':', 1)
    b.guards.require(Path(target).resolve() == (root/'harness.o').resolve(), 'unexpected harness dependency target')
    allowed = {Path(v['path']).resolve():v['sha256'] for v in
               json.loads(Path(record['compiler_inputs']['path']).read_text())}
    allowed.update({(P/n).resolve():s for n,s in record['input_sha256'].items()})
    allowed.update({Path(n).resolve():s for n,s in record['generated'].items()})
    allowed[(P/CPP).resolve()] = plan['input_sha256'][CPP]
    actual = {}
    for name in shlex.split(text):
        path = Path(name).resolve();sha = b.digest(path)
        b.guards.require(allowed.get(path) == sha, 'unfrozen harness compiler dependency: ' + str(path))
        actual[str(path)] = sha
    b.guards.require(len(actual) > 100, 'incomplete harness dependency closure')
    original = record['actual_compiled_inputs']['source_sha256']
    return dict(files=len(actual),
        canonical_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        base='actual_compiled_inputs.source_sha256',
        added_or_changed={p:s for p,s in actual.items() if original.get(p) != s},
        omitted=sorted(p for p in original if p not in actual))


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
        b.guards.require(compiled_dependencies(record, plan) == plan['actual_compiled_inputs'],
                         'actual-ID harness dependency closure changed')


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
        actual = compiled_dependencies(record, plan)
        binary = Path(plan['output_binary'])
        plan.update(stage='built', passed=True, complete=True,
            binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=b.digest(binary)),
            actual_compiled_inputs=actual)
    except BaseException as error:
        plan.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        plan['ownership_closed'] = owner.active is None and all(b.supervision.closed(v) for v in owner.commands)
        if not plan['ownership_closed']:
            plan.update(stage='failed', passed=False, complete=False, error='harness build ownership open')
        persist();b.write(RESULT, record)
    print(json.dumps({k:plan.get(k) for k in ('stage','passed','complete','ownership_closed','error')}), flush=True)
    return int(not plan['passed'])


def harness_contract():
    key = 'qsa_cell64_actual_id_benchmark_contract_20261027'
    row = c.validation.committed(b.gen.A/'research-index.json', (key,))
    b.guards.require(row['status'] == 'reviewed_accept_conditional' and row['worker_closed'] and
        row['proposal']['sha256'] == b.digest(P/CPP), 'closed source-bound harness audit required')
    review = row['review']
    b.guards.require(review['status'] == 'completed' and review['model'] == 'gpt-6-luna' and
        review['reasoning_effort'] == 'max', 'native Luna/max harness review required')
    for asset, path in ((row['report_identity'], row['report']), (review, review['path'])):
        file = P/path
        b.guards.require(file.stat().st_size == asset['bytes'] and b.digest(file) == asset['sha256'],
                         'reviewed harness reference changed')
    return dict(index_key=key, report_sha256=row['report_identity']['sha256'], review_sha256=review['sha256'],
        correction='cell<=position includes valid current-token KV;global live_cells39259,n_kv=position+1.',
        finite_reference_domain='Fixed synthetic absV<=15/128;conservative2112term accumulator bound247.5. The maximum-finite BF16 spacing edge is outside this fixture domain.',
        timing_limit='Private32/private64 capture direct events inside timing;original32 does not. Control/original wall calibration is required;this is not clean identical-production-call speed evidence.')


def observer_contract(record):
    hook = P/'include/strata/sycl/command_profile_hook.hpp'
    source = P/'src/sycl/command_profile.cpp'
    b.guards.require('thread_local Trace* active = nullptr;' in source.read_text() and
        'if (active) observe_impl(*active, stream, event);' in hook.read_text(), 'observer activation contract changed')
    # All these native files are already hash-bound by build admission. The
    # fixture has no Trace/Frame owner;only observe is called by its wrappers.
    for name in record['sources']:
        path = Path(name)
        if path == source:
            continue
        text = path.read_text()
        b.guards.require(not any(s in text for s in ('command_profile::Trace', 'command_profile::Frame',
            'command_profile::active', 'using namespace command_profile')), 'unexpected observer owner')
    b.guards.require('command_profile' not in (P/CPP).read_text(), 'harness observer owner added')
    return dict(initial_active='null', activation='Only a constructed Trace::begin sets active;none is called by the fixed native fixture/runtime/kernel entrypoints.',
        hook_sha256=b.digest(hook), source_sha256=b.digest(source), environment='STRATA_SYCL_PREFILL_COMMAND_PROFILE=0',
        scope='Source-bound inactive observer proof;no dynamic observer counter or hardware occupancy claim.')


def run_admission(record, plan):
    admit(record, record['actual_id_build'], True)
    b.guards.require(inputs(plan['source_commit']) == plan['controller_inputs'] and
        c.kernel_ids(record) == plan['kernel_ids'] and c.contract() == plan['contract'] and
        c.numerical_contract() == plan['numerical'] and harness_contract() == plan['harness_contract'] and
        observer_contract(record) == plan['observer'], 'actual-ID qualification input changed')
    _, proof = b.qualified_baseline()
    rt = c.validation.committed(b.qualified_runtime.RESULT)
    b.guards.require(proof == plan['runtime_qualification'] and rt['runtime_files'] == plan['runtime_files'] and
        rt['preload'] == plan['preload'], 'runtime/preload qualification changed')
    return rt


def prepare_run(mode):
    record = b.gen.committed(RESULT);history = record.get('actual_id_checks', [])
    position = 0 if mode == 'smoke' else 1
    b.guards.require(len(history) == position and all(v['passed'] and v['complete'] and
        v['ownership_closed'] and not v['fresh_faults'] for v in history), 'ordered successful actual-ID gates required;no retry')
    if position:
        b.guards.require(history[0]['mode'] == 'smoke' and record['actual_id_plan']['stage'] == 'closed', 'smoke gate incomplete')
    else:
        b.guards.require('actual_id_plan' not in record, 'inspect existing actual-ID plan before preparing')
    qualified = [v for v in record['oracle_checks'] if v['mode'] == 'oracle-component' and v['passed']]
    b.guards.require(len(qualified) == 1 and qualified[0]['complete'] and qualified[0]['ownership_closed'] and
        not qualified[0]['fresh_faults'], 'extended numerical/local-argument qualification required')
    root = ROOT/('actual-id-'+mode)
    b.guards.require(not root.exists(), 'fresh actual-ID stage only')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    plan = {k:record['admission_plan'][k] for k in
            ('kernel_ids', 'contract', 'runtime_qualification', 'runtime_files', 'preload')}
    plan.update(stage='prepared', mode=mode, source_commit=revision, controller_inputs=inputs(revision),
        numerical=c.numerical_contract(), harness_contract=harness_contract(), observer=observer_contract(record),
        output_root=str(root), jobs=[], budgets=dict(wall=300 if mode == 'smoke' else 600,
            cpu=600 if mode == 'smoke' else 1800, rss_cap=8 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=192 << 20),
        comparison=dict(processes=3, rounds=6, shapes=[512,6491], arms=['original32','private32','private64'],
            order='(process+round+offset)%3', repeats={'512':64,'6491':4}, minimum_batch_ns=100000000,
            minimum_clock_multiple=1000, control_original_ratio_bounds=[0.99,1.01], candidate_control_ratio_max=0.99,
            decision='All six process/shape medians must pass control/original wall calibration and candidate/control wall plus attention-service ratio gates. Preserve every eligible sample. Failed speed gate stops model advancement;no adaptive retry or changed thresholds.',
            scope='Actual selected-ID geometry with synthetic finite Q/K/V;pack+attention wrapper wall and private attention-only service. Not PCIe traffic,hardware saturation,production speed,model quality or1000/70qualification.'),
        retirement='Retain individual timing samples,numerical/delta bounds,closure/fault results and decision. Retire successful full traces and normal logs after extraction. Keep one current component generation only for this live comparison.')
    rt = run_admission(record, plan)
    base = b.gen.committed(P/'tools/sycl/measurement_environment.json')['environment']
    root.mkdir(mode=0o700)
    for process in range(3):
        folder = root/str(process);folder.mkdir()
        for name in ('tmp','cache'):
            (folder/name).mkdir()
        env = dict(base, TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'),
            LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
            ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
            STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
        if mode == 'smoke':
            env['UR_LOG_TRACING'] = 'level:info;flush:info;output:file,'+str(folder/'api.trace')
        else:
            env.pop('UR_LOG_TRACING', None);env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
        command = [record['actual_id_build']['binary']['path'], '--smoke' if mode == 'smoke' else '--bench', record['selected_fixture']['path']]
        if mode == 'perf':
            command.append(str(process))
        plan['jobs'].append(dict(process=process, command=command, environment=env))
    b.write(root/'plan.json', plan);record['actual_id_plan'] = plan;b.write(RESULT, record)
    print('Prepared actual-ID '+mode+';commit source-bound plan before execution.', flush=True)


def parsed_output(text, mode, process):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    shapes = [3] if mode == 'smoke' else [512,6491]
    b.guards.require(rows and rows[-1] == {'kind':'PASS'} and sum(v['kind'] == 'PASS' for v in rows) == 1 and
        {v['kind'] for v in rows} <= {'PASS','numerical','delta','checked','sample'}, 'incomplete/unknown actual-ID output')
    numerical = [v for v in rows if v['kind'] == 'numerical']
    b.guards.require([(v['queries'],v['arm']) for v in numerical] == [(q,a) for q in shapes for a in range(3)], 'scalar coverage differs')
    for v in numerical:
        spots = {3:3,512:6,6491:8}[v['queries']]
        b.guards.require(v['reference_cells'] == (64 if v['arm'] == 2 else 32) and v['spot_queries'] == spots and
            v['compared'] == spots*24*256 and math.isfinite(v['max_abs']) and v['max_abs'] >= 0 and
            math.isfinite(v['max_budget_ratio']) and 0 <= v['max_budget_ratio'] <= 1, 'scalar numerical bound/coverage failed')
    deltas = [v for v in rows if v['kind'] == 'delta']
    b.guards.require([v['queries'] for v in deltas] == shapes, 'full-output delta coverage differs')
    for v in deltas:
        b.guards.require(v['elements'] == v['queries']*24*256 and 0 <= v['changed_from32'] <= v['elements'] and
            all(math.isfinite(v[k]) and v[k] >= 0 for k in ('max_abs','relative_l2')), 'nonfinite output/delta')
    checked = [v for v in rows if v['kind'] == 'checked']
    b.guards.require([v['queries'] for v in checked] == shapes and all(v['cells'] == 39259 and
        all(v[k] is True for k in ('guards','inputs_immutable','cell32_exact','own_reference')) and
        v['candidate_old_bits_required'] is False for v in checked), 'guard/input/control qualification failed')
    samples = [v for v in rows if v['kind'] == 'sample']
    expected = [] if mode == 'smoke' else [(q,r,(process+r+o)%3) for q in shapes for r in range(6) for o in range(3)]
    b.guards.require([(v['queries'],v['round'],v['arm']) for v in samples] == expected, 'timing count/order differs')
    for v in samples:
        repeats = 64 if v['queries'] == 512 else 4
        b.guards.require(v['process'] == process and v['repeats'] == repeats and v['timer_floor_ns'] > 0 and
            v['batch_ns'] >= max(100000000,1000*v['timer_floor_ns']), 'timing batch ineligible')
        b.guards.require((v['arm'] == 0 and v['attention_device_ns'] is None and v['event_calls'] == 0) or
            (v['arm'] != 0 and v['event_calls'] == repeats and 0 < v['attention_device_ns'] <= v['batch_ns']+1000000), 'direct-event coverage/envelope differs')
    return dict(numerical=numerical, deltas=deltas, samples=samples,
                guards=True, inputs_immutable=True, cell32_exact=True, own_reference=True)


def performance_decision(observations, comparison):
    groups = []; passed = True
    b.guards.require([v['process'] for v in observations] == [0,1,2], 'three independent process ordinals required')
    for observation in observations:
        for queries in comparison['shapes']:
            arms = []
            for arm in range(3):
                samples = [v for v in observation['output']['samples'] if v['queries'] == queries and v['arm'] == arm]
                b.guards.require(len(samples) == 6, 'six samples per process/shape/arm required')
                wall = [v['batch_ns']/v['repeats'] for v in samples]
                service = [v['attention_device_ns']/v['repeats'] for v in samples] if arm else []
                arms.append(dict(wall_median_ns=statistics.median(wall), wall_min_ns=min(wall), wall_max_ns=max(wall),
                                 service_median_ns=statistics.median(service) if service else None))
            calibration = arms[1]['wall_median_ns']/arms[0]['wall_median_ns']
            wall = arms[2]['wall_median_ns']/arms[1]['wall_median_ns']
            service = arms[2]['service_median_ns']/arms[1]['service_median_ns']
            lo,hi = comparison['control_original_ratio_bounds'];limit = comparison['candidate_control_ratio_max']
            ok = lo <= calibration <= hi and wall <= limit and service <= limit;passed &= ok
            groups.append(dict(process=observation['process'], queries=queries, arms=arms,
                control_original_wall_ratio=calibration, candidate_control_wall_ratio=wall,
                candidate_control_service_ratio=service, performance_gate_passed=ok))
    b.guards.require(len(groups) == 6, 'all process/shape groups required')
    return dict(performance_gate_passed=passed, advance_to_model=passed, groups=groups,
                model_adopted=False, hardware_saturation_qualified=False)


def host_controls():
    """Exercise false-pass cases before using this controller for GPU evidence."""
    import copy
    rejected = []
    def reject(label, function):
        try:
            function()
        except (RuntimeError,KeyError,ValueError):
            rejected.append(label)
        else:
            raise RuntimeError('negative control accepted: '+label)
    rows = [dict(kind='numerical',queries=3,arm=a,reference_cells=64 if a == 2 else 32,
        spot_queries=3,compared=18432,max_abs=0,max_budget_ratio=0) for a in range(3)]
    rows += [dict(kind='delta',queries=3,elements=18432,changed_from32=123,max_abs=0.0001,relative_l2=0.001),
        dict(kind='checked',queries=3,cells=39259,guards=True,inputs_immutable=True,cell32_exact=True,
             own_reference=True,candidate_old_bits_required=False),dict(kind='PASS')]
    def parse(values):
        return parsed_output('\n'.join(json.dumps(v) for v in values),'smoke',0)
    parse(rows)  # Changed64output is permitted;old32bits are not its criterion.
    bad = copy.deepcopy(rows);bad[2]['reference_cells'] = 32
    reject('wrong_candidate_recurrence',lambda:parse(bad))
    bad = copy.deepcopy(rows);bad[2]['max_budget_ratio'] = math.inf
    reject('nonfinite_reference_ratio',lambda:parse(bad))
    bad = copy.deepcopy(rows);bad[-2]['guards'] = False
    reject('corrupted_guard',lambda:parse(bad))
    reject('missing_footer',lambda:parse(rows[:-1]))
    observations = [dict(process=p,output=dict(samples=[dict(queries=q,arm=a,
        batch_ns=(100 if a == 2 else 200)*1000000,repeats=1,attention_device_ns=(90 if a == 2 else 190)*1000000)
        for q in (512,6491) for a in range(3) for r in range(6)])) for p in range(3)]
    comparison = dict(shapes=[512,6491],control_original_ratio_bounds=[0.99,1.01],candidate_control_ratio_max=0.99)
    b.guards.require(performance_decision(observations,comparison)['advance_to_model'], 'eligible control refused')
    bad = copy.deepcopy(observations)
    for v in bad[2]['output']['samples']:
        if v['queries'] == 6491 and v['arm'] == 2:
            v['batch_ns'] = 210000000
    b.guards.require(not performance_decision(bad,comparison)['advance_to_model'], 'one slow group advanced')
    bad = copy.deepcopy(observations)
    for v in bad[0]['output']['samples']:
        if v['arm'] == 1:
            v['batch_ns'] = 220000000
    b.guards.require(not performance_decision(bad,comparison)['advance_to_model'], 'failed calibration advanced')
    reject('duplicate_process',lambda:performance_decision([observations[0]]*3,comparison))
    print(json.dumps(dict(passed=True,GPU_executed=False,controls=9,rejected=rejected,
        positive='Changed64bits accepted under own recurrence;all-group speed gain passes.',
        performance_failures_stopped=['one_slow_shape_process','control_original_calibration'])),flush=True)


def run_stage(mode):
    record = b.gen.committed(RESULT);plan = record['actual_id_plan'];root = Path(plan['output_root'])
    b.guards.require(plan['stage'] == 'prepared' and plan['mode'] == mode and root == ROOT/('actual-id-'+mode) and
        plan == json.loads((root/'plan.json').read_text()) and not (root/'owned.json').exists(), 'fresh committed actual-ID stage required')
    rt = run_admission(record, plan)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursors = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor'],text=True,timeout=15), re.M)
    b.guards.require(len(cursors) == 1, 'fresh fault cursor required')
    result = dict(mode=mode, controller_commit=plan['source_commit'], plan_commit=b.gen.git('rev-parse','HEAD').decode().strip(),
        passed=False, complete=False, active=False, ownership_closed=False, observations=[], model_qualified=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        b.write(root/'owned.json',dict(result,commands=owner.commands if owner else []))
    owner = b.supervision.Owner(root/'owned',persist)
    try:
        for job in plan['jobs']:
            process = job['process'];folder = root/str(process)
            b.guards.driver_guard();b.guards.no_model_owners([])
            b.guards.require(b.digest(job['command'][0]) == record['actual_id_build']['binary']['sha256'], 'actual-ID binary changed')
            initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
            def interact(io,entry):
                for stage in ('before-runtime','runtime-ready','after-cleanup'):
                    while True:
                        line = io.line(250)
                        if line.startswith('QSA_CELL64_GATE '):
                            break
                    b.guards.require(line == 'QSA_CELL64_GATE '+stage, 'actual-ID gate differs')
                    delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
                    b.guards.require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
                    native = Path('/proc',str(io.proc.pid))
                    b.guards.require(str((native/'exe').resolve()) == str(Path(job['command'][0]).resolve()), 'wrong actual-ID executable')
                    env = dict(v.split(b'=',1) for v in (native/'environ').read_bytes().split(b'\0') if b'=' in v)
                    b.guards.require(all(env.get(k.encode()) == v.encode() for k,v in job['environment'].items()), 'actual-ID environment changed')
                    mapped = b.qualified_runtime.executable_mappings((native/'maps').read_text())
                    allowed = set(rt['runtime_files']) | {str(Path(job['command'][0]).resolve())}
                    b.guards.require(mapped <= allowed, 'unknown actual-ID code mapping: '+str(sorted(mapped-allowed)))
                    if stage == 'runtime-ready':
                        b.guards.require({rt['preload']['loader'],rt['preload']['library']['path'],job['environment']['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'qualified provider missing')
                    io.send(b'X')
                b.guards.require(json.loads(io.line(20)) == {'kind':'PASS'}, 'actual-ID footer missing')
                io.proc.stdin.close();io.exit(20)
            print('Running actual-ID '+mode+' process'+str(process),flush=True)
            entry,out,err = owner.run(str(process),job['command'],job['environment'],P,interaction=interact,**plan['budgets'])
            b.guards.require(b.supervision.closed(entry) and entry['exit_code'] == 0, 'actual-ID child failed/ownership open')
            stderr = err.read_text();proof = c.validation.preload_evidence(stderr,entry['pid'])
            b.guards.require(proof['loader'] == rt['preload']['loader'], 'actual-ID UR loader mismatch')
            body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            b.guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body), 'actual-ID runtime error')
            log = folder/'validation.log'
            b.guards.require(not log.exists() or log.stat().st_size == 0, 'actual-ID validation warning/error')
            observation = dict(process=process, exit_code=0, ownership_closed=True,
                validation=proof, output=parsed_output(out.read_text(),mode,process))
            if mode == 'smoke':
                launches = c.local_launches((folder/'api.trace').read_text(),plan['kernel_ids'],[0,1,2])
                b.guards.require(all(v['calls'] == 2 for v in launches.values()), 'actual-ID smoke dispatch count differs')
                observation['local_launches'] = launches
            result['observations'].append(observation);persist()
        run_admission(record, plan)
        if mode == 'perf':
            result['decision'] = performance_decision(result['observations'],plan['comparison'])
        result.update(passed=True,complete=True,GPU_executed=True)
    except BaseException as error:
        result.update(passed=False,complete=False,error=repr(error))
    finally:
        result['ownership_closed'] = owner.active is None and all(b.supervision.closed(v) for v in owner.commands)
        if not result['ownership_closed']:
            result.update(passed=False,complete=False,error='actual-ID ownership remains open')
        c.postchecks(result,boot,cursors[0]);persist()
        if not result['passed'] and 'decision' in result:
            result['decision']['advance_to_model'] = False
        plan['stage'] = 'closed' if result['passed'] else 'failed'
        record.setdefault('actual_id_checks',[]).append(result);b.write(RESULT,record)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    if 'decision' in result:
        print(json.dumps(result['decision']),flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    args = sys.argv[1:]
    b.guards.require(args in (['prepare-build'],['build'],['host-controls'],['prepare','smoke'],['run','smoke'],['prepare','perf'],['run','perf']),
                     'prepare-build|build|host-controls|prepare smoke|run smoke|prepare perf|run perf')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.guards.no_model_owners([]);b.guards.driver_guard()
        if len(args) == 1:
            if args[0] == 'host-controls':
                raise SystemExit(host_controls())
            raise SystemExit(prepare() if args[0] == 'prepare-build' else build())
        raise SystemExit(prepare_run(args[1]) if args[0] == 'prepare' else run_stage(args[1]))
