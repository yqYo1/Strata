#!/usr/bin/env python3
"""Main-owned cell64 property and bounded oracle checks; repository LICENSE.

Property admission precedes separately planned oracle execution.
Own archived208c controller APIs are references; no XeStrata application source.
"""
import ast
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_qsa_cell64 as build
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks

P, B, ROOT, RESULT = build.P, build.B, build.ROOT, build.RESULT
KEY = 'qsa_cell64_admission_contract_20261027'


def inputs(revision):
    pending = [Path(__file__).resolve()]; seen = set(); pins = {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = str(path.relative_to(P))
        raw = build.gen.git('show', revision + ':' + name)
        guards.require(raw == path.read_bytes(), 'commit exact controller input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules = [n.name for n in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    return pins


def kernel_ids(record):
    names = []
    for arm, stride in (('original', 33), ('control', 33), ('candidate', 65)):
        asset = record['device_IR'][arm]
        guards.require(build.digest(asset['path']) == asset['sha256'], 'device IR changed')
        text = Path(asset['path']).read_text()
        headers = re.findall(r'^define [^\n]*prefix_batch_implILi' + str(stride) +
            r'ELb0ELb0ELb1ELb1ELb1ELb1ELb0EEE[^\n]*', text, re.M)
        selected = [h for h in headers if '!intel_reqd_sub_group_size ' in h]
        guards.require(len(selected) == 1, 'one exact ordered-exp attention kernel per arm')
        header = selected[0]
        metadata = re.search(r'!intel_reqd_sub_group_size !(\d+)', header)[1]
        guards.require(re.search(r'^!' + metadata + r' = !\{i32 16\}$', text, re.M),
                       'compiled SIMD contract differs')
        name = re.search(r'spir_kernel void @([^ (]+)\(', header)[1]
        guards.require(name.startswith('_ZTS'), 'unexpected kernel identity encoding')
        names.append(name)
    guards.require(len(set(names)) == 3, 'compiled arm identities overlap')
    return names


def contract():
    row = validation.committed(build.gen.A / 'research-index.json', (KEY,))
    guards.require(row['status'] == 'reviewed_accept_conditional' and row['worker_closed'] and
        row['source_revision'] == '9b59e9b511b60941c4d5dd06fb3dae91a83bcac0',
        'closed independent admission API audit required')
    review = row['review']
    guards.require(review['status'] == 'completed' and review['model'] == 'gpt-6-luna' and
        review['reasoning_effort'] == 'max', 'native Luna/max curation required')
    for asset, path in ((row['report_identity'], row['report']), (review, review['path'])):
        file = P / path
        guards.require(file.stat().st_size == asset['bytes'] and
            build.digest(file) == asset['sha256'], 'reviewed admission reference changed')
    return dict(source_revision=row['source_revision'], report_sha256=row['report_identity']['sha256'],
                review_sha256=review['sha256'], status=row['status'])


def admission(record, plan):
    guards.require(record['stage'] == 'built' and record['passed'] and record['complete'] and
        record['ownership_closed'] and not record['active'], 'closed source-bound build required')
    build.admission(record, True)
    guards.require(inputs(plan['source_commit']) == plan['controller_inputs'] and
        kernel_ids(record) == plan['kernel_ids'] and contract() == plan['contract'],
        'controller, compiled identity or independent contract changed')
    _, proof = build.qualified_baseline()
    guards.require(proof == plan['runtime_qualification'], 'runtime qualification changed')
    rt = validation.committed(build.qualified_runtime.RESULT)
    guards.require(rt['runtime_files'] == plan['runtime_files'] and
        rt['preload'] == plan['preload'], 'runtime/preload identities changed')
    return rt


def loader_bindings(text, observed):
    """Correlate glibc namespace/object evidence with the live mapped loader."""
    path = observed['path']
    bindings = re.findall(r"binding file (\S+) \[(\d+)\] to (\S+) \[(\d+)\]:[^\n]*"
        r"symbol [`'](zeDeviceGetComputeProperties|zeKernelGetProperties)['`]", text)
    guards.require(bindings and {r[4] for r in bindings} ==
        {'zeDeviceGetComputeProperties', 'zeKernelGetProperties'}, 'missing loader symbol bindings')
    for _, source_namespace, target, namespace, _ in bindings:
        guards.require(source_namespace == namespace == '0' and
            str(Path(target).resolve()) == path, 'property symbol resolved to another loader object')
    objects = re.findall(r'file=(\S*libze_loader\.so[^\s]*) \[(\d+)\]', text)
    guards.require(objects and all(namespace == '0' and
        (not name.startswith('/') or str(Path(name).resolve()) == path)
        for name, namespace in objects), 'multiple loader objects/namespaces')
    return dict(observed, namespace=0, symbols=sorted({r[4] for r in bindings}),
        evidence='Live executable map plus canonical namespace-0 glibc object/dlsym bindings;helper requires RTLD_NOLOAD. This is this child only,not a general interop guarantee.')


def prepare():
    record = build.gen.committed(RESULT)
    guards.require('admission_plan' not in record and 'admission_check' not in record,
                   'fresh admission only; inspect existing ownership before retry')
    folder = ROOT / 'admission'
    guards.require(not folder.exists(), 'admission output already exists')
    revision = build.gen.git('rev-parse', 'HEAD').decode().strip()
    rt = validation.committed(build.qualified_runtime.RESULT)
    _, proof = build.qualified_baseline()
    names = kernel_ids(record)
    loader = [p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')]
    guards.require(len(loader) == 1, 'one pinned Level Zero loader required')
    env = dict(build.gen.committed(P / 'tools/sycl/measurement_environment.json')['environment'],
        TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
        LD_PRELOAD=rt['preload']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        LD_DEBUG='files,bindings',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
    plan = dict(stage='prepared', source_commit=revision, controller_inputs=inputs(revision),
        kernel_ids=names, contract=contract(), runtime_qualification=proof,
        runtime_files=rt['runtime_files'], preload=rt['preload'], environment=env,
        command=[record['binaries']['fixture']['path'], '--admit', *names, loader[0]],
        budgets=dict(wall=120, cpu=300, rss_cap=4 << 30, text_cap=32 << 20,
                     file_cap=32 << 20, total_cap=128 << 20),
        criteria='One property-only child;all3exact kernels pass WG128/SG16/18argument and conservative per-group local budget checks. Observe one mapped qualified loader and namespace-0 dlsym bindings for both property functions. UR pre-main proof,quiet validation,no kernel enqueue in trace,closed owned process and no fresh faults.4096Breserve is heuristic;actual bound local arguments/rounding remain separate. No numerical,occupancy,performance or model qualification.',
        retention='Retain compact queried descriptors and outcome;retire successful trace/normal logs after verification. Current generation remains for the bounded independent oracle comparison.')
    admission(record, plan)
    folder.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (folder / name).mkdir()
    build.write(folder / 'plan.json', plan)
    record['admission_plan'] = plan
    build.write(RESULT, record)
    print('Prepared property-only cell64 admission;commit plan before run.', flush=True)


def run():
    record = build.gen.committed(RESULT); plan = record['admission_plan']
    folder = ROOT / 'admission'
    guards.require(plan['stage'] == 'prepared' and 'admission_check' not in record and
        plan == json.loads((folder / 'plan.json').read_text()), 'fresh committed admission plan required')
    rt = admission(record, plan)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True)
    cursor = re.findall(r'^-- cursor: (.+)$', cursor_text, re.M)
    guards.require(len(cursor) == 1, 'fresh fault cursor unavailable')
    result = dict(passed=False, complete=False, active=False, ownership_closed=False,
                  device_initialized=None, kernel_submissions=None, model_qualified=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        build.write(folder / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = supervision.Owner(folder / 'owned', persist)
    try:
        observed = {}
        def interact(io, entry):
            guards.require(io.line(30) == 'admission_stage=before-runtime', 'unexpected property entry')
            guards.require(io.line(60).startswith('admission_stage=runtime-ready device='),
                           'property runtime not ready')
            native = Path('/proc', str(io.proc.pid))
            actual = dict(v.split(b'=', 1) for v in (native / 'environ').read_bytes().split(b'\0') if b'=' in v)
            guards.require(all(actual.get(k.encode()) == v.encode() for k, v in plan['environment'].items()),
                           'property environment changed')
            mapped_text = (native / 'maps').read_text()
            mapped = build.qualified_runtime.executable_mappings(mapped_text)
            allowed = set(rt['runtime_files']) | {str(Path(plan['command'][0]).resolve())}
            guards.require(mapped <= allowed, 'unknown property code mapping: ' + str(sorted(mapped - allowed)))
            loader = plan['command'][-1]
            adapter = plan['environment']['UR_ADAPTERS_FORCE_LOAD']
            guards.require(loader in mapped and str(Path(adapter).resolve()) in mapped,
                           'qualified UR adapter/Level Zero loader not mapped')
            maps = [line.split(maxsplit=5) for line in mapped_text.splitlines()]
            maps = [r for r in maps if len(r) == 6 and 'x' in r[1] and
                    r[5].startswith('/') and str(Path(r[5]).resolve()) == loader]
            guards.require(len(maps) == 1, 'one mapped executable loader object required')
            row = maps[0]
            observed.update(path=loader, sha256=rt['runtime_files'][loader], device=row[3],
                inode=int(row[4]), executable_start=row[0].split('-')[0], file_offset=row[2])
            while True:
                line = io.line(60)
                if line.startswith('admission_status='):
                    guards.require(line.startswith('admission_status=PASS '), 'native property refusal')
                    break
            io.proc.stdin.close();io.exit(20)
        entry, out, err = owner.run('properties', plan['command'], plan['environment'], P,
                                   interaction=interact, **plan['budgets'])
        stdout, stderr = out.read_text(), err.read_text()
        result['device_initialized'] = 'admission_stage=runtime-ready ' in stdout
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0,
                       'property child failed or did not close')
        proof = validation.preload_evidence(stderr, entry['pid'])
        result['loader_object'] = loader_bindings(stderr, observed)
        guards.require(proof['loader'] == rt['preload']['loader'], 'UR provider mismatch')
        body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main', 1)[1]
        guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:', body),
                       'native runtime error')
        validation_log = folder / 'validation.log'
        guards.require(not validation_log.exists() or validation_log.stat().st_size == 0,
                       'validation warning/error')
        trace = (folder / 'api.trace').read_text()
        guards.require('urKernelGet' in trace and not re.search(r'urEnqueue\w*Kernel', trace),
                       'query trace absent or unexpected kernel submission')
        result['kernel_submissions'] = 0
        rows = []
        for line in stdout.splitlines():
            if not line.startswith('arm='):
                continue
            fields = dict(part.split('=', 1) for part in line.split())
            guards.require(fields.pop('status') == 'PASS', 'compiled resource check failed')
            rows.append({key: int(value) for key, value in fields.items()})
        guards.require([row['arm'] for row in rows] == [0, 1, 2] and
            stdout.count('admission_status=PASS kernel_submissions=0 ') == 1,
            'property coverage incomplete')
        result.update(passed=True, complete=True, device_initialized=True,
                      properties=rows, validation=proof)
        admission(record, plan)
    except BaseException as error:
        result.update(passed=False, complete=False, error=repr(error))
    finally:
        result['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(result, boot, cursor[0]);persist()
        plan['stage'] = 'closed' if result['passed'] else 'failed'
        record['admission_check'] = result
        build.write(RESULT, record)
    print(json.dumps(result), flush=True)
    return int(not result['passed'])


def numerical_contract():
    row = validation.committed(build.gen.A / 'research-index.json',
                               ('qsa_cell64_framework_contract_20261027',))
    guards.require(row['status'] == 'reviewed_accept_conditional' and row['worker_closed'],
                   'closed framework arithmetic review required')
    ref = row['primary_identity_supplement']['vllm']
    guards.require(ref['revision'] == '3d81a510f9ebc802a7864ced38832182295f4cad' and
        ref['kernel']['sha256'] == 'cac685c7b31c32064be1251570b963b360f98e323dfae3f39d4f5bb5e67f7be8',
        'model-specific vLLM source reference changed')
    return dict(reference=ref, reference_configuration='BF16 Q/K/V operands,Dim256,12real/16padded heads per KV group,same selected IDs,no split merge,pre-gate output. Source contract only;CUDA reference not executed. This fork dequantizes INT8 pools to BF16 and scales Q before BF16 conversion;vLLM scales after QK. These pre-existing differences are not attributed to32to64.',
        component='Independent scalar dot/PV terms in FP64,FP32 online max/alpha/normalizer and BF16 probability/final output.32cadence for original/control,64for candidate. Each finite output error<=one BF16 spacing(reference)+2e-6;all reported case budget ratios finite and<=1. Zero/mask/diagnostic/guard/input/lifetime checks remain exact.32controls compare bits;64does not compare old32bits.',
        limits='This verifies the claimed component recurrence,not vLLM execution parity,model/task quality or a production adoption tolerance. No speed is collected. Full physical262144model,decode handoff/RESTORE and long thinking-on quality remain separate.')


def local_launches(trace, names, allowed_arms):
    created = {}; result = {}
    for line in trace.splitlines():
        if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
            match = re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
            if match:
                created[match[2]] = match[1]
        if '<--- urEnqueueKernelLaunchWithArgsExp(' not in line or 'UR_RESULT_SUCCESS' not in line:
            continue
        handle = re.search(r'\.hKernel = (0x[0-9a-f]+)', line)[1]
        guards.require(handle in created, 'successful kernel launch without creation identity')
        name = created[handle]
        if name not in names:
            continue
        arm = names.index(name); cells = 64 if arm == 2 else 32
        guards.require(arm in allowed_arms and '.workDim = 1,' in line and
            re.search(r'\.pLocalWorkSize = 0x[0-9a-f]+ \(128\)', line) and '.numArgs = 18,' in line,
            'attention dispatch differs from admitted one-dimensional WG128/18args')
        pairs = [(int(i), int(n)) for i, n in re.findall(
            r'UR_EXP_KERNEL_ARG_TYPE_LOCAL, \.index = (\d+), \.size = (\d+)', line)]
        # Lambda capture ABI is use order,not accessor declaration order.
        # The compiled/traced score pointer is arg15;args3..5 are16-float arrays.
        expected = {0: cells * 256 * 2, 1: 16 * cells * 2, 3: 64,
                    4: 64, 5: 64, 13: cells * 8, 15: 16 * (cells + 1) * 4, 16: 64}
        guards.require(len(pairs) == 8 and dict(pairs) == expected,
                       'actual local argument indices/extents differ from compiled/source manifest')
        # IR requires align4 except rows(arg13) align8. All eight extents are
        # multiples of64,so no natural-ABI padding is needed in either order.
        guards.require(all(n % (8 if i == 13 else 4) == 0 for i, n in pairs),
                       'compiled local alignment contract differs')
        item = result.setdefault(str(arm), dict(calls=0, local_arguments=sorted(pairs),
            dynamic_local_bytes=sum(n for _, n in pairs), work_dim=1, local_size=[128]))
        guards.require(item['local_arguments'] == sorted(pairs), 'local binding changed between launches')
        item['calls'] += 1
    guards.require(set(result) == {str(a) for a in allowed_arms}, 'missing selected attention arm')
    return result


def oracle_cases(text, mode, arm):
    found = []; samples = []
    for line in text.splitlines():
        if not line.startswith('case cells='):
            continue
        fields = dict(v.split('=', 1) for v in line.split()[1:] if '=' in v)
        for name in ('packed_logical_prefix', 'packed_guards', 'guards', 'inputs', 'status'):
            guards.require(fields.get(name) == 'PASS', 'oracle failed: ' + name)
        guards.require(int(fields['tested_arm']) == arm and int(fields['oracle_tile_cells']) == (64 if arm == 2 else 32) and
            fields['exact_cell32_control'] == ('NOT_APPLICABLE' if arm == 2 else 'PASS') and
            int(fields['packed_guard_elements']) == 17 and int(fields['packed_k_mod8']) % 8 == 2 and
            int(fields['compared']) >= int(fields['queries']) * 24 * 256,
            'numerical/control/alignment coverage differs')
        error, ratio = float(fields['max_abs']), float(fields['max_budget_ratio'])
        guards.require(math.isfinite(error) and error >= 0 and math.isfinite(ratio) and 0 <= ratio <= 1,
                       'component numerical bound exceeded')
        geometry = [int(fields[k]) for k in ('cells', 'live_cells', 'cap', 'queries', 'masked')]
        found.append(geometry)
        samples.append(dict(geometry=geometry, compared=int(fields['compared']), max_abs=error, max_budget_ratio=ratio))
    expected = [[97,49,33,1,0], [97,71,33,3,1]]
    if mode == 'oracle-component':
        expected += [[2113,2113,w,3,1] for w in (1,17,31,32,63,64,65,127,128,129,2051)]
        expected += [[97,71,65,3,0], [2113,2113,2051,3,0], [262144,262144,2051,1,0], [262144,13,33,1,0]]
    guards.require(found == expected, 'complete scalar-oracle case grid required')
    extended = 3 if mode == 'oracle-component' else 1
    for marker in ('case explicit_duplicate_ids=PASS', 'case alternating_rows=PASS unique_KV_sentinels=PASS'):
        guards.require(text.count(marker) == extended, 'adversarial coverage incomplete')
    guards.require(text.count('case mixed_widths=PASS') == (2 if mode == 'oracle-component' else 0) and
                   text.count('fixture_status=PASS ') == 1, 'oracle final/width coverage incomplete')
    return dict(samples=samples, extended_cases=extended, guards=True, inputs_immutable=True,
                physical262144_final_address_component=mode == 'oracle-component', model_quality=False)


def prepare_oracle(mode):
    record = build.gen.committed(RESULT); history = record.get('oracle_checks', [])
    failed = [c for c in history if not c['passed']]
    # One explicitly diagnosed observer failure is preserved. This permits a
    # new prospective controller-only plan,not automatic retries of GPU faults
    # or a second failed run. Native source/fixtures remain unchanged.
    guards.require(len(failed) <= 1, 'multiple failures need a new explicit investigation')
    if failed:
        guards.require(history[0] == failed[0] and failed[0]['mode'] == 'oracle-smoke' and
            failed[0]['source_commit'].startswith('f755227f8') and failed[0]['ownership_closed'] and
            failed[0]['fresh_faults'] == [] and failed[0]['error'] ==
            "RuntimeError('actual local argument indices/extents differ from compiled/source manifest')",
            'unrecognized failure;no automatic continuation')
    checks = history[1:] if failed else history
    position = 0 if mode == 'oracle-smoke' else 1
    guards.require(record['admission_check']['passed'] and record['admission_check']['ownership_closed'] and
        not record['admission_check']['fresh_faults'] and len(checks) == position and
        all(c['passed'] and c['complete'] and c['ownership_closed'] and not c['fresh_faults'] for c in checks),
        'ordered closed property/smoke gates required;no retry')
    root = ROOT / (mode + '-v2' if mode == 'oracle-smoke' and failed else mode)
    guards.require(not root.exists(), 'fresh oracle stage only')
    revision = build.gen.git('rev-parse', 'HEAD').decode().strip()
    # Carry the separately qualified native source/build and exact-kernel IDs.
    plan = {k: record['admission_plan'][k] for k in
            ('kernel_ids', 'contract', 'runtime_qualification', 'runtime_files', 'preload')}
    plan.update(stage='prepared', mode=mode, output_root=str(root), source_commit=revision, controller_inputs=inputs(revision),
        numerical=numerical_contract(), budgets=dict(wall=300, cpu=600, rss_cap=8 << 30,
        text_cap=16 << 20, file_cap=32 << 20, total_cap=192 << 20), jobs=[],
        local_contract='Eight active LLVM local pointer arguments align4,rows(arg13)align8. Declared source includes one unused4B Q accessor. Require successful traced WG128/18arg/SG16source kernel with exact active argument indices/extents;raw sum20032/39744+reported32+heuristic4096<=queried131072. Hardware rounding/occupancy unmeasured;successful bounded launch is separate evidence.',
        retirement='Keep useful per-case numerical samples,actual local sizes/launch counts,source/recipe and fault/closure outcome. Retire successful full traces and routine logs after verified extraction. No model speed or quality claim.')
    if failed:
        plan['original_observer_failure'] = dict(source_commit=failed[0]['source_commit'],
            original_plan_commit='659384563', original_status='failed_preserved',
            correction='Scores arg15,not3;source/compiled lambda capture ABI. Same strict8argument extent/alignment checks and numerical bounds. Native source,binaries and generated inputs unchanged.')
    rt = admission(record, plan)
    root.mkdir(mode=0o700)
    base = build.gen.committed(P / 'tools/sycl/measurement_environment.json')['environment']
    for arm in range(3):
        folder = root / str(arm); folder.mkdir()
        for name in ('tmp', 'cache'):
            (folder / name).mkdir()
        env = dict(base, TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
            LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
            ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
            STRATA_TEST_MAIN_GATES='1', UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'api.trace'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
        plan['jobs'].append(dict(arm=arm, environment=env,
            command=[record['binaries']['oracle']['path'], '--smoke' if mode == 'oracle-smoke' else '--component', str(arm)]))
    build.write(root / 'plan.json', plan);record['oracle_plan'] = plan;build.write(RESULT, record)
    print('Prepared ' + mode + ';commit numerical/launch plan before execution.', flush=True)


def run_oracle(mode):
    record = build.gen.committed(RESULT);plan = record['oracle_plan'];root = Path(plan['output_root'])
    guards.require(root.parent == ROOT and root.name in (mode, mode+'-v2'), 'unexpected oracle output root')
    guards.require(plan['mode'] == mode and plan['stage'] == 'prepared' and
        plan == json.loads((root / 'plan.json').read_text()) and not (root / 'owned.json').exists(),
        'fresh matching committed oracle plan required')
    rt = admission(record, plan)
    guards.require(plan['numerical'] == numerical_contract(), 'prospective numerical contract changed')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True, timeout=15)
    cursor = re.findall(r'^-- cursor: (.+)$', cursor_text, re.M)
    guards.require(len(cursor) == 1, 'fresh fault cursor required')
    result = dict(mode=mode, source_commit=plan['source_commit'], passed=False, complete=False,
        observations=[], active=False, ownership_closed=False, model_qualified=False,
        GPU_kernel_execution_observed=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        build.write(root / 'owned.json', dict(result, commands=owner.commands if owner else []))
    owner = supervision.Owner(root / 'owned', persist)
    try:
        for job in plan['jobs']:
            arm = job['arm'];folder = root / str(arm)
            guards.driver_guard();guards.no_model_owners([])
            guards.require(build.digest(job['command'][0]) == record['binaries']['oracle']['sha256'],
                           'oracle executable changed')
            initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
            def interact(io, entry):
                for stage in ('before-runtime', 'runtime-ready', 'after-cleanup'):
                    while True:
                        line = io.line(250)
                        if line.startswith('QSA_BF16_READY '):
                            break
                    guards.require(line == 'QSA_BF16_READY ' + stage, 'native gate differs')
                    delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
                    guards.require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000,
                                   'suspend/clock discontinuity')
                    native = Path('/proc', str(io.proc.pid))
                    guards.require(str((native / 'exe').resolve()) == str(Path(job['command'][0]).resolve()),
                                   'wrong native executable')
                    actual = dict(v.split(b'=',1) for v in (native/'environ').read_bytes().split(b'\0') if b'=' in v)
                    guards.require(all(actual.get(k.encode()) == v.encode() for k,v in job['environment'].items()),
                                   'native oracle environment changed')
                    mapped = build.qualified_runtime.executable_mappings((native/'maps').read_text())
                    allowed = set(rt['runtime_files']) | {str(Path(job['command'][0]).resolve())}
                    guards.require(mapped <= allowed, 'unknown oracle executable mapping: ' + str(sorted(mapped-allowed)))
                    if stage == 'runtime-ready':
                        guards.require({rt['preload']['loader'], job['environment']['ZE_ENABLE_ALT_DRIVERS'],
                            rt['preload']['library']['path']} <= mapped, 'qualified provider missing')
                    io.send(b'X')
                guards.require(io.line(20).startswith('fixture_status=PASS '), 'final oracle status missing')
                io.proc.stdin.close();io.exit(20)
            print('Running ' + mode + ' arm' + str(arm), flush=True)
            entry, out, err = owner.run(str(arm), job['command'], job['environment'], P,
                                        interaction=interact, **plan['budgets'])
            guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'oracle child failed/ownership open')
            stdout, stderr = out.read_text(), err.read_text()
            proof = validation.preload_evidence(stderr, entry['pid'])
            guards.require(proof['loader'] == rt['preload']['loader'], 'oracle UR loader mismatch')
            body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:', body),
                           'native oracle runtime error')
            log = folder/'validation.log'
            guards.require(not log.exists() or log.stat().st_size == 0, 'oracle validation warning/error')
            launches = local_launches((folder/'api.trace').read_text(), plan['kernel_ids'], [2] if arm == 2 else [0,1])
            result['GPU_kernel_execution_observed'] = True
            for key, value in launches.items():
                descriptor = record['admission_check']['properties'][int(key)]
                guards.require(value['dynamic_local_bytes'] + descriptor['reported_local_bytes'] + 4096 <=
                    descriptor['per_group_local_max'], 'actual local payload exceeds query budget')
            result['observations'].append(dict(arm=arm, exit_code=0, ownership_closed=True,
                validation=proof, local_launches=launches, oracle=oracle_cases(stdout, mode, arm)))
            persist()
        admission(record, plan)
        result.update(passed=True, complete=True)
    except BaseException as error:
        result.update(passed=False, complete=False, error=repr(error))
    finally:
        result['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        postchecks(result, boot, cursor[0]);persist()
        plan['stage'] = 'closed' if result['passed'] else 'failed'
        record.setdefault('oracle_checks', []).append(result)
        record['GPU_executed'] = True if result['GPU_kernel_execution_observed'] else None if owner.commands else record['GPU_executed']
        build.write(RESULT, record)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    guards.require(len(sys.argv) in (2,3) and sys.argv[1] in ('prepare','run') and
        (len(sys.argv) == 2 or sys.argv[2] in ('oracle-smoke','oracle-component')),
        'prepare|run [oracle-smoke|oracle-component]')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([]);guards.driver_guard()
        if len(sys.argv) == 2:
            raise SystemExit(prepare() if sys.argv[1] == 'prepare' else run())
        raise SystemExit(prepare_oracle(sys.argv[2]) if sys.argv[1] == 'prepare' else run_oracle(sys.argv[2]))
