#!/usr/bin/env python3
"""Main-owned subgroup-K property,smoke,boundary and actual-ID comparison.

Own archivedcfabc1ee2 controller scaffolds;repository LICENSE applies.
No model route or decode setting is changed by these component checks.
"""
import ast
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_qsa_k_staging as b
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks
P,B,ROOT,RESULT=b.P,b.B,b.ROOT,b.RESULT
CPP='bench/sycl/qsa_k_staging.cpp'

def inputs(revision):
    pending = [Path(__file__).resolve()]; seen = set(); pins = {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = str(path.relative_to(P))
        raw = b.gen.git('show', revision + ':' + name)
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
    for arm, stride in (('original', 33), ('control', 33), ('candidate', 33)):
        asset = record['device_IR'][arm]
        guards.require(b.digest(asset['path']) == asset['sha256'], 'device IR changed')
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
    row=validation.committed(b.gen.A/'research-index.json',('qsa_k_staging_harness_contract_20261028',))
    guards.require(row['status']=='reviewed_accept_conditional' and row['worker_closed']
        and row['review']['status']=='completed' and row['review']['model']=='gpt-6-luna'
        and row['review']['reasoning_effort']=='max','closed independent harness review required')
    for name,sha in row['source_snapshot'].items():
        guards.require(b.digest(P/name)==sha,'source-reviewed native fixture changed: '+name)
    return dict(source_commit=row['source_commit'],source_snapshot=row['source_snapshot'],
        report_sha256=row['report_identity']['sha256'],review_sha256=row['review']['sha256'],status=row['status'])


def admission(record, plan):
    guards.require(record['stage'] == 'built' and record['passed'] and record['complete'] and
        record['ownership_closed'] and not record['active'], 'closed source-bound build required')
    b.admission(record, True)
    guards.require(inputs(plan['source_commit']) == plan['controller_inputs'] and
        kernel_ids(record) == plan['kernel_ids'] and contract() == plan['contract'],
        'controller, compiled identity or independent contract changed')
    _, proof = b.qualified_baseline()
    guards.require(proof == plan['runtime_qualification'], 'runtime qualification changed')
    rt = validation.committed(b.qualified_runtime.RESULT)
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
    record = b.gen.committed(RESULT)
    guards.require('admission_plan' not in record and 'admission_check' not in record,
                   'fresh admission only; inspect existing ownership before retry')
    folder = ROOT / 'admission'
    guards.require(not folder.exists(), 'admission output already exists')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    rt = validation.committed(b.qualified_runtime.RESULT)
    _, proof = b.qualified_baseline()
    names = kernel_ids(record)
    loader = [p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')]
    guards.require(len(loader) == 1, 'one pinned Level Zero loader required')
    env = dict(b.gen.committed(P / 'tools/sycl/measurement_environment.json')['environment'],
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
    b.write(folder / 'plan.json', plan)
    record['admission_plan'] = plan
    b.write(RESULT, record)
    print('Prepared property-only subgroup-K admission;commit plan before run.', flush=True)


def run():
    record = b.gen.committed(RESULT); plan = record['admission_plan']
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
        b.write(folder / 'owned.json', dict(result, commands=owner.commands if owner else []))
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
            mapped = b.qualified_runtime.executable_mappings(mapped_text)
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
        b.write(RESULT, record)
    print(json.dumps(result), flush=True)
    return int(not result['passed'])


def numerical_contract():
    return dict(recurrence_cells=32,all_candidate_output_bits_match_original32=True,
        scalar_spots='FP64dot/PVterms,FP32online32recurrence,BF16probability/output;error<=oneBF16spacing+2e-6.',
        coverage='All output words finite;guards,immutable inputs,packed operands and repeated outputs exact. Translated boundary IDs include physical262143. Not full-context model fill,model quality or decode qualification.')


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
        arm = names.index(name); cells = 32; vpitch = 256
        guards.require(arm in allowed_arms and '.workDim = 1,' in line and
            re.search(r'\.pLocalWorkSize = 0x[0-9a-f]+ \(128\)', line) and '.numArgs = 18,' in line,
            'attention dispatch differs from admitted one-dimensional WG128/18args')
        pairs = [(int(i), int(n)) for i, n in re.findall(
            r'UR_EXP_KERNEL_ARG_TYPE_LOCAL, \.index = (\d+), \.size = (\d+)', line)]
        # Lambda capture ABI is use order,not accessor declaration order.
        # The compiled/traced score pointer is arg15;args3..5 are16-float arrays.
        expected = {0: cells * vpitch * 2, 1: 16 * cells * 2, 3: 64,
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


def run_admission(record,plan):
    admission(record,plan)
    guards.require(b.compiled_inputs(record)==record['actual_compiled_inputs'],'compiled dependency closure changed')
    guards.require(numerical_contract()==plan['numerical'] and observer_contract(record)==plan['observer'],
        'numerical/observer qualification changed')
    return validation.committed(b.qualified_runtime.RESULT)


def prepare_run(mode):
    record = b.gen.committed(RESULT);history = record.get('actual_id_checks', [])
    position = {'smoke':0,'boundary':1,'perf':2}[mode]
    b.guards.require(len(history) == position and all(v['passed'] and v['complete'] and
        v['ownership_closed'] and not v['fresh_faults'] for v in history), 'ordered successful actual-ID gates required;no retry')
    if position:
        b.guards.require([v['mode'] for v in history]==['smoke','boundary'][:position] and record['actual_id_plan']['stage']=='closed','preceding native gate incomplete')
    else:
        b.guards.require('actual_id_plan' not in record, 'inspect existing actual-ID plan before preparing')
    guards.require(record['admission_check']['passed'] and record['admission_check']['complete'] and record['admission_check']['ownership_closed'] and not record['admission_check']['fresh_faults'],'property-only admission required')
    if mode == 'perf':
        checks=record.get('oracle_checks',[])
        guards.require([v['mode'] for v in checks]==['smoke','component'] and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults']
                for v in checks),'separate candidate numerical/oracle gates required before timing')
    root = ROOT/('actual-id-'+mode)
    b.guards.require(not root.exists(), 'fresh actual-ID stage only')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    plan = {k:record['admission_plan'][k] for k in
            ('kernel_ids', 'contract', 'runtime_qualification', 'runtime_files', 'preload')}
    plan.update(stage='prepared', mode=mode, source_commit=revision, controller_inputs=inputs(revision),
        numerical=numerical_contract(), observer=observer_contract(record),
        output_root=str(root), jobs=[], budgets=dict(wall=300 if mode == 'smoke' else 600,
            cpu=600 if mode == 'smoke' else 1800, rss_cap=8 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=192 << 20),
        comparison=dict(processes=3, rounds=6, shapes=[512,6491], arms=['original32','private32','subgroupK32'],
            order='(process+round+offset)%3', repeats={'512':64,'6491':4}, minimum_batch_ns=100000000,
            minimum_clock_multiple=1000, control_original_ratio_bounds=[0.99,1.01], candidate_control_ratio_max=0.99,
            decision='All six process/shape medians must pass control/original wall calibration and candidate/control wall plus attention-service ratio gates. Preserve every eligible sample. Failed speed gate stops model advancement;no adaptive retry or changed thresholds.',
            scope='Actual selected-ID geometry with synthetic finite Q/K/V;pack+attention wrapper wall and private attention-only service. Not PCIe traffic,hardware saturation,production speed,model quality or1000/70qualification.'),
        retirement='Retain individual timing samples,numerical/delta bounds,closure/fault results and decision. Retire successful full traces and normal logs after extraction. Keep one current component generation only for this live comparison.')
    rt = run_admission(record, plan)
    base = b.gen.committed(P/'tools/sycl/measurement_environment.json')['environment']
    root.mkdir(mode=0o700)
    for process in range(3 if mode == 'perf' else 1):
        folder = root/str(process);folder.mkdir()
        for name in ('tmp','cache'):
            (folder/name).mkdir()
        env = dict(base, TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'),
            LD_PRELOAD=rt['preload']['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
            ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
            STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
        if mode in ('smoke','boundary'):
            env['UR_LOG_TRACING'] = 'level:info;flush:info;output:file,'+str(folder/'api.trace')
        else:
            env.pop('UR_LOG_TRACING', None);env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
        command = [record['binaries']['fixture']['path'], '--'+mode if mode in ('smoke','boundary') else '--bench', record['selected_fixture']['path']]
        if mode == 'perf':
            command.append(str(process))
        plan['jobs'].append(dict(process=process, command=command, environment=env))
    b.write(root/'plan.json', plan);record['actual_id_plan'] = plan;b.write(RESULT, record)
    print('Prepared actual-ID '+mode+';commit source-bound plan before execution.', flush=True)


def parsed_output(text, mode, process):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    shapes = [3] if mode in ('smoke','boundary') else [512,6491]
    b.guards.require(rows and rows[-1] == {'kind':'PASS'} and sum(v['kind'] == 'PASS' for v in rows) == 1 and
        {v['kind'] for v in rows} <= {'PASS','numerical','delta','checked','sample'}, 'incomplete/unknown actual-ID output')
    numerical = [v for v in rows if v['kind'] == 'numerical']
    b.guards.require([(v['queries'],v['arm']) for v in numerical] == [(q,a) for q in shapes for a in range(3)], 'scalar coverage differs')
    for v in numerical:
        spots = {3:3,512:6,6491:8}[v['queries']]
        b.guards.require(v['reference_cells'] == 32 and v['spot_queries'] == spots and
            v['compared'] == spots*24*256 and math.isfinite(v['max_abs']) and v['max_abs'] >= 0 and
            math.isfinite(v['max_budget_ratio']) and 0 <= v['max_budget_ratio'] <= 1, 'scalar numerical bound/coverage failed')
    deltas = [v for v in rows if v['kind'] == 'delta']
    b.guards.require([v['queries'] for v in deltas] == shapes, 'full-output delta coverage differs')
    for v in deltas:
        b.guards.require(v['elements'] == v['queries']*24*256 and v['changed_from32'] == 0 and v['max_abs'] == 0 and v['relative_l2'] == 0 and
            all(math.isfinite(v[k]) and v[k] >= 0 for k in ('max_abs','relative_l2')), 'nonfinite output/delta')
    checked = [v for v in rows if v['kind'] == 'checked']
    b.guards.require([v['queries'] for v in checked] == shapes and all(v['cells'] == (262144 if mode=='boundary' else 39259) and
        all(v[k] is True for k in ('guards','inputs_immutable','cell32_exact','own_reference')) and
        v['candidate_old_bits_required'] is True for v in checked), 'guard/input/control qualification failed')
    samples = [v for v in rows if v['kind'] == 'sample']
    expected = [] if mode in ('smoke','boundary') else [(q,r,(process+r+o)%3) for q in shapes for r in range(6) for o in range(3)]
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
    rows = [dict(kind='numerical',queries=3,arm=a,reference_cells=32,
        spot_queries=3,compared=18432,max_abs=0,max_budget_ratio=0) for a in range(3)]
    rows += [dict(kind='delta',queries=3,elements=18432,changed_from32=0,max_abs=0,relative_l2=0),
        dict(kind='checked',queries=3,cells=39259,guards=True,inputs_immutable=True,cell32_exact=True,
             own_reference=True,candidate_old_bits_required=True),dict(kind='PASS')]
    def parse(values):
        return parsed_output('\n'.join(json.dumps(v) for v in values),'smoke',0)
    parse(rows)  # K-staging32 preserves arithmetic and original output bits.
    bad = copy.deepcopy(rows);bad[2]['reference_cells'] = 64
    reject('wrong_candidate_recurrence',lambda:parse(bad))
    bad = copy.deepcopy(rows);bad[2]['max_budget_ratio'] = math.inf
    reject('nonfinite_reference_ratio',lambda:parse(bad))
    bad = copy.deepcopy(rows);bad[-2]['guards'] = False
    reject('corrupted_guard',lambda:parse(bad))
    reject('missing_footer',lambda:parse(rows[:-1]))
    bad = copy.deepcopy(rows);bad[3]['changed_from32'] = 1
    reject('K_staging_changed_original_bits',lambda:parse(bad))
    reject('boundary_extent_not_exercised',lambda:parsed_output(
        '\n'.join(json.dumps(v) for v in rows),'boundary',0))
    boundary = copy.deepcopy(rows);boundary[-2]['cells'] = 262144
    parsed_output('\n'.join(json.dumps(v) for v in boundary),'boundary',0)
    names=['_ZTSoriginal','_ZTScontrol','_ZTScandidate'];trace=[]
    for arm,name in enumerate(names):
        handle=hex(arm+1)
        trace.append(f'<--- urKernelCreate( ({name}), .phKernel = value ({handle}) UR_RESULT_SUCCESS')
        sizes={0:16384,1:1024,3:64,4:64,5:64,13:256,15:2112,16:64}
        local=' '.join(f'UR_EXP_KERNEL_ARG_TYPE_LOCAL, .index = {i}, .size = {n}' for i,n in sizes.items())
        trace.append(f'<--- urEnqueueKernelLaunchWithArgsExp( .hKernel = {handle}, .workDim = 1, '
            f'.pLocalWorkSize = 0x100 (128), .numArgs = 18, {local} UR_RESULT_SUCCESS')
    bound=local_launches('\n'.join(trace),names,[0,1,2])
    guards.require([bound[str(a)]['dynamic_local_bytes'] for a in range(3)]==[20032,20032,20032],
        'traced local payload accounting differs')
    reject('wrong_candidate_local_extent',lambda:local_launches(
        '\n'.join(trace).replace('.size = 16384','.size = 16896'),names,[0,1,2]))
    reject('wrong_score_argument_index',lambda:local_launches(
        '\n'.join(trace).replace('.index = 15, .size = 2112','.index = 3, .size = 2112'),names,[0,1,2]))
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
    def oracle_text(mode):
        geometry=[(97,49,33,1),(97,71,33,3)]
        if mode=='component':
            geometry += [(2113,2113,w,3) for w in (1,17,31,32,63,65,2051)]
            geometry += [(262144,262144,2051,1),(262144,13,33,1)]
        text=[]
        for cells,live,cap,queries in geometry:
            text.append(f'case cells={cells} live_cells={live} cap={cap} queries={queries} '
                'packed_guard_elements=17 packed_k_mod8=2 max_budget_ratio=0 '
                'packed_logical_prefix=PASS packed_guards=PASS guards=PASS inputs=PASS status=PASS')
            text.append(f'case exact_prefix33_bits=PASS cells={cells} cap={cap} queries={queries}')
        return '\n'.join(text+['fixture_status=PASS rest=UNRUN'])
    for mode in ('smoke','component'):parsed_oracle(oracle_text(mode),mode)
    text=oracle_text('smoke')
    reject('oracle_wrong_alignment',lambda:parsed_oracle(text.replace('packed_k_mod8=2','packed_k_mod8=0'),'smoke'))
    reject('oracle_missing_exact_comparison',lambda:parsed_oracle(text.replace('case exact_prefix33_bits=PASS','case skipped'),'smoke'))
    reject('oracle_wrong_geometry',lambda:parsed_oracle(text.replace('live_cells=49','live_cells=71'),'smoke'))
    reject('oracle_nonfinite_budget',lambda:parsed_oracle(text.replace('max_budget_ratio=0','max_budget_ratio=nan'),'smoke'))
    print(json.dumps(dict(passed=True,GPU_executed=False,controls=21,rejected=rejected,
        oracle_positive_controls=2,oracle_negative_controls=4,
        positive='K-staging32matches original bits;all-group speed gain passes.',
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
            b.guards.require(b.digest(job['command'][0]) == record['binaries']['fixture']['sha256'], 'actual-ID binary changed')
            initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
            def interact(io,entry):
                for stage in ('before-runtime','runtime-ready','after-cleanup'):
                    while True:
                        line = io.line(250)
                        if line.startswith('QSA_K_STAGING_GATE '):
                            break
                    b.guards.require(line == 'QSA_K_STAGING_GATE '+stage, 'actual-ID gate differs')
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
            stderr = err.read_text();proof = validation.preload_evidence(stderr,entry['pid'])
            b.guards.require(proof['loader'] == rt['preload']['loader'], 'actual-ID UR loader mismatch')
            body = stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            b.guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body), 'actual-ID runtime error')
            log = folder/'validation.log'
            b.guards.require(not log.exists() or log.stat().st_size == 0, 'actual-ID validation warning/error')
            observation = dict(process=process, exit_code=0, ownership_closed=True,
                validation=proof, output=parsed_output(out.read_text(),mode,process))
            if mode in ('smoke','boundary'):
                launches = local_launches((folder/'api.trace').read_text(),plan['kernel_ids'],[0,1,2])
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
        postchecks(result,boot,cursors[0]);persist()
        if not result['passed'] and 'decision' in result:
            result['decision']['advance_to_model'] = False
        plan['stage'] = 'closed' if result['passed'] else 'failed'
        record.setdefault('actual_id_checks',[]).append(result);b.write(RESULT,record)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    if 'decision' in result:
        print(json.dumps(result['decision']),flush=True)
    return int(not result['passed'])


def parsed_oracle(text, mode):
    cases=[line for line in text.splitlines() if line.startswith('case cells=')]
    exact=[line for line in text.splitlines() if line.startswith('case exact_prefix33_bits=PASS ')]
    expected=2 if mode=='smoke' else 11
    guards.require(text.count('fixture_status=PASS ')==1 and len(cases)==len(exact)==expected,
        'oracle case,exact-bit or footer coverage differs')
    geometry=[(97,49,33,1),(97,71,33,3)]
    if mode=='component':
        geometry += [(2113,2113,w,3) for w in (1,17,31,32,63,65,2051)]
        geometry += [(262144,262144,2051,1),(262144,13,33,1)]
    fields=[dict(token.split('=',1) for token in line.split()[1:]) for line in cases]
    guards.require([tuple(int(v[k]) for k in ('cells','live_cells','cap','queries')) for v in fields]==geometry,
        'oracle geometry differs from prospective coverage')
    guards.require(all(math.isfinite(float(v['max_budget_ratio'])) and
        0<=float(v['max_budget_ratio'])<=1 for v in fields),'oracle numerical bound invalid')
    guards.require(all('packed_guard_elements=17 ' in line and 'packed_k_mod8=2 ' in line and
        all(token in line for token in ('packed_logical_prefix=PASS','packed_guards=PASS',
            'guards=PASS','inputs=PASS','status=PASS')) for line in cases), 'oracle guard/alignment/input gate failed')
    return dict(cases=expected,exact_prefix33_bits_cases=len(exact),case_summaries=cases,
        scope='Existing independent BF16/FP32 oracle and guard/mask/nonfinite/tail/admission controls through the candidate adapter;not full model or speed.')


def prepare_oracle(mode):
    record=b.gen.committed(RESULT);history=record.get('oracle_checks',[])
    guards.require(mode in ('smoke','component') and
        [v['mode'] for v in history]==([] if mode=='smoke' else ['smoke']) and
        all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults'] for v in history),
        'ordered fresh oracle stages only;inspect prior ownership before retry')
    checks=record.get('actual_id_checks',[])
    guards.require(checks and checks[0]['mode']=='smoke' and checks[0]['passed'] and
        checks[0]['complete'] and checks[0]['ownership_closed'] and not checks[0]['fresh_faults'],
        'first traced actual-ID smoke must close before oracle')
    root=ROOT/('oracle-'+mode);guards.require(not root.exists(),'oracle stage already exists')
    plan={k:record['admission_plan'][k] for k in
        ('kernel_ids','contract','runtime_qualification','runtime_files','preload')}
    plan.update(stage='prepared',mode=mode,source_commit=b.gen.git('rev-parse','HEAD').decode().strip(),
        controller_inputs=inputs(b.gen.git('rev-parse','HEAD').decode().strip()),
        numerical=numerical_contract(),observer=observer_contract(record),
        command=[record['binaries']['numerical']['path'],'--'+mode],output_root=str(root),
        budgets=dict(wall=300,cpu=600,rss_cap=8<<30,text_cap=32<<20,file_cap=64<<20,total_cap=128<<20),
        criteria='Existing independent oracle2/11cases and all prefix33bit comparisons;align2scratch,masked/missing/tail/nonfinite/alias/no-op checks,quiet validation,exact mapped providers,normal owned exit and no fresh fault. No performance/model qualification.')
    rt=run_admission(record,plan)
    base=b.gen.committed(P/'tools/sycl/measurement_environment.json')['environment']
    plan['environment']=dict(base,TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
        LD_PRELOAD=rt['preload']['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'],
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',STRATA_TEST_MAIN_GATES='1',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'api.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    root.mkdir(mode=0o700)
    for name in ('tmp','cache'):(root/name).mkdir()
    b.write(root/'plan.json',plan);record['oracle_plan']=plan;b.write(RESULT,record)
    print('Prepared candidate oracle '+mode+';commit exact plan before execution.',flush=True)


def run_oracle(mode):
    record=b.gen.committed(RESULT);plan=record['oracle_plan'];root=Path(plan['output_root'])
    guards.require(plan['stage']=='prepared' and plan['mode']==mode and
        root==ROOT/('oracle-'+mode) and plan==json.loads((root/'plan.json').read_text()) and
        not (root/'owned.json').exists(),'fresh committed oracle plan only')
    rt=run_admission(record,plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor=re.findall(r'^-- cursor: (.+)$',subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor'],text=True,timeout=15),re.M)
    guards.require(len(cursor)==1,'fresh oracle fault cursor required')
    result=dict(mode=mode,passed=False,complete=False,active=False,ownership_closed=False,GPU_executed=None)
    owner=None
    def persist():
        result['active']=bool(owner and owner.active is not None)
        b.write(root/'owned.json',dict(result,commands=owner.commands if owner else []))
    owner=supervision.Owner(root/'owned',persist)
    try:
        guards.require(b.digest(plan['command'][0])==record['binaries']['numerical']['sha256'],'oracle executable changed')
        def interact(io,entry):
            for stage in ('before-runtime','runtime-ready','after-cleanup'):
                while True:
                    line=io.line(250)
                    if line.startswith('QSA_BF16_READY '):break
                guards.require(line=='QSA_BF16_READY '+stage,'oracle gate differs')
                native=Path('/proc',str(io.proc.pid))
                guards.require(str((native/'exe').resolve())==str(Path(plan['command'][0]).resolve()),'wrong oracle executable')
                env=dict(v.split(b'=',1) for v in (native/'environ').read_bytes().split(b'\0') if b'=' in v)
                guards.require(all(env.get(k.encode())==v.encode() for k,v in plan['environment'].items()),'oracle environment changed')
                mapped=b.qualified_runtime.executable_mappings((native/'maps').read_text())
                allowed=set(rt['runtime_files'])|{str(Path(plan['command'][0]).resolve())}
                guards.require(mapped<=allowed,'unexpected oracle code mapping')
                if stage=='runtime-ready':
                    guards.require({rt['preload']['loader'],rt['preload']['library']['path'],plan['environment']['ZE_ENABLE_ALT_DRIVERS']}<=mapped,'oracle provider missing')
                io.send(b'X')
            guards.require(io.line(20).startswith('fixture_status=PASS '),'oracle footer missing')
            io.proc.stdin.close();io.exit(20)
        entry,out,err=owner.run(mode,plan['command'],plan['environment'],P,interaction=interact,**plan['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'oracle process failed or ownership open')
        stderr=err.read_text();proof=validation.preload_evidence(stderr,entry['pid'])
        guards.require(proof['loader']==rt['preload']['loader'],'oracle UR loader mismatch')
        body=stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
        guards.require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|strata UR validation failure:',body),
            'oracle runtime error')
        log=root/'validation.log'
        guards.require(not log.exists() or log.stat().st_size==0,'oracle validation warning/error')
        result.update(output=parsed_oracle(out.read_text(),mode),validation=proof,
            local_launches=local_launches((root/'api.trace').read_text(),plan['kernel_ids'],[2]))
        run_admission(record,plan);result.update(passed=True,complete=True,GPU_executed=True,exit_code=0)
    except BaseException as error:
        result.update(error=repr(error),passed=False,complete=False)
    finally:
        result['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        if not result['ownership_closed']:result.update(passed=False,complete=False,error='oracle ownership remains open')
        postchecks(result,boot,cursor[0]);persist();plan['stage']='closed' if result['passed'] else 'failed'
        record.setdefault('oracle_checks',[]).append(result);b.write(RESULT,record)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return int(not result['passed'])


if __name__=='__main__':
    args=sys.argv[1:]
    guards.require(args in (['prepare-admission'],['run-admission'],['host-controls'],
        ['prepare-oracle','smoke'],['run-oracle','smoke'],['prepare-oracle','component'],['run-oracle','component'],
        ['prepare','smoke'],['run','smoke'],['prepare','boundary'],['run','boundary'],['prepare','perf'],['run','perf']),
        'prepare-admission|run-admission|host-controls|prepare|run smoke|boundary|perf')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        if len(args)==1:
            raise SystemExit(host_controls() if args[0]=='host-controls' else prepare() if args[0]=='prepare-admission' else run())
        if args[0] in ('prepare-oracle','run-oracle'):
            raise SystemExit(prepare_oracle(args[1]) if args[0]=='prepare-oracle' else run_oracle(args[1]))
        raise SystemExit(prepare_run(args[1]) if args[0]=='prepare' else run_stage(args[1]))
