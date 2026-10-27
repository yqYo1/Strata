#!/usr/bin/env python3
"""Main-owned logical-gather property,smoke and topology sensitivity screen.

Own archived4f2fffe03 controller scaffolds;repository LICENSE applies.
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
import build_qsa_locality as b
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks
P,B,ROOT,RESULT=b.P,b.B,b.ROOT,b.RESULT
CPP='bench/sycl/qsa_locality.cpp'

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
    for arm, stride in (('original', 33), ('control', 33)):
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
    guards.require(len(set(names)) == 2, 'compiled arm identities overlap')
    return names


def contract():
    row=validation.committed(b.gen.A/'research-index.json',('qsa_locality_harness_contract_20261027',))
    guards.require(row['worker_closed'] and row['review']['status']=='completed' and
        row['review']['model']=='gpt-6-luna' and row['review']['reasoning_effort']=='max' and
        row['main_acceptance']=='eligible_native_gates','closed independent locality review required')
    for name,sha in row['source_snapshot'].items():
        guards.require(b.digest(P/name)==sha,'source-reviewed locality input changed: '+name)
    return dict(source_snapshot=row['source_snapshot'],report_sha256=row['report_identity']['sha256'],
        review_sha256=row['review']['sha256'],main_acceptance=row['main_acceptance'])


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
        criteria='One property-only child;both exact kernels pass WG128/SG16/18argument and conservative per-group local budget checks. Observe one mapped qualified loader and namespace-0 dlsym bindings for both property functions. UR pre-main proof,quiet validation,no kernel enqueue in trace,closed owned process and no fresh faults.4096Breserve is heuristic;actual bound local arguments/rounding remain separate. No numerical,occupancy,performance or model qualification.',
        retention='Retain compact queried descriptors and outcome;retire successful trace/normal logs after verification. Current generation remains for the bounded independent oracle comparison.')
    admission(record, plan)
    folder.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (folder / name).mkdir()
    b.write(folder / 'plan.json', plan)
    record['admission_plan'] = plan
    b.write(RESULT, record)
    print('Prepared property-only locality admission;commit plan before run.', flush=True)


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
        guards.require([row['arm'] for row in rows] == [0, 1] and
            stdout.count('admission_status=PASS kernel_submissions=0\n') == 1,
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
    return dict(recurrence_cells=32,source_derived_counts=2051,
        oracle='Logical-cell-constant K/V perKVhead makes all2051scores equal. BF16output must exactly match the dequantized BF16V for all queries,24heads and256dimensions. Every arm and repetition is checked.',
        coverage='All output bits,guards,immutable inputs and packed rows. Same source count and launch mask. Synthetic Q/K/V and deliberate H/S topology;not model-selector rankings,physical262144model fill,quality or decode qualification.')


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
    position = {'smoke':0,'perf':1}[mode]
    b.guards.require(len(history) == position and all(v['passed'] and v['complete'] and
        v['ownership_closed'] and not v['fresh_faults'] for v in history), 'ordered successful actual-ID gates required;no retry')
    if position:
        b.guards.require([v['mode'] for v in history]==['smoke'][:position] and record['actual_id_plan']['stage']=='closed','preceding native gate incomplete')
    else:
        b.guards.require('actual_id_plan' not in record, 'inspect existing actual-ID plan before preparing')
    guards.require(record['admission_check']['passed'] and record['admission_check']['complete'] and record['admission_check']['ownership_closed'] and not record['admission_check']['fresh_faults'],'property-only admission required')
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
        comparison=dict(processes=3,rounds=8,shapes=[6491],arms=['originalP','privateP','privateHot','privateSpread'],
            order=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]],order_row='(round+process)%4',
            repeats=8,minimum_batch_ns=100000000,minimum_clock_multiple=1000,
            control_original_ratio_bounds=[0.99,1.01],sensitivity_ratio_bounds=[0.97,1.03],
            decision='All3processes must pass privateP/originalPwall calibration. A diagnostic response requires the same H/P,S/P or S/H service-ratio direction beyond3percent in all3process medians. A flat/inconsistent result is inconclusive. Never a model-adoption or saturation gate.',
            scope='Exact6491query/source-derived2051counts,synthetic repeated-value Q/K/V. Only gather-address topology changes among private arms. Wall includes pack and wrapper;direct event covers attention. No physical-traffic claim.'),
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
        command = [record['binaries']['fixture']['path'], '--'+mode if mode == 'smoke' else '--bench', record['selected_fixture']['path']]
        if mode == 'perf':
            command.append(str(process))
        plan['jobs'].append(dict(process=process, command=command, environment=env))
    b.write(root/'plan.json', plan);record['actual_id_plan'] = plan;b.write(RESULT, record)
    print('Prepared actual-ID '+mode+';commit source-bound plan before execution.', flush=True)


def parsed_output(text,mode,process):
    rows=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
    guards.require(rows and rows[-1]=={'kind':'PASS'},'one complete native footer required')
    guards.require(all(v.get('kind') in ('numerical','checked','sample','PASS') for v in rows),'unknown result row')
    queries=3 if mode=='smoke' else 6491
    numerical=[v for v in rows if v.get('kind')=='numerical']
    guards.require([(v['queries'],v['arm']) for v in numerical]==[(queries,a) for a in range(4)] and
        all(v['compared']==queries*24*256 and v['all_bits_exact'] is True for v in numerical),'complete exact uniform-output oracle required')
    checked=[v for v in rows if v.get('kind')=='checked']
    guards.require(len(checked)==1 and checked[0]==dict(kind='checked',queries=queries,cells=39259,
        guards=True,inputs_immutable=True,all_outputs_exact=True),'guard/input/packed/output coverage differs')
    samples=[v for v in rows if v.get('kind')=='sample']
    order=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]
    expected=[] if mode=='smoke' else [(r,a) for r in range(8) for a in order[(r+process)%4]]
    guards.require([(v['round'],v['arm']) for v in samples]==expected,'sample extent/order differs')
    for v in samples:
        guards.require(v['queries']==6491 and v['process']==process and v['repeats']==8 and
            type(v['batch_ns']) is int and type(v['timer_floor_ns']) is int and v['timer_floor_ns']>0 and
            v['batch_ns']>=max(100000000,1000*v['timer_floor_ns']),'ineligible timing batch')
        if v['arm']:
            guards.require(v['event_calls']==8 and type(v['attention_device_ns']) is int and
                0<v['attention_device_ns']<=v['batch_ns']+1000000,'direct-event coverage/envelope differs')
        else:guards.require(v['attention_device_ns'] is None and v['event_calls']==0,'original event attribution invented')
    guards.require(sum(v.get('kind')=='PASS' for v in rows)==1,'duplicate footer')
    return dict(numerical=numerical,checked=checked,samples=samples)


def performance_decision(observations,comparison):
    guards.require([v['process'] for v in observations]==[0,1,2],'three distinct independent processes required')
    groups=[]
    for observation in observations:
        samples=observation['output']['samples'];process=observation['process']
        guards.require(len(samples)==32,'complete32sampleprocess required')
        wall={a:statistics.median(v['batch_ns'] for v in samples if v['arm']==a) for a in range(4)}
        dev={a:statistics.median(v['attention_device_ns'] for v in samples if v['arm']==a) for a in range(1,4)}
        groups.append(dict(process=process,control_original_wall_ratio=wall[1]/wall[0],
            hot_actual_service_ratio=dev[2]/dev[1],spread_actual_service_ratio=dev[3]/dev[1],
            spread_hot_service_ratio=dev[3]/dev[2],
            hot_actual_wall_ratio=wall[2]/wall[1],spread_actual_wall_ratio=wall[3]/wall[1]))
    lo,hi=comparison['control_original_ratio_bounds'];low,high=comparison['sensitivity_ratio_bounds']
    calibration=all(lo<=v['control_original_wall_ratio']<=hi for v in groups)
    response={}
    for key in ('hot_actual_service_ratio','spread_actual_service_ratio','spread_hot_service_ratio'):
        values=[v[key] for v in groups]
        response[key]='lower' if calibration and all(v<=low for v in values) else \
            'higher' if calibration and all(v>=high for v in values) else 'inconclusive'
    return dict(calibration_passed=calibration,groups=groups,responses=response,
        address_sensitivity_observed=any(v!='inconclusive' for v in response.values()),
        advance_to_model=False,model_adopted=False,hardware_saturation_qualified=False,
        interpretation='Only logical-address topology sensitivity at this synthetic equivalent-value6491shape. Flat response does not rule out memory limits. No DRAM/cache/saturation/limiting-resource or model-speed conclusion.')


def host_controls():
    import copy
    rows=[dict(kind='numerical',arm=a,queries=3,compared=18432,all_bits_exact=True) for a in range(4)]
    rows += [dict(kind='checked',queries=3,cells=39259,guards=True,inputs_immutable=True,all_outputs_exact=True),dict(kind='PASS')]
    def render(r):return '\n'.join(json.dumps(v) for v in r)
    parsed_output(render(rows),'smoke',0);rejected=[]
    def reject(name,call):
        try:call()
        except (ValueError,RuntimeError,KeyError,AssertionError):rejected.append(name)
        else:raise RuntimeError('invalid control accepted:'+name)
    for key in ('all_bits_exact','compared'):
        bad=copy.deepcopy(rows);bad[0][key]=False if key=='all_bits_exact' else 1
        reject(key,lambda:parsed_output(render(bad),'smoke',0))
    bad=copy.deepcopy(rows);bad[-2]['inputs_immutable']=False
    reject('input_changed',lambda:parsed_output(render(bad),'smoke',0))
    reject('missing_footer',lambda:parsed_output(render(rows[:-1]),'smoke',0))
    order=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]
    observations=[]
    for process in range(3):
        samples=[dict(kind='sample',process=process,round=r,arm=a,queries=6491,repeats=8,
            batch_ns=1000000000 if a<2 else 900000000,attention_device_ns=None if not a else 800000000 if a==1 else 700000000,
            event_calls=8 if a else 0,timer_floor_ns=20) for r in range(8) for a in order[(r+process)%4]]
        numer=[dict(v,queries=6491,compared=6491*24*256) for v in rows[:4]]
        checked=dict(rows[-2],queries=6491)
        parsed=parsed_output(render(numer+[checked]+samples+[rows[-1]]),'perf',process)
        observations.append(dict(process=process,output=parsed))
    comparison=dict(control_original_ratio_bounds=[0.99,1.01],sensitivity_ratio_bounds=[0.97,1.03])
    decision=performance_decision(observations,comparison)
    guards.require(decision['calibration_passed'] and decision['address_sensitivity_observed'] and
        not decision['advance_to_model'],'diagnostic direction/adoption distinction')
    bad=copy.deepcopy(observations)
    for sample in bad[0]['output']['samples']:
        if sample['arm']==1:sample['batch_ns']=1030000000
    guards.require(not performance_decision(bad,comparison)['address_sensitivity_observed'],'failed calibration cannot support sensitivity')
    rejected.append('calibration_blocks_interpretation')
    reject('duplicate_process',lambda:performance_decision([observations[0]]*3,comparison))
    bad=copy.deepcopy(observations[0]['output']['samples']);bad[0]['batch_ns']=99999999
    reject('short_timer_batch',lambda:parsed_output(render(numer+[checked]+bad+[rows[-1]]),'perf',0))
    print(json.dumps(dict(passed=True,rejected=rejected,valid_processes=3,valid_samples=96,GPU_executed=False)),flush=True)



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
                        if line.startswith('QSA_LOCALITY_GATE '):
                            break
                    b.guards.require(line == 'QSA_LOCALITY_GATE '+stage, 'actual-ID gate differs')
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
            if mode == 'smoke':
                launches = local_launches((folder/'api.trace').read_text(),plan['kernel_ids'],[0,1])
                b.guards.require(launches['0']['calls']==2 and launches['1']['calls']==6, 'actual-ID smoke dispatch count differs')
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


if __name__=='__main__':
    args=sys.argv[1:]
    guards.require(args in (['prepare-admission'],['run-admission'],['host-controls'],
        ['prepare','smoke'],['run','smoke'],['prepare','perf'],['run','perf']),
        'prepare-admission|run-admission|host-controls|prepare|run smoke|perf')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        if len(args)==1:
            raise SystemExit(host_controls() if args[0]=='host-controls' else prepare() if args[0]=='prepare-admission' else run())
        raise SystemExit(prepare_run(args[1]) if args[0]=='prepare' else run_stage(args[1]))
