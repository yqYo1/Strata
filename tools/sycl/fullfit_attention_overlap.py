#!/usr/bin/env python3
"""Owned QSA/copy overlap experiment. Repository LICENSE.

Build/provenance checks follow this fork's daca8f47e fixture control.
The engine sources and kernel entry are identical across all arms.
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
import coherent_model_basis as basis
import current_pti_admission as pti
import current_pti_probe as pti_probe
import measurement_guards as guards
import measurement_owner as owner
import qsa_cached_subgroup as qsa
import request_profile_reader as reader
import ur_validation as validation

P, B, RESULT = basis.P, basis.B, basis.RESULT
ROOT = B / 'fullfit-attention-overlap-v1'
KEY = 'fullfit_attention_overlap_v1'
ELF = ROOT / 'build/sycl_fullfit_attention_overlap'
PYTHON = basis.reference.model.PYTHON
COMPILER = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
ADAPTER = 'bench/sycl/fullfit_attention_overlap.hpp'
TEST = 'tools/test_fullfit_attention_overlap.py'
STAGES = ('host', 'build', 'smoke', 'trace', '0', '1', '2')
ARMS = ('compute_only', 'copy_only', 'serial_shared', 'serial_separate', 'deferred_wait')
COPY = re.compile(r'^zeCommandListAppendMemoryCopy\(([HDMS]2[HDMS])\)\[(\d+)\]$')
BUILD_LIMIT = dict(wall=300, cpu=360, rss_cap=10 << 30, text_cap=4 << 20,
                   file_cap=96 << 20, total_cap=384 << 20)
RUN_LIMIT = dict(wall=300, cpu=450, rss_cap=8 << 30, text_cap=16 << 20,
                 file_cap=64 << 20, total_cap=96 << 20)
SOURCE = qsa.source
require, sha, signature = basis.require, basis.sha, basis.signature
write = basis.reference.core.write


def pins(revision):
    sources = basis.comparison.build.controls(revision, controller=Path(__file__).resolve())
    names = {ADAPTER, BENCH, TEST, *qsa.NATIVE, *basis.POLICIES}
    names.update(SOURCE.git(P, 'ls-tree', '-r', '--name-only', revision, 'include').decode().splitlines())
    for name, raw in SOURCE.blobs(P, revision, sorted(names)):
        require((P/name).read_bytes() == raw, 'commit exact component input: '+name)
        sources[name] = hashlib.sha256(raw).hexdigest()
    require(names <= sources.keys(), 'incomplete native and fixture source pins')
    return sources


def platform():
    native, proof, speed = basis.basis()
    cycle = validation.committed(RESULT, ('qsa_normalizer_128k_v1',))
    require(cycle['stage'] in ('closed', 'failed') and cycle['ownership_closed'] and not cycle['active'],
        'close the preceding model comparison before building or launching')
    retired = validation.committed(RESULT, ('qsa_v_tile_v2', 'artifact_retirement'))
    require(retired['complete'] and not (B/'qsa-v-tile-v2').exists(), 'prior disposable generation remains')
    build_review = validation.committed(RESULT, ('retained_builds_review_20261103',
        'live_consumer_findings', 'current_boundary_recheck', 'window_candidate_retirement'))
    decode = build_review['retained_decode']
    require(sha(decode['path']) == decode['sha256']
        and build_review['complete'] and all(not Path(p).exists() for p in build_review['roots']),
        'independent decode or retention identity changed')
    resource = validation.committed(RESULT, ('eager_request_profile_v2',))
    require(resource['stage'] == 'closed' and resource['decision']['fresh_pair_within_observer_budget'],
        'closed 64K resource evidence required')
    tools = pti.verify_tools(pti.committed_json(pti.RESULT, P))
    probe = validation.committed(pti.RESULT, ('unitrace_current_runtime_probe_v2',))
    require(probe['stage'] == 'closed' and probe['ownership_closed']
        and probe['passed'] and len(probe['results']) == 2
        and probe['preload'] == speed['native_validation_probe'] and probe['boot'] == speed['boot'],
        'surviving collector requires its closed provider and clock probe')
    return native, dict(original_binary=proof['same_build_executables']['strata'],
        original_source=proof['original_native_source'], decode_binary=decode,
        external=native['external_inputs'], compiler=COMPILER,
        compiler_sha256=native['toolchain'][COMPILER], boot=speed['boot'], driver=speed['driver'],
        runtime_manifest=speed['runtime_manifest'], preload=speed['native_validation_probe'],
        environment_template=speed['environment_template'], tools=tools,
        resource_sha256=signature(resource))


def compile_commands():
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-fsycl', '-fsycl-id-queries-range=size_t',
        '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-DSTRATA_ENABLE_SYCL=1',
        '-DSTRATA_CPU_SERVICE_PROFILE=1', '-fno-fast-math', '-ffp-contract=off',
        '-I'+str(P/'include'), '-I'+str(P/'include/strata/sycl/compat'), '-I'+str(P/'bench/sycl')]
    commands, objects = [], []
    for i, name in enumerate([*qsa.NATIVE, BENCH]):
        obj = ROOT/'build'/f'{i}.o'
        extra = ['-DSTRATA_QSA_BENCH_FULLFIT_OVERLAP=1'] if name == BENCH else []
        commands.append([COMPILER, *flags, *extra, '-MD', '-MF', str(obj.with_suffix('.d')),
                         '-c', str(P/name), '-o', str(obj)])
        objects.append(str(obj))
    commands.append([COMPILER, *flags, *objects, '-Wl,-t', '-pthread', '-ldl', '-o', str(ELF)])
    return commands


def case_environment(pin, mode):
    path = ROOT/mode
    template = pin['environment_template']
    old = template['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(path)) for k,v in template.items()}
    for key in ('STRATA_TRACE', 'UR_LOG_TRACING', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(STRATA_TEST_MAIN_GATES='1', STRATA_UR_PROBE_EXPECTED_LOADER=pin['preload']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(path/'validation.log'),
        STRATA_FULLFIT_OVERLAP_OBSERVE_CLOCK='1' if mode == 'trace' else '0')
    shim = pin['preload']['library']['path']
    if mode == 'trace':
        env.update(STRATA_UNITRACE_TARGET_PRELOAD=shim, UNITRACE_SystemTime='1',
            UNITRACE_FollowChildProcess='1', UNITRACE_TraceOutputDir=str(path/'trace'))
    else:
        env['LD_PRELOAD'] = shim
    if mode == 'smoke':
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr')
    required = dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
        EnableDirectSubmission='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1', ZES_ENABLE_SYSMAN='0')
    require(all(env[k] == v for k,v in required.items()) and
        not any('xestrata' in v.lower() for v in env.values()), 'qualified safety or provenance changed')
    return env


def case_command(pin, mode):
    target = [str(ELF), '--smoke'] if mode == 'smoke' else [str(ELF), '--component', mode if mode.isdecimal() else '0']
    if mode == 'trace':
        return [pin['tools']['artifacts']['unitrace']['path'], '--device-timing',
            '--chrome-device-logging', '--chrome-event-buffer-size', '128', *target]
    return target


def audit_build(row, link_log, native):
    allowed = {str((P/p).resolve()): h for p,h in row['sources'].items()}
    ext = json.loads(Path(row['platform']['external']['path']).read_text())
    allowed.update(ext)
    compiled = {}
    deps = sorted((ROOT/'build').glob('*.d'))
    require(len(deps) == len(qsa.NATIVE)+1, 'one dependency file per fresh translation unit')
    for dep in deps:
        for name in shlex.split(dep.read_text().replace('\\\n', ' ').split(':',1)[1]):
            path = Path(name).resolve()
            digest = sha(path)
            require(allowed.get(str(path)) == digest, 'unfrozen compiler input: '+str(path))
            if path.is_relative_to(P/'include'):
                require(native['project']['source_sha256'].get(str(path.relative_to(P))) == digest,
                    'actual production header differs from original: '+str(path))
            compiled[str(path)] = digest
    require(len(compiled) > 600, 'incomplete compiler dependency closure')
    links = {}
    for line in link_log.splitlines():
        name = line.strip()
        if not name.startswith('/') or any(c.isspace() for c in name):
            continue
        path = Path(name).resolve()
        if path.is_relative_to(ROOT/'tmp') and not path.exists():
            require(path.suffix == '.o', 'unrecognized ephemeral link input')
            continue
        digest = sha(path)
        require((path.is_relative_to(ROOT/'build') and path.suffix == '.o') or ext.get(str(path)) == digest,
            'unfrozen link input: '+str(path))
        links[str(path)] = digest
    require(len(links) >= 12, 'actual link input capture incomplete')
    row.update(actual_inputs=dict(compiled=compiled, links=links,
        objects={str(path): sha(path) for path in (ROOT/'build').glob('*.o')}),
        binary=dict(path=str(ELF), bytes=ELF.stat().st_size, sha256=sha(ELF)))


def samples(text, mode):
    smoke = mode == 'smoke'
    traced = mode == 'trace'
    process = int(mode) if mode.isdecimal() else 0
    rounds = 1 if smoke else len(ARMS)
    rows, clocks, cases = [], [], []
    for line in text.splitlines():
        if line.startswith('fullfit_attention_overlap process='):
            pairs = [v.split('=', 1) for v in line.split()[1:]]
            data = dict(pairs)
            require(len(data) == len(pairs), 'duplicate sample field')
            rows.append(data)
        elif line.startswith('fullfit_attention_overlap clock_pair_max_s='):
            clocks.append(float(line.rsplit('=',1)[1]))
        elif line.startswith('case cells='):
            cases.append(line)
    shape = 'cells=97 queries=3 cap=33' if smoke else 'cells=65536 queries=8192 cap=2051'
    require(cases == [f'case {shape} pattern=gather samples={len(ARMS)*rounds} full_bits=PASS guards=PASS inputs=PASS status=PASS']
        and len(rows) == rounds * len(ARMS) and len(clocks) == 1
        and math.isfinite(clocks[0]) and clocks[0] >= 0,
        'fixture shape, sample count or clock-overhead screen changed')
    require(not smoke or 'fullfit_attention_overlap cancel_after_copy=PASS' in text.splitlines(),
        'missing accepted-copy cancellation check')
    result, previous = [], 0
    for i, row in enumerate(rows):
        round_n, order = divmod(i, len(ARMS))
        arm = ARMS[(order+round_n+2*process) % len(ARMS)]
        copies = (4 if smoke else 512) if arm != 'compute_only' else 0
        requested = copies * (256 if smoke else 1971200)
        attention = int(arm != 'copy_only')
        elapsed = float(row['milliseconds'])
        lo, hi = int(row['raw_start']), int(row['raw_end'])
        require((int(row['process']),int(row['round']),int(row['order']),row['arm'],
            int(row['copies']),int(row['requested_bytes']),int(row['attention_calls']))
            == (process,round_n,order,arm,copies,requested,attention)
            and row['payload'] == row['guards'] == row['output'] == 'PASS'
            and math.isfinite(elapsed) and elapsed > 0
            and (smoke or elapsed >= 10 and clocks[0]*1000 <= elapsed/1000)
            and ((previous <= lo < hi and lo > 0) if traced else lo == hi == 0),
            'timing bounds, arm, payload or exact output changed')
        previous = hi
        result.append(dict(round=round_n, arm=arm, milliseconds=elapsed, copies=copies,
            requested_bytes=requested, attention_calls=attention, raw_start=lo, raw_end=hi))
    return dict(clock_pair_max_s=clocks[0], samples=result)


def timeline(trace, points, lifetime):
    events = []
    for event in trace['traceEvents']:
        if event.get('cat') != 'gpu_op':
            continue
        lo = reader.nanoseconds(event['ts'])
        hi = lo + reader.nanoseconds(event['dur'])
        name = event['name']
        require(event.get('ph') == 'X' and not name.startswith('Tile #')
            and lifetime[0] <= lo <= hi <= lifetime[1], 'invalid device event or native RAW bounds')
        events.append((lo,hi,name))
    require(0 < len(events) <= 100000, 'bounded nonempty GPU event list')
    out = []
    for point in points:
        start, end = point['raw_start'], point['raw_end']
        require(lifetime[0] <= start < end <= lifetime[1], 'fixture window outside native owner')
        copies, kernels, classifications = [], [], {}
        nbytes, other = 0, 0
        for lo, hi, name in events:
            if hi <= start or lo >= end:
                continue
            require(start <= lo <= hi <= end, 'device event crosses timed boundary')
            match = COPY.fullmatch(name)
            if match:
                direction, size = match.groups()
                require(direction in ('M2D','H2D') and int(size) == 1971200, 'unexpected timed copy kind or extent')
                classifications[direction] = classifications.get(direction,0) + 1
                nbytes += int(size)
                copies.append((lo,hi))
            elif name.startswith('zeCommandList'):
                other += 1
            else:
                require('prefix_batch_impl' in name, 'unexpected timed kernel: '+name)
                kernels.append((lo,hi))
        require(len(copies) == point['copies'] and nbytes == point['requested_bytes']
            and len(kernels) == point['attention_calls']*2, 'copy or QSA launches do not conserve')
        copy_ns, kernel_ns = reader.union_ns(copies), reader.union_ns(kernels)
        overlap_ns = copy_ns+kernel_ns-reader.union_ns(copies+kernels)
        out.append(dict(arm=point['arm'], round=point['round'], classifications=classifications,
            copy_union_ns=copy_ns, qsa_union_ns=kernel_ns, overlap_ns=overlap_ns, other_commands=other))
    return out


def assessment(outcomes):
    trace = next(row for row in outcomes if row['mode'] == 'trace')
    eligible = [row for row in outcomes if row['mode'].isdecimal()]
    require([p['mode'] for p in eligible] == ['0','1','2'] and len({p['pid'] for p in eligible}) == 3,
        'three independent clean processes required')
    def medians(row):
        return {arm: statistics.median(p['milliseconds'] for p in row['timing']['samples'] if p['arm'] == arm)
                for arm in ARMS}
    clean = [medians(row) for row in eligible]
    trace_medians = medians(trace)
    ratios = {}
    for name, arm, reference in (
        ('queue_choice', 'serial_separate', 'serial_shared'),
        ('wait_placement', 'deferred_wait', 'serial_separate'),
        ('net', 'deferred_wait', 'serial_shared'),
    ):
        values = [p[arm]/p[reference] for p in clean]
        ratios[name] = dict(paired=values, geometric=statistics.geometric_mean(values),
                            all_three_faster=all(v < 1 for v in values))
    overlap = {arm: statistics.median(p['overlap_ns'] for p in trace['timeline'] if p['arm'] == arm)
               for arm in ARMS}
    trace_cost = {arm: trace_medians[arm]/clean[0][arm] for arm in ARMS}
    return dict(ratios=ratios, clean_process_medians_ms=clean, traced_vs_first_clean=trace_cost,
        observed_device_overlap_median_ns=overlap,
        estimated_serial_equivalent_ms=[p['copy_only']+p['compute_only'] for p in clean],
        eligible_for_separate_reader_ownership_test=ratios['net']['all_three_faster']
            and ratios['wait_placement']['all_three_faster'] and overlap['deferred_wait'] > overlap['serial_separate'],
        model_qualified=False, physical_bandwidth_or_saturation_qualified=False)


def compact(row):
    public = {k: v for k,v in row.items() if k not in ('sources','actual_inputs')}
    public['source_inputs'] = dict(files=len(row['sources']), sha256=signature(row['sources']))
    if 'actual_inputs' in row:
        public['actual_inputs_summary'] = {k:dict(files=len(v),sha256=signature(v))
            for k,v in row['actual_inputs'].items()}
    return public


def save(row):
    write(ROOT/'record.json', row)
    public = json.loads(RESULT.read_text())
    public[KEY] = compact(row)
    write(RESULT, public)


def frozen_cases(pin):
    return {mode: dict(command=case_command(pin,mode), environment_sha256=signature(case_environment(pin,mode)))
        for mode in STAGES[2:]}


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh plan only')
    revision = SOURCE.git(P, 'rev-parse', 'HEAD').decode().strip()
    inputs = pins(revision)
    native, pin = platform()
    require(all(inputs[p] == native['project']['source_sha256'][p] for p in qsa.NATIVE),
        'maintain the native14508 sources for all nonfixture translation units')
    env = {k:v for k,v in native['recipe']['environment'].items()
           if k in ('PATH','LANG','LC_ALL','LD_LIBRARY_PATH')}
    env['TMPDIR'] = str(ROOT/'tmp')
    ROOT.mkdir(mode=0o700)
    for name in ('build','tmp'): (ROOT/name).mkdir(mode=0o700)
    save(dict(stage='prepared', source_commit=revision, sources=inputs, platform=pin,
        compile_environment=env, commands=compile_commands(), cases=frozen_cases(pin), results=[],
        active=False, passed=False, complete=False, ownership_closed=True, whole_bytes=768<<20,
        plan=dict(order=STAGES, compiled_translation_units=6,
            unit='Same original ordered-exp 65536-cell/8192-query/cap2051 QSA, same 512 ordinary RAM copies at layer-2 stride. Two serial arms isolate queue choice; two same-second-queue arms isolate wait position. Compare net against serial shared queue. Compute/copy alone are controls.',
            safety='Compile the disposable third generation from scratch. Check cold startup, owner/provider identities, small traced output and cancellation, then bounded PTI device-only timeline. Clean processes keep identical pre-exec safety settings and remove PTI/RAW-clock observer.',
            decision='Five rotated arm orders in each of three independent clean processes. Require all 25 clean samples per process with identical bytes, exact output and guards, at least 10 ms and clock-pair overhead below 0.1%. Retain each contrast and every sample. Same-source overlap eligibility also needs traced positive overlap over serial on the second queue.',
            limits='Single synthetic weight slab and one real kernel entry, not model speed or physical PCIe measurement. Chosen data do not represent all layers. No engine/decode/quality/full-context qualification.',
            retention='Keep the timing and device-window aggregates, source/recipe/fault proof and decisions. Retire instrumented logs and the one fixture generation after closure and exact provenance recovery.')))
    print('Prepared isolated copy/attention component; commit exact plan before running.')


def run_device(row, mode, scope, path, result):
    env = case_environment(row['platform'], mode)
    plan = row['cases'][mode]
    require(plan['environment_sha256'] == signature(env), 'pre-exec environment changed')
    art = row['platform']['tools']['artifacts']
    mapping_row = dict(admission=dict(runtime_manifest=row['platform']['runtime_manifest'], tools=dict(artifacts=art)),
        preload=row['platform']['preload'], fixture=row['binary'])
    maps = []
    def observe(io, entry):
        for phase in ('before-runtime','runtime-ready','after-cleanup'):
            while (line := io.line(240)) != 'QSA_BF16_READY '+phase:
                require(line.startswith(('device=', 'case ', 'fullfit_attention_overlap ')),
                    'unexpected fixture gate output')
            require(str(Path('/proc',str(entry['pid']),'exe').resolve()) == row['binary']['path'],
                'owned native executable differs at '+phase)
            if phase == 'before-runtime':
                environ = Path('/proc',str(entry['pid']),'environ').read_bytes().split(b'\0')
                keys = ('STRATA_TEST_MAIN_GATES', 'STRATA_FULLFIT_OVERLAP_OBSERVE_CLOCK',
                    'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
                    'STRATA_SYCL_PREFILL_SAME_QUEUE', 'UR_ENABLE_LAYERS', 'UR_LOG_VALIDATION',
                    'ZES_ENABLE_SYSMAN', 'ONEAPI_DEVICE_SELECTOR')
                require(all(f'{key}={env[key]}'.encode() in environ for key in keys),
                    'actual native environment differs')
            else:
                maps.append(dict(phase=phase,**pti_probe.mapped(entry['pid'],mapping_row,mode)))
            if phase == 'runtime-ready':
                result['GPU_work_admitted'] = True
            if phase == 'after-cleanup':
                result['GPU_executed'] = True
            io.send(b'X')
        io.exit(45)
    before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    result.update(GPU_work_admitted=False, GPU_executed=None)
    entry, out, err = scope.run(mode, plan['command'], env, P, interaction=observe, **RUN_LIMIT)
    after = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    result.update(pid=entry['pid'], native_exit_code=entry['exit_code'], providers=maps)
    require(owner.closed(entry) and entry['exit_code'] == 0, 'fixture failed: '+err.read_text()[-2400:])
    text, logs = out.read_text(), err.read_text()
    probe = validation.preload_evidence(logs,entry['pid'])
    qsa.diagnostics(logs,entry['pid'])
    require(probe['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER']
        and len(maps) == 2 and 'fixture_status=PASS' in text
        and (not (path/'validation.log').exists() or not (path/'validation.log').stat().st_size)
        and (mode != 'smoke' or 'urEnqueue' in logs), 'native validation, loader or fixture gate differs')
    timing = samples(text,mode)
    result.update(validation=probe, timing=timing)
    if mode == 'trace':
        traces = sorted((path/'trace').glob('*.json'))
        require(len(traces) == 1 and traces[0].stat().st_size < RUN_LIMIT['file_cap'], 'one bounded device trace required')
        result.update(timeline=timeline(reader.load_json(traces[0]),timing['samples'],(before,after)),
            trace=dict(sha256=sha(traces[0]), bytes=traces[0].stat().st_size))


def run_stage(mode):
    row = json.loads((ROOT/'record.json').read_text())
    require(validation.committed(RESULT,(KEY,)) == compact(row)
        and row['stage'] in ('prepared','running') and not row['active']
        and mode == STAGES[len(row['results'])]
        and all(p['passed'] and p['ownership_closed'] for p in row['results']),
        'commit preceding closed stage and launch only the next declared arm')
    native, pin = platform()
    require(pins(row['source_commit']) == row['sources'] and row['platform'] == pin
        and row['commands'] == compile_commands() and row['cases'] == frozen_cases(pin)
        and owner.artifact_bytes(ROOT) < row['whole_bytes'], 'frozen source, recipe, providers or budget differ')
    if mode in STAGES[2:]:
        require(sha(row['binary']['path']) == row['binary']['sha256']
            and all(sha(p) == h for group in row['actual_inputs'].values() for p,h in group.items()),
            'fixture executable or actual native inputs changed')
    path = ROOT/mode
    path.mkdir(mode=0o700)
    for name in ('tmp','cache','trace'): (path/name).mkdir(mode=0o700)
    scope = owner.Owner(path/'owned',lambda: write(path/'execution.json',scope.commands))
    result = dict(mode=mode, passed=False, GPU_executed=False)
    journal = None
    row.update(active=True, ownership_closed=False)
    save(row)
    try:
        if mode == 'host':
            env = dict(PATH='/usr/bin:/bin', LANG='C', TMPDIR=str(path/'tmp'))
            entry, _, err = scope.run('controls',[PYTHON,'-B','-m','unittest','tools.test_fullfit_attention_overlap'],
                env,P,wall=45,cpu=30,rss_cap=512<<20,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
            require(owner.closed(entry) and entry['exit_code'] == 0, 'host checks failed: '+err.read_text()[-2400:])
            result['host_checks'] = True
        elif mode == 'build':
            for i, command in enumerate(row['commands']):
                print('Compile/link',i+1,'of',len(row['commands']),flush=True)
                record, out, err = scope.run(str(i),command,row['compile_environment'],P,**BUILD_LIMIT)
                require(owner.closed(record) and record['exit_code'] == 0,
                    'build command '+str(i)+' failed: '+err.read_text()[-2400:])
            audit_build(row,out.read_text(),native)
        else:
            log = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
                text=True,timeout=15)
            journal = next(s.split(': ',1)[1] for s in log.splitlines() if s.startswith('-- cursor: '))
            run_device(row,mode,scope,path,result)
        result['passed'] = True
    except BaseException as error:
        result.update(error=repr(error),passed=False)
    finally:
        if journal:
            validation.finish(result,scope,journal,row['platform']['boot'])
            result['passed'] = bool(result['passed'] and result['driver_after'] == row['platform']['driver'])
        else:
            result['ownership_closed'] = scope.active is None and all(owner.closed(c) for c in scope.commands)
        result['passed'] = bool(result['passed'] and result['ownership_closed'])
        row['results'].append(result)
        row.update(stage='running' if result['passed'] else 'failed', active=bool(scope.active),
            ownership_closed=result['ownership_closed'])
        if mode == STAGES[-1] and result['passed']:
            try:
                decision = assessment(row['results'])
            except (RuntimeError, ValueError, KeyError) as error:
                result.update(passed=False,error=repr(error))
                row['stage'] = 'failed'
            else:
                row.update(stage='closed',passed=True,complete=True,decision=decision)
        save(row)
    require(result['passed'], result.get('error','component stage failed'))
    print(mode,'completed with closed ownership.',flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON and len(sys.argv) == 2 and sys.argv[1] in ('prepare',*STAGES),
        'use the pinned Python and prepare|host|build|smoke|trace|0|1|2')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            run_stage(sys.argv[1])
