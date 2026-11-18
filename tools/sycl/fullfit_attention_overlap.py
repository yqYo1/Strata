#!/usr/bin/env python3
"""Owned source comparison for actual layer weights on the copy queue. Repository LICENSE.

Build/provenance checks follow this fork's daca8f47e fixture control.
The engine sources and kernel entry are identical across all arms.
"""
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

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import check_qsa16_stream as native_build
import current_pti_admission as pti
import current_pti_probe as pti_probe
import measurement_guards as guards
import measurement_owner as owner
import qsa_cached_subgroup as qsa
import request_profile_reader as reader
import ur_validation as validation

P, B, RESULT = basis.P, basis.B, basis.RESULT
ROOT = B / 'fullfit-attention-arena-source-v1'
KEY = 'fullfit_attention_arena_source_v1'
COMPILE_LOGS = ROOT / 'compile'
TARGET = 'sycl_fullfit_arena_source_bench'
ELF = ROOT / 'build' / TARGET
PYTHON = basis.reference.model.PYTHON
COMPILER = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
ZE_LOADER = '/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0'
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
ADAPTER = 'bench/sycl/fullfit_attention_overlap.hpp'
TEST = 'tools/test_fullfit_attention_overlap.py'
TREE_INPUTS = ('cmake','src','include','third_party','tests','bench/sycl','tools/sycl')
SOURCES = ('allocated','staged','borrowed')
CLEAN_BLOCKS = (SOURCES, ('staged','borrowed','allocated'), ('borrowed','allocated','staged'))
CLEAN_ORDER = tuple(kind for block in CLEAN_BLOCKS for kind in block)
CASES = {
    'metadata-borrowed': ('metadata','borrowed',0),
    **{f'{phase}-{kind}': (phase, kind, 0)
       for phase in ('smoke','trace') for kind in SOURCES},
    **{f'clean-{i}': ('clean', kind, i//len(SOURCES)) for i, kind in enumerate(CLEAN_ORDER)},
}
STAGES = ('host', 'configure', 'build', *CASES)
ARMS = ('compute_only', 'copy_only', 'serial_separate', 'deferred_wait')
FULL_ROUNDS = 5
COPY = re.compile(r'^zeCommandListAppendMemoryCopy\(([HDMS]2[HDMS])\)\[(\d+)\]$')
SOURCE_SETUP = re.compile(r'^fullfit_attention_overlap source=(allocated|staged|borrowed) '
                          r'bytes=(\d+) prepare_ms=(\S+) allocation_id=(\d+) type=PASS$')
SOURCE_RETIRE = re.compile(r'^fullfit_attention_overlap retire_ms=(\S+) source_unchanged=PASS$')
ARENA_SETUP = re.compile(r'^fullfit_attention_overlap arena=production layers=48 bytes=50292326400 '
                         r'load_ms=(\S+) layout=PASS$')
QUEUE_SETUP = re.compile(r'^fullfit_attention_overlap copy_api=native '
                         r'ordinal=(\d+) index=(\d+) mode=(\d+) flags=(\d+) priority=(\d+) domain=PASS$')
OVERLAP_FLOOR_NS = 1_000_000
BUILD_LIMIT = dict(wall=1800, cpu=5400, rss_cap=12 << 30, text_cap=24 << 20,
                   file_cap=512 << 20, total_cap=4 << 30)
RUN_LIMIT = dict(wall=420, cpu=900, rss_cap=56 << 30, text_cap=16 << 20,
                 file_cap=64 << 20, total_cap=96 << 20)
SOURCE = qsa.source
require, sha, signature = basis.require, basis.sha, basis.signature
write = basis.reference.core.write


def pins(revision):
    sources = basis.comparison.build.controls(revision, controller=Path(__file__).resolve())
    names = {'CMakeLists.txt',ADAPTER,BENCH,TEST,*basis.POLICIES}
    names.update(SOURCE.git(P, 'ls-tree', '-r', '--name-only', revision, *TREE_INPUTS).decode().splitlines())
    for name, raw in SOURCE.blobs(P, revision, sorted(names)):
        require((P/name).read_bytes() == raw, 'commit exact component input: '+name)
        sources[name] = hashlib.sha256(raw).hexdigest()
    require(names <= sources.keys(), 'incomplete native and fixture source pins')
    return sources


def weight_asset(assets, layout_proof):
    layout = Path(layout_proof['path'])
    require(sha(layout) == assets[str(layout)]['sha256'] == layout_proof['sha256'],
        'actual model layout changed')
    line, = [line for line in layout.read_text().splitlines() if line.split()[:1] == ['2']]
    values = [int(v) for v in line.split()]
    require(len(values)==8 and values[:5]==[2,22,20,1887436800,1971200]
        and layout_proof['experts_per_layer']==512 and layout_proof['page_bytes']==4096,
        'actual layer-2 format, source or extent differs')
    shard, = [Path(p) for p in assets if p.endswith('-00001-of-00002.gguf')]
    st = shard.stat()
    identity = assets[str(shard)]['identity']
    require(all(getattr(st, 'st_'+a) == identity[b] for a,b in
        (('dev','dev'),('ino','ino'),('size','bytes'),('mtime_ns','mtime_ns'),('ctime_ns','ctime_ns'))),
        'actual GGUF shard identity changed')
    roles = [dict(name=name,offset=offset,bytes=size*512) for name,offset,size in
             zip(('gate','up','down'),values[5:],(524800,524800,921600))]
    require(all(0 < r['offset'] <= st.st_size-r['bytes'] for r in roles), 'real weights exceed shard extent')
    return dict(path=str(shard),identity=identity,sha256=assets[str(shard)]['sha256'],roles=roles,
                layout=dict(path=str(layout),sha256=layout_proof['sha256']),layer=2,expert_count=512)


def hash_roles(asset):
    hashes = []
    with Path(asset['path']).open('rb') as shard:
        for role in asset['roles']:
            shard.seek(role['offset'])
            remaining = role['bytes']
            digest = hashlib.sha256()
            while remaining:
                data = shard.read(min(remaining,8 << 20))
                require(data, 'short actual GGUF role read')
                digest.update(data)
                remaining -= len(data)
            hashes.append(digest.hexdigest())
    return hashes


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
    ordinal = validation.committed(RESULT, ('fullfit_attention_copy_ordinal_v1',))
    require(ordinal['stage'] == 'closed' and ordinal['ownership_closed']
        and not ordinal['active'] and ordinal['decision']['eligible_native_ordinal_overlap']
        and ordinal['artifact_retirement']['complete']
        and not (B/'fullfit-attention-copy-ordinal-v1').exists(),
        'require successful fixed-API ordinal test and retire the old generation')
    model_weights = weight_asset(speed['model_assets'],ordinal['next_source_layout_check'])
    prior_source = validation.committed(RESULT, ('fullfit_attention_layer_source_v1',))
    require(prior_source['stage']=='closed' and prior_source['ownership_closed'] and not prior_source['active']
        and prior_source['decision']['eligible_borrowed_layer_source'] and prior_source['role_sha256_after_matches']
        and prior_source['artifact_retirement']['complete'] and not (B/'fullfit-attention-layer-source-v1').exists(),
        'require a retired actual-layer comparison with a qualified borrowed lifecycle')
    shards = {p:spec for p,spec in speed['model_assets'].items()
              if p.endswith(('-00001-of-00002.gguf','-00002-of-00002.gguf'))}
    require(len(shards)==2 and model_weights['path'] in shards, 'exact two production model shards required')
    copy_engine = validation.committed(pti.RESULT, ('prefill_copy_engine_v6',))
    ze_check = copy_engine['validation_provider']
    require(copy_engine['stage'] == 'closed' and copy_engine['ownership_closed']
        and not copy_engine['active'] and copy_engine['boot'] == speed['boot']
        and copy_engine['driver'] == speed['driver']
        and sha(ze_check['path']) == ze_check['sha256'],
        'qualified Level Zero validation provider required')
    runtime = json.loads(Path(speed['runtime_manifest']['path']).read_text())
    require(runtime.get(ZE_LOADER) == sha(ZE_LOADER), 'pinned Level Zero linker input changed')
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
        resource_sha256=signature(resource), prior_evidence_sha256=signature(prior_source),
        model_weights=model_weights, model_shards=shards,
        ze_loader=dict(path=ZE_LOADER,sha256=runtime[ZE_LOADER]), ze_validation=ze_check)


def build_context():
    ctx = native_build.context()
    ctx.ROOT = ROOT
    return ctx


def compile_commands(native):
    old = str(Path(native['binaries']['strata']['path']).parent.parent)
    build = str(ROOT/'build')
    configure = [arg.replace(old,str(ROOT)) for arg in native['recipe']['configure']]
    require(configure[configure.index('-S')+1] == str(P) and configure[configure.index('-B')+1] == build,
        'exact original source and private build directory required')
    return dict(configure=configure,planned=['/usr/bin/ninja','-C',build,'-t','commands',TARGET],
        compile=['/usr/bin/cmake','--build',build,'--target',TARGET,'-j','4'],
        dependencies=['/usr/bin/ninja','-C',build,'-t','deps'])


def compile_environment(native):
    old = str(Path(native['binaries']['strata']['path']).parent.parent)
    return {k:v.replace(old,str(ROOT)) for k,v in native['recipe']['environment'].items()}


def case_environment(pin, mode):
    phase, kind, _ = CASES[mode]
    path = ROOT/mode
    template = pin['environment_template']
    old = template['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(path)) for k,v in template.items()}
    for key in ('STRATA_TRACE', 'UR_LOG_TRACING', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(STRATA_TEST_MAIN_GATES='1', STRATA_UR_PROBE_EXPECTED_LOADER=pin['preload']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(path/'validation.log'),
        STRATA_FULLFIT_OVERLAP_OBSERVE_CLOCK='1' if phase == 'trace' else '0',
        STRATA_FULLFIT_OVERLAP_SOURCE=kind, STRATA_FULLFIT_WEIGHT_SHARD=pin['model_weights']['path'],
        STRATA_FULLFIT_WEIGHT_PACK=str(Path(pin['model_weights']['layout']['path']).parent),
        STRATA_ARENA_LOCK='0', STRATA_NO_LARGEPAGES='1')
    shim = pin['preload']['library']['path']
    if phase == 'trace':
        env.update(STRATA_UNITRACE_TARGET_PRELOAD=shim, UNITRACE_SystemTime='1',
            UNITRACE_FollowChildProcess='1', UNITRACE_TraceOutputDir=str(path/'trace'))
    else:
        env['LD_PRELOAD'] = shim
    if phase in ('metadata','smoke'):
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',
            ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
            ZE_ENABLE_THREADING_VALIDATION='1', ZE_ENABLE_HANDLE_LIFETIME='0',
            ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOG_CONSOLE='1',
            ZEL_LOADER_LOGGING_LEVEL='warn', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0')
    required = dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
        EnableDirectSubmission='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1', ZES_ENABLE_SYSMAN='0')
    require(all(env[k] == v for k,v in required.items()) and
        not any('xestrata' in v.lower() for v in env.values()), 'qualified safety or provenance changed')
    return env


def case_command(pin, mode):
    phase, _, process = CASES[mode]
    short_modes = {'metadata':'--source-admission','smoke':'--smoke'}
    target = [str(ELF), short_modes[phase]] if phase in short_modes else [str(ELF), '--component', str(process)]
    if phase == 'trace':
        return [pin['tools']['artifacts']['unitrace']['path'], '--device-timing',
            '--chrome-device-logging', '--chrome-event-buffer-size', '128', *target]
    return target


def audit_build(row, dependency_output, link_output, native):
    inputs = dict(project=dict(root=str(P),revision=row['source_commit'],source_sha256=row['sources']),
        dependencies=native['dependencies'], external_inputs=native['external_inputs'],
        toolchain={**native['toolchain'],ZE_LOADER:row['platform']['ze_loader']['sha256']},
        **{k:row[k] for k in ('generated_inputs','planned_compilation','planned_translation_units')})
    census = build_context().census(inputs,dependency_output,link_output,minimum_translation_units=6)
    manifest = json.loads(Path(census['path']).read_text())
    require(manifest['links'].get(ZE_LOADER)==row['platform']['ze_loader']['sha256'],
        'native copy loader missing from actual link')
    objects = [ROOT/'build'/tu['output'] for tu in row['planned_compilation']]
    archives = [Path(p) for p in manifest['owned_link_products'] if p.endswith('.a')]
    require(archives and all(path.is_relative_to(ROOT/'build') for path in archives),
        'fresh owned static libraries required')
    row.update(build_census=census,
        actual_inputs=dict(compiled=manifest['compiled'],generated=manifest['generated'],links=manifest['links'],
            products={str(path):sha(path) for path in [*objects,*archives]}),
        binary=dict(path=str(ELF),bytes=ELF.stat().st_size,sha256=sha(ELF)))


def audit_configuration(row, planned_output):
    planned = build_context().planned_compilation(planned_output.read_text())
    required = {str(P/name) for name in (BENCH, 'src/core/expert_source.cpp',
        'src/core/pinned.cu', 'src/kernels/cpu/expert_layout.cpp', 'src/kernels/cpu/native_expert.cpp',
        'src/sycl/runtime.cpp')}
    units = sorted(tu['file'] for tu in planned)
    require(required <= set(units), 'the production ArenaExpertSource and format libraries must be built')
    build = ROOT/'build'
    paths = [build/name for name in ('build.ninja','CMakeCache.txt','compile_commands.json')]
    paths += sorted((build/'CMakeFiles').glob('*.ninja'))+[planned_output]
    row.update(planned_compilation=planned,planned_translation_units=units,
        configured_files={str(path):sha(path) for path in paths},
        generated_inputs={str(path):sha(path) for path in build.rglob('*')
            if path.is_file() and path.suffix in ('.h','.hpp','.c','.cpp','.cxx','.inc')})


def samples(text, mode):
    phase, kind, process = CASES[mode]
    smoke = phase == 'smoke'
    traced = phase == 'trace'
    rounds = 1 if smoke else FULL_ROUNDS
    rows, clocks, cases, setup, retired, queues = [], [], [], [], [], []
    arena, staging_setup, admission_layers = [], [], []
    for line in text.splitlines():
        if line.startswith('fullfit_attention_overlap process='):
            pairs = [v.split('=', 1) for v in line.split()[1:]]
            data = dict(pairs)
            require(len(data) == len(pairs), 'duplicate sample field')
            rows.append(data)
        elif line.startswith('fullfit_attention_overlap clock_pair_max_s='):
            clocks.append(float(line.rsplit('=',1)[1]))
        elif line.startswith('fullfit_attention_overlap source='):
            match = SOURCE_SETUP.fullmatch(line)
            require(match is not None, 'invalid source setup and allocation type proof')
            source_kind, size, prepare, alloc_id = match.groups()
            setup.append(dict(kind=source_kind,bytes=int(size),
                              prepare_ms=float(prepare),allocation_id=int(alloc_id)))
        elif line.startswith('fullfit_attention_overlap retire_ms='):
            match = SOURCE_RETIRE.fullmatch(line)
            require(match is not None, 'missing retired-source integrity proof')
            retired.append(float(match[1]))
        elif line.startswith('fullfit_attention_overlap arena='):
            match = ARENA_SETUP.fullmatch(line)
            require(match is not None, 'missing full ArenaExpertSource load proof')
            arena.append(float(match[1]))
        elif line.startswith('fullfit_attention_overlap staging_startup_ms='):
            staging_setup.append(float(line.rsplit('=',1)[1]))
        elif line.startswith('fullfit_attention_overlap admission_layer='):
            admission_layers.append(int(line.rsplit('=',1)[1]))
        elif line.startswith('fullfit_attention_overlap copy_api='):
            match = QUEUE_SETUP.fullmatch(line)
            require(match is not None, 'invalid queue setup proof')
            ordinal, index, queue_mode, flags, priority = match.groups()
            queues.append(dict(api='native',ordinal=int(ordinal),index=int(index),mode=int(queue_mode),
                               flags=int(flags),priority=int(priority)))
        elif line.startswith('case cells='):
            cases.append(line)
    require(len(arena)==1 and math.isfinite(arena[0]) and arena[0]>0,
        'one verified full ArenaExpertSource load required')
    expected = 48 if phase=='metadata' else 1
    require(len(setup)==len(retired)==expected and all(
        src['kind']==kind and bool(src['allocation_id'])==(kind=='borrowed') and
        math.isfinite(src['prepare_ms']) and src['prepare_ms']>0 and
        math.isfinite(close_ms) and (close_ms==0 if kind=='staged' else close_ms>0)
        for src,close_ms in zip(setup,retired)), 'wrong source kind, timing or allocation proof')
    if phase == 'metadata':
        require(not (rows or clocks or cases or queues or staging_setup)
            and admission_layers == list(range(48))
            and sum(src['bytes'] for src in setup)==50292326400
            and all(src['bytes']>0 and src['bytes']%4096==0 for src in setup),
            'source admission must cover the entire actual arena without application work')
        return dict(arena_load_ms=arena[0],admitted_layers=len(setup),
            min_layer_bytes=min(src['bytes'] for src in setup),max_layer_bytes=max(src['bytes'] for src in setup),
            import_ms_range=[min(src['prepare_ms'] for src in setup),max(src['prepare_ms'] for src in setup)],
            retire_ms_range=[min(retired),max(retired)],edge_fingerprints_unchanged=True,samples=[])
    require(not admission_layers and setup[0]['bytes']==(7884800 if smoke else 1009254400)
        and len(staging_setup)==1 and math.isfinite(staging_setup[0])
        and (staging_setup[0]>0 if kind=='staged' else staging_setup[0]==0),
        'wrong source extent or reusable staging startup')
    source = dict(setup[0],retire_ms=retired[0],arena_load_ms=arena[0],staging_startup_ms=staging_setup[0])
    shape = 'cells=97 queries=3 cap=33' if smoke else 'cells=65536 queries=8192 cap=2051'
    require(cases == [f'case {shape} pattern=gather samples={len(ARMS)*rounds} full_bits=PASS guards=PASS inputs=PASS status=PASS']
        and len(rows) == rounds * len(ARMS) and len(clocks) == 1
        and math.isfinite(clocks[0]) and clocks[0] >= 0,
        'fixture shape, sample count or clock-overhead screen changed')
    require(not smoke or 'fullfit_attention_overlap cancel_after_copy=PASS' in text.splitlines(),
        'missing accepted-copy cancellation check')
    require(len(queues) == 1
        and (queues[0]['ordinal'],queues[0]['index'],queues[0]['mode']) == (1,0,2)
        and queues[0]['flags'] & 2 and not queues[0]['flags'] & 4,
        'wrong native queue configuration')
    result, previous = [], 0
    for i, row in enumerate(rows):
        round_n, order = divmod(i, len(ARMS))
        arm = ARMS[(order+round_n+2*process) % len(ARMS)]
        copies = (4 if smoke else 512) if arm != 'compute_only' else 0
        requested = copies * 1971200
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
    return dict(clock_pair_max_s=clocks[0], source=source, queue=queues[0], samples=result)


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
                require(direction == 'H2D' and int(size) == 1971200, 'unexpected timed copy kind or extent')
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
        require(bool(copy_ns) == bool(copies) and bool(kernel_ns) == bool(kernels),
            'device timestamp resolution did not establish submitted work')
        overlap_ns = copy_ns+kernel_ns-reader.union_ns(copies+kernels)
        out.append(dict(arm=point['arm'], round=point['round'], classifications=classifications,
            copy_union_ns=copy_ns, qsa_union_ns=kernel_ns, overlap_ns=overlap_ns, other_commands=other))
    return out


def assessment(outcomes):
    clean = [row for row in outcomes if row['mode'].startswith('clean-')]
    traces = [row for row in outcomes if row['mode'].startswith('trace-')]
    require([p['mode'] for p in clean] == [f'clean-{i}' for i in range(len(CLEAN_ORDER))]
        and [p['mode'] for p in traces] == [f'trace-{kind}' for kind in SOURCES]
        and len({p['pid'] for p in [*clean,*traces]}) == len(clean)+len(traces)
        and all(row['timing']['source']['kind']==CASES[row['mode']][1]
                and row['timing']['queue']['ordinal']==1 for row in [*clean,*traces]),
        'matched source, fixed copy queue, order and independent processes required')
    def medians(row):
        return {arm: statistics.median(p['milliseconds'] for p in row['timing']['samples']
                    if p['arm'] == arm and p['round'] > 0)
                for arm in ARMS}
    warmed = [medians(row) for row in clean]
    first = [{p['arm']:p['milliseconds'] for p in row['timing']['samples'] if p['round'] == 0}
             for row in clean]
    blocks = [{kind:index for index,kind in enumerate(block,start=i*len(SOURCES))}
              for i,block in enumerate(CLEAN_BLOCKS)]
    def device_medians(trace):
        return {arm: {key: statistics.median(p[key] for p in trace['timeline']
                        if p['arm'] == arm and p['round'] > 0)
                    for key in ('copy_union_ns','qsa_union_ns','overlap_ns')} for arm in ARMS}
    trace_cost, overlap, device = {}, {}, {}
    for trace in traces:
        kind = CASES[trace['mode']][1]
        control = warmed[CLEAN_ORDER.index(kind)]
        observed = medians(trace)
        trace_cost[kind] = {arm: observed[arm]/control[arm] for arm in ARMS}
        device[kind] = device_medians(trace)
        windows = [p for p in trace['timeline'] if p['arm']=='deferred_wait' and p['round']>0]
        require([p['round'] for p in windows] == list(range(1,FULL_ROUNDS)),
            'actual warmed source trace order differs')
        overlap[kind] = [p['overlap_ns'] for p in windows]
    sustained_overlap = {kind:all(v>=OVERLAP_FLOOR_NS for v in spans) for kind,spans in overlap.items()}
    lifecycle = []
    for median, row in zip(warmed,clean):
        source = row['timing']['source']
        lifecycle.append(source['prepare_ms']+median['deferred_wait']+source['retire_ms'])
    comparisons = {}
    for candidate, reference in (('staged','allocated'), ('borrowed','staged'), ('borrowed','allocated')):
        def ratios(values):
            paired = [values[block[candidate]]/values[block[reference]] for block in blocks]
            return dict(paired=paired,geometric=statistics.geometric_mean(paired),
                        all_three_faster=all(v<1 for v in paired))
        layer = ratios(lifecycle)
        comparisons[f'{candidate}_vs_{reference}'] = dict(
            layer_lifecycle=layer, eligible=layer['all_three_faster'] and sustained_overlap[candidate],
            warmed_by_arm={arm:ratios([sample[arm] for sample in warmed]) for arm in ARMS},
            first_cycle_by_arm={arm:ratios([sample[arm] for sample in first]) for arm in ARMS})
    return dict(clean_progression=list(CLEAN_ORDER), matched_clean_indices=blocks,
        clean_process_warmed_medians_ms=warmed, clean_first_cycle_ms=first,
        traced_vs_first_matching_clean=trace_cost,
        observed_device_interval_medians_ns=device, warmed_deferred_overlap_ns=overlap,
        overlap_floor_ns=OVERLAP_FLOOR_NS,sustained_overlap=sustained_overlap,
        clean_layer_lifecycle_ms=lifecycle, source_comparisons=comparisons,
        eligible_borrowed_layer_source=all(comparisons['borrowed_vs_'+ref]['eligible'] for ref in ('staged','allocated')),
        next_comparison_prerequisite='Measure actual expert demand before evaluating a full-layer model route.',
        model_qualified=False, physical_bandwidth_or_saturation_qualified=False)


def compact(row):
    public = {k: v for k,v in row.items()
              if k not in ('sources','actual_inputs','planned_compilation','planned_translation_units')}
    public['source_inputs'] = dict(files=len(row['sources']), sha256=signature(row['sources']))
    if 'actual_inputs' in row:
        public['actual_inputs_summary'] = {k:dict(files=len(v),sha256=signature(v))
            for k,v in row['actual_inputs'].items()}
    if 'planned_compilation' in row:
        public['planned_compilation_summary'] = dict(units=len(row['planned_compilation']),
            sha256=signature(row['planned_compilation']))
    return public


def save(row):
    write(ROOT/'record.json', row)
    public = json.loads(RESULT.read_text())
    public[KEY] = compact(row)
    write(RESULT, public)


def frozen_cases(pin):
    return {mode: dict(command=case_command(pin,mode), environment_sha256=signature(case_environment(pin,mode)))
        for mode in CASES}


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh plan only')
    revision = SOURCE.git(P, 'rev-parse', 'HEAD').decode().strip()
    inputs = pins(revision)
    native, pin = platform()
    require(all(inputs[p] == native['project']['source_sha256'][p] for p in qsa.NATIVE),
        'maintain the native14508 sources for all nonfixture translation units')
    role_hashes = hash_roles(pin['model_weights'])
    ROOT.mkdir(mode=0o700)
    for name in ('build','tmp'): (ROOT/name).mkdir(mode=0o700)
    save(dict(stage='prepared', source_commit=revision, sources=inputs, platform=pin,
        compile_environment=compile_environment(native), commands=compile_commands(native),
        cases=frozen_cases(pin),role_sha256_before=role_hashes, results=[],
        active=False, passed=False, complete=False, ownership_closed=True, whole_bytes=4<<30,
        plan=dict(order=STAGES,
            unit='Use the same private production ArenaExpertSource and native layout in all arms. Compare staged vs allocated to isolate moving host USM allocation and release out of the layer cycle. Compare borrowed vs staged to isolate replacing the reused host USM copy with read-only registration of the original layer pages. Compare borrowed vs allocated for the net source choice. Fix native copy ordinal 1, QSA, waits, transfer extent and runtime for each comparison. The full arena load is a common production prerequisite with separate timing and admission.',
            safety='Admit and retire every full layer sequentially with zero application enqueues; check layer endpoints against the static owner and fingerprints after unimport. Then three real four-expert bounded cases check exact bytes, output and cancellation. Advance to full traces and clean comparisons only after closed quiet checks. Hold RAM and GPU owners through checked completion. Limit native RSS to 56GiB and require 64GiB host availability before admitting a full-arena process.',
            inputs='Freeze the production layout, both GGUF shard identities and known hashes. Rehash the three exact layer-2 role slices before and after. Native parser and packer supply the offsets without fixture-specific copies. Build the analyzer as one disposable fresh CMake target with all production libraries compiled from pinned source.',
            decision='Nine independent clean processes in ABC/BCA/CAB order, with the same application schedule within each block. Require 20 validated samples per process at >=10ms and <=0.1% timer overhead. Record first cycle, four warmed cycles, common arena loading and one-time warm staging preparation separately. Layer cost is source preparation plus warmed deferred median plus per-layer release. A comparison is eligible if its candidate costs less in all three blocks and sustains >=1ms device overlap in each warmed deferred window. Borrowing must pass against both USM references.',
            limits='Use the actual ArenaExpertSource and full model expert bytes with fixture-owned QSA state and GPU bank. Model dispatch, demand selection, full-engine speed, decode, quality and physical context require their own tests.',
            retention='Keep individual timings, allocation and fill costs, device-window aggregates and source/build/fault proof. Retire the fixture generation and normal output after closure and exact provenance recovery.')))
    print('Prepared actual layer source comparison; commit exact plan before running.')


def run_device(row, mode, scope, path, result):
    phase, _, _ = CASES[mode]
    provider_check = 'trace' if phase == 'trace' else mode
    env = case_environment(row['platform'], mode)
    plan = row['cases'][mode]
    require(plan['environment_sha256'] == signature(env), 'pre-exec environment changed')
    art = row['platform']['tools']['artifacts']
    mapping_row = dict(admission=dict(runtime_manifest=row['platform']['runtime_manifest'], tools=dict(artifacts=art)),
        preload=dict(row['platform']['preload'],level_zero_loader=row['platform']['ze_loader']), fixture=row['binary'])
    checked = phase in ('metadata','smoke')
    if checked:
        mapping_row['preload']['level_zero_validation'] = row['platform']['ze_validation']
    maps = []
    def observe(io, entry):
        for gate in ('before-runtime','runtime-ready','after-cleanup'):
            while (line := io.line(240)) != 'QSA_BF16_READY '+gate:
                require(line.startswith(('device=', 'case ', 'fullfit_attention_overlap ')),
                    'unexpected fixture gate output')
            require(str(Path('/proc',str(entry['pid']),'exe').resolve()) == row['binary']['path'],
                'owned native executable differs at '+gate)
            if gate == 'before-runtime':
                environ = Path('/proc',str(entry['pid']),'environ').read_bytes().split(b'\0')
                keys = ('STRATA_TEST_MAIN_GATES', 'STRATA_FULLFIT_OVERLAP_OBSERVE_CLOCK',
                    'STRATA_FULLFIT_OVERLAP_SOURCE','STRATA_FULLFIT_WEIGHT_SHARD',
                    'STRATA_FULLFIT_WEIGHT_PACK',
                    'STRATA_ARENA_LOCK','STRATA_NO_LARGEPAGES',
                    'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
                    'STRATA_SYCL_PREFILL_SAME_QUEUE', 'UR_ENABLE_LAYERS', 'UR_LOG_VALIDATION',
                    'ZES_ENABLE_SYSMAN', 'ONEAPI_DEVICE_SELECTOR')
                if checked:
                    keys += ('ZE_ENABLE_VALIDATION_LAYER', 'ZE_ENABLE_PARAMETER_VALIDATION',
                             'ZE_ENABLE_THREADING_VALIDATION', 'ZE_ENABLE_HANDLE_LIFETIME')
                require(all(f'{key}={env[key]}'.encode() in environ for key in keys),
                    'actual native environment differs')
            else:
                maps.append(dict(phase=gate,**pti_probe.mapped(entry['pid'],mapping_row,provider_check)))
            if gate == 'runtime-ready':
                result['GPU_work_admitted'] = phase != 'metadata'
            if gate == 'after-cleanup':
                result['GPU_executed'] = phase != 'metadata'
            io.send(b'X')
        require(io.line(45) == 'fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED',
            'fixture did not report normal completion after the cleanup gate')
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
        and (not checked or (('urEnqueue' in logs)==(phase=='smoke') and not re.search(
            r'\b(?:VALIDATION ERROR|ERROR_INVALID|THREADING VIOLATION)\b',logs,re.I))),
        'native validation, loader or fixture gate differs')
    timing = samples(text,mode)
    if phase != 'metadata' and mode != 'smoke-allocated':
        smoke = next(stage for stage in row['results'] if stage['mode'] == 'smoke-allocated')
        fields = ('api','ordinal','index','mode','flags','priority')
        require(all(timing['queue'][name] == smoke['timing']['queue'][name] for name in fields),
            'queue configuration differs from the bounded staged control')
    result.update(validation=probe, timing=timing)
    if phase == 'trace':
        traces = sorted((path/'trace').glob('*.json'))
        require(len(traces) == 1 and traces[0].stat().st_size < RUN_LIMIT['file_cap'], 'one bounded device trace required')
        result.update(timeline=timeline(reader.load_json(traces[0]),timing['samples'],(before,after)),
            trace=dict(sha256=sha(traces[0]), bytes=traces[0].stat().st_size))
    result['host_probe_accepted'] = True


def run_stage(mode):
    row = json.loads((ROOT/'record.json').read_text())
    require(validation.committed(RESULT,(KEY,)) == compact(row)
        and row['stage'] in ('prepared','running') and not row['active']
        and mode == STAGES[len(row['results'])]
        and all(p['passed'] and p['ownership_closed'] for p in row['results']),
        'commit preceding closed stage and launch only the next declared arm')
    native, pin = platform()
    require(pins(row['source_commit']) == row['sources'] and row['platform'] == pin
        and row['commands'] == compile_commands(native) and row['compile_environment'] == compile_environment(native)
        and row['cases'] == frozen_cases(pin)
        and owner.artifact_bytes(ROOT) < row['whole_bytes'], 'frozen source, recipe, providers or budget differ')
    if mode != 'configure':
        require(all(sha(path)==h for path,h in row.get('configured_files',{}).items()),
            'configured build recipe or plan changed')
    if mode in CASES:
        require(sha(row['binary']['path']) == row['binary']['sha256']
            and all(sha(p) == h for group in row['actual_inputs'].values() for p,h in group.items()),
            'fixture executable or actual native inputs changed')
    path = COMPILE_LOGS if mode == 'build' else ROOT/mode
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
        elif mode in ('configure','build'):
            outputs = {}
            for action in (('configure','planned') if mode=='configure' else ('compile','dependencies')):
                record,out,err = scope.run(action,row['commands'][action],row['compile_environment'],P,**BUILD_LIMIT)
                require(owner.closed(record) and record['exit_code'] == 0,
                    action+' failed: '+err.read_text()[-2400:]+out.read_text()[-1200:])
                outputs[action] = out
            if mode == 'configure':
                audit_configuration(row,outputs['planned'])
            else:
                audit_build(row,outputs['dependencies'],outputs['compile'],native)
        else:
            available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))
            require(available >= 64<<20, 'actual full arena requires 64 GiB available before launch')
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
                require(hash_roles(row['platform']['model_weights']) == row['role_sha256_before'],
                    'actual GGUF role slices changed during comparison')
                row['role_sha256_after_matches'] = True
                decision = assessment(row['results'])
            except (RuntimeError, ValueError, KeyError, OSError) as error:
                result.update(passed=False,error=repr(error))
                row['stage'] = 'failed'
            else:
                row.update(stage='closed',passed=True,complete=True,decision=decision)
        save(row)
    require(result['passed'], result.get('error','component stage failed'))
    print(mode,'completed with closed ownership.',flush=True)


if __name__ == '__main__':
    require(sys.executable == PYTHON and len(sys.argv) == 2 and sys.argv[1] in ('prepare',*STAGES),
        'use the pinned Python and a declared stage: prepare or '+', '.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            run_stage(sys.argv[1])
