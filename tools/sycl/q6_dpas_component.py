#!/usr/bin/env python3
"""Owned compressed-Q6 C4 DPAS component; Main-authored, repository LICENSE.

No inference dispatch is changed. Compare the existing public column entry
against the new packed entry through ordinary production graphs. Keep the
failed model observer and closed row-copy decisions; neither proves a current
physical bottleneck. All stages require committed frozen source/recipes.
"""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import select
import shlex
import statistics
import subprocess
import sys
import tarfile
sys.dont_write_bytecode = True
import benchmark_runtime as python_runtime
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation
import build_compat_profile as selected
P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'q6-dpas-component-v3'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'q6_dpas_component_v3'
TARGET = 'sycl_q6_dpas_component'
HELPERS = ('tools/sycl/q6_dpas_component.py','tools/sycl/test_q6_dpas_component.py',
    'tools/sycl/benchmark_runtime.py','tools/sycl/measurement_guards.py',
    'tools/sycl/measurement_owner.py','tools/sycl/ur_validation.py',
    'AGENTS.md','docs/ARTIFACT_RETENTION.md')
UNITS = ('src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp',
    'src/kernels/sycl/native_mmvq.cpp','src/kernels/sycl/native_q5_head.cpp',
    'src/kernels/sycl/native_q3_esimd.cpp','src/kernels/sycl/native_q4_esimd.cpp',
    'src/kernels/sycl/native_q2_esimd.cpp','src/kernels/sycl/native_iq4xs_esimd.cpp',
    'src/kernels/sycl/native_iq_esimd.cpp','src/kernels/sycl/native_mmq_xmx.cpp',
    'src/kernels/sycl/iq.cpp','src/kernels/sycl/q6_dpas_component.cpp','bench/sycl/q6_dpas.cpp')
PACKAGES = ('libc6','libc6-dev','linux-libc-dev','libstdc++6',
            'libstdc++-13-dev','libgcc-s1','libgcc-13-dev','binutils')
FIELDS = ('wall_ns','cpu_ns','submit_wall_ns','submit_cpu_ns')
require, sha = guards.require, owner.sha


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)

def signature(pins):
    return dict(files=len(pins), aggregate_sha256=hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode()).hexdigest())

def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'runtime_pins', 'dependency_pins', 'commands')}
    out['source_inputs'] = signature(row['sources'])
    out['runtime_inputs'] = signature(row['runtime_pins'])
    out['dependency_inputs'] = signature(row['dependency_pins'])
    return out

def save(row):
    (ROOT / 'record.json').write_text(json.dumps(row, indent=2) + '\n')
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def packages():
    return subprocess.check_output(['/usr/bin/dpkg-query', '-W',
        '-f=${binary:Package} ${Version}\n', *PACKAGES], text=True, timeout=15).splitlines()

def dependency_pins(roots):
    pending, visited, pins, total, entries = list(map(Path, roots)), set(), {}, 0, 0
    while pending:
        root = pending.pop().resolve()
        if root in visited: continue
        require(root.is_dir(), 'dependency include/config directory missing')
        visited.add(root)
        for path in root.rglob('*'):
            entries += 1
            require(entries <= 32768, 'dependency traversal budget')
            if path.is_symlink() and path.is_dir():
                pending.append(path.resolve())
            elif path.is_file():
                name = str(path.resolve())
                if name in pins: continue
                total += path.stat().st_size
                require(len(pins) < 16384 and total <= 256 << 20, 'finite dependency inventory')
                pins[name] = sha(path)
    return pins

def prepare():
    data = validation.committed(RESULT)
    for key in ('q6_dpas_component_v1', 'q6_dpas_component_v2'):
        failed = data[key]
        require(failed['stage'] == 'failed' and not failed['passed'] and not failed['complete']
            and failed['ownership_closed'] and not failed['active'] and not failed['GPU_executed']
            and failed['artifact_retirement']['state'] == 'deleted_verified',
            'preserve and retire each pre-native compiler failure first')
    previous = data['mapped_row_copy_graph_v1']
    require(previous['stage']=='closed' and previous['passed'] and previous['complete'] and
        previous['ownership_closed'] and not previous['active'] and
        not previous['decision']['host_source_component_gain'] and
        previous['artifact_retirement']['state']=='deleted_verified',
        'close and retire graph-copy comparison first')
    cohort = data['dflash2_device_service_v2']
    require(cohort['stage'] == 'closed' and cohort['complete'] and cohort['ownership_closed']
        and not cohort['active'] and len(cohort['cases']) == 3 and
        all(c['passed'] and c['complete'] and c['ownership_closed'] and
            c['native_exit_code'] == c['benchmark_exit_code'] == 0 and not c['fresh_faults']
            for c in cohort['cases']), 'close the current frozen DFlash2 cohort before component preparation')
    require(not ROOT.exists() and KEY not in data, 'fresh component attempt only')
    revision = git('rev-parse', 'HEAD').decode().strip()
    helpers = {n: sha(P / n) for n in HELPERS}
    require(all((P / n).read_bytes() == git('show', revision + ':' + n) for n in HELPERS),
        'commit exact component controls first')
    native, identity = selected.basis()
    require(native['binaries']['strata'] == cohort['binary'], 'same selected90f production input required')
    existing=[n for n in UNITS if n not in ('src/kernels/sycl/q6_dpas_component.cpp','bench/sycl/q6_dpas.cpp')]
    control_inputs={n:sha(P/n) for n in existing}
    require(len(existing)==12 and all(native['project']['source_sha256'].get(n)==h for n,h in control_inputs.items()),
        'all12production control units must match selected591c native source')
    recipe = cohort['recipe']
    original = json.loads(git('show', recipe['commit'] + ':' + recipe['path']))
    template = original['current_sync']['dflash2_cache_loan_long_v1']['plans'][0]['environment']
    env = dict(native['recipe']['environment'])
    env = {k: v for k, v in env.items() if not k.startswith('STRATA_IQ') and k != 'STRATA_CPU_SERVICE'}
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        ZE_ENABLE_ALT_DRIVERS=template['ZE_ENABLE_ALT_DRIVERS'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr')
    require(not env.get('LD_PRELOAD') and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
        env['EnableDirectSubmission'] == '0' and env['NEOReadDebugKeys'] == '1' and
        env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and
        env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and env['ZES_ENABLE_SYSMAN'] == '0',
        'preserve selected immutable runtime safety')
    provider = cohort['runtime_manifest']
    require(sha(provider['path']) == provider['sha256'], 'qualified provider inventory differs')
    runtime = json.loads(Path(provider['path']).read_text())
    require(len(runtime) == provider['files'] and all(sha(n) == h for n, h in runtime.items()),
        'qualified runtime files differ')
    for asset in cohort['native_validation_probe'].values():
        require(runtime.get(str(Path(asset['path']).resolve())) == asset['sha256'],
            'qualified validation preload/loader differs')
    tools = {name: dict(path=path, realpath=str(Path(path).resolve()), sha256=sha(path)) for name, path in
        dict(compiler='/opt/intel/oneapi/compiler/2026.2/bin/icpx', cmake='/usr/bin/cmake',
             ninja='/usr/bin/ninja', assembler='/usr/bin/as', linker='/usr/bin/ld', python=sys.executable).items()}
    versions = {n: subprocess.check_output([tools[n]['path'], '--version'], env=env,
        text=True, timeout=15).splitlines()[0] for n in ('compiler', 'cmake', 'ninja')}
    dependencies = ['/opt/intel/oneapi/compiler/2026.2/include',
        '/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl', '/usr/include/c++/13',
        '/usr/include/x86_64-linux-gnu/c++/13', '/usr/lib/gcc/x86_64-linux-gnu/13/include',
        '/usr/include/level_zero']
    dependencies += sorted(str(p) for p in Path('/opt/intel/oneapi/compiler/2026.2/lib/clang').glob('*/include'))
    require(len(dependencies) > 6, 'compiler resource headers required')
    dependency_inputs = dependency_pins(dependencies)
    external = native['external_inputs']
    require(sha(external['path']) == external['sha256'], 'original external-input manifest changed')
    external_inputs = json.loads(Path(external['path']).read_text())
    require(all(sha(p) == h for p, h in external_inputs.items()), 'original external compiler/link inputs changed')
    names = git('ls-files', '-z', '.gitattributes', 'CMakeLists.txt', 'LICENSE', 'cmake', 'src', 'include',
        'tests', 'bench', 'tools', 'third_party').decode().split('\0')
    names = sorted(n for n in names if n and not n.startswith('bench/results/')
                   and n != 'bench/performance-records.csv')
    require(0 < len(names) <= 4096, 'finite source snapshot')
    archive = git('archive', '--format=tar', revision, *names)
    require(len(archive) <= 64 << 20, 'source snapshot byte budget')
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isdir(): continue
            require(item.isfile() and item.name in names, 'only named regular source files')
            sources[item.name] = hashlib.sha256(source.extractfile(item).read()).hexdigest()
    require(set(sources) == set(names), 'complete archive source identity')
    configure = [tools['cmake']['path'], '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'),
        '-G', 'Ninja', '-DCMAKE_CXX_COMPILER=' + tools['compiler']['path'],
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DMKL_DIR=/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl',
        '-DSTRATA_ENABLE_SYCL=ON', '-DSTRATA_ENABLE_CUDA=OFF', '-DSTRATA_ENABLE_HIP=OFF',
        '-DSTRATA_NATIVE_EXPERTS=OFF', '-DSTRATA_BUILD_TESTS=OFF', '-DSTRATA_SYCL_TESTS=OFF',
        '-DSTRATA_BUILD_PREFILL_HOST_TESTS=OFF', '-DSTRATA_SYCL_VLLM_XE2_NATIVE16=OFF',
        '-DSTRATA_WERROR=ON',
        '-DCMAKE_CXX_FLAGS=-fsycl-id-queries-range=size_t -Wno-error=deprecated-declarations --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t']
    codegen = [tools['compiler']['path'],'-fsycl','-fsycl-device-only','-S','-emit-llvm',
        '-O3','-std=c++20','-ffp-contract=off','-fno-fast-math','-fsycl-id-queries-range=size_t',
        '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-I'+str(ROOT/'source/include'),'-I'+str(ROOT/'source/include/strata/sycl/compat'),
        '-I'+str(ROOT/'source/src/kernels/sycl'),str(ROOT/'source/src/kernels/sycl/q6_dpas_component.cpp'),
        '-o',str(ROOT/'q6-dpas.ll')]
    row = dict(stage='preparing',source_commit=revision,sources=sources,helpers=helpers,
        source_generation=dict(command='git archive --format=tar <source_commit> <named source paths>',
            attributes_path='.gitattributes',attributes_sha256=sources['.gitattributes']),
        provenance=dict(local_reference='src/kernels/sycl/native_mmq_xmx.cpp;existing public Q6 columns entry',
            external_reference='ggml-org/llama.cpp@34ca79122b44d43eda7efb203b82ec91db94c55c:ggml/src/ggml-sycl/esimd.hpp',
            license='MIT notice retained;new pack/kernel/controller/fixture authored here under repository LICENSE;no XeStrata source.',
            distinction='Own uint16 interleave and explicit C4 entry;not upstream in-place SOA reorder or engine dispatch.'),
        dependency_roots=dependencies,dependency_pins=dependency_inputs,tools=tools,tool_versions=versions,
        production_control_inputs=dict(source_commit=native['project']['revision'],**signature(control_inputs)),
        system_packages=packages(),runtime_pins=runtime,external_inputs=external,selected_native=identity,
        runtime_basis=dict(key='dflash2_device_service_v2',observer_screen_passed=False,
            scope='Normal current-model execution/provider/preload provenance only;failed observer screen supplies a hypothesis,not representative attribution or saturation.'),
        native_validation_probe=cohort['native_validation_probe'],environment=env,configure=configure,
        build=[tools['cmake']['path'],'--build',str(ROOT/'build'),'--target',TARGET,'--parallel','1'],
        codegen_command=codegen,build_budget=budget(600,2400,file_cap=128<<20),
        host_budget=dict(budget(90,90,file_cap=16<<20),rss_cap=1<<30),
        host_command=[tools['python']['path'],'-B','-m','unittest','-v','tools.sycl.test_q6_dpas_component'],
        native_budget=budget(600,1200,file_cap=16<<20),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),driver=guards.driver_guard(),
        plan=dict(order='10CPU reader controls,build+device IR,12CPU pack/shape controls,small traced3shape/2seed check,full2shape/2seed check,three timing processes0/1/2.Commit each closed stage;stop first failure,no retries.',
            shapes=[[2560,640],[2560,248320]],columns=4,arms=['public_columns','packed_dpas'],orders=[[0,1],[1,0],[0,1]],
            timing_samples_per_shape_arm=5,warmup_batches=2,batch_by_rows={'640':4096,'248320':256},
            minimum_batch_ns=10000000,minimum_timer_multiple=300,sample_fields_ns=FIELDS,
            numerical='Canonical Q6_K and Q8_1 quantization unchanged,per16 signed scale and FP16 super-scale/d preserved;s unused. FP32 accumulation association changes. All outputs finite and DP4A/DPAS difference bounded by2*gamma(8*(K/32)+64)*row L1 upper envelope+1e-12. Independent byte scalar double oracle on every small row and >=512distributed/head-tail rows per large shape;each arm within gamma*actual L1+1e-12. This checks this FP32-accumulating component,not a universal FP32 model precision or task-quality rule.',
            lifetime='Both entries capture one ordinary node and replay through graph_launch_with_completion. Explicit completion and both-queue drains precede CPU writes,readback,graph or allocation retirement. Immutable pre-exec environment;no dynamic fallback.',
            placement='Canonical and independently interleaved Device allocations both live. Record exact payload and successful maximum allocation,host pack and uploaded setup separately. Does not qualify10GiB model placement,cold total model latency or physical traffic.',
            acceptance='Exact input/ELF/provider/pre-main/ownership/boot pins;normal0,quiet validation,no fresh fault. CPU roundtrip,C1 rejection before runtime and unchanged C1 reference;all small/full numeric/input/canary checks. Codegen must retain>=2DPAS intrinsic calls;this is compiler-IR evidence,not native ISA or utilization. All60timed samples meet order/replay count/timer/conservation gates. Compare each shape independently:all3candidate/reference process-median whole-batch wall ratios<1,no percentage floor. Only a winning full head cell can motivate a separately qualified model proposal;no adoption.',
            limits='Parameter-validation/preload-instrumented component wall/threadCPU including graph submission/completion;not clean or sustained model speed. Source-selected cost hypothesis;physical bottleneck remains unknown.'),
        retention='Useful individual samples,setup/numeric outcomes,source/recipe and decision only.Retire successful raw/build output after this named component/model-proposal consumer ends;no additional inference engine.',
        results=[],timing=[],active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,model_executed=False,
        materialization=dict(complete=False,completed_source_files=0))
    ROOT.mkdir(mode=0o700)
    try:
        # Persist intent before extraction. A partial tree is a failed attempt,
        # never a fresh root to repair or retry by refreshing its input hashes.
        save(row)
        for name in ('source', 'tmp', 'cache'): (ROOT / name).mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for item in source.getmembers():
                if item.isdir(): continue
                path = ROOT / 'source' / item.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.extractfile(item).read()); path.chmod(item.mode & 0o777)
                row['materialization']['completed_source_files'] += 1
        row['materialization']['complete'] = True
        row['stage'] = 'prepared'
        save(row)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, preparation_error=repr(error))
        try:
            save(row)
        except BaseException as persist_error:
            raise RuntimeError('Preparation failed and its receipt could not be saved; '
                               'preserve the partial component root. Original error: ' +
                               repr(error)) from persist_error
        raise
    print('Prepared committed-source Q6 DPAS component;commit its exact plan before build.', flush=True)

def current(row):
    require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact component state')
    require(all((P / n).read_bytes() == git('show', row['source_commit'] + ':' + n)
                and sha(P / n) == h for n, h in row['helpers'].items()), 'component control drift')
    require(all(sha(ROOT / 'source' / n) == h for n, h in row['sources'].items()), 'native snapshot drift')
    require(all(sha(P / n) == h for n, h in row['sources'].items()), 'live component/helper source drift')
    _, identity = selected.basis()
    require(identity == row['selected_native'], 'selected production provenance drift')
    external = row['external_inputs']
    require(sha(external['path']) == external['sha256'] and
        all(sha(p) == h for p, h in json.loads(Path(external['path']).read_text()).items()),
        'external compiler/link input drift')
    require(all(sha(n) == h for n, h in row['runtime_pins'].items()), 'runtime provider drift')
    require(dependency_pins(row['dependency_roots']) == row['dependency_pins'], 'compiler/header/config dependency drift')
    require(all(str(Path(t['path']).resolve()) == t['realpath'] and sha(t['path']) == t['sha256']
                for t in row['tools'].values()) and packages() == row['system_packages'], 'toolchain drift')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot drift')
    if 'binary' in row:
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'component executable drift')
        actual = row['actual_inputs']
        require(sha(actual['path']) == actual['sha256'], 'actual component input census drift')
        census = json.loads(Path(actual['path']).read_text())
        require(all(sha(p) == h for k in ('compiled', 'generated', 'links') for p, h in census[k].items()),
            'actual component compiler/link bytes changed')
    if 'codegen' in row:
        require(sha(row['codegen']['path'])==row['codegen']['sha256'],'DPAS IR drift')


def census(row, dependencies, link_text):
    """Reject actual compiler/link files outside the precommitted inventories."""
    allowed = {str((ROOT / 'source' / p).resolve()): h for p, h in row['sources'].items()}
    allowed.update({str(Path(p).resolve()): h for p, h in row['dependency_pins'].items()})
    allowed.update({str(Path(p).resolve()): h for p, h in
        json.loads(Path(row['external_inputs']['path']).read_text()).items()})
    allowed.update({v['realpath']: v['sha256'] for v in row['tools'].values()})
    compiled, generated, links, owned = {}, {}, {}, set()
    for line in dependencies.splitlines():
        if not line.startswith('    ') or not line.strip(): continue
        p = Path(line.strip()); p = (p if p.is_absolute() else ROOT / 'build' / p).resolve()
        value = sha(p)
        if p.is_relative_to(ROOT / 'build'): generated[str(p)] = value
        else:
            require(allowed.get(str(p)) == value, 'unfrozen actual compiler input: ' + str(p))
            compiled[str(p)] = value
    expected = {str((ROOT / 'source' / p).resolve()) for p in UNITS}
    observed = {p for p in compiled if Path(p).suffix in ('.cpp', '.c', '.cc', '.cu')}
    require(observed == expected and len(compiled) > 100 and not generated,
        'exact component-unit compiler census with no uncovered/generated input required')
    for line in link_text.splitlines():
        name = line.strip(); p = Path(name)
        if not name or any(c.isspace() for c in name) or not (
            p.is_absolute() or p.suffix in ('.o', '.a') or '.so' in p.name): continue
        p = (p if p.is_absolute() else ROOT / 'build' / p).resolve()
        if any(p.is_relative_to(ROOT / n) for n in ('build', 'tmp')):
            owned.add(str(p)); continue
        require(p.is_file() and allowed.get(str(p)) == sha(p), 'unfrozen actual linker input: ' + str(p))
        links[str(p)] = sha(p)
    require(len(links) > 8, 'actual external link census incomplete')
    path = ROOT / 'actual-inputs.json'
    path.write_text(json.dumps(dict(compiled=compiled, generated=generated, links=links,
        owned_link_products=sorted(owned)), indent=2) + '\n')
    return dict(path=str(path), sha256=sha(path), compiler_files=len(compiled),
        generated_files=len(generated), external_link_files=len(links), uncovered=0,
        translation_units=UNITS, source_commit=row['source_commit'])

def budget(wall, cpu, *, file_cap):
    # Owner includes the private source/build tree,not just raw output.
    return dict(wall=wall, cpu=cpu, rss_cap=8 << 30, text_cap=16 << 20,
                file_cap=file_cap, total_cap=512 << 20)


def collect_timing(text, index):
    lines=text.splitlines()
    clocks=[v for v in lines if v.startswith('CLOCK ')]
    require(len(clocks)==1 and re.fullmatch(r'CLOCK median_stamp_ns=\d+ floor_ns=10000000 process='+str(index),clocks[0]),'timing calibration')
    timer=int(clocks[0].split()[1].split('=')[1])
    samples=[v for v in lines if v.startswith('SAMPLE ')]
    order=(1,0) if index==1 else (0,1)
    expected=[(n,arm,s) for n in (640,248320) for arm in order for s in range(5)]
    require(len(samples)==len(expected)==20 and lines.count('TIMING passed=1 samples=20 model_speed=0 physical_traffic=0')==1,'complete timing extent')
    cells={}
    for line,(n,arm,s) in zip(samples,expected):
        pairs=[v.split('=',1) for v in line.split()[1:]];f=dict(pairs)
        require(len(f)==len(pairs) and set(f)=={'process','arm','k','n','columns','sample','launches','graph_launches','nodes_per_graph','route',*FIELDS,'wait_wall_ns','wait_cpu_ns'},'timing fields')
        require(f['route']=='compat_graph','actual graph route differs')
        ints={k:int(v) for k,v in f.items() if k!='route'}
        batch=4096 if n==640 else 256
        require(all(re.fullmatch(r'\d+',f[k]) for k in ints) and all(ints[k]==v for k,v in
            dict(process=index,arm=arm,k=2560,n=n,columns=4,sample=s,launches=batch,graph_launches=batch,nodes_per_graph=1).items()),'actual timing shape/order/replay count')
        require(ints['wall_ns']>=max(10000000,300*timer),'timing floor/overhead')
        require(ints['submit_wall_ns']+ints['wait_wall_ns']==ints['wall_ns'] and ints['submit_cpu_ns']+ints['wait_cpu_ns']==ints['cpu_ns'],'timing conservation')
        cells.setdefault(str(n),{}).setdefault(str(arm),[]).append([ints[k] for k in FIELDS])
    numeric=[v for v in lines if v.startswith('NUMERIC ')]
    require(len(numeric)==4 and all(v.startswith('NUMERIC passed=1 ') and v.endswith(' guards=1') for v in numeric),'complete timing numerical checks')
    return dict(index=index,timer_stamp_median_ns=timer,cells=cells)


def decision(timing):
    require(len(timing)==3 and [v['index'] for v in timing]==[0,1,2],'three fresh timing processes')
    cells={}
    for n in ('640','248320'):
        ratios=[statistics.median(a[0] for a in v['cells'][n]['1'])/statistics.median(a[0] for a in v['cells'][n]['0']) for v in timing]
        cells[n]=dict(candidate_over_reference_median_wall=ratios,component_gain_qualified=all(r<1 for r in ratios))
    return dict(cells=cells,head_component_gain=cells['248320']['component_gain_qualified'],model_speed_qualified=False,
        inference_adopted=False,physical_traffic_qualified=False,next='Only a winning full-head cell permits a new decode-only model proposal;placement/setup,phase preservation,task/agent and physical262144 remain unqualified.')


def execute(mode):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    expected={'controls':'prepared','build':'cpu-controlled','host':'built','check-small':'host-checked','check':'small-checked'}
    if mode.startswith('time'):
        index=int(mode[-1]);require(row['stage'] in ('checked','timing') and len(row['timing'])==index,'fresh ordered timing only')
    else:require(row['stage']==expected[mode],'fresh ordered component stage only')
    scope=None;cursor=None;outcome=dict(mode=mode,passed=False,host_probe_accepted=False)
    def persist():
        (ROOT/'execution.json').write_text(json.dumps(dict(mode=mode,active=bool(scope and scope.active),commands=scope.commands if scope else []),indent=2)+'\n')
    try:
        scope=owner.Owner(ROOT/('owned-'+mode),persist)
        if mode=='controls':
            e,out,err=scope.run('reader-controls',row['host_command'],row['environment'],P,**row['host_budget'])
            text=out.read_text()+err.read_text()
            require(owner.closed(e) and e['exit_code']==0 and 'Ran 10 tests' in text and '\nOK\n' in text and 'skipped' not in text,'all10CPU reader controls')
            outcome.update(tests=10,skipped=0,native_execution_attempted=False);row['stage']='cpu-controlled'
        elif mode=='build':
            outcome['build_commands']=[];outputs={}
            for name,command in [('configure',row['configure']),('build',row['build']),('codegen',row['codegen_command'])]:
                e,out,err=scope.run(name,command,row['environment'],ROOT,**row['build_budget'])
                require(owner.closed(e) and e['exit_code']==0,
                    name+' failed:'+(out.read_text()+err.read_text())[-2500:])
                outputs[name]=out;outcome['build_commands'].append(dict(name=name,exit_code=0,ownership_closed=True))
            path=ROOT/'q6-dpas.ll';ir=path.read_text()
            calls=[v for v in ir.splitlines() if 'call ' in v and ('__esimd_dpas' in v or 'llvm.genx.dpas' in v)]
            require(len(calls)>=2,'device IR must retain the two DPAS half-dot intrinsics')
            row['codegen']=dict(path=str(path),sha256=sha(path),dpas_intrinsic_calls=len(calls),scope='Compiler device IR,not final native ISA or utilization.')
            e,out,_=scope.run('dependencies',[row['tools']['ninja']['path'],'-C',str(ROOT/'build'),'-t','deps'],row['environment'],ROOT,**row['build_budget'])
            require(owner.closed(e) and e['exit_code']==0,'actual dependency listing failed')
            row['actual_inputs']=census(row,out.read_text(),outputs['build'].read_text())
            path=ROOT/'build'/TARGET;row['binary']=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
            database=json.loads((ROOT/'build/compile_commands.json').read_text())
            commands=[c for c in database if TARGET+'.dir' in c['command'] or 'strata_sycl_runtime.dir' in c['command']]
            require(len(commands)==len(UNITS)==14 and {str(Path(c['file']).relative_to(ROOT/'source')) for c in commands}==set(UNITS)
                and all('-fsycl' in shlex.split(c['command']) and '-fsycl-id-queries-range=size_t' in shlex.split(c['command']) for c in commands),'exact14-unit build')
            require(all('-fno-fast-math' in shlex.split(c['command']) and '-ffp-contract=off' in shlex.split(c['command']) for c in commands if TARGET+'.dir' in c['command']),'numerical build flags')
            row['compiled_commands']=commands;row['stage']='built'
        elif mode=='host':
            e,out,err=scope.run('host-pack',[row['binary']['path'],'--host'],row['environment'],ROOT,**row['host_budget'])
            require(owner.closed(e) and e['exit_code']==0 and out.read_text().splitlines().count('HOST passed=1 cases=12 GPU=0')==1,'all12CPU pack/shape controls')
            outcome.update(cases=12,native_execution_attempted=True,GPU_executed=False);row['stage']='host-checked'
        else:
            env=dict(row['environment'],STRATA_Q6_DPAS_GATE='1',LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/(mode+'-validation.log')))
            if mode!='check-small':env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
            args=['--time',str(index)] if mode.startswith('time') else ['--'+mode]
            cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
            def interact(io,command):
                while True:
                    line=io.line(600)
                    if line.startswith('STRATA_Q6_DPAS_GATE '):break
                identity=owner.identity(command['pid'])
                require(identity and identity['pid']==identity['sid']==command['pid'] and all(identity[k]==command['direct_identity'][k] for k in ('pid','start_ticks','sid'))
                    and line=='STRATA_Q6_DPAS_GATE pid='+str(command['pid']),'actual native owner')
                require(str(Path('/proc',str(command['pid']),'exe').resolve())==row['binary']['path'],'actual executable')
                actual=dict(v.split('=',1) for v in Path('/proc',str(command['pid']),'environ').read_text().split('\0') if '=' in v)
                require(actual==env,'immutable native environment')
                mapped=set(python_runtime.code_maps(command['pid']));required={str(Path(env[k]).resolve()) for k in ('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER','UR_ADAPTERS_FORCE_LOAD','ZE_ENABLE_ALT_DRIVERS')}
                require(required<=mapped<=set(row['runtime_pins'])|{row['binary']['path']},'actual provider mappings')
                outcome.update(native_identity=identity,mapped_provider_count=len(mapped));row['GPU_executed']=True
                io.send(b'continue\n');io.proc.stdin.close()
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
                io.exit(1)
            outcome['native_execution_attempted']=True
            if not row['GPU_executed']:row['GPU_executed']=None
            e,out,err=scope.run('native',[row['binary']['path'],*args],env,ROOT,interaction=interact,**row['native_budget'])
            require(owner.closed(e) and e['exit_code']==0,'owned native component failed')
            outcome['native_exit_code']=0;text,diagnostic=out.read_text(),err.read_text()
            outcome['validation']=validation.preload_evidence(diagnostic,e['pid'])
            require(outcome['validation']['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'native validation provider')
            log=ROOT/(mode+'-validation.log');require(not log.exists() or log.stat().st_size==0,'parameter validation diagnostic')
            require(any(v.startswith('DEVICE name=') and 'B570' in v for v in text.splitlines()),'actual device')
            outcome['setup']=[v for v in text.splitlines() if v.startswith(('SETUP ','ADMISSION '))]
            outcome['numerical']=[v for v in text.splitlines() if v.startswith(('NUMERIC ','C1_CONTROL '))]
            if mode.startswith('time'):
                row['timing'].append(collect_timing(text,index));row['stage']='timing'
                if index==2:row.update(stage='closed',decision=decision(row['timing']),complete=True)
            else:
                small=mode=='check-small';cases=6 if small else 4
                require(text.splitlines().count('CHECK passed=1 small='+str(int(small))+' cases='+str(cases)+' route=compat_graph columns=4')==1
                    and sum(v.startswith('NUMERIC passed=1 ') for v in text.splitlines())==cases
                    and text.splitlines().count('C1_CONTROL passed=1 dpas_launched=0')==(3 if small else 2),'complete numeric/negative-control checks')
                if small:require('urEnqueueCommandBufferExp' in diagnostic and 'Q6DpasC4' in diagnostic,'actual small DPAS kernel graph trace')
                row['stage']='small-checked' if small else 'checked'
            outcome['host_probe_accepted']=True
        current(json.loads((ROOT/'record.json').read_text()));outcome['passed']=True
    except BaseException as error:
        outcome['error']=repr(error);row.update(stage='failed',passed=False,complete=False)
    finally:
        if cursor is not None and scope is not None:
            validation.finish(outcome,scope,cursor,row['boot'])
            if not outcome['passed'] or outcome.get('driver_after')!=row['driver'] or 'error' in outcome:
                outcome['passed']=False;row.update(stage='failed',passed=False,complete=False)
        row['results'].append(outcome);row['commands']=scope.commands if scope else []
        row['active']=bool(scope and scope.active);row['ownership_closed']=scope is None or (scope.active is None and all(owner.closed(e) for e in scope.commands))
        row['passed']=bool(row['complete'] and row['ownership_closed'] and outcome['passed']);persist();save(row)
    print(json.dumps(dict(mode=mode,stage=row['stage'],passed=outcome['passed'],error=outcome.get('error'))),flush=True)
    require(outcome['passed'],outcome.get('error','component stage failed'))


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','controls','build','host','check-small','check','time0','time1','time2'),'component stage argument')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        prepare() if sys.argv[1]=='prepare' else execute(sys.argv[1])
