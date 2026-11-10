#!/usr/bin/env python3
"""Main-owned cached-query/vector-KV/subgroup component build and comparison.

Independent composition of this repository's MIT kernel; no model dispatch.
The current full engine remains untouched. Commit each stage before executing.
"""
import ast
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
import build_engine as source
import measurement_guards as guards
import measurement_owner as owner
import check_qsa16_stream as shared

P, B = shared.P, shared.B
ROOT = B / 'qsa-cached-subgroup-component-v1'
RESULT = shared.RESULT
KEY = 'qsa_cached_subgroup_component_v1'
SEED = B / 'qsa16-bulk-mirror-model-native-v1/record.json'
NATIVE = ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
          'src/kernels/sycl/prompt_attention.cpp', 'src/kernels/sycl/qsa_bf16_prefix.cpp']
FIXTURES = ['tests/sycl/qsa_bf16_prefix.cpp', 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp']
require, sha = guards.require, shared.sha

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def inputs(revision):
    names = set(NATIVE + FIXTURES + ['cmake/sycl_backend.cmake', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md'])
    names.update(source.git(P, 'ls-tree', '-r', '--name-only', revision, '--', 'include').decode().splitlines())
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path); names.add(str(path.relative_to(P)))
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [v.name for v in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file(): pending.append(local)
    pins = {}
    for name, raw in source.blobs(P, revision, sorted(names)):
        require((P / name).read_bytes() == raw, 'commit exact component input first: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins

def compact(row):
    result = {k:v for k,v in row.items() if k not in ('input_sha256', 'actual_inputs')}
    result['source_inputs'] = dict(files=len(row['input_sha256']),
        aggregate_sha256=sha_json(row['input_sha256']))
    if 'actual_inputs' in row:
        result['actual_input_summary'] = {k:len(v) for k,v in row['actual_inputs'].items()}
    return result

def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def save(row):
    write(ROOT / 'record.json', row)
    public = json.loads(RESULT.read_text()); public[KEY] = compact(row); write(RESULT, public)

def current(row, committed=True):
    require(inputs(row['source_commit']) == row['input_sha256'], 'frozen source changed')
    if committed:
        public = json.loads(source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
        require(public.get(KEY) == compact(row), 'commit exact preceding component stage first')
    for item in row['basis'].values():
        require(sha(item['path']) == item['sha256'], 'compiler/build basis changed')
    for path, expected in row['compiler'].items():
        require(sha(path) == expected, 'compiler changed')
    require(shared.probe_qualification() == (row['native_validation_probe'], row['validation_qualification']),
        'native pre-main probe qualification changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'platform changed')
    for item in row.get('binaries', {}).values():
        require(sha(item['path']) == item['sha256'], 'fixture executable changed')

def allowed_inputs(row):
    external = json.loads(Path(row['basis']['sdk']['path']).read_text())
    census = json.loads(Path(row['basis']['census']['path']).read_text())
    allowed = {}
    for name, expected in {**external, **census['compiled'], **census['links']}.items():
        if not Path(name).resolve().is_relative_to(P):
            allowed[str(Path(name).resolve())] = expected
    allowed.update({str((P / n).resolve()):h for n,h in row['input_sha256'].items()})
    return allowed

def prepare():
    require(not ROOT.exists(), 'fresh component only; inspect previous receipt, no retry')
    revision = source.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision); native = json.loads(SEED.read_text())
    require(native['passed'] and native['complete'] and native['ownership_closed'] and not native['active'],
        'closed retained engine required')
    for item in native['binaries'].values():
        require(sha(item['path']) == item['sha256'], 'retained fallback ELF changed')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); (ROOT / 'cache').mkdir()
    compiler = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-g0', '-fsycl',
        '-fsycl-id-queries-range=size_t', '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-fno-fast-math', '-ffp-contract=off', '-DSTRATA_ENABLE_SYCL=1',
        '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat')]
    commands, objects = [], []
    for i, name in enumerate(NATIVE + FIXTURES):
        obj = ROOT / ('input-' + str(i) + '.o'); objects.append(obj)
        extra = (['-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_cached_subgroup_batch',
            '-DSTRATA_QSA_COMPARE_SUBGROUP=1', '-DSTRATA_QSA_PACKED_GUARD=17'] if name == FIXTURES[0]
            else ['-DSTRATA_QSA_BENCH_CACHED_SUBGROUP=1'] if name == FIXTURES[1] else [])
        commands.append([compiler, *flags, *extra, '-MD', '-MF', str(obj.with_suffix('.d')),
            '-c', str(P / name), '-o', str(obj)])
    for name, obj in [('numerical', objects[5]), ('fixture', objects[6])]:
        commands.append([compiler, *flags, *map(str, [*objects[:5], obj]),
            '-pthread', '-ldl', '-Wl,-t', '-o', str(ROOT / name)])
    probe, qualification = shared.probe_qualification()
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins,
        basis=dict(native=dict(path=str(SEED), sha256=sha(SEED)), sdk=native['external_inputs'],
            census=native['actual_inputs']),
        compiler={compiler:native['toolchain'][compiler]}, commands=commands,
        native_validation_probe=probe, validation_qualification=qualification,
        environment=dict(native['recipe']['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache')),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        preserved_engine=native['binary'], passed=False, complete=False, active=False, ownership_closed=True,
        outcomes={}, GPU_executed=False,
        plan='Commit source/recipe before build. Smoke then full numerical component,then3fresh processes with6rotated timed samples per arm at each of3existing geometries. Full entry wall includes admission,prefix pack,attention and checked drain;upload/readback/oracles outside. Each timed production-shaped sample must exceed2ms. Candidate must exactly equal the existing uncached subgroup export and pass the unchanged independent CPU BF16 operand oracle,nonfinite/domain/alias/guard/input tests. Ordered-versus-subgroup output differences are recorded,never called bit-identical or model-quality evidence. No model/decode dispatcher changes.',
        performance_gate='All3pooled shape-median candidate/base ratios<=1 and geometric mean<1;at least2of3process medians improve for each cap2051pattern. No blanket minimum gain. Numerical/ownership/runtime/fault and timing-completeness gates must all pass before a separate prospective model plan.',
        numerical_policy='The independent BF16 operand oracle retains its existing1BF16-ULP+2e-6 tolerance. A failure remains failed. Any framework-contract criterion revision requires source-based justification and a new committed prospective plan;no model advance from self-consistency alone.',
        retention='One component fixture/numerical generation for this imminent comparison,beside3named retained full engines56cad9/552b/a347. No new full engine or overwrite. Commit compact useful samples,verify original source/recipe and retire unsuccessful/superseded component bytes/logs at decision closure. Physical262144model/RESTORE,task/agent,sustained TG,default adoption remain open.')
    current(row, committed=False); save(row)
    print('Prepared7TUs/2executables;commit exact source/recipe before build.', flush=True)

def build(row):
    require(row['stage'] == 'prepared', 'fresh prepared build only')
    scope = shared.owned(ROOT / 'build')
    result = dict(passed=False, complete=False)
    actual = dict(compiled={}, links={})
    try:
        allow = allowed_inputs(row); deadline = time.monotonic() + 1200
        for i, command in enumerate(row['commands']):
            print('compile/link', i + 1, '/', len(row['commands']), flush=True)
            entry, out, err = scope.run(str(i), command, row['environment'], P,
                wall=min(420, max(1, deadline-time.monotonic())), cpu=900,
                rss_cap=12<<30, text_cap=8<<20, file_cap=512<<20, total_cap=1<<30)
            require(owner.closed(entry) and entry['exit_code'] == 0, 'compile/link failed: ' + err.read_text()[-4096:])
            if i >= 7:
                for line in out.read_text().splitlines():
                    path = Path(line.strip())
                    if path.is_absolute() and path.is_file() and not path.resolve().is_relative_to(ROOT):
                        resolved = str(path.resolve()); h = sha(path)
                        require(allow.get(resolved) == h, 'unfrozen external link input: ' + resolved)
                        actual['links'][resolved] = h
        for dep in sorted(ROOT.glob('*.d')):
            for name in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]):
                path = Path(name).resolve(); h = sha(path)
                require(allow.get(str(path)) == h, 'unfrozen actual compiler input: ' + str(path))
                actual['compiled'][str(path)] = h
        require(len(actual['compiled']) > 500 and len(actual['links']) >= 10, 'incomplete dependency/link census')
        current(row)
        row['actual_inputs'] = actual
        row['binaries'] = {name:dict(path=str(ROOT/name), bytes=(ROOT/name).stat().st_size,
            sha256=sha(ROOT/name)) for name in ('numerical','fixture')}
        result.update(passed=True, complete=True)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result['ownership_closed'] = scope.active is None and all(owner.closed(c) for c in scope.commands)
        row.update(stage='built' if result.get('passed') else 'failed', passed=result.get('passed',False),
            ownership_closed=result['ownership_closed'], build_outcome=result)
        save(row)
    require(result.get('passed'), result.get('error','build failed'))
    print('PASS component build;GPU/model/speed remain unrun.', flush=True)

def run(row, mode):
    prior = list(row['outcomes'])
    order = ['smoke','component','perf0','perf1','perf2']
    require(mode == order[len(prior)] and row['stage'] in ('built','checked'),
        'fresh next stage only; commit preceding result, no retry')
    root = ROOT / mode; root.mkdir(mode=0o700)
    env = dict(row['environment'], STRATA_TEST_MAIN_GATES='0',
        LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if mode=='smoke' else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    if mode != 'smoke': env.pop('UR_LOG_TRACING',None)
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission'):
        require(env[key] == {'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1','NEOReadDebugKeys':'1','EnableDirectSubmission':'0'}[key],
            'qualified safety changed')
    cursor = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor, 'kernel cursor missing')
    scope = shared.owned(root); result = dict(passed=False,complete=False,checks=[])
    try:
        jobs = [('numerical','--smoke'),('fixture','--smoke')] if mode=='smoke' else \
            [('numerical','--component')] if mode=='component' else [('fixture','--component')]
        for name, argument in jobs:
            command = [row['binaries'][name]['path'], argument]
            if mode.startswith('perf'): command.append(mode[-1])
            entry, out, err = scope.run(name, command, env, P, wall=240,cpu=720,
                rss_cap=4<<30,text_cap=64<<20,file_cap=64<<20,total_cap=80<<20)
            text, errors = out.read_text(), err.read_text()
            proof = shared.validation.preload_evidence(errors,entry['pid'])
            samples = [dict(token.split('=',1) for token in line.split()[1:])
                for line in text.splitlines() if line.startswith('sample ')]
            check = dict(binary=name,command=command,exit_code=entry['exit_code'],
                ownership_closed=owner.closed(entry),validation=proof,
                cases=[line for line in text.splitlines() if line.startswith(('case ','fixture_status='))],
                references=[line for line in text.splitlines() if line.startswith('reference ')],samples=samples)
            result['checks'].append(check)
            require(owner.closed(entry) and entry['exit_code']==0 and 'fixture_status=PASS' in text,
                'component failed: '+(errors+text)[-4096:])
            require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|cleanup failed|terminate called',errors),
                'native diagnostic error')
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,
                'parameter validation warning')
            if name=='numerical':
                require(sum(line.startswith('case cells=') for line in text.splitlines()) ==
                    (2 if mode=='smoke' else 11), 'incomplete numerical cases')
            if mode.startswith('perf'):
                require(len(samples)==36 and all(float(s['milliseconds'])>=2 and
                    math.isfinite(float(s['milliseconds'])) for s in samples), 'timing eligibility/count')
        current(row); result.update(passed=True,complete=True,host_probe_accepted=True)
    except BaseException as error:
        result['error']=repr(error)
    finally:
        shared.validation.finish(result,scope,cursor,row['boot'])
        result['passed']=bool(result.get('passed') and result['complete'] and row['driver']==guards.driver_guard())
        row['outcomes'][mode]=result
        row.update(stage='checked' if result['passed'] else 'failed',passed=result['passed'],
            ownership_closed=result['ownership_closed'],GPU_executed=True,
            complete=mode=='perf2' and result['passed'])
        if row['complete']:
            samples=[dict(s,process=i) for i in range(3)
                for c in row['outcomes']['perf'+str(i)]['checks'] for s in c['samples']]
            shapes=[]
            for cap,pattern in ((33,'gather'),(2051,'gather'),(2051,'clustered')):
                arms=[]
                for arm in (0,1):
                    selected=[s for s in samples if int(s['cap'])==cap and s['pattern']==pattern and int(s['variant'])==arm]
                    values=[float(s['milliseconds']) for s in selected]
                    require(len(values)==18,'complete shape/arm samples')
                    arms.append(dict(median_ms=statistics.median(values),sample_sd_ms=statistics.stdev(values),
                        process_medians_ms=[statistics.median(float(s['milliseconds']) for s in selected if s['process']==i) for i in range(3)]))
                shapes.append(dict(cap=cap,pattern=pattern,arms=arms,ratio=arms[1]['median_ms']/arms[0]['median_ms']))
            ratio=statistics.geometric_mean(s['ratio'] for s in shapes)
            row['performance']=dict(shapes=shapes,geometric_mean_ratio=ratio,
                component_advance_eligible=ratio<1 and all(s['ratio']<=1 for s in shapes) and all(
                    sum(a<b for a,b in zip(s['arms'][1]['process_medians_ms'],s['arms'][0]['process_medians_ms']))>=2
                    for s in shapes if s['cap']==2051),model_speed_measured=False,model_adopted=False)
            row['stage']='closed'
        save(row)
    print(json.dumps({k:result.get(k) for k in ('passed','complete','ownership_closed','error')}),flush=True)
    require(result['passed'],result.get('error','component failed'))

if __name__ == '__main__':
    require(len(sys.argv)==2,'use prepare|build|smoke|component|perf0|perf1|perf2')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare': prepare()
        else:
            row=json.loads((ROOT/'record.json').read_text());current(row)
            if sys.argv[1]=='build': build(row)
            else: run(row,sys.argv[1])
