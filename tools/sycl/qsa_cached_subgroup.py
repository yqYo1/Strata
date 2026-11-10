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
import statistics
import subprocess
import sys
sys.dont_write_bytecode = True
import build_engine as source
import measurement_guards as guards
import measurement_owner as owner
import check_qsa16_stream as shared

P, B = shared.P, shared.B
ROOT = B / 'qsa-cached-subgroup-component-v2'
RESULT = shared.RESULT
KEY = 'qsa_cached_subgroup_component_v2'
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
    for group in row.get('actual_inputs', {}).values():
        for name, expected in group.items():
            require(sha(name) == expected, 'original actual compiler/link input changed')

def diagnostics(errors, pid):
    marker = re.search(r'^strata UR validation: pid=' + str(pid) + r' phase=before-main .*$', errors, re.M)
    require(marker is not None, 'missing before-main marker')
    expected = re.compile(r'^\s*<--- urPlatformGetApiVersion\(\.hPlatform = nullptr, '
        r'\.pVersion = 0x[0-9a-fA-F]+ \([^\n]*\)\) -> UR_RESULT_ERROR_INVALID_NULL_HANDLE;$')
    canaries = 0
    for match in re.finditer(r'^.*(?:UR_RESULT_ERROR_|ZE_RESULT_ERROR_|cleanup failed|terminate called).*$', errors, re.M):
        require(match.start() < marker.start() and expected.fullmatch(match[0]) is not None,
            'unexpected native diagnostic outside the qualified canary')
        canaries += 1
    require(canaries <= 1, 'duplicate canary diagnostic')

def diagnostic_controls():
    marker = 'strata UR validation: pid=123 phase=before-main checked=46\n'
    expected = '   <--- urPlatformGetApiVersion(.hPlatform = nullptr, .pVersion = 0x1234 (-1.65535)) -> UR_RESULT_ERROR_INVALID_NULL_HANDLE;\n'
    diagnostics(expected + marker, 123); diagnostics(marker, 123)
    cases = [marker + expected, expected + expected + marker,
        expected.replace('urPlatformGetApiVersion','urQueueCreate') + marker,
        expected.replace('INVALID_NULL_HANDLE','INVALID_ARGUMENT') + marker,
        'UR_RESULT_ERROR_UNKNOWN\n' + marker, expected]
    for case in cases:
        try: diagnostics(case, 123)
        except RuntimeError: continue
        raise RuntimeError('diagnostic refusal control unexpectedly accepted')
    return dict(valid_controls=2, refused_controls=len(cases), passed=True)

def prepare():
    require(not ROOT.exists(), 'fresh V2 only; inspect existing receipt, no retry')
    original_path = B / 'qsa-cached-subgroup-component-v1/record.json'
    previous = json.loads(original_path.read_text())
    public = json.loads(source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(public['qsa_cached_subgroup_component_v1'] == compact(previous),
        'commit original failed V1 first')
    failure = previous['outcomes']['smoke']
    require(previous['stage']=='failed' and not previous['passed'] and previous['ownership_closed'] and
        failure['error']=="RuntimeError('native diagnostic error')" and failure['fresh_faults']==[] and
        failure['ownership_closed'] and len(failure['checks'])==1 and
        failure['checks'][0]['exit_code']==0 and failure['checks'][0]['ownership_closed'],
        'original failed controller result must remain unchanged')
    recovered = {n:hashlib.sha256(raw).hexdigest() for n,raw in
        source.blobs(P,previous['source_commit'],previous['input_sha256'])}
    require(recovered==previous['input_sha256'], 'original native/controller source unrecoverable')
    revision=source.git(P,'rev-parse','HEAD').decode().strip(); pins=inputs(revision)
    require(set(pins)==set(recovered) and {n for n in pins if pins[n]!=recovered[n]}==
        {'tools/sycl/qsa_cached_subgroup.py'}, 'only declared controller may change')
    for item in previous['binaries'].values():
        require(sha(item['path'])==item['sha256'], 'original component ELF changed')
    for group in previous['actual_inputs'].values():
        for name,h in group.items(): require(sha(name)==h,'original actual compiler/link input changed')
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
    row={k:v for k,v in previous.items() if k not in ('outcomes','error')}
    row.update(stage='built',source_commit=revision,native_source_commit=previous['source_commit'],
        input_sha256=pins,outcomes={},passed=True,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,diagnostic_controls=diagnostic_controls(),
        controller_reuse=dict(original_record=dict(path=str(original_path),sha256=sha(original_path)),
            original_failed_evidence_commit='4d5167c88d1b58ba7ff12f0881e6b1d07cd03781',
            only_changed_input='tools/sycl/qsa_cached_subgroup.py',
            original_native_source_recovered=True,all722compiled_and25link_inputs_unchanged=True,
            same_two_executable_hashes_verified=True,new_native_build_or_GPU_qualification_claimed=False,
            reason='V1 controller rejected exactly the pre-main invalid-null-handle canary,although the numerical child completed0. Keep V1 failed. V2 admits at most one exact known function/error line strictly before the checked pre-main marker,and refuses all native errors afterward. Fresh smoke/numerical/timing evidence required.'),
        environment=dict(previous['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')))
    row['basis']['original_component']=dict(path=str(original_path),sha256=sha(original_path))
    row['plan'] += ' V2 changes only the diagnostic controller and private temp/cache paths;reuses exact original compiled native/fixture bytes7350b16d,without relabelling them as the controller source. Original V1 stays failed.'
    current(row,committed=False);save(row)
    print('Prepared V2 with exact original7TU/2ELF reuse and6diagnostic-refusal controls;commit before fresh smoke.',flush=True)

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
            diagnostics(errors,entry['pid'])
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
    require(len(sys.argv)==2,'use prepare|smoke|component|perf0|perf1|perf2')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare': prepare()
        else:
            row=json.loads((ROOT/'record.json').read_text());current(row)
            run(row,sys.argv[1])
