#!/usr/bin/env python3
"""Main-owned production-linked upload/dequant/native16 dependency comparison.

Repository LICENSE applies. No engine rebuild or model-speed claim. Commit
source, preparation, scanned compiler inputs and each result before advancing.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import check_qsa_cached_model as device
import qsa_cached_subgroup as shared

P, B, RESULT = device.P, device.B, device.RESULT
ROOT = B / 'expert-upload-chain-v1'
KEY = 'expert_upload_chain_v1'
THIS = 'tools/sycl/check_expert_upload_chain.py'
CPP = 'tests/sycl/expert_upload_chain.cpp'
OBSERVERS = {device.THIS, 'tools/sycl/qsa_cached_model_entry.py', THIS, CPP}
STAGES = ('smoke', 'check', 'perf0', 'perf1', 'perf2', 'perf3', 'perf4', 'perf5')
ARMS = (0, 1, 1, 0, 0, 1)
require, sha = device.require, device.sha
signature = shared.sha_json


def native():
    n = device.native(observers=OBSERVERS)
    prior = device.original.validation.committed(RESULT, (device.KEY,))
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
        and prior['ownership_closed'] and prior['binaries'] == n['binaries']
        and prior['native_source_commit'] == n['native_source_commit']
        and len(prior['results']) == 9 and all(v['passed'] and v['ownership_closed']
            and v['native_exit_code'] == 0 for v in prior['results']),
        'fresh same-build nine-stage qualification required')
    return n


def compact(row):
    result = {k:v for k,v in row.items() if k not in ('headers', 'actual_headers')}
    for name in ('headers', 'actual_headers'):
        if name in row: result[name + '_summary'] = dict(files=len(row[name]), sha256=signature(row[name]))
    return result


def save(row):
    shared.write(ROOT / 'record.json', row)
    public = json.loads(RESULT.read_text()); public[KEY] = compact(row)
    shared.write(RESULT, public)


def current(row, committed=True):
    require(native()['observer_admission'] == row['native_proof'], 'same-build source/recipe/ELF proof changed')
    require(all(sha(P/p) == h for p,h in row['inputs'].items()), 'new fixture/controller changed')
    recovered = {p:hashlib.sha256(raw).hexdigest() for p,raw in
        device.original.common.source.blobs(P, row['source_commit'], row['inputs'])}
    require(recovered == row['inputs'], 'exact fixture/controller sources not recoverable')
    require(all(sha(p) == h for p,h in row['linked_objects'].items()), 'production library/image changed')
    if 'headers' in row:
        require(all(sha(p) == h for p,h in row['headers'].items()), 'scanned compiler input changed')
    if 'binary' in row:
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'fixture ELF changed')
    if committed:
        require(device.original.validation.committed(RESULT, (KEY,)) == compact(row),
            'commit preceding stage first')


def dependencies(path, cwd):
    text = path.read_text().replace('\\\n', ' ')
    require(': ' in text, 'compiler dependency target missing')
    paths = [Path(p) for p in shlex.split(text.split(': ', 1)[1])]
    return {str((p if p.is_absolute() else cwd/p).resolve()):
        sha(p if p.is_absolute() else cwd/p) for p in paths}


def prepare():
    require(not ROOT.exists(), 'fresh bounded comparison only; no overwrite/retry')
    n = native(); build = device.build.ROOT/'build'
    database = json.loads((build/'compile_commands.json').read_text())
    template, = [v for v in database if v['file'].endswith('/tests/sycl/mmvq.cpp')]
    obj, binary = ROOT/'expert-upload-chain.o', ROOT/'sycl_expert_upload_chain'
    compile_argv = shlex.split(template['command'])
    compile_argv[compile_argv.index('-o')+1] = str(obj)
    compile_argv[-1] = str(P/CPP)
    compile_argv += ['-MD', '-MF', str(ROOT/'actual.d')]
    scan = [a for a in compile_argv[:-3] if a != '-c']
    i = scan.index('-o'); del scan[i:i+2]
    scan += ['-M', '-MF', str(ROOT/'planned.d'), '-MT', str(obj)]
    line = subprocess.check_output(['ninja','-C',str(build),'-t','commands','strata'], text=True).splitlines()[-1]
    link = shlex.split(line)
    require(link[:2] == [':','&&'] and link[-2:] == ['&&',':'], 'exact original link template required')
    link = link[2:-2]
    old_main = ['CMakeFiles/strata.dir/src/program/generate.cpp.o',
                'CMakeFiles/strata.dir/src/program/dense_prefill_nll.cpp.o']
    require(all(link.count(p) == 1 for p in old_main), 'two original main objects required')
    link = [a for a in link if a not in old_main]
    link[link.index('-o')+1] = str(binary)
    link.insert(link.index('-o'), str(obj))
    linked = {}
    for i,a in enumerate(link):
        p = Path(a)
        if p.suffix in ('.a','.o') or (p.is_absolute() and '.so' in p.name):
            if p == obj: continue
            p = (p if p.is_absolute() else build/p).resolve()
            linked[str(p)] = sha(p); link[i] = str(p)
    revision = device.original.common.source.git(P,'rev-parse','HEAD').decode().strip()
    inputs = {p:sha(P/p) for p in (THIS,CPP)}
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'): (ROOT/name).mkdir(mode=0o700)
    env = dict(n['recipe']['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    for key in ('UR_ENABLE_LAYERS','UR_LOG_TRACING'): env.pop(key,None)
    row = dict(stage='prepared', source_commit=revision, native_source_commit=n['native_source_commit'],
        inputs=inputs, native_proof=n['observer_admission'], linked_objects=linked,
        recipe=dict(cwd=str(build), scan=scan, compile=compile_argv, link=link, environment=env),
        native_validation_probe=n['native_validation_probe'],
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=shared.guards.driver_guard(),
        outcomes={}, passed=False, complete=False, ownership_closed=True, GPU_executed=False,
        plan=dict(order=['scan','compile',*STAGES],
            source='New single TU links the exact existing eight-target engine libraries and vLLM image objects. No engine, CMake, dispatch, queue, or runtime change.',
            shapes='Actual N2560/F640,GU22/Down20=1971200 bytes and GU23/Down20=2662400 bytes;synthetic valid blocks,not actual model weights. Rows640/1280 are production unfused jobs;32/1 are numerical/small checks. No fused>=2334 claim.',
            memory='Ordinary prefaulted immutable512-expert slab;three private bank slots(two retained,one scratch),same custom in-order queue;four Host-USM slots only for staged copies. This probes the dependency path,not the actual384-slot model scheduler.',
            correctness='17queued output readbacks per arm,bitwise equality for every final tensor,ring wrap,cache hits/scratch overwrite,source/tensor/bank/host guards. Full final-expert GU/Down dequant against ggml FP16 bits. Stagewise FP64 sampled dot oracle uses BF16-RNE FP16 operands,FP32 gammaK forward-error bound plus2BF16ULPs. Every SwiGLU output uses2e-6+0.002abs(reference) tolerance with FP16 saturation. Actual previous-stage tensors are each next-stage inputs;not a full CPU end-to-end model oracle.',
            timing='Six separate immutable parameter-validated processes ABBAAB;each has32warmup jobs per shape,then two256-job batches with192misses/64hits;checked queue completion included. Every batch>=100ms. Setup,allocations,prefaulting,oracles,readbacks and final guards excluded;all production wrapper admission,CPU stage copy,slot waits,upload,dequant,GU,SwiGLU,Down and final drain included.',
            decision='Pair process0/1,3/2,4/5;compare per-shape medians of two batches. All three staged/direct ratios must be<1 for all four shapes before component advancement. Collect all six unless correctness,fault,ownership,budget or eligibility fails. No speed-based stopping. No automatic retries,hash refresh or threshold changes.',
            limits='Parameter-validation overhead is charged equally;not clean model speed or physical PCIe saturation. No model advancement/default adoption from correctness alone. Then separately commit model32K/64K repeated bench.sh plan and independent decode/full262144/RESTORE/task gates.'),
        prior_performance=dict(path='bench/results/2026-10-15-pr6-recovery-audit/prefill-resources-current.json',
            key='event_reconciliation.fused_command_profile.prefill_upload_staging_v5',
            source='617e1845a4477994f812d9bf0f4cbe4fc80d045c',
            outcome='failed_component_gain_gate_preserved',
            large_process_ratios=[.943385240,1.049832685,.525887815],
            reason='Earlier byte-consumer fixture did not qualify full dequant/GEMM or model advancement. This prospective current-SDK full-chain question does not rewrite that outcome.'),
        retention='One small fixture alongside the existing engine,not another engine generation. Keep individual timings and decisions in Git. At closure retire normal trace,depfiles,object,private receipts and unused fixture after exact-source recovery;never remove original native proof dependencies or physical component holds here.')
    current(row, committed=False); save(row)
    print('Prepared production-linked chain;commit before scan.',flush=True)


def build(stage, row):
    require(row['stage'] == ('prepared' if stage == 'scan' else 'scanned'), 'next build stage only')
    scope = shared.shared.owned(ROOT/stage)
    result = dict(passed=False, complete=False)
    try:
        for name in (['scan'] if stage == 'scan' else ['compile','link']):
            e,out,err = scope.run(name,row['recipe'][name],row['recipe']['environment'],
                Path(row['recipe']['cwd']),wall=240,cpu=720,rss_cap=8<<30,total_cap=256<<20)
            require(shared.owner.closed(e) and e['exit_code'] == 0, name+' failed: '+err.read_text()[-4096:])
            if name == 'scan': row['headers'] = dependencies(ROOT/'planned.d',Path(row['recipe']['cwd']))
            if name == 'compile':
                row['actual_headers'] = dependencies(ROOT/'actual.d',Path(row['recipe']['cwd']))
                require(row['actual_headers'] == row['headers'], 'actual compiler dependencies differ from committed scan')
            if name == 'link':
                allowed = dict(row['linked_objects'])
                n = json.loads((device.build.ROOT/'record.json').read_text())
                census = json.loads(Path(n['actual_inputs']['path']).read_text())
                allowed.update(census['links'])
                observed = {}
                for line in out.read_text().splitlines():
                    s = line.strip(); p = Path(s)
                    if not s or any(c.isspace() for c in s) or not p.is_absolute(): continue
                    if p.is_relative_to(ROOT): continue
                    require(p.is_file() and allowed.get(str(p.resolve())) == sha(p), 'unfrozen external link input:'+s)
                    observed[str(p.resolve())] = sha(p)
                require(len(observed) >= len(row['linked_objects']), 'incomplete actual link trace')
                row['actual_link_inputs'] = observed
        current(row,committed=False)
        if stage == 'compile':
            p=ROOT/'sycl_expert_upload_chain'
            row['binary']=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
        result.update(passed=True,complete=True)
    except BaseException as e: result['error'] = repr(e)
    result['ownership_closed'] = scope.active is None and all(shared.owner.closed(v) for v in scope.commands)
    result['commands'] = [dict(argv=v['argv'],exit_code=v.get('exit_code'),closed=shared.owner.closed(v)) for v in scope.commands]
    row['outcomes'][stage] = result
    row.update(stage=('scanned' if stage=='scan' else 'built') if result['passed'] else 'failed',
        passed=result['passed'],ownership_closed=result['ownership_closed'])
    save(row); print(json.dumps(result),flush=True)
    require(result['passed'], result.get('error','build stage failed'))


def execute(stage,row):
    done = [k for k in row['outcomes'] if k in STAGES]
    require(row['stage'] in ('built','checked') and done == list(STAGES[:STAGES.index(stage)])
        and all(row['outcomes'][k]['passed'] for k in done), 'fresh next GPU stage only')
    root=ROOT/stage; root.mkdir(mode=0o700)
    for name in ('tmp','cache'): (root/name).mkdir()
    env=dict(row['recipe']['environment'], TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
        STRATA_SYCL_PREFILL_NATIVE16='bf16',STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
        STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',STRATA_SYCL_PREFILL_GU_SWIGLU='large',
        STRATA_SYCL_PREFILL_EXPERT_UPLOAD_STAGING=str(ARMS[int(stage[-1])]) if stage.startswith('perf') else '0',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if stage=='smoke' else 'UR_LAYER_PARAMETER_VALIDATION',
        LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    if stage=='smoke': env['UR_LOG_TRACING']='level:info;flush:info;output:stderr'
    cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor,'kernel cursor missing')
    scope=shared.shared.owned(root); result=dict(passed=False,complete=False)
    try:
        argument='--bench' if stage.startswith('perf') else '--'+stage
        e,out,err=scope.run(stage,[row['binary']['path'],argument],env,P,
            wall=240,cpu=720,rss_cap=6<<30,total_cap=128<<20)
        text,errors=out.read_text(),err.read_text()
        result['native_exit_code']=e['exit_code']
        result['validation']=device.original.validation.preload_evidence(errors,e['pid'])
        require(shared.owner.closed(e) and e['exit_code']==0 and 'fixture_status=PASS' in text,
            'chain failed: '+(errors+text)[-4096:])
        shared.diagnostics(errors,e['pid'])
        require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,
            'parameter validation warning')
        if stage=='smoke': require('urEnqueue' in errors,'actual small trace missing')
        result['checks']=[v for v in text.splitlines() if v.startswith('CHECK ')]
        result['samples']=[dict(v.split('=',1) for v in line.split()[1:])
            for line in text.splitlines() if line.startswith('TIME ')]
        if stage.startswith('perf'):
            require(len(result['samples'])==8 and all(math.isfinite(float(v['ms'])) and float(v['ms'])>=100
                and int(v['staged'])==ARMS[int(stage[-1])] and v['jobs']=='256'
                and v['uploads']=='192' and v['hits']=='64' for v in result['samples']), 'timing eligibility/count')
        else: require(len(result['checks'])==(1 if stage=='smoke' else 6),'complete correctness shapes required')
        current(row); result.update(passed=True,complete=True,host_probe_accepted=True)
    except BaseException as e: result['error']=repr(e)
    finally:
        device.original.validation.finish(result,scope,cursor,row['boot'])
        result['passed']=bool(result.get('passed') and result['complete'] and shared.guards.driver_guard()==row['driver'])
        row['outcomes'][stage]=result
        row.update(stage='checked' if result['passed'] else 'failed',passed=result['passed'],
            ownership_closed=result['ownership_closed'],GPU_executed=True)
        if stage=='perf5' and result['passed']:
            shapes=[]
            for t in (640,1280):
                for gu in (22,23):
                    medians=[]
                    for i in range(6):
                        values=[float(s['ms']) for s in row['outcomes']['perf'+str(i)]['samples']
                            if int(s['rows'])==t and int(s['gu'])==gu]
                        require(len(values)==2,'two batches per process/shape required')
                        medians.append(statistics.median(values))
                    ratios=[medians[b]/medians[a] for a,b in ((0,1),(3,2),(4,5))]
                    shapes.append(dict(rows=t,gu_type=gu,process_median_ms=medians,
                        paired_ratios=ratios,geometric_mean_ratio=statistics.geometric_mean(ratios)))
            row['performance']=dict(shapes=shapes,component_advance_eligible=all(
                all(v<1 for v in s['paired_ratios']) for s in shapes),model_speed_measured=False,
                default_adopted=False,physical_resource_saturation_qualified=False)
            row.update(stage='closed',complete=True)
        save(row)
    print(json.dumps({k:result.get(k) for k in ('passed','complete','ownership_closed','error')}),flush=True)
    require(result['passed'],result.get('error','chain stage failed'))


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','scan','compile',*STAGES),'use prepare|scan|compile|smoke|check|perf0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); shared.guards.no_model_owners([])
        if sys.argv[1]=='prepare': prepare()
        else:
            row=json.loads((ROOT/'record.json').read_text()); current(row)
            if sys.argv[1] in ('scan','compile'): build(sys.argv[1],row)
            else: execute(sys.argv[1],row)
