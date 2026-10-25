#!/usr/bin/env python3
"""Main-owned, same-binary prefill allocation diagnostic; never adopts placement.

Repository LICENSE. Uses this fork's owned benchmark and Linux readers only.
The owned128 case is a diagnostic, not a production decode configuration.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_capacity as assets
import generation_engine as gen
import measurement_owner as owners
import runtime_probe as runtime

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'prefill-placement-probe-v1'
RESULT = A / 'prefill-runtime-comparison-current.json'
TEMPLATE = A / 'phase-prefill-speed-current.json'
READER = P / 'bench/sycl/client_resources.py'
KEY = 'placement_diagnostic'
CASES = (('owned-gate', 32768, 1, True), ('borrowed', 98304, 4, False), ('owned', 98304, 4, True))
require, digest, write = runtime.require, runtime.digest, runtime.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(row):
    return {k: v for k, v in row.items() if k not in ('models', 'benchmark_pins', 'environment_template')}


def publish(row):
    base = json.loads(RESULT.read_text()); base[KEY] = compact(row); write(RESULT, base)


def environment(row, root, owned=False):
    template = row['environment_template']; old = template['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(root)) for k, v in template.items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'STRATA_TRACE', 'ZE_ENABLE_ALT_DRIVERS',
        'STRATA_STATE_HASH', 'STRATA_SNAPSHOT_VERIFY', 'STRATA_CPU_WORK_CENSUS')), 'unexpected observer')
    env.update(STRATA_TRACE='1', UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root/'validation.log'))
    if owned:
        # The native admission forbids a cache-tail lease without borrowing.
        # These allocations have their own owners; no cache lease exists.
        env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] = '0'
    return env


def current(row):
    require(gen.committed(RESULT)[KEY] == compact(row), 'commit exact diagnostic plan/outcome')
    require(gen.benchmark_pins(row['source_commit']) == row['benchmark_pins'] and
        digest(READER) == row['reader']['sha256'] and digest(TEMPLATE) == row['template_sha256'], 'source/template changed')
    rt = gen.committed(runtime.RESULT); runtime.current(rt)
    require(digest(runtime.RESULT) == row['runtime_sha256'], 'qualified runtime changed')
    require(row['environment_template'] == gen.committed(TEMPLATE)['environment_template'], 'template differs')
    require(fingerprint(row['models']) == row['model_fingerprint'], 'model proof differs')
    for plan in row['cases']:
        root = Path(plan['root'])
        require(all(digest(p) == sha for p, sha in plan['inputs'].items()) and
            json.loads((root/'plan.json').read_text()) == plan['native_plan'], 'generated inputs changed')
    assets.model_pins(row['configuration'], row['models'])
    return rt


def check_requests(samples, hashes=None):
    for i, s in enumerate(samples):
        require(s['ok'] and s['eligible'] and s['stream_complete'] and
            s['thinking_requested'] and s['reasoning_observed'] and s['actual_reused_tokens'] == 0 and
            s['actual_read_tokens'] == s['actual_prompt_tokens'] >= 32768 and s['generated_tokens'] == 24 and
            s['prompt_tps_source'] == 'engine' and type(s['prompt_ms']) in (int, float) and
            10000 <= s['prompt_ms'] < 1500000, 'ineligible request')
        if hashes is not None:
            require(s['actual_read_tokens'] == 117489 and s['request_sha256'] == hashes[i], 'long workload differs')


def request_scopes(log, samples, owned):
    footers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', log, re.M))
    require(len(footers) == len(samples)+1 and int(footers[0][1]) < 8192, 'warmup/request extent differs')
    result = []
    for first, last, sample in zip(footers, footers[1:], samples):
        require(tuple(map(int, last.groups())) == (sample['actual_prompt_tokens'], 0, sample['actual_read_tokens'], 24), 'native footer differs')
        part = log[first.end():last.start()]; cells = sample['actual_read_tokens']-1
        required = f'strata prefill: layer-major candidate, tokens={cells} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={cells*20480} residual_gpu_rows={cells} residual_gpu={cells*20480} residual_host=0 residual_staging=0'
        require(part.count(required) == 1 and part.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 1,
            'long route/actual GPU residual admission differs')
        require(part.count('strata prefill dense: phase=restored residency=Full activations=48') == 1, 'dense restoration differs')
        batch = re.findall(r'^strata trace: read (\d+) tokens \(batched\) in ([\d.]+) ms$', part, re.M)
        require(len(batch) == 1 and int(batch[0][0]) == cells, 'batched interval differs')
        refills = re.findall(r'^strata trace: refilled (\d+) slots on (\d+) stage\(s\) in ([\d.]+) ms$', part, re.M)
        if owned: require(not refills and 'strata cache loan: phase=masked' not in part, 'owned case borrowed cache')
        else: require(len(refills) == 1 and refills[0][:2] == ('1124', '1'), 'borrowed refill differs')
        result.append(dict(batched_wrapper_ms=float(batch[0][1]), expert_refill_ms=0.0 if owned else float(refills[0][2]),
            limits='Batch includes native entry/run/restore;refill includes RAM lookup/enqueue/drain/publication. Not GPU service or PCIe bytes.'))
    return result


def host_snapshot():
    start = time.monotonic_ns()
    policy = Path('/sys/devices/system/cpu/cpufreq')
    gpu = Path('/sys/bus/pci/devices/0000:05:00.0/tile0/gt0/freq0')
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    available, = [int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')]
    result = dict(begin=start,
        cpu_khz={p.name: int((p/'scaling_cur_freq').read_text()) for p in sorted(policy.glob('policy*'))},
        gpu_act_mhz=int((gpu/'act_freq').read_text()), gpu_requested_mhz=int((gpu/'cur_freq').read_text()),
        throttle=int((gpu/'throttle/status').read_text()),
        pswpin=int(vm['pswpin']), pswpout=int(vm['pswpout']), pgmajfault=int(vm['pgmajfault']),
        memory_available=available)
    result['end'] = time.monotonic_ns()
    return result


def host_summary(records, interval):
    rows = [r for r in records if interval['request_begin_monotonic_ns'] <= r['begin'] and r['end'] <= interval['first_output_monotonic_ns']]
    require(len(rows) >= 2, 'host-state coverage missing')
    gpu = [r['gpu_act_mhz'] for r in rows]
    return dict(samples=len(rows), gpu_act_mhz=dict(min=min(gpu), median=statistics.median(gpu), max=max(gpu)),
        throttle_samples=sum(bool(r['throttle']) for r in rows),
        cpu0_reported_khz_median=statistics.median(r['cpu_khz']['policy0'] for r in rows),
        system_vm_delta={k: rows[-1][k]-rows[0][k] for k in ('pswpin','pswpout','pgmajfault')},
        min_available_RAM_bytes=min(r['memory_available'] for r in rows),
        limits='Wholly contained pre-first-output snapshots;not native phase boundaries. Reported frequency is not an arithmetic ceiling;system paging counters are not per-process attribution.')


def prepare():
    require(not ROOT.exists() and KEY not in gen.committed(RESULT), 'fresh diagnostic only')
    rt = gen.committed(runtime.RESULT); runtime.current(rt)
    baseline = json.loads(gen.git('show', '8573fa7fb:' + str(RESULT.relative_to(P))))
    cfg = copy.deepcopy(baseline['configuration']); cfg['args'][cfg['args'].index('--expert-profile')+1] = str(P/'data/expert-profile.bin')
    require(digest(P/'data/expert-profile.bin') == '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf', 'profile changed')
    original_assets = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    require(fingerprint(original_assets) == baseline['model_fingerprint'], 'original model inventory differs')
    models = assets.model_pins(cfg, original_assets, full=True)
    source = gen.git('rev-parse','HEAD').decode().strip(); pins = gen.benchmark_pins(source)
    ROOT.mkdir(mode=0o700)
    row = dict(stage='planned',source_commit=source, native_source_commit=rt['original_build']['source_commit'],
        binary=baseline['binary'], preload=baseline['preload'], configuration=cfg,
        template_sha256=digest(TEMPLATE), environment_template=gen.committed(TEMPLATE)['environment_template'],
        runtime_sha256=digest(runtime.RESULT), boot=rt['boot'], runtime_reference=str(runtime.RESULT.relative_to(P)),
        models=models,model_fingerprint=fingerprint(models),benchmark_pins=pins,
        reader=dict(path=str(READER),sha256=digest(READER)),request_hashes=baseline['request_hashes'],
        prior_preparation_failure=dict(source='76bf87d1a',error="KeyError:models",GPU_executed=False,
            boundary='Before root creation/model reads;compact historical record excludes models. Recover original15-asset map from1e676d554 and verify the8573 fingerprint. No native/model outcome relabelled.'),
        plan='One owned128/32K+ gate,then two separate observed117K processes:borrowed1256 then owned128;four fresh requests each,first separate. Fixed same6e40 binary/current runtime/math. Placement/cache geometry change together;order and observers prevent clean causal/adoption claims. Never change production decode. No retries or speed-based stopping.',
        source_audit='6e40 generate own_fits uses exact bytes_needed+512MiB;chunk fallback rejected by actual8192 route. Owned carve separately owns emb/R/region;borrowed lease uses checked tail. Native admission forbids lease1 with no-prefill-borrow,so owned-only uses lease0 for its independently owned buffers;borrowed keeps lease1. Shared long-pass GPU BF16/mode10 and48-layer dense restore must match. Same runtime safety/queue/parameter validation. Actual all-GPU residual allocation verified per request;32K gate does not establish117K or262144 capacity.',
        limits='Diagnostic only. No clean speed,observer overhead,hardware saturation,task/decode performance,physical262144 or ordinary-runtime-teardown qualification.',
        retention='At closed replay,keep all useful request clocks/work/placement/host summaries;retire normal logs,raw sampling,JIT and single-use controller. Selected engine+12fixtures and separate pending oneMKL consumer remain.',
        budgets=dict(wall=1800,cpu=72000,rss_cap=104<<30,text_cap=32<<20,file_cap=64<<20,total_cap=128<<20),
        cases=[],outcomes=[],active=False,ownership_closed=True)
    ports = set()
    for name, tokens, runs, owned in CASES:
        root=ROOT/name;root.mkdir();(root/'cache').mkdir();(root/'tmp').mkdir()
        config=copy.deepcopy(cfg);config['log']=str(root/'engine.stderr')
        if owned:config['args'][config['args'].index('--expert-cache')+1]='128';config['args'].append('--no-prefill-borrow')
        collection=dict(source=row['reader'],seconds=900,interval=1)
        native=dict(binary=row['binary'],native_validation_probe=row['preload'],resource_collection=collection)
        write(root/'config.json',config);write(root/'plan.json',native)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        require(port not in ports,'port collision');ports.add(port)
        command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),'--prefill-only','--prefill-tokens',str(tokens),'--think','--runs',str(runs),'--profile-request-gates','--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]
        row['cases'].append(dict(name=name,root=str(root),tokens=tokens,runs=runs,owned=owned,command=command,native_plan=native,
            inputs={str(p):digest(p) for p in (root/'config.json',root/'plan.json')}))
    write(ROOT/'record.json',row);publish(row)


def run():
    row=json.loads((ROOT/'record.json').read_text());rt=current(row)
    i=len(row['outcomes']);require(i<3 and all(o['passed'] for o in row['outcomes']),'qualified ordered cases only')
    plan=row['cases'][i];root=Path(plan['root']);config=json.loads((root/'config.json').read_text());env=environment(row,root,plan['owned'])
    require(not (root/'owned').exists(),'existing attempt;inspect owner,never retry')
    lines=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in lines.splitlines() if s.startswith('-- cursor: '))
    outcome=dict(name=plan['name'],passed=False,complete=False);owner=None
    host_file=(root/'host-state.jsonl').open('x');next_sample=0.0;native_target=None
    def persist():
        nonlocal next_sample
        if native_target and owner and owner.active and time.monotonic()>=next_sample:
            now=owners.identity(native_target['pid'])
            if now and now['start_ticks']==native_target['start_ticks']:
                require(host_file.tell()<8<<20,'host-state budget')
                host_file.write(json.dumps(host_snapshot())+'\n');host_file.flush();next_sample=time.monotonic()+1
        write(root/'record.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def interaction(io,leader):
        nonlocal native_target
        for request in range(plan['runs']):
            for stage in ('begin','end'):
                while True:
                    line=io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='prefill-only',run=request),'request gate differs')
                require(not (root/'validation.log').exists() or not (root/'validation.log').stat().st_size,'parameter warning')
                claim=json.loads((root/'engine-start.json').read_text());native_target=claim['native'];pid=native_target['pid'];ident=owners.identity(pid)
                require(ident and ident['start_ticks']==native_target['start_ticks'] and ident['sid']==leader['pid'] and
                    str(Path('/proc',str(pid),'exe').resolve())==str(Path(row['binary']['path']).resolve()),'native owner differs')
                actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                expected=dict(env,LD_PRELOAD=row['preload']['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'])
                require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'actual environment differs')
                mapped=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                require(mapped<=set(rt['runtime_files'])|{str(Path(row['binary']['path']).resolve())} and
                    any('libze_intel_gpu' in p for p in mapped),'runtime mapping differs')
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=owners.Owner(root/'owned',persist);entry,_,_=owner.run('bench',plan['command'],env,P,interaction=interaction,**row['budgets'])
        require(owners.closed(entry) and entry['exit_code']==0,'benchmark/owner failed')
        result_file,=(root/'bench').glob('*/results.json');result=json.loads(result_file.read_text());samples=result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples)==plan['runs'],'incomplete benchmark')
        check_requests(samples,None if i==0 else row['request_hashes'])
        sys.path.insert(0,str(P/'tools'));from strata_bench import verify_request_journal
        journal,=(root/'bench').glob('*/requests.jsonl');verified=verify_request_journal(journal.parent)
        events=[json.loads(v) for v in journal.read_text().splitlines()];longs=[v for v in events if v['kind']=='result' and v['phase']=='prefill-only']
        require(verified['valid'] and verified['complete'] and [dict(v['sample'],run=v['run']) for v in longs]==samples,'journal binding differs')
        claim=gen.native_end(root,dict(binary=row['binary'],native_validation_probe=row['preload'],resource_collection=plan['native_plan']['resource_collection']),dict(config=config),row['boot'],entry['pid'])
        log=(root/'engine.stderr').read_text();activation=runtime.validation.preload_evidence(log,claim['native']['pid'])
        require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|teardown uncertain|Strata SYCL cleanup:',log),'runtime failure')
        scopes=request_scopes(log,samples,plan['owned'])
        host_file.flush();hosts=[json.loads(s) for s in (root/'host-state.jsonl').read_text().splitlines()]
        sys.path.insert(0,str(READER.parent));import client_resources as reader
        captures=[json.loads(s) for s in (root/'client-resources.jsonl').read_text().splitlines()]
        outcome['requests']=[]
        for s,scope in zip(samples,scopes):
            resource=reader.summarize_capture(captures,claim['native'],row['boot'],s['client_interval'])
            before=resource['before_first_output']
            outcome['requests'].append(dict(prompt_ms=s['prompt_ms'],read_tokens=s['actual_read_tokens'],request_sha256=s['request_sha256'],
                scope=scope,host=host_summary(hosts,s['client_interval']),
                pre_first_output_cpu_seconds=before['process_cpu_seconds'],pre_first_output_cpu_core_equivalents=before['process_cpu_core_equivalents'],
                top_scheduled_threads=before['top_scheduled_threads'][:2],ccs=[e for e in before['engines'] if e['engine']=='ccs'],
                collector_coverage_verified=resource['request_coverage_verified']))
        current(row)
        if i==2:assets.model_pins(config,row['models'],full=True)
        outcome.update(complete=True,host_probe_accepted=True,native_identity=claim['native'],validation=activation,journal_sha256=digest(journal))
    except BaseException as error:
        outcome['error']=repr(error)
        if (root/'engine.stderr').exists():outcome['error_excerpt']=(root/'engine.stderr').read_text()[-4096:]
    finally:
        if owner:runtime.validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete']);persist();host_file.close()
        row['outcomes'].append(outcome);row['ownership_closed']=outcome.get('ownership_closed',True)
        row['stage']='closed' if not outcome['passed'] or i==2 else 'collecting';write(ROOT/'record.json',row);publish(row)
    print(json.dumps({k:outcome.get(k) for k in ('name','passed','complete','ownership_closed','error')}),flush=True)
    return int(not outcome['passed'])


def analyze_owned():
    """Closed-data plan: pre-first-output only; preserve the original failure.

    Original collection and full source/runtime admission are archived at80bb79887.
    This never resumes its closed cohort or relabels its failed original outcome.
    """
    row=json.loads((ROOT/'record.json').read_text())
    frozen=json.loads(gen.git('show','80bb79887:' + str(RESULT.relative_to(P))))[KEY]
    require(compact(row)==frozen==gen.committed(RESULT)[KEY] and row['stage']=='closed' and
        row['ownership_closed'] and not row['active'] and len(row['outcomes'])==1 and
        row['outcomes'][0]['error']=="ValueError('insufficient contained CPU/CCS observation spans for the client interval')" and
        not row['outcomes'][0]['passed'] and row['outcomes'][0]['fresh_faults']==[], 'original closed parser failure required')
    proof=row['post_closure_admission'];require(proof['original_failure_unchanged'] and proof['model_assets_stat_verified']==15,'original post-closure proof required')
    require(gen.benchmark_pins(row['source_commit'],match_worktree=False)==row['benchmark_pins'],'original source cannot recover')
    changed=[name for name,sha in row['benchmark_pins'].items() if digest(P/name)!=sha]
    require(changed==['tools/sycl/prefill_placement_probe.py'],'unrelated benchmark changes')
    rt=gen.committed(runtime.RESULT);runtime.current(rt);require(digest(runtime.RESULT)==row['runtime_sha256'],'runtime changed')
    assets.model_pins(row['configuration'],row['models'])
    root=ROOT/'owned-gate'
    for name,sha in proof['raw_hashes'].items():require(digest(root/name)==sha,'closed input changed')
    record=json.loads((root/'record.json').read_text())
    for command in record['commands']:
        require(owners.closed(command) and command['exit_code']==0,'owner did not close')
        for identity in command['owners']:
            now=owners.identity(identity['pid']);require(not now or now['start_ticks']!=identity['start_ticks'],'recorded owner still live')
    native=json.loads((root/'engine-exit.json').read_text());result=json.loads(next((root/'bench').glob('*/results.json')).read_text())
    samples=result['results']['prefill_only']['requests'];check_requests(samples)
    require(len(samples)==1 and result['complete'] and not result['results']['failures'] and native['exit_code']==native['resource_collector_exit_code']==0,'closed collection differs')
    sample=samples[0];scopes=request_scopes((root/'engine.stderr').read_text(),samples,True)
    sys.path.insert(0,str(READER.parent));import client_resources as reader
    captures=[json.loads(s) for s in (root/'client-resources.jsonl').read_text().splitlines()]
    resources=reader.summarize_capture(captures,native['native'],row['boot'],sample['client_interval'],before_only=True)
    before=resources['before_first_output'];hosts=[json.loads(s) for s in (root/'host-state.jsonl').read_text().splitlines()]
    row['closed_analysis']=dict(source_commit=gen.git('rev-parse','HEAD').decode().strip(),collection_archive='80bb79887',
        source_inputs_changed=changed,reader_sha256=digest(READER),GPU_executed=False,attribution_passed=True,
        original_passed=False,native_ownership_closed=True,actual_read_tokens=sample['actual_read_tokens'],
        prompt_ms=sample['prompt_ms'],instrumented_tok_s=sample['actual_read_tokens']*1000/sample['prompt_ms'],
        request_sha256=sample['request_sha256'],scope=scopes[0],host=host_summary(hosts,sample['client_interval']),
        client_coverage_verified=resources['request_coverage_verified'],
        pre_first_output_cpu_seconds=before['process_cpu_seconds'],
        pre_first_output_cpu_core_equivalents=before['process_cpu_core_equivalents'],
        top_scheduled_threads=before['top_scheduled_threads'][:2],ccs=[x for x in before['engines'] if x['engine']=='ccs'],
        limits='Closed39K owned128 observation only;first use,trace and passive collectors charged. Not clean speed,117K/262144 capacity,decode,host/GPU/bus saturation or historical decline attribution.',
        next='New committed two-process117K plan may reuse this bounded owned-shape gate with unchanged native/runtime/inputs. Original failed collection and original three-case cohort stay closed;never resume or overwrite them.')
    write(ROOT/'record.json',row);publish(row);print(json.dumps(row['closed_analysis']),flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run','analyze-owned'),'prepare,run or analyze-owned required')
    with (B/'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime.guards.no_model_owners([]);runtime.guards.driver_guard()
        sys.exit(prepare() if sys.argv[1]=='prepare' else analyze_owned() if sys.argv[1]=='analyze-owned' else run())
