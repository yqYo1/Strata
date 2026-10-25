#!/usr/bin/env python3
"""Owned official long-prefill comparison after restored-model state admission.

Repository LICENSE. Main-authored from this fork's owned benchmark interfaces
and archived Main phase controller1e676d554; no XeStrata application source.
Prefill selection never qualifies decode preservation or inference adoption.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import restored_model as state
import measurement_clock as clock
import measurement_owner as owners

native, gen, guards = state.native, state.gen, state.guards
runtime, models = native.runtime, state.models
validation = runtime.validation
P, B = state.P, state.B
ROOT = B / 'restored-phase-bench-v1'
RESULT = gen.A / 'restored-phase-bench-current.json'
TEMPLATE = gen.A / 'phase-prefill-speed-current.json'
ORDER = (0, 1, 1, 0, 0, 1)
PAIRS = ((0, 1), (3, 2), (4, 5))
WORKLOAD = 'native_gdn_20261018'
require, digest, fingerprint, write = guards.require, state.digest, state.fingerprint, native.write


def plan():
    return dict(order=list(ORDER), pairs=[list(p) for p in PAIRS], requests_per_process=4,
        execution='Six fresh native processes,original6e40/restored1baa/restored1baa/original6e40/original6e40/restored1baa. '
            'Official bench.sh --prefill-only --prefill-tokens98304 --think --runs4:one short warmup then four '
            'fresh117489-token requests. Keep first-use and subsequent-three requests separate.',
        constants='131072 resident INT8 context,1256 expert slots,8192 chunk,5 workers,pcie0,adapt0,spec4, '
            'suffix0,prompt/conversation caches0,Q8 MTP,no coupled draft or token graph. Both arms use the same '
            'original BF16/vLLM/mode10 arithmetic and placement,with separate immutable pre-exec environments.',
        observer='Qualified native-only retained-loader preload and silent parameter validation remain charged '
            'in both arms. No trace,state/snapshot/cache readbacks,CPU census,resource collector,profiler or phase timers.',
        decision='Use exact actual read tokens/engine prompt milliseconds. Every long request must last>=10000ms. '
            'Compare subsequent-three medians in each paired process;no-decline requires all three candidate/control '
            'ratios>=1. Faster screen additionally requires every ratio>1 and geometric mean>=1.0005. Report all '
            'individual samples and spreads,600 and1000tok/s milestones independently. No decode/adoption claim.',
        timing='Before each native launch,run65536 adjacent C++ steady_clock pairs against the actual mapped '
            'native libstdc++ provider. Require1ns steady period,CLOCK_MONOTONIC resolution<=1us and observed '
            '(2*max_pair_ns/1e6+0.05ms)/10000ms<=0.0001. This screens endpoint cost/rounding,not a worst-case bound '
            'or an overhead subtraction.',
        stopping='Stop first source,model,clock,route,journal,provider,owner,fault or budget failure. No retries, '
            'replacements or speed-based early stopping. Commit every closed process before the next. Reuse the '
            'just-closed full15 model hashes only with identical inventory/stat/ctime;full hashes again after process5.',
        next='Independent repeated thinking-on long decode for both actual MTP classes,and official long/agent '
            'sessions. Any decode impact is a separation defect. Task quality,physical262144 final slots/session/RESTORE '
            'and ordinary runtime teardown remain unqualified.',
        interpretation='Restored-package throughput comparison on updated NEO,not a hardware-saturation or causal '
            'kernel attribution. Old-runtime1003tok/s medians are historical context,not pooled samples.')


def pins(revision):
    values = gen.benchmark_pins(revision)
    name = str(clock.SOURCE.relative_to(P)); raw = gen.git('show', revision + ':' + name)
    require(raw == clock.SOURCE.read_bytes(), 'uncommitted C++ clock source')
    values[name] = hashlib.sha256(raw).hexdigest()
    return values


def qualification(revision):
    row, component = state.qualification(revision)
    original, fallback = native.native.selected.baseline()
    rt = gen.committed(runtime.RESULT)
    loader = rt['preload']['loader']
    preload = dict(library=rt['preload']['library'],loader=dict(path=loader,sha256=rt['runtime_files'][loader]),
        retained_reference_until_exit=True,ordinary_runtime_teardown_qualified=False)
    return row, dict(native_sources=[original['source_commit'],row['native_source_commit']],
        binaries=[original['binary'],row['binary']],fallback=fallback,preload=preload,
        model_state_sha256=digest(state.RESULT),component_sha256=digest(state.COMPONENT),
        runtime_sha256=digest(runtime.RESULT),boot=rt['boot'],driver=rt['driver'])


def environment(root):
    template = gen.committed(TEMPLATE)['environment_template']; old = template['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old,str(root)) for k,v in template.items()}
    require(not any(k in env for k in ('LD_PRELOAD','STRATA_TRACE','STRATA_STATE_HASH','STRATA_SNAPSHOT_VERIFY',
        'STRATA_DECODE_TIMING','STRATA_VERIFY_PROFILE','STRATA_SYCL_CACHE_LOAN_AUDIT',
        'STRATA_CPU_WORK_CENSUS','STRATA_SYCL_PREFILL_SEPARATE_LAYOUT')), 'heavy observer or retired layout in template')
    env.update(UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root/'parameter.validation'))
    qualified = gen.committed(runtime.RESULT)['environment']
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission',
            'STRATA_SYCL_PREFILL_SAME_QUEUE','SYCL_CACHE_PERSISTENT'):
        require(env[k] == qualified[k], 'qualified safety setting differs: ' + k)
    return env


def config(root):
    cfg = copy.deepcopy(models.setup()['config'])
    cfg.update(exe=str(P/'tools/sycl/engine_once.py'),cwd=str(P),host='127.0.0.1',log=str(root/'engine.stderr'))
    args = cfg['args']
    for key,value in (('--expert-profile',str(P/'data/expert-profile.bin')),('--conversation-cache-mib','0'),
            ('--prompt-cache','0')):
        args[args.index(key)+1] = value
    require(all(args[args.index(k)+1] == v for k,v in (
        ('--max-context','131072'),('--kv','int8'),('--kv-resident','131072'),('--expert-cache','1256'),
        ('--prefill','8192'),('--pool-workers','5'),('--pcie-frac','0'),('--adapt-swaps','0'),('--spec','4'))),
        'original independent decode placement/configuration differs')
    return cfg


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('models','benchmark_pins')}
    out['clock'] = dict(clock.compact(row['clock']),**{k:v for k,v in row['clock'].items()
        if k in ('binary','passed','ownership_closed','calibration','host_tests')})
    out['plans'] = [{k:v for k,v in v.items() if k not in ('environment',)} for v in row['plans']]
    return out


def current(row):
    require(gen.committed(RESULT) == compact(row) and plan() == row['plan'], 'commit exact phase plan/result')
    model, proof = qualification(row['source_commit'])
    require(proof == row['qualification'] and pins(row['source_commit']) == row['benchmark_pins'] and
        fingerprint(row['benchmark_pins']) == row['benchmark_fingerprint'] and
        model['model_assets'] == row['models'] and digest(TEMPLATE) == row['template_sha256'],
        'phase source/qualification/model/template changed')
    clock.check(row['clock'])
    for i,v in enumerate(row['plans']):
        root = ROOT/str(i)
        require(v['arm'] == ORDER[i] and v['environment'] == environment(root) and
            json.loads((root/'config.json').read_text()) == config(root) and
            json.loads((root/'plan.json').read_text()) == dict(binary=proof['binaries'][v['arm']],
                native_validation_probe=proof['preload']) and
            all(digest(p) == sha for p,sha in v['inputs'].items()), 'generated process inputs differ')
    if row['clock'].get('binary'):
        require(digest(row['clock']['binary']['path']) == row['clock']['binary']['sha256'], 'clock executable changed')
    return model


def prepare():
    require(not ROOT.exists() and not RESULT.exists(), 'fresh single-use comparison only')
    revision = gen.git('rev-parse','HEAD').decode().strip(); model, proof = qualification(revision)
    benchmark = pins(revision)
    ROOT.mkdir(mode=0o700); host = ROOT/'host'; host.mkdir()
    rt = gen.committed(runtime.RESULT); provider, = [dict(path=p,sha256=h) for p,h in rt['runtime_files'].items() if 'libstdc++' in p]
    compiler = dict(path=str(Path('/usr/bin/g++').resolve())); compiler['sha256'] = digest(compiler['path'])
    clocks = [clock.prepare(b,compiler,environment(ROOT/'0'),host,provider) for b in proof['binaries']]
    require(clocks[0] == clocks[1], 'native packages use incompatible clock contracts')
    plans,ports = [],set()
    for i,arm in enumerate(ORDER):
        root=ROOT/str(i);root.mkdir()
        for n in ('tmp','cache'): (root/n).mkdir()
        write(root/'config.json',config(root))
        write(root/'plan.json',dict(binary=proof['binaries'][arm],native_validation_probe=proof['preload']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        require(port not in ports,'duplicate port;no retry');ports.add(port)
        plans.append(dict(arm=arm,root=str(root),port=port,environment=environment(root),
            inputs={str(root/n):digest(root/n) for n in ('config.json','plan.json')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--prefill-only','--prefill-tokens','98304','--think','--runs','4','--profile-request-gates',
                '--workload-id',WORKLOAD,'--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')]))
    row=dict(stage='prepared',source_commit=revision,plan=plan(),qualification=proof,clock=clocks[0],
        benchmark_pins=benchmark,benchmark_fingerprint=fingerprint(benchmark),models=model['model_assets'],
        model_fingerprint=model['model_fingerprint'],template_sha256=digest(TEMPLATE),configuration=config(ROOT/'0'),
        plans=plans,processes=[],active=False,ownership_closed=True,
        budgets=dict(wall=3000,cpu=72000,rss_cap=104<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20),
        retention='Keep compact individual samples,route/clocks/journal/conservation/owner/fault decisions. '
            'Retire closed successful raw/JIT data after replay;only original fallback and restored candidate retained '
            'for this imminent phase comparison.No new engine generation or inference adoption.')
    write(ROOT/'record.json',row);write(RESULT,compact(row))
    print('Prepared six-process official phase comparison;commit before CPU/GPU execution.',flush=True)


def rates(samples):
    require(len(samples)==4,'four long requests required')
    out=[]
    for i,s in enumerate(samples):
        require(s['run']==i and all(s.get(k) is True for k in
            ('ok','eligible','stream_complete','thinking_requested','reasoning_observed')) and
            s['actual_prompt_tokens']==s['actual_read_tokens']==117489 and s['actual_reused_tokens']==0 and
            type(s['generated_tokens']) is int and 1<=s['generated_tokens']<=24 and s['prompt_tps_source']=='engine' and
            type(s['prompt_ms']) in (float,int) and math.isfinite(s['prompt_ms']) and s['prompt_ms']>=clock.MIN_TIME_MS,
            'ineligible actual long-prefill sample')
        out.append(117489000/s['prompt_ms'])
    return out


def summarize(rows):
    require(len(rows)==6 and [v['arm'] for v in rows]==list(ORDER) and all(v['passed'] and
        v['ownership_closed'] and v['fresh_faults']==[] for v in rows),'six successful independent processes required')
    require(len({(v['native_identity']['pid'],v['native_identity']['start_ticks']) for v in rows})==6,'native process reused')
    speeds=[rates(v['requests']) for v in rows]; medians=[statistics.median(v[1:]) for v in speeds]
    for a,b in PAIRS:
        require([s['request_sha256'] for s in rows[a]['requests']]==
            [s['request_sha256'] for s in rows[b]['requests']],'paired workloads changed')
    ratios=[medians[b]/medians[a] for a,b in PAIRS]; geo=math.exp(statistics.mean(map(math.log,ratios)))
    return dict(individual_tok_s=speeds,first_use_tok_s=[v[0] for v in speeds],subsequent_median_tok_s=medians,
        pair_ratios=ratios,ratio_geomean=geo,prefill_no_decline_screen=all(v>=1 for v in ratios),
        faster_screen=all(v>1 for v in ratios) and geo>=1.0005,
        candidate600_each_process=all(medians[i]>600 for i in (1,2,5)),
        candidate1000_each_process=all(medians[i]>=1000 for i in (1,2,5)),
        decode_preservation_qualified=False,adoption_qualified=False)


def host(row):
    current(row);require(row['stage']=='prepared' and not (ROOT/'host/owned').exists(),'single-use clock qualification')
    owner=None; result=dict(passed=False,GPU_executed=False)
    def persist():write(ROOT/'host/record.json',dict(result,active=bool(owner and owner.active is not None),
        commands=owner.commands if owner else []))
    try:
        owner=owners.Owner(ROOT/'host/owned',persist);args=dict(wall=60,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=8<<20)
        entry,_,_=owner.run('clock-tests',['/usr/bin/python3','-B',str(P/'tools/sycl/test_measurement_clock.py')],
            row['clock']['environment'],P,**args)
        require(owners.closed(entry) and entry['exit_code']==0,'clock corruption controls failed')
        entry,_,_=owner.run('compile',row['clock']['command'],row['clock']['environment'],P,**args)
        require(owners.closed(entry) and entry['exit_code']==0,'clock compile failed')
        entry,out,_=owner.run('calibrate',[str(ROOT/'host/clock_probe')],row['clock']['environment'],P,**args)
        require(owners.closed(entry) and entry['exit_code']==0,'clock calibration failed')
        calibration=clock.evidence(out.read_text(),row['clock']['provider']);clock.check(row['clock']);current(row)
        row['clock'].update(binary=dict(path=str(ROOT/'host/clock_probe'),sha256=digest(ROOT/'host/clock_probe')),
            passed=True,ownership_closed=True,calibration=calibration,host_tests=2)
        result.update(passed=True,ownership_closed=True);row['stage']='host-closed'
    except BaseException as error:result['error']=repr(error);row['stage']='failed'
    finally:
        result['ownership_closed']=bool(owner and owner.active is None and all(owners.closed(e) for e in owner.commands))
        result['passed']=bool(result['passed'] and result['ownership_closed']);row['host_observation']=result
        persist();write(ROOT/'record.json',row);write(RESULT,compact(row))
    print(json.dumps(result),flush=True);return int(not result['passed'])


def run(row):
    current(row)
    require(row['stage'] in ('host-closed','collecting') and row['clock']['passed'] and
        len(row['processes'])<6 and all(v['passed'] for v in row['processes']),'ordered qualified cohort;no retry')
    i=len(row['processes']);selected=row['plans'][i];root=Path(selected['root'])
    require(not (root/'owned').exists(),'existing attempt;inspect its owner')
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    initial=copy.deepcopy(row);outcome=dict(arm=selected['arm'],passed=False,complete=False);owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None)
        write(root/'record.json',dict(outcome,active=row['active'],commands=owner.commands if owner else []))
    def interaction(io,leader):
        for request in range(4):
            for stage in ('begin','end'):
                while True:
                    line=io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='prefill-only',run=request),'official gate differs')
                require(not (root/'parameter.validation').exists() or (root/'parameter.validation').stat().st_size==0,'parameter validation warning')
                if stage=='begin':
                    claim=json.loads((root/'engine-start.json').read_text());ident=owners.identity(claim['native']['pid'])
                    require(ident and ident['start_ticks']==claim['native']['start_ticks'] and ident['sid']==leader['pid'] and
                        Path('/proc',str(ident['pid']),'exe').resolve()==Path(row['qualification']['binaries'][selected['arm']]['path']).resolve(), 'native identity differs')
                    env=dict(v.split(b'=',1) for v in Path('/proc',str(ident['pid']),'environ').read_bytes().split(b'\0') if b'=' in v)
                    probe=row['qualification']['preload'];expected=dict(selected['environment'],LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
                    require(all(env.get(k.encode())==v.encode() for k,v in expected.items()) and
                        set(env)-{k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'},
                        'actual immutable environment differs')
                    mapped=runtime.executable_mappings(Path('/proc',str(ident['pid']),'maps').read_text());allowed=gen.committed(runtime.RESULT)['runtime_files']
                    require(all(p in allowed or p==str(Path(row['qualification']['binaries'][selected['arm']]['path']).resolve()) for p in mapped) and
                        any('libze_intel_gpu.so.1.17.39758' in p for p in mapped) and row['clock']['provider']['path'] in mapped,'actual native provider differs')
                    outcome['verified_request_boundaries']=request+1
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'benchmark teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=owners.Owner(root/'owned',persist)
        entry,out,_=owner.run('clock',[row['clock']['binary']['path']],row['clock']['environment'],P,
            wall=30,cpu=30,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=128<<20)
        require(owners.closed(entry) and entry['exit_code']==0,'pre-native clock failed')
        outcome['clock']=clock.evidence(out.read_text(),row['clock']['provider'])
        entry,_,_=owner.run('bench',selected['command'],selected['environment'],P,interaction=interaction,**row['budgets'])
        require(owners.closed(entry) and entry['exit_code']==0,'official benchmark/owner failed')
        result_file,=(root/'bench').glob('*/results.json');result=json.loads(result_file.read_text())
        outcome['requests']=result['results']['prefill_only']['requests'];rates(outcome['requests'])
        require(result['complete'] and not result['results']['failures'] and result['params']['mode']=='prefill-only' and
            result['params']['runs']==4 and result['params']['think'] is True,'official benchmark incomplete')
        sys.path.insert(0,str(P/'tools'));from strata_bench import verify_request_journal
        journal,=(root/'bench').glob('*/requests.jsonl');verified=verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'],'request journal incomplete')
        events=[json.loads(v) for v in journal.read_text().splitlines()];samples=[v['sample'] for v in events if v['kind']=='result']
        longs=[v for v in events if v['kind']=='result' and v['phase']=='prefill-only']
        require(len(samples)==5 and len(longs)==4 and [dict(v['sample'],run=v['run']) for v in longs]==outcome['requests'] and
            outcome.get('verified_request_boundaries')==4,'journal/request binding differs')
        claim=gen.native_end(root,dict(binary=row['qualification']['binaries'][selected['arm']],native_validation_probe=row['qualification']['preload']),
            dict(config=config(root)),row['qualification']['boot'],entry['pid'])
        text=(root/'engine.stderr').read_text();activation=validation.preload_evidence(text,claim['native']['pid'])
        boundary=re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n',text,re.M)
        require(boundary and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed',text[boundary.end():]),'runtime error after canary')
        require(not (root/'parameter.validation').exists() or (root/'parameter.validation').stat().st_size==0,'parameter validation warning')
        headers=list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',text,re.M))
        require(len(headers)==5 and all(tuple(map(int,h.groups()))==(s['prompt_tokens'],s['reused_tokens'],s['read_tokens'],s['generated_tokens'])
            for h,s in zip(headers,samples)),'native extents differ')
        for j in range(1,5):
            frame=text[headers[j-1].end():headers[j].start()];n=117488;prefix=n*2056
            route=re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=0 residual_staging=0 ',frame,re.M)
            require(route==[(str(n),str(n*20480),str(n),str(n*20480))] and
                'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920 mixed_fp16_bytes=0' in frame and
                headers[j].group().endswith(', 0 checkpoints'),'actual long GPU/BF16/mode10 dispatch differs')
            arithmetic=re.findall(r'^strata prefill: QSA BF16 operands, FP32 online state/accumulation; KV=(\S+) transient_bytes=(\d+); source=resident-Mode0 score_pitch=(\d+) operand_layout=(\S+) softmax=(\S+)$',frame,re.M)
            require(arithmetic==[('active-logical-prefix',str(prefix),'33','row-major','parallel-exp-ascending-sum')],'actual QSA math differs')
        require(not any(v in text for v in ('strata trace:','strata loan audit:','strata decode timing:',
            'strata UR memory:','strata serve: STATE_HASH','strata serve: SNAPSHOT_VERIFY')),'unexpected heavy observer')
        if row['processes']:
            require([s['request_sha256'] for s in outcome['requests']]==[s['request_sha256'] for s in row['processes'][0]['requests']],'request bodies changed')
        current(initial)
        if i==5:models.model_pins(config(root),row['models'],full=True)
        outcome.update(native_identity=claim['native'],journal_sha256=digest(journal),validation=activation,
            route=dict(requests=4,rows=117488,full_GPU_BF16=True,MoE_mode=10,QSA_prefix_bytes=117488*2056),
            host_probe_accepted=True,passed=True,complete=True)
    except BaseException as error:
        outcome['error']=repr(error)
        if (root/'engine.stderr').exists():outcome['error_excerpt']=(root/'engine.stderr').read_text()[-4096:]
    finally:
        if owner:validation.finish(outcome,owner,cursor,row['qualification']['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome.get('complete') and outcome.get('driver_after')==row['qualification']['driver'])
        persist();row['processes'].append(outcome);row['ownership_closed']=outcome.get('ownership_closed',False)
        row['stage']='collecting' if outcome['passed'] and i<5 else 'closed'
        if outcome['passed'] and i==5:row['analysis']=summarize(row['processes'])
        write(ROOT/'record.json',row);write(RESULT,compact(row))
    print(json.dumps({k:outcome.get(k) for k in ('arm','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return int(not outcome['passed'])


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['host'],['run']),'prepare|host|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']:prepare()
        else:
            record=json.loads((ROOT/'record.json').read_text())
            sys.exit(host(record) if sys.argv[1:]==['host'] else run(record))
