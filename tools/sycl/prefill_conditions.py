#!/usr/bin/env python3
"""Main-owned, bounded original6e40 hardware-conditions diagnostic.

Repository LICENSE. Reuses our historical-crossover ownership/request pattern,
not third-party application code. No engine rebuild, setting change or adoption.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_capacity as models
import generation_engine as gen
import measurement_owner as owners
import runtime_probe as runtime

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'prefill-conditions-117k-v1'
RESULT = A / 'prefill-resources-current.json'
SCOPE = 'execution_conditions'
TEMPLATE = A / 'phase-prefill-speed-current.json'
HISTORY = A / 'prefill-runtime-comparison-current.json'
READER = P / 'bench/sycl/client_resources.py'
EXTRA = ('bench/sycl/client_resources.py','bench/sycl/hardware_conditions.py',
         'tests/core/test_client_resources.py')
require, digest, write = runtime.require, runtime.digest, runtime.write


def pins(revision):
    found=gen.benchmark_pins(revision)
    for name in EXTRA:
        raw=gen.git('show',revision+':'+name)
        require((P/name).read_bytes()==raw,'commit exact condition input:'+name)
        found[name]=hashlib.sha256(raw).hexdigest()
    return found


def scope():
    return gen.committed(RESULT)[SCOPE]


def compact(row):
    return {k:v for k,v in row.items() if k not in ('source_pins','models')}


def save(row):
    container=json.loads(RESULT.read_text());container[SCOPE]=compact(row);write(RESULT,container)


def current(row):
    require(scope()==compact(row),'commit frozen condition plan/outcome')
    require(pins(row['source_commit'])==row['source_pins'],'condition source changed')
    require(digest(TEMPLATE)==row['template_sha256'] and digest(HISTORY)==row['history_sha256'] and
            digest(runtime.RESULT)==row['runtime_sha256'],'qualified reference changed')
    full=runtime.current(gen.committed(runtime.RESULT))
    require(full['binary']==row['binary'],'original selected binary changed')
    for name,sha in row['generated_inputs'].items():require(digest(name)==sha,'generated input changed')
    models.model_pins(row['configuration'],row['models'])
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'],'boot changed')
    return gen.committed(runtime.RESULT)


def prepare():
    container=gen.committed(RESULT)
    require(not ROOT.exists() and SCOPE not in container,'fresh diagnostic only;no retry')
    full,proof=runtime.selected.baseline();rt=gen.committed(runtime.RESULT);runtime.current(rt)
    history=gen.committed(HISTORY)['historical_regression_audit']['userland_crossover']
    require(history['stage']=='closed' and len(history['processes'])==6 and
            all(p['passed'] and p['ownership_closed'] for p in history['processes']), 'close/replay crossover first')
    config=copy.deepcopy(history['configuration']);config['log']=str(ROOT/'engine.stderr')
    original=json.loads(gen.git('show','1e676d554:'+str(TEMPLATE.relative_to(P))))
    model_pins=models.model_pins(config,original['model_assets'])
    env=gen.committed(TEMPLATE)['environment_template'];old=env['STRATA_MEASUREMENT_ROOT']
    env={k:v.replace(old,str(ROOT)) for k,v in env.items()}
    require(not any(k in env for k in ('LD_PRELOAD','STRATA_TRACE','STRATA_RESOURCE_CONDITIONS',
                                     'ZE_ENABLE_ALT_DRIVERS')),'undeclared observer in template')
    driver=next(p for p in rt['runtime_files'] if '/libze_intel_gpu.so.' in p)
    env.update(STRATA_TRACE='1',STRATA_RESOURCE_CONDITIONS='1',ZE_ENABLE_ALT_DRIVERS=driver,
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'validation.log'))
    preload=rt['preload'];preload=dict(library=preload['library'],loader=dict(
        path=preload['loader'],sha256=rt['runtime_files'][preload['loader']]))
    source=gen.git('rev-parse','HEAD').decode().strip()
    row=dict(stage='prepared',source_commit=source,source_pins=pins(source),binary=full['binary'],
        original_build_proof=proof,boot=rt['boot'],driver=rt['driver'],configuration=config,environment=env,
        template_sha256=digest(TEMPLATE),history_sha256=digest(HISTORY),runtime_sha256=digest(runtime.RESULT),
        models=model_pins,model_fingerprint=hashlib.sha256(json.dumps(model_pins,sort_keys=True).encode()).hexdigest(),
        preload=preload,resource_collection=dict(source=dict(path=str(READER),sha256=digest(READER)),seconds=900,interval=1),
        requests_expected=history['request_hashes'][:2],generated_inputs={},active=False,ownership_closed=True,
        passed=False,complete=False,clean_speed_qualified=False,hardware_saturation_qualified=False,
        question='Which loaded GPU clocks/throttles/power/temperature,CPU reported clocks/CPU time and paging '
            'accompany the original6e40 historical speed decline? No counterfactual hardware control.',
        plan='One owned process,one short warmup,two fresh117489-input/24-output thinking-on requests through '
            'official bench.sh. First use and subsequent request separate. Original1256decode layout,8192chunks, '
            '131072INT8 context,BF16/vLLM/mode10,384packed expert bank,Q8MTP and safety settings unchanged. '
            'Reuse exact native/runtime qualification;collector is CPU-only. Run committed synthetic observer controls '
            'before model exec. Stop first source/provider/owner/stream/coverage/fault failure,no retry.',
        interpretation='Read-only1Hz diagnostic plus STRATA_TRACE1,charged parameter validation and retained-loader '
            'preload. No clean-speed,observer-overhead,causality or saturation claim. Require closed collector and '
            '>=10 wholly contained hardware samples per pre-output window,>=95% primaryGT clock coverage. '
            'Keep zero frequency distinct from missing;counter reset invalidates energy averages. CPU scaling_cur_freq '
            'is reported,not APERF/MPERF effective frequency. Global host/device signals can include other work.',
        retention='Retain two individual requests,compact GPU/CPU/paging/energy/coverage outcomes,source and '
            'native/runtime identities and decision. Raw timeline/JIT/logs live only through replay and attribution '
            'of this diagnostic;retire after that boundary. No tensors/fullAPI trace/new native generation.',
        budgets=dict(wall=1800,cpu=72000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20))
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir()
    write(ROOT/'config.json',config);write(ROOT/'plan.json',dict(binary=row['binary'],native_validation_probe=preload,
        resource_collection=row['resource_collection']))
    row['generated_inputs']={str(ROOT/name):digest(ROOT/name) for name in ('config.json','plan.json')}
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    row['command']=['/bin/sh',str(P/'bench.sh'),'--config',str(ROOT/'config.json'),'--port',str(port),
        '--prefill-only','--prefill-tokens','98304','--think','--runs','2','--profile-request-gates',
        '--workload-id','native_gdn_20261018','--fail-fast-engine-error','--ready-timeout','600','--out',str(ROOT/'bench')]
    write(ROOT/'record.json',row);save(row)
    print('Prepared;commit exact generated diagnostic plan before tests/model execution.',flush=True)


def compact_conditions(summary):
    # Drop static repeated PCI/CPU policy cells after checking changes; preserve
    # the useful gauges/counters rather than every sampled state or histogram.
    gauges=summary.pop('gauges');text=summary.pop('text')
    summary['GPU']={k:v for k,v in gauges.items() if not k.startswith('cpu/')}
    summary['CPU_reported_kHz']={k:v for k,v in gauges.items() if k.endswith('/scaling_cur_freq')}
    summary['throttle_reasons']={k:v for k,v in text.items() if k.endswith('/throttle/reasons')}
    policies={k:sorted(v['counts']) for k,v in text.items() if k.startswith('cpu/')}
    summary['CPU_policies']={name:sorted(set(v[0] for k,v in policies.items() if k.endswith('/'+name) and len(v)==1))
        for name in ('scaling_driver','scaling_governor','energy_performance_preference')}
    summary['changed_text_sources']={k:v for k,v in text.items() if len(v['counts'])>1}
    summary['PCI_link_observations']={k:sorted(v['counts']) for k,v in text.items() if k.startswith('pci/')}
    return summary


def run(row):
    rt=current(row);require(row['stage']=='prepared' and not (ROOT/'owned').exists(),'single-use diagnostic')
    initial=copy.deepcopy(row);owner=None;outcome=dict(host_probe_accepted=False,complete=False)
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist():
        write(ROOT/'live.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def interaction(io,leader):
        for request in range(2):
            for stage in ('begin','end'):
                while True:
                    line=io.line(1200)
                    if line.startswith('STRATA_PROFILE_GATE '):break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='prefill-only',run=request),'request gate differs')
                require(not (ROOT/'validation.log').exists() or not (ROOT/'validation.log').stat().st_size,'parameter warning')
                if stage=='begin':
                    start=json.loads((ROOT/'engine-start.json').read_text());pid=start['native']['pid'];ident=owners.identity(pid)
                    require(ident and ident['start_ticks']==start['native']['start_ticks'] and ident['sid']==leader['pid'] and
                        str(Path('/proc',str(pid),'exe').resolve())==str(Path(row['binary']['path']).resolve()),'native identity differs')
                    actual=dict(v.split(b'=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
                    expected=dict(row['environment'],LD_PRELOAD=row['preload']['library']['path'],
                        STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'])
                    require(all(actual.get(k.encode())==v.encode() for k,v in expected.items()) and
                        set(actual)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment differs')
                    paths=runtime.executable_mappings(Path('/proc',str(pid),'maps').read_text())
                    allowed=set(rt['runtime_files'])|{str(Path(row['binary']['path']).resolve())}
                    require(paths<=allowed and row['environment']['ZE_ENABLE_ALT_DRIVERS'] in paths,'native provider differs')
                    outcome['verified_request_boundaries']=request+1
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'diagnostic teardown deadline')
            if select.select([io.proc.stdout],[],[],.05)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=owners.Owner(ROOT/'owned',persist)
        entry,_,_=owner.run('host-controls',['/usr/bin/python3','-B',str(P/'tests/core/test_client_resources.py')],
            dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1'),P,
            wall=60,cpu=60,rss_cap=1<<30,text_cap=1<<20,file_cap=1<<20,total_cap=64<<20)
        require(owners.closed(entry) and entry['exit_code']==0,'observer host controls failed')
        outcome['host_controls_passed']=True
        entry,_,_=owner.run('bench',row['command'],row['environment'],P,interaction=interaction,**row['budgets'])
        require(owners.closed(entry) and entry['exit_code']==0,'official diagnostic/owner failed')
        rf,=(ROOT/'bench').glob('*/results.json');result=json.loads(rf.read_text());samples=result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples)==2 and
            outcome.get('verified_request_boundaries')==2,'official diagnostic incomplete')
        for s,h in zip(samples,row['requests_expected']):
            require(s['ok'] and s['eligible'] and s['stream_complete'] and s['thinking_requested'] and
                s['reasoning_observed'] and s['request_sha256']==h and
                s['actual_prompt_tokens']==s['actual_read_tokens']==117489 and s['actual_reused_tokens']==0 and
                s['generated_tokens']==24 and s['prompt_tps_source']=='engine' and
                10000<=s['prompt_ms']<1500000,'workload/clock differs')
        sys.path.insert(0,str(P/'tools'));from strata_bench import verify_request_journal
        jf,=(ROOT/'bench').glob('*/requests.jsonl');journal=verify_request_journal(jf.parent)
        events=[json.loads(v) for v in jf.read_text().splitlines()]
        require(journal['valid'] and journal['complete'] and
            [dict(v['sample'],run=v['run']) for v in events if v['kind']=='result' and v['phase']=='prefill-only']==samples,'journal binding differs')
        native_plan=json.loads((ROOT/'plan.json').read_text())
        native=gen.native_end(ROOT,native_plan,dict(config=row['configuration']),row['boot'],entry['pid'])
        log=(ROOT/'engine.stderr').read_text();runtime.validation.preload_evidence(log,native['native']['pid'])
        require(log.count('strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16')==2 and
            log.count('strata MoE window configuration: requested_mode=10 effective_mode=10')==2 and
            not any(v in log for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),'native route/fault differs')
        sys.path.insert(0,str(READER.parent));import client_resources,hardware_conditions
        captures=[json.loads(v) for v in (ROOT/'client-resources.jsonl').read_text().splitlines()]
        analyses=[]
        for s in samples:
            span=s['client_interval']
            base=client_resources.summarize_capture(captures,native['native'],row['boot'],span,before_only=True)
            conditions=hardware_conditions.summarize(captures,span['request_begin_monotonic_ns'],span['first_output_monotonic_ns'])
            analyses.append(dict(client=base,conditions=compact_conditions(conditions)))
        current(initial);models.model_pins(row['configuration'],row['models'],full=True)
        fields=('run','request_sha256','actual_prompt_tokens','actual_read_tokens','actual_reused_tokens',
                'generated_tokens','prompt_ms','thinking_requested','client_interval')
        outcome.update(complete=True,host_probe_accepted=True,
            requests=[{k:s[k] for k in fields} for s in samples],tok_s=[117489000/s['prompt_ms'] for s in samples],
            observations=analyses,collector_source=captures[0]['conditions'],
            collector_terminal=captures[-1],journal_sha256=digest(jf),capture_sha256=digest(ROOT/'client-resources.jsonl'),
            native_identity=native['native'],collector_identity=native['resource_collector'])
    except BaseException as error:
        outcome['error']=repr(error)
        if (ROOT/'engine.stderr').exists():outcome['error_excerpt']=(ROOT/'engine.stderr').read_text()[-4096:]
    finally:
        if owner:runtime.validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete']);persist()
        row.update(stage='closed',active=False,ownership_closed=outcome.get('ownership_closed',True),
            complete=outcome['complete'],passed=outcome['passed'],outcome=outcome)
        write(ROOT/'record.json',row);save(row)
    print(json.dumps({k:outcome.get(k) for k in ('passed','complete','ownership_closed','tok_s','error')}),flush=True)
    return int(not outcome['passed'])


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare();return 0
        if sys.argv[1:]==['run']:return run(json.loads((ROOT/'record.json').read_text()))
        raise RuntimeError('use prepare or run')


if __name__=='__main__':sys.exit(main())
