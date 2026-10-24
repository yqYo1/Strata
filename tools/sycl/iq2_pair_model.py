#!/usr/bin/env python3
"""Main-owned repeated IQ2 pair comparison on one qualified native binary.

The closed single-process pilot and its launcher are recoverable at bedfdf312.
Prepare once; commit the plan and each ordered process outcome before run.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import shutil
import socket
import statistics
import sys
import time

sys.dont_write_bytecode = True
import cpu_capacity as assets
import cpu_iq2 as component
import cpu_census_engine as census_run
import cpu_work_census as census
import generation_compare as comparison
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import query_profile as q
from build_engine import census_model_evidence

P, B, A = gen.P, gen.B, gen.A
ROOT = B/'iq2-pair-model-v2'
RESULT = A/'iq2-pair-model-current.json'
PILOT = dict(commit='bedfdf312fe3cf28842c78263f1e5b69cdf5ec5c',
    path=str(RESULT.relative_to(P)), sha256='336a926be0252d3c6f1ebddaafd6a227ac2af82042099ee534b4bb6d97879255')
WORKLOADS = ('regular-code', 'varied-prose')
ORDER = (0, 1, 1, 0, 0, 1)
PAIRS = ((0, 1), (3, 2), (4, 5))
STATIC_BIN = ('request','bin','H','FF','blob_bytes','gu_act','down_act',
              'gu_row','down_row','act_bytes','h_bytes','layer_count')


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def pilot():
    raw = gen.git('show', PILOT['commit']+':'+PILOT['path'])
    guards.require(hashlib.sha256(raw).hexdigest() == PILOT['sha256'], 'original IQ2 pilot not recoverable')
    return json.loads(raw)


def qualification():
    full, count = q.current_build()
    pool = gen.committed(component.RESULT)
    guards.require(pool['stage'] == 'closed' and pool['passed'] and pool['complete'] and pool['ownership_closed'] and
        not pool['active'] and pool['eligible_down_formats'] == [20,42] and
        all(d['performance_gate_passed'] for d in pool['decisions']) and
        pool['plan']['environment']['STRATA_IQ_SINGLE_EXACT'] == '0' and
        component.base.source_pin(P, pool['plan']['source']['revision'], component.PREFIXES) == pool['plan']['source'],
        'same native CPU source and actual-baseline format gates required')
    setup = pilot()
    proof = dict(original_inputs_recovered=count, build_receipt_sha256=gen.common.digest(q.BUILD_RECORD),
        shared_component_sha256=gen.common.digest(q.COMPONENT), pool_result_sha256=gen.common.digest(component.RESULT),
        pilot_reference=PILOT)
    census_model_evidence(full, gen.committed(q.COMPONENT), setup,
        proof['build_receipt_sha256'], proof['shared_component_sha256'])
    guards.require(setup['qualification']['pool_result_sha256'] == proof['pool_result_sha256'] and
        setup['analysis']['native_review']['worker_closed'] is True and
        setup['environment']['STRATA_IQ_SINGLE_EXACT'] == '0' and
        setup['environment']['STRATA_IQ2_PAIR'] == setup['environment']['STRATA_IQ2_PAIR_AUDIT'] == '1' and
        len(setup['iq2_dispatch']) == len(setup['census']) == 2, 'reviewed actual-baseline pilot required')
    cpus = component.base.machine()['physical_cpus']
    for audit, work in zip(setup['iq2_dispatch'], setup['census']):
        guards.require(audit['row_ops'] == 640*work['cpu_entries'] and
            audit['pair_rows'] == audit['nt1_rows'] == 640*sum(x['jobs_NT'][0] for x in work['format_bins'] if x['gu'] == 22) and
            audit['pair_rows'] > 0 and audit['active_threads'] == audit['pinned_threads'] == len(cpus) and
            audit['cpus'] == cpus, 'pilot actual IQ2 work/pins differ')
    return full, proof, setup


def plan():
    return dict(question='Does exact IQ2_S NT1 gate/up pairing improve repeated model generation on the same native binary?',
        order=list(ORDER), pairs=[list(p) for p in PAIRS], requests_per_process=2, independent_processes=6,
        workloads=list(WORKLOADS), workload_id=comparison.WORKLOAD_ID,
        constants='Same native binary/model/profile/runtime/config:131072 resident INT8 context,1256 expert slots,8192 prefill,5 workers+host,pcie0,adapt0,spec4,suffix0,caches0,greedy,no token graph/no coupled draft,Q8 MTP,packed bank. Only PAIR0/1 differs; SINGLE_EXACT0,AUDIT1,CENSUS1,TIMING1 and validation/trace identical before exec.',
        eligibility='Four official warmups then both thinking-on workloads per fresh native process. Each long request >=32768 fresh input,1024..16384 output including thinking,>=512 answer chars,normal stream/native exit,observed pair dispatch,six pinned physical CPUs and exact target census conservation. No output/hash equality requirement.',
        decision='For each workload and each process pair use candidate/base token throughput computed from actual tokens/engine milliseconds. Judge prefill and decode separately. Faster screen: all three ratios>1 and geomean>1.01; slower screen: all<1 and geomean<1/1.01; otherwise inconclusive. Nonregression:geomean>=0.98 and every pair>=0.95. Advancement to clean comparison needs both decode workloads nonregressing,at least one faster,and both prefill workloads nonregressing. Confirm MTP classes separately in both arms:all three code acceptances>=0.60 and prose<=0.40. No speed-based stopping.',
        stopping='Stop on first source/runtime/config/owner/fault/budget/native/request/dispatch/conservation failure. No retries,replaced samples or arm-order changes. Commit each process outcome before the next launch.',
        limits='Counter/census/clock/parameter-validation observers are charged equally in both arms. This is an observed same-native screen,not clean speed,observer overhead,hardware saturation,task/agent quality,physical262144/session/RESTORE or adoption qualification. Two baseline-first pairs and one candidate-first pair cover both directions without exact chronology balance.',
        retention='Keep twelve individual samples and compact format/NT/pair/clock/route/owner/fault evidence. Recover shared setup from pinned pilot Git reference. Successful logs,JIT and owner transcripts retire after each committed closed process; keep frozen generated inputs through cohort closure.')


def source_pins(revision):
    pins = gen.benchmark_pins(revision)
    for name in ('data/expert-profile.bin', 'tests/core/test_iq2_pair_model.py'):
        raw = gen.git('show', revision+':'+name)
        guards.require((P/name).read_bytes() == raw, 'uncommitted IQ2 comparison input: '+name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def config(setup, root):
    result = copy.deepcopy(setup['config'])
    args = result['args']; args[args.index('--expert-profile')+1] = str(root/'expert-profile.bin')
    result.update(log=str(root/'engine.stderr'), env={})
    return result


def environment(setup, root, arm):
    old_root = setup['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old_root, str(root)) for k,v in setup['environment'].items()}
    env['STRATA_IQ2_PAIR'] = str(arm)
    return env


def command(root, port):
    return ['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
        '--long-generation-only', '--think', '--runs', '1', '--profile-request-gates',
        '--workload-id', comparison.WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
        '--out', str(root/'bench')]


def current(record, full_models=False):
    full, proof, setup = qualification()
    guards.require(record['binary'] == full['binary'] and record['build_source_commit'] == full['source_commit'] and
        record['qualification'] == proof and record['plan'] == plan(), 'IQ2 comparison qualification/plan drift')
    pins = source_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['benchmark_fingerprint'] and
        gen.common.digest(gen.SETTINGS) == record['settings_sha256'], 'IQ2 comparison source/settings drift')
    guards.require(len(record['plans']) == 6, 'IQ2 comparison plan count')
    for i, row in enumerate(record['plans']):
        root = ROOT/str(i)
        guards.require(row['cohort'] == i and row['arm'] == ORDER[i] and row['root'] == str(root) and
            type(row['port']) is int and 0 < row['port'] < 65536 and row['command'] == command(root, row['port']) and
            json.loads((root/'config.json').read_text()) == config(setup, root) and
            set(row['generated_inputs']) == {str(root/n) for n in ('config.json','plan.json','expert-profile.bin')},
            'IQ2 comparison configuration/command drift')
        for path, sha in row['generated_inputs'].items():
            guards.require(gen.common.digest(path) == sha, 'IQ2 comparison generated input drift')
    guards.require(assets.machine_identity() == record['machine_identity'] and
        assets.fingerprint(assets.model_pins(setup['config'], setup['model_assets'], full=full_models)) ==
            record['model_assets_fingerprint'], 'IQ2 comparison model/platform drift')
    gen.runtime_assets_unchanged(record)
    return full, setup


def prepare():
    guards.require(not ROOT.exists() and gen.committed(RESULT) == pilot(), 'closed pilot and fresh single-use comparison root required')
    full, proof, setup = qualification()
    revision = gen.git('rev-parse', 'HEAD').decode().strip(); pins = source_pins(revision)
    models = assets.model_pins(setup['config'], setup['model_assets'], full=True)
    ROOT.mkdir(mode=0o700); plans=[]; ports=set()
    for i, arm in enumerate(ORDER):
        root = ROOT/str(i); root.mkdir(mode=0o700)
        for name in ('cache','tmp'): (root/name).mkdir()
        shutil.copyfile(P/'data/expert-profile.bin', root/'expert-profile.bin')
        old_profile = setup['config']['args'][setup['config']['args'].index('--expert-profile')+1]
        guards.require(gen.common.digest(root/'expert-profile.bin') == setup['generated_inputs'][old_profile], 'IQ2 profile differs')
        write(root/'config.json', config(setup, root))
        write(root/'plan.json', dict(binary=full['binary'], native_validation_probe=None))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        guards.require(port not in ports, 'duplicate comparison port; never retry root'); ports.add(port)
        plans.append(dict(cohort=i, arm=arm, root=str(root), port=port, command=command(root,port),
            generated_inputs={str(root/n):gen.common.digest(root/n) for n in ('config.json','plan.json','expert-profile.bin')}))
    record=dict(mode='engine-compare', stage='prepared', source_commit=revision, build_source_commit=full['source_commit'],
        binary=full['binary'], qualification=proof, plan=plan(), plans=plans, processes=[],
        native_validation_probe=None, runtime_asset_delta=setup['runtime_asset_delta'],
        model_assets_fingerprint=assets.fingerprint(models), source_count=len(pins), benchmark_fingerprint=assets.fingerprint(pins),
        machine_identity=assets.machine_identity(), settings_sha256=gen.common.digest(gen.SETTINGS),
        complete=False, passed=False, active=False, ownership_closed=True,
        clean_speed_qualified=False, hardware_saturation_qualified=False, model_adoption_qualified=False,
        model_hash_policy='Full hashes at preparation and after each process; original model identities checked before every launch.',
        budgets=dict(wall=3600, cpu=108000, rss_cap=96<<30, text_cap=64<<20, file_cap=64<<20, total_cap=64<<20))
    current(record); write(ROOT/'frozen-plan.json',record); write(RESULT,record)
    print('Prepared six independent same-native processes; commit before the first run.')


def work_evidence(text, samples, cfg, arm, root):
    timing=gen.decode_timing_evidence(text,samples)
    bindings=[dict(prompt_tokens=s['prompt_tokens'],windows=t['windows'],rows=t['work']['target_rows'],
        cpu_jobs=t['work']['cpu_unique_jobs'],cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run']*t['windows']) for s,t in zip(samples,timing)]
    args=cfg['args']; formats=census.manifest_formats(Path(args[args.index('--pack')+1])/'native_experts.txt')
    work=census.extract(text,bindings,formats,census.expected_membership(root/'expert-profile.bin'))
    paired=gen.iq2_pair_evidence(text,samples,arm,component.base.machine()['physical_cpus'])
    for audit, observed in zip(paired,work):
        guards.require(audit['row_ops'] == 640*observed['cpu_entries'] and
            audit['nt1_rows'] == 640*sum(b['jobs_NT'][0] for b in observed['format_bins'] if b['gu'] == 22),
            'IQ2 callback rows do not conserve actual target census')
    return paired, work, timing


def compact_work(work):
    result=copy.deepcopy(work)
    for row in result:
        row.pop('logical_job_blob_bytes',None)
        for b in row['format_bins']:
            for key in STATIC_BIN: b.pop(key,None)
    return result


def metric(ratios):
    guards.require(len(ratios)==3 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in ratios), 'three finite positive independent ratios required')
    geo=math.exp(statistics.mean(math.log(x) for x in ratios))
    label=('faster_screen' if all(x>1 for x in ratios) and geo>1.01 else
        'slower_screen' if all(x<1 for x in ratios) and geo<1/1.01 else 'inconclusive')
    return dict(candidate_base_ratios=ratios, geomean=geo, decision=label,
        nonregression=geo>=.98 and all(x>=.95 for x in ratios))


def summarize(processes):
    guards.require(len(processes)==6 and all(type(r['cohort']) is type(r['arm']) is int for r in processes) and
        [r['cohort'] for r in processes]==list(range(6)) and
        [r['arm'] for r in processes]==list(ORDER), 'complete frozen six-process order required')
    guards.require(len({(r['native_identity']['pid'],r['native_identity']['start_ticks']) for r in processes})==6,
        'six independent native processes required')
    for key in ('binary_sha256','build_source_commit','benchmark_fingerprint','qualification_sha256',
                'machine_identity_sha256','model_assets_fingerprint','observer_environment_sha256'):
        guards.require(all(r[key]==processes[0][key] for r in processes), 'cross-process identity changed: '+key)
    for row in processes:
        guards.require(all(row.get(k) is True for k in ('passed','complete','ownership_closed','boot_unchanged',
                'fault_probe_successful','effective_environment_verified')) and row['active'] is False and
            row['native_exit_code']==0 and row['fresh_faults']==[] and
            [s['workload'] for s in row['requests']]==list(WORKLOADS) and
            all(gen.sample_eligible(s) and s['prompt_tps_source']=='engine' and
                type(s['prompt_ms']) in (int,float) and math.isfinite(s['prompt_ms']) and s['prompt_ms']>0
                for s in row['requests']) and
            len(row['iq2_dispatch'])==len(row['census'])==len(row['routes'])==len(row['decode_timing'])==2 and
            row['warmups']['valid'] is True and row['warmups']['successful_warmups']==4 and
            row['warmups']['successful_long_requests']==2 and
            row['warmups']['request_sha256'][4:]==[s['request_sha256'] for s in row['requests']],
            'closed eligible process/requests required')
        for sample,route,audit,work in zip(row['requests'],row['routes'],row['iq2_dispatch'],row['census']):
            guards.require(audit['row_ops']==640*work['cpu_entries'] and audit['nt1_rows']>0 and
                audit['nt1_rows']==640*sum(b['jobs_NT'][0] for b in work['format_bins'] if b['gu']==22) and
                audit['pair_rows']==audit['nt1_rows']*row['arm'] and audit['cpus']==list(range(6)) and
                audit['active_threads']==audit['pinned_threads']==6 and
                route['actual_GPU_BF16_rows']==sample['read_tokens']-1 and route['actual_MoE_mode']==10 and
                route['restored_dense_layers']==route['private_bank']['actual_private_layers']==48 and
                route['cache_payload_and_map_match'] is True,'actual comparison dispatch/work/route differs')
    cases=[]
    for i,name in enumerate(WORKLOADS):
        speeds={}
        for phase,tokens,clock in [('prefill','read_tokens','prompt_ms'),('decode','generated_tokens','decode_ms')]:
            values=[1000*r['requests'][i][tokens]/r['requests'][i][clock] for r in processes]
            for a,b in PAIRS:
                guards.require(processes[a]['requests'][i]['request_sha256']==processes[b]['requests'][i]['request_sha256'], 'paired workload body changed')
            speeds[phase]=dict(metric([values[b]/values[a] for a,b in PAIRS]), individual_tok_s=values)
        acceptance={str(arm):[r['requests'][i]['acceptance_rate'] for r in processes if r['arm']==arm] for arm in (0,1)}
        confirmed=all(all(x>=.60 if i==0 else x<=.40 for x in values) for values in acceptance.values())
        cases.append(dict(workload=name,**speeds,acceptance_by_arm=acceptance,mtp_class_confirmed=confirmed))
    return dict(cases=cases, MTP_pair_confirmed=all(c['mtp_class_confirmed'] for c in cases),
        eligible_for_clean_comparison=all(c['mtp_class_confirmed'] and c['prefill']['nonregression'] and
            c['decode']['nonregression'] for c in cases) and any(c['decode']['decision']=='faster_screen' for c in cases),
        clean_speed_qualified=False, hardware_saturation_qualified=False, model_adoption_qualified=False,
        limits=plan()['limits'])


def run():
    record=gen.committed(RESULT)
    guards.require(record['stage'] in ('prepared','collecting') and not record['active'] and record['ownership_closed'] and
        len(record['processes'])<6 and all(r['passed'] for r in record['processes']), 'closed/failed or active IQ2 comparison')
    frozen=json.loads((ROOT/'frozen-plan.json').read_text())
    mutable={'stage','processes','complete','passed','active','ownership_closed','decision','error'}
    guards.require({k:v for k,v in record.items() if k not in mutable}==
        {k:v for k,v in frozen.items() if k not in mutable}, 'frozen IQ2 comparison changed')
    i=len(record['processes']); p=record['plans'][i]; root=Path(p['root']); arm=p['arm']
    guards.require(not (root/'record.json').exists(), 'existing process attempt; never retry')
    try: full,setup=current(record)
    except BaseException as error:
        record.update(stage='closed',error='Prelaunch admission: '+str(error),passed=False,complete=False)
        write(RESULT,record);raise
    cfg=config(setup,root);env=environment(setup,root,arm)
    boot,cursor=census_run.start_fault_scope();own=None
    row=dict(cohort=i,arm=arm,passed=False,complete=False,active=False,ownership_closed=False,
        binary_sha256=record['binary']['sha256'],build_source_commit=record['build_source_commit'],
        benchmark_fingerprint=record['benchmark_fingerprint'],qualification_sha256=assets.fingerprint(record['qualification']),
        machine_identity_sha256=assets.fingerprint(record['machine_identity']),
        model_assets_fingerprint=record['model_assets_fingerprint'],
        observer_environment_sha256=assets.fingerprint({k:v.replace(str(root),'{process_root}')
            for k,v in env.items() if k!='STRATA_IQ2_PAIR'}))
    def persist():
        row['active']=bool(own and own.active is not None)
        write(root/'record.json',dict(row,commands=own.commands if own else []))
    def interact(io,leader):
        for j,name in enumerate(WORKLOADS):
            for stage in ('begin','end'):
                while True:
                    line=io.line(3300)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==
                    dict(stage=stage,phase='long-generation/'+name,run=0), 'IQ2 official request gate differs')
                if stage=='begin':
                    native=json.loads((root/'engine-start.json').read_text())['native']; identity=supervision.identity(native['pid'])
                    guards.require(identity and identity['sid']==leader['pid'] and identity['start_ticks']==native['start_ticks'] and
                        Path('/proc',str(native['pid']),'exe').resolve()==Path(record['binary']['path']).resolve(), 'IQ2 native owner differs')
                    actual=dict(x.split(b'=',1) for x in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in x)
                    guards.require(all(actual.get(k.encode())==v.encode() for k,v in env.items()) and b'LD_PRELOAD' not in actual,
                        'IQ2 immutable native environment differs')
                    path,=(root/'bench').glob('*/requests.jsonl');events=[json.loads(s) for s in path.read_text().splitlines()]
                    guards.require(len(events)==10+2*j and events[-1]['kind']=='start' and events[-1]['phase']=='long-generation/'+name,
                        'IQ2 warmup/prior journal extent differs')
                    gen.warmup_reconciliation(events[:-1],WORKLOADS[:j])
                    samples=[e['sample'] for e in events[:-1] if e['kind']=='result'];text=(root/'engine.stderr').read_text(errors='replace')
                    guards.require(all(gen.sample_eligible(s) for s in samples[4:]), 'prior long request ineligible')
                    gen.request_evidence(text,samples);work_evidence(text,samples,cfg,arm,root)
                    guards.require(not re.search(comparison.FAILURE,text), 'IQ2 small/prior native check failed')
                    fault=census_run.fault_evidence(cursor,boot)
                    guards.require(fault['boot_unchanged'] and not fault['fresh_faults'], 'fresh IQ2 fault before long request')
                    row.update(native_identity=native,effective_environment_verified=True);persist()
                io.send(b'continue\n')
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();guards.require(time.monotonic()<deadline, 'IQ2 benchmark teardown budget')
            if select.select([io.proc.stdout],[],[],.05)[0]: io.read();io.pending.clear()
        io.exit(1)
    own=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=own.run('bench',p['command'],env,P,interaction=interact,**record['budgets'])
        paths=list((root/'bench').glob('*/results.json'))
        if paths: row['requests']=gen.collect_samples(json.loads(paths[0].read_text())['results'].get('long_generation',{}))
        guards.require(supervision.closed(entry) and entry['exit_code']==0 and len(paths)==1, 'IQ2 official benchmark failed')
        samples,journal,events=comparison.official(root,json.loads(paths[0].read_text()))
        native=gen.native_end(root,record,dict(config=cfg),boot,entry['pid']);text=(root/'engine.stderr').read_text(errors='replace')
        guards.require(not re.search(comparison.FAILURE,text), 'IQ2 native error after long generation')
        paired,work,timing=work_evidence(text,[e['sample'] for e in events if e['kind']=='result'],cfg,arm,root)
        for s in samples: s.pop('client_interval',None)
        # Packed-bank layout stays enabled in both IQ2 CPU arms.
        row.update(requests=samples,warmups=journal,routes=comparison.native_routes(text,events,1),
            iq2_dispatch=paired[4:],census=compact_work(work[4:]),decode_timing=timing[4:],native_exit_code=native['exit_code'])
        current(record,full_models=True);row.update(passed=True,complete=True)
    except BaseException as error:
        row['error']=str(error)
        if (root/'engine.stderr').exists(): row['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row['ownership_closed']=own.active is None and all(supervision.closed(c) for c in own.commands)
        try: row.update(census_run.fault_evidence(cursor,boot))
        except Exception as error: row.update(fault_probe_error=str(error),passed=False)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row.get('boot_unchanged') and
            row.get('fault_probe_successful') and row.get('fresh_faults')==[])
        persist();record['processes'].append(row)
        record.update(active=row['active'],ownership_closed=row['ownership_closed'],stage='collecting' if row['passed'] else 'closed')
        if len(record['processes'])==6 and row['passed']:
            try: record['decision']=summarize(record['processes']);record.update(stage='closed',passed=True,complete=True)
            except Exception as error: record.update(stage='closed',error=str(error),passed=False,complete=False)
        write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ('cohort','arm','passed','complete','ownership_closed','error')}))
    return 0 if row['passed'] and not record.get('error') else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__': sys.exit(main())
