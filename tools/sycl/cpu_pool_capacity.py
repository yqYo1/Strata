#!/usr/bin/env python3
"""Clean CPU-pool capacity diagnostic on the retained native engine.

Main-authored from owned Strata benchmark/controller interfaces; repository
LICENSE. No XeStrata source, new native build, profiler or inference adoption.
"""
import copy
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import re
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import compare_coherent_generations as preceding
import decode_kv_bench as kv

P, B, RESULT = kv.P, kv.B, kv.RESULT
ROOT, KEY = B/'cpu-pool-capacity-v1', 'cpu_pool_capacity_v1'
THIS, TEST = 'tools/sycl/cpu_pool_capacity.py', 'tools/test_cpu_pool_capacity.py'
PYTHON, ARMS = kv.PYTHON, kv.ARMS
# Preserve the existing long-generation bodies; this is a new CPU question.
WORKLOAD_ID = kv.WORKLOAD_ID
OBSERVERS = preceding.OBSERVERS | {THIS}
UNIT_TESTS = ['tools.test_cpu_pool_capacity.CapacityTests']
require, sha, signature, compact = kv.require, kv.sha, kv.signature, kv.compact


def save(row):
    kv.core.write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    kv.core.write(RESULT, data)


def pinned(revision):
    values = kv.core.pins(revision)
    for name in (THIS, TEST):
        raw = kv.shared.gen.git('show', revision+':'+name)
        require(raw == (P/name).read_bytes(), 'commit exact capacity controller/tests')
        values[name] = sha(P/name)
    return values


def workers(plan):
    require(type(plan['index']) is int and 0 <= plan['index'] < len(ARMS)
        and plan['kernel'] == ARMS[plan['index']]
        and type(plan['pool_workers']) is int
        and plan['pool_workers'] == (5 if plan['kernel'] == 'reference' else 1),
        'unplanned worker case')
    return plan['pool_workers']


def configuration(row, plan):
    value = kv.configuration(row,plan); args = value['args']
    require(args.count('--pool-workers') == 1
        and args[args.index('--pool-workers')+1] == '5'
        and '--no-host-worker' not in args
        and value['exe'] == str(P/'tools/sycl/engine_once.py'),
        'preserve supervised native entry and host participation')
    args[args.index('--pool-workers')+1] = str(workers(plan))
    return value


def environment(row, plan):
    workers(plan); env = kv.base.environment(row,plan)
    env.update(STRATA_SYCL_DECODE_KV_COALESCED='0',STRATA_SYCL_KV_COPY_GROUPS='0')
    return env


def invariant(row, plan):
    value = dict(config=configuration(row,plan),environment=environment(row,plan))
    args=value['config']['args'];args[args.index('--pool-workers')+1]='<workers>'
    return signature(json.dumps(value,sort_keys=True).replace(plan['root'],'<process>'))


def pool_binding(text, expected):
    require(type(expected) is int and expected in (1,5),'unplanned actual worker count')
    lines = [v for v in text.splitlines() if v.startswith('strata generate:')
        and 'expert-pool workers' in v]
    require(lines == [f'strata generate: {expected} expert-pool workers + the host thread'],
        'actual native worker/host binding differs')
    return dict(workers=expected,host_participates=True,potential_participants=expected+1,
        source_defined_row_tasks_per_native_GU_or_Down_phase=3*(expected+1),
        source_contract='Native T1 and wider windows use row-split multi,not canonical inline.',
        actual_thread_service_or_memory_traffic_measured=False)


def basis(expected=None):
    native = kv.device.native(expected,observers=OBSERVERS,retained_frontend=kv.FRONTEND)
    prior = json.loads((preceding.ROOT/'record.json').read_text())
    require(kv.device.common.validation.committed(RESULT,(preceding.KEY,)) == compact(prior)
        and prior['stage']=='closed' and not prior['active']
        and all(prior[k] for k in ('passed','complete','ownership_closed'))
        and len(prior['processes'])==12
        and prior['decision']==preceding.decision(prior['processes'])
        and all(p['passed'] and p['complete'] and p['ownership_closed']
            and p['native_exit_code']==0 and not p['fresh_faults'] for p in prior['processes']),
        'closed unchanged cross-build comparison required,including its original decision')
    require(native['binaries']['strata']==prior['native_builds']['candidate']['binary']
        and native['native_source_commit']==prior['native_builds']['candidate']['native_source_commit']
        and native['native_validation_probe']==prior['native_validation_probe'],
        'same qualified native generation/provider required')
    for key in (kv.device.KEY,kv.model.KEY):
        gate=kv.device.common.validation.committed(RESULT,(key,))
        require(gate['stage']=='closed' and not gate['active']
            and all(gate[k] for k in ('passed','complete','ownership_closed'))
            and gate['native_source_commit']==native['native_source_commit'],
            'closed same-build safety/handoff qualification required')
    manifest=prior['runtime_manifest'];files=json.loads(Path(manifest['path']).read_text())
    require(sha(manifest['path'])==manifest['sha256']
        and all(sha(p)==h for p,h in files.items())
        and kv.build.guards.driver_guard()==prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==prior['boot'],
        'qualified runtime/driver/boot changed')
    kv.shared.models.model_pins(prior['configuration'],prior['model_assets'],full=False)
    return native,prior,files


def current(row):
    require(kv.device.common.validation.committed(RESULT,(KEY,))==compact(row)
        and pinned(row['source_commit'])==row['sources']
        and all(sha(P/p)==h for p,h in row['policies'].items()),
        'freeze committed capacity source,plan and evidence')
    native,prior,files=basis(row['native_observer_admission'])
    require(row['binary']==native['binaries']['strata']
        and row['native_source_commit']==native['native_source_commit']
        and row['preceding_prefill_sha256']==signature(compact(prior))
        and all(row[k]==prior[k] for k in ('configuration','environment_template','template',
            'boot','driver','runtime_manifest','model_assets','native_validation_probe')),
        'native generation or preceding evidence changed')
    contract,manifest=kv.runner.model.python_manifest()
    require(contract==row['python_contract'] and manifest['selected']==row['python_selected']
        and sha(row['template']['path'])==row['template']['sha256']
        and kv.build.supervision.artifact_bytes(ROOT)<=row['whole_bytes']
        and len(row['plans'])==len(ARMS),'frontend/template/output budget differs')
    for index,plan in enumerate(row['plans']):
        root=Path(plan['root']);workers(plan)
        require(plan['index']==index and root==ROOT/str(index) and plan['heads']==1 and plan['adapt_swaps']==0
            and plan['command']==command(plan)
            and invariant(row,plan)==row['invariant_signature']
            and json.loads((root/'config.json').read_text())==configuration(row,plan)
            and json.loads((root/'plan.json').read_text())==dict(binary=row['binary'],
                native_validation_probe=row['native_validation_probe'])
            and all(sha(root/p)==h for p,h in plan['generated'].items())
            and kv.runner.comparison.fingerprint(environment(row,plan))==plan['environment_sha256'],
            'frozen worker/native launcher/configuration/environment changed')
    if row['host_passed']:
        unit=row.get('host_unit_evidence',{})
        require(unit.get('passed') and unit.get('ownership_closed') and unit.get('exit_code')==0
            and unit.get('tests')==UNIT_TESTS and unit.get('source_commit')==row['source_commit'],
            'closed source-pinned capacity controls required')
    return files,manifest


def command(plan):
    root=Path(plan['root'])
    return ['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),
        '--port',str(plan['port']),'--long-generation-only','--think','--runs','1',
        '--profile-request-gates','--workload-id',WORKLOAD_ID,'--fail-fast-engine-error',
        '--ready-timeout','600','--out',str(root/'bench')]


def decision(processes):
    require(len(processes)==len(ARMS)
        and all(p['kernel']==ARMS[i] and p['passed'] and p['complete']
            and p['ownership_closed'] and p['native_exit_code']==0
            and not p['fresh_faults'] and len(p['samples'])==2
            and len(p['API_native_binding'])==2 for i,p in enumerate(processes)),
        'complete ordered capacity cohort required')
    identities=[(p['native_identity']['pid'],p['native_identity']['start_ticks']) for p in processes]
    require(len(set(identities))==len(ARMS),'independent native processes required')
    result={}
    for wi,name in enumerate(kv.WORKLOADS):
        pairs=[]
        for i in (0,2,4):
            group={p['kernel']:p for p in processes[i:i+2]}
            for arm,p in group.items():
                expected=5 if arm=='reference' else 1
                require(p['API_native_binding'][wi]['cpu_pool']['workers']==expected,
                    'timed actual pool count differs')
                s=p['samples'][wi]
                require(type(s['native_generated_tokens']) is int and s['native_generated_tokens']>=1024
                    and all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>0
                        for k in ('exact_native_decode_tps','prompt_ms','ttft_s')),
                    'ineligible sustained capacity sample')
            work={}
            for arm,p in group.items():
                n=p['samples'][wi]['native_generated_tokens'];routes=p['API_native_binding'][wi]['routes']
                entries=sum(v['CPU_entries'] for v in routes);jobs=sum(v['CPU_unique_jobs'] for v in routes)
                require(0<jobs<=entries,'CPU work conservation differs')
                work[arm]=dict(CPU_entries=entries,CPU_unique_jobs=jobs,emitted_tokens=n,
                    entries_per_emitted=entries/n,jobs_per_emitted=jobs/n)
            r,c=[group[a]['samples'][wi] for a in ('reference','candidate')]
            require(r['request_sha256']==c['request_sha256'],'matched request bodies required')
            pairs.append(dict(worker1_over_worker5_TG=c['exact_native_decode_tps']/r['exact_native_decode_tps'],
                prompt_time=c['prompt_ms']/r['prompt_ms'],TTFT=c['ttft_s']/r['ttft_s'],
                reference_tokens=r['native_generated_tokens'],candidate_tokens=c['native_generated_tokens'],
                reference_acceptance=r['acceptance_rate'],candidate_acceptance=c['acceptance_rate'],
                CPU_work=work,worker1_over_worker5_jobs_per_emitted=
                    work['candidate']['jobs_per_emitted']/work['reference']['jobs_per_emitted']))
        ratios=[p['worker1_over_worker5_TG'] for p in pairs]
        reference_acceptance=[p['samples'][wi]['acceptance_rate'] for p in processes if p['kernel']=='reference']
        classified=all(v>=.60 for v in reference_acceptance) if name=='regular-code' else all(v<=.40 for v in reference_acceptance)
        result[name]=dict(pairs=pairs,reference_MTP_class_confirmed=classified,geometric_mean_TG_ratio=math.exp(statistics.mean(math.log(v) for v in ratios)),
            fewer_workers_repeatedly_slower=all(v<1 for v in ratios),
            fewer_workers_repeatedly_faster=all(v>1 for v in ratios),
            mixed_or_equal_directions=not(all(v<1 for v in ratios) or all(v>1 for v in ratios)))
    return dict(workloads=result,inference_adopted=False,physical_saturation_qualified=False,
        limits='CPU-pool parallelism/scheduling sensitivity,not arithmetic-versus-DRAM attribution. No rejected prefill screen is revised. Task/agent,physical262144/RESTORE,ordinary teardown and70TG remain unqualified.')


def context():
    spec=importlib.util.spec_from_file_location('capacity_owned_driver',kv.__file__)
    d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    for key,value in dict(ROOT=ROOT,KEY=KEY,THIS=THIS,ARMS=ARMS,WORKLOAD_ID=WORKLOAD_ID,
            current=current,configuration=configuration,environment=environment,
            invariant=invariant,decision=decision,save=save).items():setattr(d,key,value)
    def api_window(text,sample,row,arm):
        index=len(row['processes']);plan=row['plans'][index]
        out=kv.api_window(text,sample,row,'reference')
        out['cpu_pool']=pool_binding((Path(plan['root'])/'engine.stderr').read_text(),workers(plan))
        return out
    d.api_window=api_window
    return d


def host():
    d=context();kv.runner.comparison.host(d)
    row=json.loads((ROOT/'record.json').read_text())
    owner=kv.build.supervision.Owner(ROOT/'unit')
    evidence=dict(passed=False,source_commit=row['source_commit'],tests=UNIT_TESTS)
    try:
        entry,out,err=owner.run('unit',[PYTHON,'-B','-m','unittest',*UNIT_TESTS],
            environment(row,row['plans'][0]),P,wall=90,cpu=60,rss_cap=1<<30,
            text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
        text=out.read_text()+err.read_text()
        require(kv.build.supervision.closed(entry) and entry['exit_code']==0
            and re.search(r'Ran 3 tests in [0-9.]+s',text) and text.rstrip().endswith('OK'),
            'capacity controls failed')
        evidence.update(passed=True,exit_code=0)
    except BaseException as error:evidence['error']=repr(error)
    finally:
        evidence['ownership_closed']=owner.active is None and all(kv.build.supervision.closed(v) for v in owner.commands)
        row.update(host_unit_evidence=evidence,host_passed=bool(evidence['passed'] and evidence['ownership_closed']))
        if not row['host_passed']:row.update(stage='failed',passed=False,complete=False)
        save(row)
    require(row['host_passed'],'capacity host controls or closure failed')


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh committed CPU question only')
    native,prior,_=basis();contract,manifest=kv.runner.model.python_manifest()
    row={k:copy.deepcopy(prior[k]) for k in ('configuration','environment_template','template',
        'boot','driver','runtime_manifest','model_assets','native_validation_probe')}
    revision=kv.shared.gen.git('rev-parse','HEAD').decode().strip()
    row.update(source_commit=revision,sources=pinned(revision),
        policies={p:sha(P/p) for p in ('AGENTS.md','docs/ARTIFACT_RETENTION.md')},
        binary=native['binaries']['strata'],native_source_commit=native['native_source_commit'],
        native_observer_admission=native['observer_admission'],preceding_prefill_sha256=signature(compact(prior)),
        python_contract=contract,python_selected=manifest['selected'],insertions=kv.insertion_recipe(row['configuration']),
        stage='prepared',plans=[],processes=[],host_passed=False,passed=False,complete=False,
        active=False,ownership_closed=True,whole_bytes=448<<20,
        budgets=dict(wall=5400,cpu=36000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        plan=dict(question='Does reducing the native CPU pool from5workers+host to1worker+host change clean sustained decode?',
            evidence='Current closed model CPUentries/jobs and source-bound demand motivate a capacity discriminator;physical saturation is unknown. Native T1/T4 both use row-split multi,with18versus6tasks per GU/Down phase;this is not canonical inline.',
            order='Six fresh processes worker5,1,1,5,5,1. Each uses four official warmups then regular-code and varied-prose once,thinking on,cap16384. Existing fixed decode_kv_20261114 bodies retained. Commit each closed process before advancing.',
            constants='Same330405/1eec13ELFs,safety/quiet retained preload,262144logical/20480residentK8V8,1256static slots,pcie0/adapt0,spec4,8192throughput BF16/largeGU/bulk1/staging0,head2/dense1. KVcoalesced0/copygroups0 in both. Only --pool-workers5/1 changes;host remains enabled and startup footer/actual argv are checked.',
            acceptance='All12long requests:>=32768fresh input,>=1024native generated including thought and an answer,normal SSE/durable journals,valid TTFT,mixed actual code/prose acceptance retained. Preserve API/native segment/count/draft/rounded-clock conservation,actual scalar decode4,typed long GPU prefill,zero expert copies and CPUjobs/entries. Exact committed controller/tests/native sources/recipe/census/ELFs/providers/Python/configuration/models;normal0/reaped empty owner,quiet validation/no freshfault/unchangedboot-driver. Reuse preceding final15full asset hashes only after identity checks;rehash after last process.',
            decision='Retain every sample and all three adjacent worker1/worker5 TG,prompt-time andTTFT ratios per workload;report geometric means and all three directions. Allthree TG<1 supports sensitivity to reducing CPU-pool parallelism;allthree>1 supports a scheduling/partitioning follow-up;mixed/equal is inconclusive. No minimum effect threshold,posthoc exclusions or extra samples. Reconcile actual jobs/entries/tokens/acceptance before causal claims;this does not distinguish arithmetic,cache/DRAM or synchronization and does not adopt worker1.',
            stopping='First input/provider/owner/count/clock/route/worker-footer/validation/fault/budget/eligibility failure stops;exclusive attempt before admission,no retry or refreshed pins.',
            retention='No new native generation. Keep compact individual samples/API-native work/source/configuration/decision;successful raw/cache ends at verified closure. Keep14508fallback until a qualified replacement and330405for this named CPUdiagnostic;three legacy provenance holds remain before another engine build. Previous failed preservation screens remain unchanged. No full-model quality,agent,physical262144/RESTORE,ordinary teardown,hardware saturation or throughput-target qualification.'))
    ROOT.mkdir(mode=0o700)
    for i,arm in enumerate(ARMS):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for name in ('tmp','cache'):(root/name).mkdir()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=i,kernel=arm,heads=1,adapt_swaps=0,pool_workers=5 if arm=='reference' else 1,root=str(root),port=port)
        kv.core.write(root/'config.json',configuration(row,plan))
        kv.core.write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        plan.update(command=command(plan),generated={n:sha(root/n) for n in ('config.json','plan.json')})
        plan['environment_sha256']=kv.runner.comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_signature']=invariant(row,row['plans'][0])
    require(all(invariant(row,p)==row['invariant_signature'] for p in row['plans']),'undeclared arm difference')
    save(row);print('Prepared CPU capacity comparison;commit before host/native.',flush=True)


if __name__=='__main__':
    require(sys.executable==PYTHON and len(sys.argv)==2
        and sys.argv[1] in ('prepare','host',*map(str,range(len(ARMS)))),'qualified Python;prepare|host|0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);kv.build.guards.no_model_owners([])
        action=sys.argv[1]
        if action=='prepare':prepare()
        else:
            with (ROOT/('host-attempt.json' if action=='host' else action+'/attempt.json')).open('x') as f:
                json.dump(dict(action=action,result_key=KEY),f)
            try:host() if action=='host' else context().run(int(action))
            except BaseException as error:
                row=json.loads((ROOT/'record.json').read_text())
                if row['stage']!='failed':row.update(stage='failed',passed=False,complete=False,error=repr(error),failed_action=action);save(row)
                raise
