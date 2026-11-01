#!/usr/bin/env python3
"""Current same-engine MTP16/Q8 long-generation diagnostic; repository LICENSE.

Main-authored from owned exact-input benchmark controls. The closed117K prefill
comparison is recoverable at90e4f00c7. Its24-output samples do not qualify
sustained decode;this fresh bounded diagnostic uses both official long tasks. Main owns serialized execution.
"""
import copy
import fcntl
import hashlib
import json
import re
import select
import subprocess
import time
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
import compare_prefill_strategies as gdn

model, comparison, shared, build = gdn.model, gdn.comparison, gdn.shared, gdn.build
P, B, RESULT = gdn.P, gdn.B, gdn.RESULT
ROOT, KEY = model.gdn16.ROOT / 'mtp16-generation-v1', 'gdn16_mtp16_generation_v1'
ARMS = ('candidate', 'reference')
WORKLOADS = ('regular-code', 'varied-prose')
WORKLOAD_ID = 'current_mtp16_generation_20261102'
SPEED_KEY = 'gdn16_mtp16_model_speed_v2'
CHANGED = {'tools/sycl/mtp_prefill.py', 'tools/sycl/check_prefill_strategy_model.py'}
COLLECTOR_HOST = {'tools/sycl/profile_split.py', 'tools/sycl/test_profile_split.py'}
require, digest, write, compact = model.require, model.digest, model.write, model.compact


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def source_delta(current, original, declared, reused):
    """Reused observer bytes must match their own qualified original inputs."""
    require(set(current) == set(original)
            and {n for n in current if current[n] != original[n]} == declared | set(reused)
            and all(current[n] == sha for n, sha in reused.items()),
            'undeclared or changed source outside the qualified reuse scope')


def proof(revision):
    model.configure('gdn16-mtp16-throughput')
    native, _, providers = model.proof()
    data = build.committed(RESULT)
    probe = json.loads((model.ROOT / 'record.json').read_text())
    require(data[model.KEY] == compact(probe) and probe['passed'] and probe['complete']
            and probe['host_passed'] and probe['ownership_closed'] and not probe['active']
            and probe['binary'] == native['binaries']['strata']
            and probe['native_source_commit'] == native['source_commit']
            and probe['outcome']['native_exit_code'] == 0 and not probe['outcome']['fresh_faults']
            and len(probe['outcome']['routes']) == 2
            and all(r['GDN_FP16_storage_calls'] == 144
                    and r['GDN_FP16_storage_tokens'] == 36 * 32768
                    and r['MTP_prefill']['callbacks'] == 4
                    and r['MTP_prefill']['nominal_cells'] == 32768
                    for r in probe['outcome']['routes']), 'closed current GDN16/MTP16 production model check required')
    prior = json.loads((model.gdn16.ROOT / 'model-speed-v2/record.json').read_text())
    require(data['gdn16_model_speed_v2'] == compact(prior) and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active']
            and prior['arms']['candidate']['binary'] == native['binaries']['strata'],
            'closed same-engine GDN16 speed basis required')
    recovered = {n: hashlib.sha256(raw).hexdigest()
                 for n, raw in shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = model.pinned_sources(revision)
    require(recovered == prior['sources'], 'original speed-controller source recovery failed')
    profile = json.loads((model.gdn16.ROOT / 'resource-profile-v3/record.json').read_text())
    require(data['gdn16_resource_profile_v3'] == compact(profile)
            and profile['passed'] and profile['complete'] and profile['ownership_closed'] and not profile['active']
            and profile['binary'] == native['binaries']['strata']
            and profile['native_source_commit'] == native['source_commit'],
            'closed same-build observer qualification required')
    profile_sources = {n: hashlib.sha256(raw).hexdigest()
                       for n, raw in shared.gen.blobs(P, profile['source_commit'], profile['sources'])}
    require(profile_sources == profile['sources'], 'original observer-controller source recovery failed')
    reuse = {n: profile_sources[n] for n in COLLECTOR_HOST}
    source_delta(current, recovered, CHANGED, reuse)
    probe_sources = {n: hashlib.sha256(raw).hexdigest()
                     for n, raw in shared.gen.blobs(P, probe['source_commit'], probe['sources'])}
    require(probe_sources == probe['sources'], 'original MTP16 probe-controller source recovery failed')
    source_delta(current, probe_sources, {'tools/sycl/mtp_prefill.py'}, {})
    def draft_function(rev):
        text = shared.gen.git('show', rev + ':src/prefill/prefill.cpp').decode()
        return text[text.index('bool Prefill::draft_kv('):text.index('bool Prefill::set_peer(')].encode()
    original = draft_function('3b4998f326101fa913788fd368e1674b111dea03')
    require(original == draft_function(native['source_commit']), 'previously executed MTP function changed')
    return native, providers, prior, probe, dict(
        previously_executed_draft_function=dict(bytes=len(original), sha256=hashlib.sha256(original).hexdigest()),
        collector_source=profile['source_commit'], collector_original_files=len(profile_sources),
        reused_collector_host_sources=reuse, probe_original_files=len(probe_sources),
        probe_controller_only_change='tools/sycl/mtp_prefill.py')



def closed_speed(revision, native):
    row = json.loads((model.gdn16.ROOT / 'mtp16-speed-v2/record.json').read_text())
    require(build.committed(RESULT)[SPEED_KEY] == compact(row) and row['passed'] and row['complete']
            and row['ownership_closed'] and not row['active'] and len(row['processes']) == 6
            and row['decision']['reproducible_prompt_gain']
            and row['decision']['all_candidate_processes_at_least1000']
            and not row['decision']['inference_adopted']
            and all(p['native_exit_code'] == 0 and not p['fresh_faults'] for p in row['processes'])
            and row['arms']['candidate']['binary'] == native['binaries']['strata'],
            'closed current-engine repeated1000 prefill evidence required')
    recovered = {n: hashlib.sha256(raw).hexdigest()
                 for n, raw in shared.gen.blobs(P, row['source_commit'], row['sources'])}
    require(recovered == row['sources'], 'original117K benchmark source recovery failed')
    source_delta(model.pinned_sources(revision), recovered, {'tools/sycl/mtp_prefill.py'}, {})
    return row


def environment(row, plan):
    require(plan['kernel'] == ARMS[plan['index']] and plan['heads'] == 1, 'undeclared long-generation arm')
    env = comparison.environment(row, plan)
    env.update(STRATA_SYCL_PREFILL_GDN_BARRIERS3='1', STRATA_SYCL_PREFILL_GDN_FP16='1',
               STRATA_SYCL_PREFILL_GDN_CHUNK64='0', STRATA_MTP_BATCH_F16=str(int(plan['kernel'] == 'candidate')),
               STRATA_SNAPSHOT_VERIFY='1', STRATA_TRACE='1')
    require(not env.get('LD_PRELOAD') and not env.get('STRATA_DRAFT_TIMING')
            and env['CUDA_MODULE_LOADING'] == 'EAGER', 'private validation and trace-only diagnostic required')
    return env


def invariant(row, plan):
    value = dict(config=json.loads((Path(plan['root']) / 'config.json').read_text()), environment=environment(row, plan))
    value['environment']['STRATA_MTP_BATCH_F16'] = '<arm>'
    return hashlib.sha256(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>').encode()).hexdigest()


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact generation state')
    native, providers, _, _, _ = proof(row['source_commit'])
    basis = closed_speed(row['source_commit'], native)
    require(row['binary'] == native['binaries']['strata'] and row['native_source_commit'] == native['source_commit']
            and row['native_validation_probe'] == basis['native_validation_probe']
            and row['model_assets'] == basis['model_assets'], 'native/model qualification differs')
    require(model.pinned_sources(row['source_commit']) == row['sources'], 'frozen generation/controller source differs')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers,
            'runtime inventory differs')
    require(digest(row['template']['path']) == row['template']['sha256']
            and build.guards.driver_guard() == row['driver'], 'template/driver differs')
    require(build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'generation output budget')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    contract, manifest = model.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected'], 'frontend provider differs')
    require(all(invariant(row, plan) == row['invariant_signature'] and
                comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'] and
                all(digest(Path(plan['root']) / n) == sha for n, sha in plan['generated'].items())
                for plan in row['plans']), 'arm/configuration differs')
    return providers, manifest


def prepare():
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh long-generation diagnostic only')
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    native, providers, _, _, scope = proof(revision)
    prior = closed_speed(revision, native)
    contract, manifest = model.python_manifest()
    require(contract == prior['python_contract'] and manifest['selected'] == prior['python_selected'], 'frontend changed')
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    controls = selfcheck()
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', providers)
    row = dict(stage='prepared', source_commit=revision, sources=model.pinned_sources(revision),
        binary=native['binaries']['strata'], native_source_commit=native['source_commit'],
        native_validation_probe=prior['native_validation_probe'], configuration=copy.deepcopy(prior['configuration']),
        environment_template=copy.deepcopy(prior['environment_template']), model_assets=assets,
        template=prior['template'], boot=prior['boot'], driver=prior['driver'], python_contract=contract,
        python_selected=manifest['selected'], runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        qualification_reuse=dict(scope, current_prefill_source=prior['source_commit'],
            current_prefill_evidence='90e4f00c76b8d2b6a65bea92104efdfc393ec3d0', new_native_build=False),
        plans=[], processes=[], host_passed=False, passed=False, complete=False, active=False, ownership_closed=True,
        controls=controls, whole_bytes=256 << 20,
        budgets=dict(wall=5400, cpu=36000, rss_cap=104 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        plan=dict(question='Do current MTP16 and Q8 routes each complete both thinking-on >=1K long-generation tasks with safe handoff and observed acceptance?',
            order='Two fresh independent processes:candidate then reference;four official warmups then regular-code and varied-prose once each. Commit each process before next.',
            constants='Same nine ELFs/model/profile,131072 INT8 context,1256 slots,5workers,spec4,8192chunks,throughput native16/GDN3/GDN16,heads1,Top-k0,originalDown,no token graph,caches0. Only MTP_BATCH_F16 differs between arms. Standard-v4-33k tasks,16384 output cap,xhigh thinking without a separate cap.',
            observers='Parameter validation with warning/error logs,private native activation preload,STRATA_TRACE1 and nominal MTP callback markers. No full API trace,CPU/GPU clocks,census or profiler. This diagnostic is not clean speed or an observer-overhead qualification.',
            acceptance='Same frozen bodies and native/provider/clock/journal/owner validation. Each long request needs >=32768 fresh input,1024..16384 generated tokens including thought,>=512 actual answer characters,normal stream and engine decode clocks,>=256 offered drafts. Actual ordinary GPU BF16/GDN16 extent,checked MoE checkpoints and restored dense/head owners must pass. Do not require identical text,generated lengths or acceptance.',
            decision='One observation perworkload/arm is descriptive only. Keep all eligible/failed samples,actual starting contexts,generated lengths,TTFT,PP,TG and draft acceptance. No strong/weak classification,speed/nonregression,inference adoption,resource saturation,physical262144/RESTORE or70target claim. Freeze repeated matched comparison after the pilot.',
            stopping='Stop first source/provider/owner/clock/validation/fault/budget/route/eligibility failure;no retry,replacement,exclusion or hash refresh. Full15 model hashes before/after;no speed-based stop.',
            retention='Keep compact individual samples,route/clock/source/configuration and closure decisions. Retire successful raw journals,traces,caches after useful evidence verification. Both retained engine generations unchanged.'))
    for index, arm in enumerate(ARMS):
        root=ROOT/str(index);root.mkdir(mode=0o700)
        for sub in ('tmp','cache'): (root/sub).mkdir()
        config=copy.deepcopy(row['configuration']);config['log']=str(root/'engine.stderr')
        for flag in ('--prompt-cache','--conversation-cache-mib'):
            require(config['args'][config['args'].index(flag)+1]=='0','snapshot caches must remain disabled')
        write(root/'config.json',config);write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        plan=dict(index=index,kernel=arm,heads=1,root=str(root),generated={n:digest(root/n) for n in ('config.json','plan.json')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--think','--runs','1','--profile-request-gates','--workload-id',WORKLOAD_ID,
                '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')])
        plan['environment_sha256']=comparison.fingerprint(environment(row,plan));row['plans'].append(plan)
    row['invariant_signature']=invariant(row,row['plans'][0])
    require(all(invariant(row,p)==row['invariant_signature'] for p in row['plans']),'undeclared arm difference')
    save(row);print('Prepared current two-process/two-workload diagnostic;commit before frontend probes.',flush=True)


def native_route(text, read):
    # Main-owned ordinary-route counterpart of official_model_entry.progress;
    # variable actual33K extent,no global mutation or change to native math.
    require(type(read) is int and 32768 <= read <= 65537,'unplanned ordinary long-generation extent')
    n=read-1
    def fields(prefix):
        lines=[v for v in text.splitlines() if v.startswith(prefix)]
        require(len(lines)==1,'missing/duplicate native route:'+prefix)
        pairs=re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)',lines[0])
        require(len({k for k,_ in pairs})==len(pairs),'duplicate native field')
        return dict(pairs)
    main=fields('strata prefill: layer-major candidate, ');window=fields('strata MoE window configuration: ')
    work=fields('strata MoE accumulator window: ');split=fields('strata MoE split admission: ')
    bank=fields('strata MoE expert bank: ');end=fields('strata MoE expert bank complete: ')
    require(main['tokens']==main['residual_gpu_rows']==str(n) and main['chunk']=='8192'
            and main['residual_dtype']=='bf16' and main['residual_element_bytes']=='2'
            and int(main['residual_total'])==int(main['residual_gpu'])==n*20480
            and main['residual_host']==main['residual_staging']=='0'
            and window['requested_mode']==window['effective_mode']=='10'
            and window['maximum_rows']==work['maximum_rows']==str(n) and work['windows']=='48'
            and work['operator_max']=='8192' and work['GPU_resident_windows_only']=='1'
            and split['active']=='0' and split['rows']==str(n) and split['decode_changed']=='0'
            and split['primary_rows']==split['tail_payload_bytes']==split['extra_allocation_bytes']=='0',
            'ordinary GPU residual/window/split extent differs')
    require(bank['requested_slots']==bank['effective_slots']==end['total_slots']==split['prefill_slots']=='384'
            and bank['staged']=='1' and bank['residual_gpu_rows']==str(n)
            and bank['general_map']=='private_scratch_never_published'
            and int(bank['device_bytes'])==int(main['layer_cache'])>0
            and int(bank['allocation_limit_bytes'])>=int(bank['device_bytes'])
            and end['final_drain']=='checked','ordinary staged bank/checked drain differs')
    gdn.native16_route(text,read);gdn.gdn_selection(text,'candidate')
    chunks=(n+8191)//8192
    stored=[int(v) for v in re.findall(r'^strata GDN storage: fp16;state=fp32;tokens=(\d+)$',text,re.M)]
    calls=[dict(re.findall(r'(\w+)=([^\s;]+)',v)) for v in text.splitlines() if v.startswith('strata GDN dispatch: ')]
    require(len(stored)==len(calls)==36*chunks and sum(stored)==sum(int(v['tokens']) for v in calls)==36*n
            and all(0<v<=8192 for v in stored) and all(v['schedule']=='barriers3' and v['partitions']=='4' for v in calls),
            'actual GDN16/three-barrier calls differ')
    pending,groups,pairs=None,{},0
    for line in text.splitlines():
        if not line.startswith('strata MoE progress: '): continue
        item=dict(re.findall(r'(\w+)=([^\s;]+)',line));phase=item.pop('phase')
        require(item['interval']=='8' and int(item['submitted'])%8==0,'checkpoint interval')
        if phase=='before_drain': require(pending is None,'overlapping checkpoint');pending=item
        else:
            require(phase=='after_drain' and pending==item,'checkpoint return differs')
            group=groups.setdefault((item['layer'],item['first']),dict(total=int(item['total']),submitted=[],queue=item['queue']))
            require(group['total']==int(item['total']) and group['queue']==item['queue'],'checkpoint owner changes')
            group['submitted'].append(int(item['submitted']));pairs+=1;pending=None
    require(pending is None and len(groups)==48 and sum(v['total'] for v in groups.values())==int(work['jobs'])
            and all(v['submitted']==list(range(8,v['total']+1,8)) and 0<v['total']<=655 for v in groups.values()),
            'checkpoint/window/job conservation differs')
    target=fields('strata prefill heads: target detached ');draft=fields('strata prefill heads: draft detached ')
    retired=fields('strata prefill dense: phase=retired ');dense=fields('strata prefill dense: phase=bank ')
    restored=fields('strata prefill dense: phase=restored ')
    require(int(target['bytes'])==int(target['D2H'])==int(target['H2D-on-restore'])==comparison.TARGET_BYTES
            and target['source']=='owned-RAM' and int(draft['bytes'])==comparison.DRAFT_BYTES
            and int(draft['retained-map-bytes'])==comparison.MAP_BYTES
            and retired['residency']==restored['residency']=='Full' and dense['residency']=='Bank'
            and restored['activations']=='48','head/dense restore differs')
    markers=('strata prefill dense: phase=retired ','strata prefill heads: target detached ',
        'strata prefill heads: draft detached ','strata prefill dense: phase=bank ',
        'strata prefill: layer-major candidate, ','strata MoE accumulator window: ','strata prefill dense: phase=restored ')
    positions=[text.index(v) for v in markers];require(positions==sorted(positions),'owner phase order differs')
    return dict(read_tokens=read,rows=n,GPU_residual_rows=n,ordinary_staged_slots=384,split=False,
        GDN_FP16_calls=len(stored),GDN_FP16_rows=sum(stored),GDN_barriers=3,
        checkpoint_pairs=pairs,windows=len(groups),jobs=int(work['jobs']),restored_dense_layers=48,
        MTP_prefill=model.mtp_route(text,n),physical_resource_saturation_qualified=False)


def official(root,result):
    suite=result['results']['long_generation'];rows=shared.gen.collect_samples(suite)
    shared.gen.require_complete_generation(suite,rows,WORKLOADS)
    require(result['complete'] and not result['results']['failures'] and result['params']['runs']==1
            and result['params']['think'] is True and result['params']['long_generation_only'] is True
            and suite['selected_workloads']==list(WORKLOADS) and not suite['diagnostic_subset'],'incomplete official diagnostic')
    journal,journal_rows=shared.gen.journal_evidence(root)
    require(journal['valid'] and journal['complete'] and
            [{k:v for k,v in r.items() if k not in ('controller_eligible','reasoning_tokens')} for r in rows]==journal_rows,
            'official results/journal differ')
    path,=(root/'bench').glob('*/requests.jsonl');events=[json.loads(v) for v in path.read_text().splitlines()]
    warm=shared.gen.warmup_reconciliation(events,WORKLOADS)
    identity=events[0];binding=result['comparison_identity']
    from strata_bench import client_fingerprint
    workload=dict(workload_id=WORKLOAD_ID,nonce=WORKLOAD_ID,nonce_policy='fixed-id')
    require(identity['mode']==binding['mode']=='suite' and identity['requested_runs']==1
            and identity['thinking'] is binding['thinking'] is True
            and identity['source']==binding['client']==client_fingerprint()
            and identity['context']==binding['context']==result['machine']['context']==131072
            and identity['workload']==binding['workload']==result['params']['workload']==workload
            and all(identity[k]==binding[k]==result['machine'][k] for k in ('model','quant')),'client/model/body binding differs')
    return rows,events,dict(warm,valid=True,journal_sha256=digest(path))


def run(index):
    row=json.loads((ROOT/'record.json').read_text());files,manifest=current(row)
    require(row['host_passed'] and row['stage'] in ('ready','running') and len(row['processes'])==index<len(ARMS)
            and all(p['passed'] for p in row['processes']),'fresh ordered process only;no retry')
    plan=row['plans'][index];root=Path(plan['root']);env=environment(row,plan)
    require(not (root/'execution.json').exists() and not (root/'launch.json').exists(),'existing native attempt')
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome,owner=dict(index=index,kernel=plan['kernel'],complete=False,passed=False,host_probe_accepted=False,model_executed=False),None
    windows=[];live=[];identity=None;initial=(time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns())
    def persist(): write(root/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    def interact(io,entry):
        nonlocal identity
        for i,name in enumerate(WORKLOADS):
            offsets={}
            for stage in ('begin','end'):
                while True:
                    line=io.line(3600)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE '))==dict(stage=stage,phase='long-generation/'+name,run=0),'request gate differs')
                delta=[a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME),time.monotonic_ns()),initial)]
                require(min(delta)>=0 and abs(delta[0]-delta[1])<=50000000,'clock/suspend discontinuity')
                claim=json.loads((root/'engine-start.json').read_text())['native'];actual=build.supervision.identity(claim['pid'])
                require(actual and actual['start_ticks']==claim['start_ticks'] and actual['sid']==entry['pid']
                        and str(Path('/proc',str(claim['pid']),'exe').resolve())==row['binary']['path']
                        and (identity is None or identity==claim),'native identity differs');identity=claim
                expected=model.engine_once.native_environment(row,env)
                actual_env=dict(v.split(b'=',1) for v in Path('/proc',str(claim['pid']),'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual_env.get(k.encode())==v.encode() for k,v in expected.items())
                        and set(actual_env)-{k.encode() for k in expected}<={b'PWD',b'OLDPWD',b'SHLVL',b'_'},'native environment differs')
                mapped=shared.runtime.executable_mappings(Path('/proc',str(claim['pid']),'maps').read_text())
                require(mapped<=set(files)|{row['binary']['path']} and {row['native_validation_probe']['loader']['path'],
                    row['native_validation_probe']['library']['path'],env['ZE_ENABLE_ALT_DRIVERS']}<=mapped,'native provider differs')
                shared.python_runtime.check_running(entry['pid'],P/'tools/strata_bench.py',manifest['selected'],env,
                    {v['realpath'] for v in manifest['inventory']['files'].values()},plan['command'][2:])
                comparison.require_quiet_validation(root);offsets[stage]=(root/'engine.stderr').stat().st_size
                if stage=='begin':
                    path,=(root/'bench').glob('*/requests.jsonl');events=[json.loads(v) for v in path.read_text().splitlines()]
                    require(len(events)==10+2*i and events[-1]['kind']=='start' and events[-1]['phase']=='long-generation/'+name,'warmup/prior journal differs')
                    shared.gen.warmup_reconciliation(events[:-1],WORKLOADS[:i])
                    prior=[e['sample'] for e in events[:-1] if e['kind']=='result']
                    require(all(shared.gen.sample_eligible(v) for v in prior[4:]),'prior long sample ineligible')
                    shared.gen.request_evidence((root/'engine.stderr').read_text(),prior)
                else:
                    text=(root/'engine.stderr').read_bytes()[offsets['begin']:offsets['end']].decode()
                    headers=re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',text,re.M)
                    require(len(headers)==1 and headers[0][1]=='0' and headers[0][0]==headers[0][2],'native long footer differs')
                    live.append(native_route(text,int(headers[0][2])))
                outcome.update(model_executed=True,native_identity=claim,request_routes_checked_before_continue=len(live));persist()
                print(f'Long-generation process{index} arm={plan["kernel"]} {name} {stage}',flush=True);io.send(b'continue\n')
            windows.append(offsets)
        io.proc.stdin.close();deadline=time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll();require(time.monotonic()<deadline,'native teardown deadline')
            if select.select([io.proc.stdout],[],[],.02)[0]: io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=build.supervision.Owner(root/'owned',persist)
        entry,_,_=owner.run('bench',plan['command'],env,P,interaction=interact,**row['budgets'])
        outputs=list((root/'bench').glob('*/results.json'))
        if outputs: outcome['samples']=shared.gen.collect_samples(json.loads(outputs[0].read_text())['results'].get('long_generation',{}))
        require(build.supervision.closed(entry) and entry['exit_code']==0 and len(outputs)==1,'owned official benchmark failed')
        result=json.loads(outputs[0].read_text());samples,events,journal=official(root,result)
        for sample in samples: comparison.require_valid_client_ttft(sample)
        require(len(windows)==len(samples)==len(live)==2,'long-generation request count')
        if row['processes']:
            require([s['request_sha256'] for s in samples]==[s['request_sha256'] for s in row['processes'][0]['samples']],'paired bodies differ')
        comparison.require_quiet_validation(root)
        end=shared.gen.native_end(root,row,dict(config=json.loads((root/'config.json').read_text())),row['boot'],entry['pid'])
        raw=(root/'engine.stderr').read_bytes();text=raw.decode();gdn.validate_model_text(text,plan)
        native_requests=shared.gen.request_evidence(text,[e['sample'] for e in events if e['kind']=='result'])
        require(not any(v in text for v in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','teardown uncertain')),'native error')
        validation=shared.runtime.validation.preload_evidence(text,end['native']['pid'])
        require(validation['loader']==row['native_validation_probe']['loader']['path'],'validation loader differs')
        routes=[native_route(raw[w['begin']:w['end']].decode(),s['read_tokens']) for w,s in zip(windows,samples)]
        require(routes==live,'live/final routes differ')
        current(row)
        if index==len(ARMS)-1: shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
        fields=('workload','prompt_tokens','read_tokens','reused_tokens','generated_tokens','reasoning_tokens','output_chars','reasoning_chars',
            'prompt_ms','prompt_tps','decode_ms','decode_tps','ttft_s','total_s','drafts_offered','drafts_accepted',
            'acceptance_rate','request_sha256','output_sha256','reasoning_sha256','finish_reason','controller_eligible')
        outcome.update(complete=True,host_probe_accepted=True,native_exit_code=0,validation=validation,routes=routes,journal=journal,
                       native_request_binding=dict(requests=len(native_requests),draft_cells=sum(r['draft_cells'] for r in native_requests)),
                       samples=[{k:s[k] for k in fields if k in s} for s in samples])
    except BaseException as error: outcome['error']=repr(error)
    finally:
        if owner: shared.runtime.validation.finish(outcome,owner,cursor,row['boot'])
        if outcome.get('driver_after')!=row['driver']: outcome.update(passed=False,final_driver_mismatch=True)
        outcome['passed']=bool(outcome.get('passed') and outcome['complete'])
        row['processes'].append(outcome);done=index==len(ARMS)-1
        row.update(stage='closed' if outcome['passed'] and done else 'running' if outcome['passed'] else 'failed',
            passed=bool(outcome['passed'] and done),complete=bool(outcome['passed'] and done),
            active=bool(owner and owner.active),ownership_closed=outcome.get('ownership_closed',False))
        if row['passed']: row['decision']=dict(pilot_complete=True,observations_per_workload_per_arm=1,
            MTP_strong_weak_classification_qualified=False,speed_nonregression_qualified=False,inference_adopted=False,
            physical262144_qualified=False,physical_resource_saturation_qualified=False,
            next='Inspect actual generation lengths,answer and acceptance;freeze repeated clean comparison before speed/adoption claims.')
        save(row)
    require(outcome['passed'],outcome.get('error','long-generation diagnostic failed'))
    print(json.dumps({k:outcome.get(k) for k in ('index','kernel','passed','complete','fresh_faults','error')}),flush=True)


def selfcheck():
    # Exercise the actual variable-shape route reader with owned protocol fixtures.
    def fixture(n):
        lines=['strata prefill dense: phase=retired residency=Full',
            f'strata prefill heads: target detached bytes={comparison.TARGET_BYTES} D2H={comparison.TARGET_BYTES} H2D-on-restore={comparison.TARGET_BYTES} source=owned-RAM',
            f'strata prefill heads: draft detached bytes={comparison.DRAFT_BYTES} retained-map-bytes={comparison.MAP_BYTES}',
            'strata prefill dense: phase=bank residency=Bank',
            f'strata prefill: layer-major candidate, tokens={n} chunk=8192 residual_dtype=bf16 residual_element_bytes=2 residual_total={n*20480} residual_gpu_rows={n} residual_gpu={n*20480} residual_host=0 residual_staging=0 layer_cache=4096',
            'strata prefill: native16 BF16 HC residual/projection IO',
            'strata prefill: GDN columns=4 workgroups=192 lanes=128 barriers=3 epilogue=GPU_FP32_same_order scratch=existing_y',
            f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={n}',
            f'strata MoE split admission: active=0 rows={n} primary_rows=0 tail_payload_bytes=0 extra_allocation_bytes=0 decode_changed=0 prefill_slots=384',
            f'strata MoE expert bank: requested_slots=384 effective_slots=384 staged=1 residual_gpu_rows={n} general_map=private_scratch_never_published device_bytes=4096 allocation_limit_bytes=8192']
        chunks=[min(8192,n-i) for i in range(0,n,8192)]
        for _ in range(36):
            for count in chunks: lines.extend([f'strata GDN storage: fp16;state=fp32;tokens={count}',f'strata GDN dispatch: schedule=barriers3 partitions=4 tokens={count}'])
        for i in range(48):
            for phase in ('before_drain','after_drain'): lines.append(f'strata MoE progress: phase={phase} layer={i} first=0 submitted=8 total=8 interval=8 queue=0x100')
        lines.extend([f'strata MoE accumulator window: windows=48 jobs=384 maximum_rows={n} operator_max=8192 GPU_resident_windows_only=1',
            'strata MoE expert bank complete: total_slots=384 final_drain=checked',
            'strata prefill dense: phase=restored residency=Full activations=48'])
        lines.extend(f'strata serve: DRAFT_PREFILL path=batched mode=2 cells={count}' for count in chunks)
        return '\n'.join(lines)
    for rows in (32767,32768,33017,65536): native_route(fixture(rows),rows+1)
    good=fixture(33017)
    bad=[good.replace('residual_host=0','residual_host=1'),good.replace('final_drain=checked','final_drain=failed'),
         good.replace('activations=48','activations=47'),good.replace('submitted=8 total=8','submitted=16 total=8',1),
         good.replace('state=fp32;tokens=8192','state=fp32;tokens=8191',1),good.replace('mode=2 cells=','mode=0 cells=',1),
         good+'\nstrata prefill heads: target detached bytes=1',good.replace('effective_slots=384','effective_slots=128')]
    for text in bad:
        try: native_route(text,33018)
        except (RuntimeError,ValueError,KeyError): pass
        else: raise RuntimeError('invalid long-generation route accepted')
    source_delta(dict(a='new',b='fixed'),dict(a='old',b='fixed'),{'a'},{})
    try: source_delta(dict(a='new',b='wrong'),dict(a='old',b='fixed'),{'a'},{})
    except RuntimeError: pass
    else: raise RuntimeError('undeclared source accepted')
    return dict(passed=True,GPU_executed=False,valid_route_shapes=4,invalid_routes_rejected=len(bad),source_controls=2)


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);build.guards.no_model_owners([])
        args=sys.argv[1:]
        if args==['selfcheck']: print(json.dumps(selfcheck()))
        elif args==['prepare']: prepare()
        elif args==['host']: comparison.host(sys.modules[__name__])
        elif len(args)==1 and args[0] in ('0','1'): run(int(args[0]))
        else: raise RuntimeError('mtp_prefill.py selfcheck|prepare|host|INDEX0..1')
