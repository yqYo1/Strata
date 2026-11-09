#!/usr/bin/env python3
"""Bound CPU leaf observer cost on the current dense-DPAS decode candidate.

Main-authored; repository LICENSE. Reuses the owned official benchmark runner,
native input census and independently tested leaf reader. No engine rebuild,
static residency census or attribution of leaf clocks to physical RAM traffic.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import socket
import sys

sys.dont_write_bytecode = True
import benchmark_runtime
import profile_adaptive_cpu_leaves as leaf
import q6_dense_model_speed as dense

P, B, RESULT = dense.P, dense.B, dense.RESULT
ROOT, KEY = B / 'q6-dense-cpu-leaves-v3', 'q6_dense_cpu_leaves_v3'
THIS = 'tools/sycl/profile_q6_cpu_leaves.py'
TEST = 'tools/test_q6_cpu_leaves.py'
HOST_KEY = 'q6_dense_cpu_leaves_host_v1'
ORDER = leaf.ORDER
require, sha, write, signature = dense.require, dense.sha, dense.write, dense.signature


def closed_cohort():
    row = json.loads((dense.ROOT / 'record.json').read_text())
    require(dense.validation.committed(RESULT, (dense.KEY, dense.SCOPE)) == dense.compact(row) and
        row['stage'] == 'closed' and row['passed'] and row['complete'] and not row['active'] and
        row['ownership_closed'] and len(row['processes']) == 6 and
        all(p['passed'] and p['complete'] and p['ownership_closed'] and
            p['native_exit_code'] == 0 and not p['fresh_faults'] for p in row['processes']),
        'closed six-process official dense cohort required')
    require(all(row['decision']['decode'][k]['gain_in_all_three_pairs'] for k in
        ('regular-code', 'varied-prose')),
        'focused dense decode choice needs its repeated workload gains')
    return row


def native(reference):
    """New observer admission; the closed V2 reader/outcome remains unchanged."""
    ctx = dense.build.context()
    n = json.loads((dense.build.ROOT / 'record.json').read_text())
    require(dense.validation.committed(RESULT, (dense.build.KEY,)) == ctx.compact(n) and
        n['stage'] == 'built' and n['passed'] and n['complete'] and n['ownership_closed'] and
        not n['active'] and n['policy_check']['passed'], 'closed native build/policy required')
    project = ctx.source.pin(P, ctx.PREFIXES)
    current = project['source_sha256']; previous = {p: h for p, h in current.items() if p != THIS}
    proof = reference['native_admission']
    require(THIS in current and len(previous) == proof['current_source_files'] and
        signature(previous) == proof['current_source_sha256'],
        'only this committed, noncompiled observer may extend the closed native admission')
    original = n['admission_project']
    require(original['source_sha256'] == {p: hashlib.sha256(raw).hexdigest() for p, raw in
        ctx.source.blobs(P, original['revision'], original['source_sha256'])},
        'recover original native admission inputs from Git')
    require(n['native_source_commit'] == n['project']['revision'] == n['actual_inputs']['source_commit'] ==
        reference['native_source_commit'] and n['binaries']['strata'] == reference['binary'] and
        dense.build.prerequisites() == n['prerequisites'] and
        '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in n['recipe']['configure'],
        'same original profile-enabled engine/source/recipe required')
    ctx.build.check_inputs(dict(n, project=project))
    require(all(sha(p) == h for p, h in (n['generated_inputs'] | n['configured_files']).items()),
        'generated/configured native recipe changed')
    actual = n['actual_inputs']
    require(sha(actual['path']) == actual['sha256'] and actual['uncovered'] == 0,
        'actual compiler/link census changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for group in ('compiled', 'generated', 'links')
        for p, h in census[group].items()) and
        all(sha(v['path']) == v['sha256'] and Path(v['path']).stat().st_size == v['bytes']
            for v in n['binaries'].values()), 'exact compiled inputs/five ELFs changed')
    producers = {p: n['project']['source_sha256'][p] for p in leaf.PRODUCERS}
    require(all(sha(P / p) == h for p, h in producers.items()), 'compiled leaf producer changed')
    return n, dict(native_source_commit=n['native_source_commit'],
        closed_dense_source_sha256=proof['current_source_sha256'],
        observer={THIS: current[THIS]}, exact_five_ELFs_census_recipe=True,
        compiled_leaf_producers=producers, native_recompiled=False)


def pins(revision):
    result = dense.head_speed.pins(revision)
    require(result.get(THIS) == sha(P / THIS) and result.get(TEST) == sha(P / TEST),
        'commit exact observer and CPU controls')
    return result


def environment(root, enabled):
    env = dense.environment(root, {})
    env.update(STRATA_SYCL_DECODE_Q6_DPAS_DENSE='1', STRATA_CPU_SERVICE=str(int(enabled)),
        STRATA_CPU_WORK_CENSUS='0', STRATA_SYCL_DECODE_SERVICE_PROFILE='0')
    require(env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '2' and
        env['STRATA_PREFILL_HEAD_HOST_BACKING'] == '0' and not env.get('LD_PRELOAD') and
        env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
        env['ZES_ENABLE_SYSMAN'] == '0', 'fixed head/backing/runtime safety required')
    return env


def invariant(config, env, root):
    env = dict(env, STRATA_CPU_SERVICE='<leaf-arm>')
    return signature(json.loads(json.dumps(dict(config=config, environment=env)).replace(str(root), '<process>')))


def contract(generation, rows):
    require(generation.get('workload_version') == 5 and generation.get('thinking') is True and
        generation.get('reasoning_budget_tokens') == 0 and generation.get('max_tokens') == 16384 and
        generation.get('diagnostic_subset') is True and
        generation.get('selected_workloads') == ['regular-code'] and len(rows) == 1 and
        (rows[0].get('workload'), rows[0].get('run'), rows[0].get('order')) == ('regular-code', 0, 0) and
        rows[0].get('controller_eligible') is True, 'focused full-output official v5 contract differs')


def dense_route(text, expected):
    """One actual long request; optional Verifier prompt capture is not required."""
    base, _ = dense.model.previous.route_tables(text, 2)
    require({k[0] for k in base} == {'5'}, 'only actual long request5 allowed')
    selected, legacy = {k: {} for k in base}, {k: {} for k in base}
    counts = dict(zip(dense.model.SHAPES, (40, 14, 22, 5)))
    for line in text.splitlines():
        if not line.startswith(('strata Q6_SHAPE ', 'strata Q6_DPAS_DENSE ')):
            continue
        f = dense.model.fields(line); key = tuple(f.get(k) for k in dense.model.IDENTITY)
        require(key in base, 'orphaned Q6 shape')
        if line.startswith('strata Q6_DPAS_DENSE '):
            require(set(f) == set(dense.model.IDENTITY) | {'ni','no','nc','layout',
                'extra_weight_bytes','canonical_table_excludes_dpas','calls','complete'} and
                f['nc'] == '4' and f['layout'] == 'canonical' and f['extra_weight_bytes'] == '0' and
                f['canonical_table_excludes_dpas'] == f['complete'] == '1' and
                all(f[k].isdecimal() for k in ('ni','no','calls')), 'dense shape schema differs')
            shape = int(f['ni']), int(f['no']); calls = int(f['calls'])
            require(shape in counts and shape not in selected[key] and
                calls == (counts[shape] if key[2:4] == ('decode','4') else 0),
                'dense route/count escaped its four decode-C4 cells')
            selected[key][shape] = calls
        else:
            names = ('ni','no','nc','reference_calls','columns_calls','centered_calls','native_half_calls')
            require(set(f) == set(dense.model.IDENTITY) | set(names) and
                all(f[k].isdecimal() for k in names), 'canonical shape schema differs')
            shape = tuple(int(f[k]) for k in names[:3]); calls = tuple(int(f[k]) for k in names[3:])
            require(shape not in legacy[key] and shape[0] > 0 and shape[1] > 0 and
                1 <= shape[2] <= 4 and sum(calls) > 0 and calls[2:] == (0,0) and
                (key[2] == 'decode' or calls[1] == 0), 'canonical shape policy differs')
            legacy[key][shape] = calls
    observed, completed = {}, []
    for key, fields in base.items():
        require(set(selected[key]) == set(counts) and fields.get('shapes','').isdecimal() and
            int(fields['shapes']) == len(legacy[key]), 'incomplete actual shape tables')
        table = legacy[key]
        if key[2:4] == ('decode','4'):
            require(all((*s,4) not in table for s in counts) and
                table.get((2560,512,4)) == (21,0,0,0) and int(fields['replay_completed']) > 0,
                'dense double count/small fallback/no actual replay')
            completed.append(dict(capture_id=int(key[4]), replay_completed=int(fields['replay_completed'])))
        label = '/'.join(key[2:4]); value = signature({'x'.join(map(str,k)):list(v) for k,v in sorted(table.items())})
        require(expected.get(label) == value and (label not in observed or observed[label] == value),
            'canonical route differs from the closed official candidate')
        observed[label] = value
    require(len(completed) == 1 and 'decode/1' in observed,
        'actual decode-C1 and exactly one decode-C4 capture required')
    return dict(canonical_phase_width_sha256=observed, decode_C4=completed[0],
        selected_C4_calls={f'{k[0]}x{k[1]}':v for k,v in counts.items()}, generic_small_C4_calls=21)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = dense.head_speed.protocol.compact(row); write(RESULT, data)


def current(row):
    require(dense.validation.committed(RESULT, (KEY,)) == dense.head_speed.protocol.compact(row) and
        pins(row['source_commit']) == row['sources'], 'commit exact leaf comparison state/source')
    reference = closed_cohort(); n, proof = native(reference)
    require(row['native_admission'] == proof and row['binary'] == n['binaries']['strata'] and
        row['native_source_commit'] == n['native_source_commit'] and
        row['model_assets'] == reference['model_assets'] and row['configuration'] == reference['configuration'] and
        row['driver'] == dense.guards.driver_guard() and row['boot'] == reference['boot'],
        'native/model/placement/driver qualification changed')
    contract_, manifest = dense.frontend.python_manifest()
    require(contract_ == row['python_contract'] and manifest['selected'] == row['python_selected'] and
        sha(row['template']['path']) == row['template']['sha256'], 'frontend/template changed')
    require(sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        dense.owner.artifact_bytes(ROOT) <= row['whole_bytes'],
        'runtime inventory/output budget changed')
    dense.suite.d.models.model_pins(row['configuration'], row['model_assets'])
    for plan in row['plans']:
        root=Path(plan['root']); config=copy.deepcopy(row['configuration']);config['log']=str(root/'engine.stderr')
        env=environment(root, ORDER[plan['index']] == 'candidate')
        require((plan['kernel'],plan['adapt_swaps']) == (ORDER[plan['index']],96) and plan['heads'] == 1 and
            json.loads((root/'config.json').read_text()) == config and
            all(sha(root/p) == h for p,h in plan['generated'].items()) and
            dense.suite.d.fingerprint(env) == plan['environment_sha256'] and
            invariant(config,env,root) == row['invariant_signature'], 'frozen leaf arm inputs differ')
    return json.loads(Path(row['runtime_manifest']['path']).read_text()), manifest


def context(index=None):
    ctx = dense.head_speed.context(1)
    ctx.ROOT,ctx.KEY,ctx.MODE = ROOT,KEY,'cohort'
    ctx.ARMS=tuple((arm,96) for arm in ORDER);ctx.WORKLOADS=('regular-code',)
    ctx.WORKLOAD_ID='q6_dense_cpu_leaves_v1_20261110'
    ctx.current,ctx.save,ctx.generation_contract = current,save,contract
    ctx.environment=lambda row,plan:environment(Path(plan['root']),plan['kernel']=='candidate')
    original_route,original_boundary=ctx.route,ctx.cache_boundary
    ctx.boundary_calls=0
    def route(text,read):
        result=original_route(text,read)
        reference=closed_cohort()['processes'][5]['evidence']['native_dense_route']['canonical_phase_width_sha256']
        result['dense_DPAS']=dense_route(text,reference)
        return result
    def boundary(root,expected):
        require(ctx.boundary_calls < 2 and expected == (4,5)[ctx.boundary_calls],
            'leaf warmup/long boundary order')
        raw,history=original_boundary(root,expected)
        # Bench emits the end gate before appending the completed client
        # sample. Bind the four warmups now; join all five only after normal
        # benchmark/native exit in run(). The independent native cache footer
        # still must be complete before either gate is acknowledged.
        if ctx.boundary_calls == 0:
            path,=(root/'bench').glob('*/requests.jsonl')
            samples=[e['sample'] for e in map(json.loads,path.read_text().splitlines()) if e['kind']=='result']
            require(len(samples)==4, 'native/client warmup leaf prefix differs')
            leaf.evidence(raw.decode(),samples,4,ORDER[index]=='candidate')
        ctx.boundary_calls+=1
        return raw,history
    ctx.route,ctx.cache_boundary=route,boundary
    ctx.decision=lambda _:dict(model_execution_closed=True,leaf_binding_pending=True,
        leaf_observer_cost_qualified=False,inference_adopted=False)
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in dense.build.primary.public(), 'fresh leaf cohort only')
    reference=closed_cohort();n,proof=native(reference)
    revision=dense.build.primary.source.git(P,'rev-parse','HEAD').decode().strip();sources=pins(revision)
    host_proof=dense.validation.committed(RESULT,(HOST_KEY,))
    require(host_proof['stage']=='closed' and host_proof['passed'] and host_proof['ownership_closed'] and
        not host_proof['GPU_executed'] and host_proof['source_signature_sha256']==signature(sources) and
        pins(host_proof['source_commit'])==sources,'fresh qualified current CPU reader/controller controls required')
    prior=json.loads((dense.model.ROOT/'record.json').read_text())
    assets=dense.suite.d.models.model_pins(reference['configuration'],reference['model_assets'],full=True)
    row={k:copy.deepcopy(reference[k]) for k in ('binary','native_source_commit','native_validation_probe',
        'configuration','boot','driver','python_contract','python_selected')}
    row.update(stage='prepared',source_commit=revision,sources=sources,model_assets=assets,
        native_admission=proof,template=copy.deepcopy(json.loads(dense.TEMPLATE.read_text())['template']),
        host_controls=dict(source_commit=host_proof['source_commit'],tests=host_proof['tests'],
            source_signature_sha256=host_proof['source_signature_sha256']),
        runtime_manifest=dict(path=prior['runtime_manifest']['path'],sha256=sha(prior['runtime_manifest']['path'])),
        plans=[],processes=[],host_passed=False,passed=False,complete=False,active=False,ownership_closed=True,
        whole_bytes=384<<20,budgets=dict(wall=1800,cpu=18000,rss_cap=104<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20),
        plan='Six fresh same-a347/221e processes leaf0/1,1/0,0/1. Dense1/head2/backing0,adapt96/every1,spec4,128KresidentINT8,1256cache,8192chunk,384bank,5workers,20ms spin fixed. Four official warmups then one unchanged v5 regular-code thinking-on >=32Kfresh/>=1Kgenerated request with >=512answer characters,16Koutput allowance. Only immutable pre-exec CPU_SERVICE differs. Static census,caller/decode/full tracing stay off;common route audits/parameter validation/safety remain. Current CPU reader/controller controls then fresh frontend probes;check actual native/provider/environment,complete native/client prefixes,all3leaf operations/format/NT/thread conservation,mode10/BF16/GDN16/MTP16 and dense-C4 routing,normal0/closed owners/quiet validation/no fresh fault and all15asset hashes before/after. Cost qualifies only if all3on/off per-token latency ratios<=1.05 and aggregate CPU/token/draft/output hashes match. A hash/work mismatch is inconclusive for observer cost,not task-quality failure;collect all samples without retries/replacement. Commit every closed stage before next. No prefill/whole-route adoption or physical traffic/saturation claim.',
        retention='Keep individual samples,operation totals,format/NT jobs,top3GU/Down bins and work hashes for the immediate current working-set decision. Retire successful raw output/caches after verified closure and source recovery. No new native generation;552b prefill fallback and a347 five ELFs remain unchanged.')
    ROOT.mkdir(mode=0o700)
    for i,arm in enumerate(ORDER):
        root=ROOT/str(i);root.mkdir(mode=0o700)
        for p in ('tmp','cache'):(root/p).mkdir(mode=0o700)
        config=copy.deepcopy(row['configuration']);config['log']=str(root/'engine.stderr')
        leaf.serialized_configuration(config)
        write(root/'config.json',config);write(root/'plan.json',dict(binary=row['binary'],native_validation_probe=row['native_validation_probe']))
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        env=environment(root,arm=='candidate');inv=invariant(config,env,root)
        if i==0:row['invariant_signature']=inv
        require(row['invariant_signature']==inv,'only leaf flag/private roots may differ')
        row['plans'].append(dict(index=i,kernel=arm,adapt_swaps=96,heads=1,root=str(root),
            generated={p:sha(root/p) for p in ('config.json','plan.json')},
            command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
                '--long-generation-only','--long-generation-workload','regular-code','--think','--runs','1',
                '--profile-request-gates','--workload-id','q6_dense_cpu_leaves_v1_20261110',
                '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench')],
            environment_sha256=dense.suite.d.fingerprint(env)))
    save(row);print('Prepared six fresh current-engine leaf cost processes;commit before host gates.',flush=True)


def controls():
    revision=dense.build.primary.source.git(P,'rev-parse','HEAD').decode().strip();sources=pins(revision)
    root=B/('q6-dense-cpu-leaves-host-'+revision[:12]);require(not root.exists(),'one CPU control attempt per committed source')
    root.mkdir(mode=0o700)
    scope=dense.owner.Owner(root/'owned');env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    row=dict(stage='running',source_commit=revision,source_signature_sha256=signature(sources),
        source_count=len(sources),GPU_executed=False,passed=False,ownership_closed=False)
    try:
        command=['/usr/bin/python3','-X','faulthandler','-B','-m','unittest',
            'tools.test_q6_cpu_leaves','tools.test_adaptive_cpu_leaf_evidence','-v']
        e,out,err=scope.run('controls',command,env,P,wall=30,cpu=20,rss_cap=1<<30,
            text_cap=256<<10,file_cap=1<<20,total_cap=4<<20)
        text=out.read_text()+err.read_text();count=re.findall(r'^Ran (\d+) tests? in ',text,re.M)
        row.update(command=command,exit_code=e['exit_code'],tests=int(count[0]) if len(count)==1 else None)
        require(dense.owner.closed(e) and e['exit_code']==0 and len(count)==1 and
            'skipped=' not in text and '\nOK\n' in text,'CPU reader/controller controls failed')
        row.update(stage='closed',passed=True)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row['ownership_closed']=not scope.active and all(dense.owner.closed(e) for e in scope.commands)
        row['owner_results']=[{k:e.get(k) for k in ('pid','direct_identity','exit_code','observation_complete',
            'direct_child_reaped','session_empty','error','errors','cleanup')} for e in scope.commands]
        data=json.loads(RESULT.read_text());previous=data.get(HOST_KEY,{})
        row['prior_attempts']=previous.get('prior_attempts',[])+([
            {k:v for k,v in previous.items() if k!='prior_attempts'}] if previous else [])
        data[HOST_KEY]=row;write(RESULT,data)
    require(row['passed'] and row['ownership_closed'],row.get('error','CPU controls failed'))
    print(json.dumps(row,sort_keys=True),flush=True)


def host():
    row=json.loads((ROOT/'record.json').read_text())
    scope=None
    try:
        _,manifest=current(row)
        require(row['stage']=='prepared' and not row['host_passed'],'fresh host gates only')
        scope=dense.owner.Owner(ROOT/'host-owned');root=ROOT/'host-probe';root.mkdir(mode=0o700)
        for p in ('tmp','cache'):(root/p).mkdir(mode=0o700)
        env=environment(root,False);allowed={v['realpath'] for v in manifest['inventory']['files'].values()}
        reports=[]
        for role,path in (('benchmark','tools/strata_bench.py'),('server','serve/server.py')):
            command=[manifest['selected']['path'],str(benchmark_runtime.SOURCE),'probe',str(P/path),str(Path(row['configuration']['tokenizer'])/'chat_template.jinja')]
            e,out,err=scope.run(role,command,env,P,wall=45,cpu=30,rss_cap=512<<20,text_cap=512<<10,file_cap=2<<20,total_cap=8<<20)
            require(dense.owner.closed(e) and e['exit_code']==0 and err.stat().st_size==0,'frontend probe failed')
            observed=json.loads(out.read_text());require(str(Path(observed['executable']).resolve())==manifest['selected']['file']['realpath'],'frontend Python differs')
            for name in [*observed['modules'],*observed['code_maps']]:
                p=Path(name).resolve();require(str(p.relative_to(P)) in row['sources'] if p.is_relative_to(P) else str(p) in allowed,'unqualified frontend import/map')
            reports.append(dict(role=role,passed=True,modules=len(observed['modules']),code_maps=len(observed['code_maps'])))
        row.update(stage='ready',host_passed=True,host_results=reports)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row['ownership_closed']=scope is None or not scope.active and all(dense.owner.closed(v) for v in scope.commands)
        row['active']=bool(scope and scope.active);save(row)
    require(row['host_passed'] and row['ownership_closed'],row.get('error','host gates failed'))
    print('Both current frontend probes pass;commit before GPU execution.',flush=True)


def run(index):
    ctx=context(index);ctx.run(index)
    row=json.loads((ROOT/'record.json').read_text());process=row['processes'][index]
    try:
        require(ctx.boundary_calls==2,'four warmups and long leaf records required')
        root=Path(row['plans'][index]['root']);path,=(root/'bench').glob('*/requests.jsonl')
        samples=[e['sample'] for e in map(json.loads,path.read_text().splitlines()) if e['kind']=='result']
        require(len(samples)==5,'exactly five completed requests required')
        enabled=ORDER[index]=='candidate'
        native_,parsed=leaf.evidence((root/'engine.stderr').read_text(),samples,4,enabled)
        long=process['samples'][0]
        work=dict(native_cpu_work=native_,tokens=[{k:s[k] for k in leaf.leaves.FIELDS} for s in samples],
            long_output={k:long[k] for k in ('request_sha256','output_sha256','reasoning_sha256')})
        process['resources']=dict(passed=True,leaf_enabled=enabled,completed_requests=5,warmup_prefix=4,
            benchmark_clients=1,work_signature_sha256=signature(work),long_request=leaf.summarize(parsed[-1]) if enabled else None)
        if index==5:row['decision']=leaf.decision(row['processes'])
    except BaseException as error:
        process.update(passed=False,complete=False,resource_error=repr(error));row.update(stage='failed',passed=False,complete=False)
    finally:save(row)
    require(process['passed'],process.get('resource_error','leaf resource binding failed'))


if __name__=='__main__':
    require(len(sys.argv)==2 and (sys.argv[1] in ('controls','prepare','host') or sys.argv[1].isdigit() and 0<=int(sys.argv[1])<6),
        'profile_q6_cpu_leaves.py controls|prepare|host|0..5')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);dense.guards.no_model_owners([])
        if sys.argv[1]=='controls':controls()
        elif sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='host':host()
        else:run(int(sys.argv[1]))
