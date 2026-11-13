#!/usr/bin/env python3
"""Source-pinned CPU scheduling comparison;Main authored,repository LICENSE.

Fresh CPU/GGML objects and unchanged native math. No GPU or model adoption.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode=True
import cpu_compact_pool as base

P=Path(__file__).resolve().parents[2];hot,inputs=base.hot,base.inputs
base.P=hot.P=inputs.P=P
B=base.B;ROOT=B/'cpu-native-whole-down42-v2';PLAN=P/'bench/sycl/cpu_native_schedule/plan.json'
base.ROOT,base.PLAN=ROOT,PLAN
THIS='tools/sycl/cpu_native_schedule.py';KEY='native_whole_down42_v2'
ORDER=['reference','selected','selected','reference','reference','selected']
base.ORDER=ORDER
CPU_RESULT='bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json'
PRIOR_COMMIT='be8d35f85193ff6247924f91cf820c613a521d32'

def prior_qualification():
    def view(value):
        result={k:value[k] for k in ['source_commit','plan_commit','plan_sha256','case_columns','cases','sample_columns','samples','summary','gates','complete','active','ownership_closed','order','pairs','numerical_results','eligibility','execution_receipts','build']}
        result['binary_sha256']=value['binary']['sha256']
        return result
    old=json.loads(hot.git('show',PRIOR_COMMIT+':'+CPU_RESULT))['native_whole_selected_v1']
    committed=json.loads(hot.git('show','HEAD:'+CPU_RESULT))['native_whole_selected_v1']
    actual=json.loads((P/CPU_RESULT).read_text())['native_whole_selected_v1']
    encode=lambda value:json.dumps(view(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    need(encode(old)==encode(committed)==encode(actual),'prior qualification fields/type changed')
    need(old['complete'] and old['ownership_closed'] and not old['active'] and not old['gates']['performance'] and not old['gates']['model_trial_allowed'],'preserve original failed eight-cell gate')
    chosen=[r for r in old['summary'] if old['cases'][r['case']][0]==42]
    need(len(chosen)==4 and all(r['all_three_pairs_gain'] and all(v>1 for v in r['paired_reference_over_candidate']) for r in chosen),'all four original Down42parts must pass independently')
    return dict(commit=PRIOR_COMMIT,key='native_whole_selected_v1',consumed_fields_sha256=hashlib.sha256(encode(old).encode()).hexdigest(),selected_cases=[r['case'] for r in chosen],scope='Only these consumed committed/live fields are frozen;no whole-file equality. Unrelated rows/artifact retirement may change,but missing/altered qualification fails.')
need,sha,write=base.need,base.sha,base.write
PREFIXES=('include/strata','src/kernels/cpu','third_party/ggml','bench/sycl/cpu_native_schedule',
    THIS,'src/core/expert_source.cpp','src/program/generate.cpp','tools/sycl/cpu_compact_pool.py','tools/sycl/cpu_compact_numeric.py',
    'tools/sycl/cpu_compact_hot.py','tools/sycl/measurement_owner.py',
    'tools/sycl/measurement_guards.py','AGENTS.md','docs/ARTIFACT_RETENTION.md')

def fixture_hash(row):
    before=inputs.identity(row['shard']);digest=hashlib.sha256()
    with Path(row['shard']).open('rb',buffering=0) as f:
        for expert in range(64):
            for start,count in zip(row['offsets'],row['extents']):
                pos=start+expert*count;need(0<=pos<=before['bytes'] and count<=before['bytes']-pos,'64expert source bounds')
                f.seek(pos);raw=f.read(count);need(len(raw)==count,'64expert short read');digest.update(raw)
    need(inputs.identity(row['shard'])==before,'model changed during source hash');return digest.hexdigest()
inputs.fixture_hash=fixture_hash

def prepare():
    need(not ROOT.exists() and not PLAN.exists(),'fresh source/plan/root')
    prior=prior_qualification();revision=hot.git('rev-parse','HEAD').decode().strip();paths=hot.git('ls-tree','-r','--name-only',revision,'--',*PREFIXES).decode().splitlines();sources={}
    for name in paths:
        raw=hot.git('show',revision+':'+name);need(raw==(P/name).read_bytes(),'commit input first:'+name);sources[name]=hashlib.sha256(raw).hexdigest()
    need(THIS in sources and 'bench/sycl/cpu_native_schedule/pool.cpp' in sources,'complete controller/fixture snapshot')
    native_path=B/'coherent-prefill-native-v1/record.json';native=json.loads(native_path.read_text());need(native['ownership_closed'] and not native['active'],'retained coherent recipe')
    ggml,=[x for x in native['dependencies'] if x['revision']=='3cf03257f219afbe7334045ff7c6a06ac68c627d']
    recipe=dict(path=str(native_path),sha256=sha(native_path));census=native['actual_inputs'];external=native['external_inputs']
    need(sha(census['path'])==census['sha256'] and sha(external['path'])==external['sha256'],'original native recipe inputs')
    flags,=[x.split('=',1)[1] for x in native['recipe']['configure'] if x.startswith('-DCMAKE_CXX_FLAGS=')];flags=flags.replace('-fsycl-id-queries-range=size_t','').strip()
    previous=B/'cpu-compact-small-batch-pool-v1';old=json.loads((previous/'build.json').read_text());check=json.loads((previous/'check.json').read_text());need(old['passed'] and check['passed'] and old['ownership_closed'] and check['ownership_closed'] and not old['active'] and not check['active'],'qualified CPUfallback')
    fallback=old['binaries']['pool'];need(sha(fallback['path'])==fallback['sha256'],'retained CPUfallback ELF')
    env=dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin',LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib:/usr/lib/x86_64-linux-gnu',LANG='C.UTF-8',LC_ALL='C.UTF-8',TMPDIR=str(ROOT/'tmp'))
    child=dict(env,STRATA_IQ3_PAIR='1',STRATA_IQ3_PAIR_AUDIT='1',STRATA_IQ2_PAIR='0',STRATA_IQ2_PAIR_AUDIT='0',STRATA_IQ_SINGLE_EXACT='0',STRATA_IQ_MT_MIN='2',STRATA_CPU_SERVICE='0',STRATA_CPU_WORK_CENSUS='0')
    headers=base.std_headers();formats=inputs.model_rows()
    for row in formats:row['sha256']=fixture_hash(row);row['fixture_experts']=64
    plan=dict(schema='native-whole-down42-v2',prior_component=prior,source_commit=revision,sources=sources,native_recipe=recipe,compiler_inputs=census,external_inputs=external,
        generated_inputs=inputs.coherent_generated(recipe,census),GGML=ggml,cmake_inputs=inputs.cmake_snapshot(),standard_headers=dict(count=len(headers),sha256=base.signature(headers)),
        extra_headers={x:sha(x) for x in base.EXTRA_HEADERS},topology=base.topology(),tools={x:sha(x) for x in (hot.CXX,inputs.CC,inputs.CM,inputs.NJ,hot.READELF,hot.PRLIMIT,hot.PYTHON)},
        metadata=dict(path=str(inputs.METADATA),sha256=sha(inputs.METADATA)),formats=formats,environment=env,native_environment=child,speed_environment=dict(child,STRATA_IQ3_PAIR_AUDIT='0'),
        qualified_fallbacks=dict(CPU_binary=fallback,scope='Retained original CPUauxiliary generation;no project archive/object reuse.'),
        configure=[inputs.CM,'-S',str(P/'bench/sycl/cpu_native_schedule'),'-B',str(ROOT/'build'),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_C_COMPILER='+inputs.CC,'-DCMAKE_CXX_COMPILER='+hot.CXX,'-DCMAKE_CXX_FLAGS='+flags,'-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t','-DSTRATA_GGML_DIR='+ggml['root']],
        build=[inputs.CM,'--build',str(ROOT/'build'),'--target','cpu_native_schedule','--parallel','3'],expected=dict(compiled_objects=36,exported_commands=38),
        expected_ELF=dict(elf_class='ELF64',data="2's complement, little endian",machine='Advanced Micro Devices X86-64',elf_type='EXEC (Executable file)',interpreter='/lib64/ld-linux-x86-64.so.2'),
        order=ORDER,pairs=[[0,1],[3,2],[4,5]],scope='Actual production run_multi_native entry in both arms;explicit policy false versus true. Whole scheduling only for Decode,5workers+host,24/36jobs,mixed NT1..4,GU18/Down42 and no cache leases. Down20 stays row. Core wiring is committed but no full engine built here. Fresh11CPU+24GGML+onefixture;no GPU or optional compact cache.',
        basis='Original production-wrapper evidencebe8d35f85keeps its eight-cell gate failed. Independently,all four Down42parts gain in all3pairs(1.052688..1.095976),and those parts also passed the earlier whole-method cohort. This new source excludesDown20and requires new actual-entry measurements;it does not relabel the original failure. CPU/RAM saturation and model coverage remain unknown',
        numerical='Both actual Down20/42layers,independent64expert/4activation row oracle. Preserve180direct-method cases/40whole calls per layer. Add360production-selector cases with both policy settings,1or5workers,both phases,n1/6/7/12/24/36/37/97/193,NT1..4orobservedmix;exactly0selected whole calls onDown20and2onDown42. Four lease/span controls preserve existing cache bounds. Two wrong-activation controls,guards/immutable inputs and actual native phase/NT counters;empty call. No model numerical/quality inference.',
        timing='Sixfresh native processes in fixedRCCRRCorder,Down42only. Four mixed24/36job reuse/stream64cells per process through actual production entry. Mixed87/64token-expert rows/job,complete64round workload cycles. Calibrate64times powers2until>=0.5s;3batches each>=0.25s,timer<0.1percent. Six distinct physical cores,5workers+host. Include binding,selector guards,GU/quant/Down and waits;exclude I/O,oracle/checks. Verify actual selected counters each timed batch. No literal model trace or DRAM-cold claim',
        acceptance='Source/build/ELF/numeric/owner gates first,including Down20and Prefill fallback. Each of four selected Down42cells must gain in all3predeclaredR/Cpairs,without minimum percentage. Any failed selected cell blocks this model trial;preserve failure and require new source/plan/evidence for a changed rule. Passing only permits a separately committed coherent full-engine/model-admission and matched long/agent/thinking/prefill-preservation plan;no model adoption or target claim. Original eight-cell failure remains failed',
        stopping='Single-use build/check/measure-0..5;stop first source,recipe,binary,dispatch,numeric,budget,affinity,timer or owner failure. No retry/hash refresh or altered criteria.',
        budgets=dict(configure_wall=90,build_wall=300,check_wall=300,measure_wall=300,cpu=2400,rss=3<<30,output=2<<20,total=384<<20),
        retention='One additional CPUfixture generation for this immediate comparison;keep existing CPUauxiliary and GPUfallbacks. Preserve compact individual speed/setup samples and exact source/recipe,then retire raw output. Work directly on feature/sycl;prior measured child is integrated and retired. After the decision,retain committed useful samples and retire this extra fixture when its immediate full-engine build consumer ends. Remove rejected unused code/exports/fixtures after preserving evidence.')
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();write(PLAN,plan);print('Prepared prospective source-pinned CPU scheduling plan;commit before build.',flush=True)

def verify(plan):
    need(plan['order']==ORDER and prior_qualification()==plan['prior_component'],'predeclared order/prior qualification');base.verify(plan)

def execute(action):
    with (ROOT/(action+'-attempt.json')).open('x') as f:json.dump(dict(action=action,no_retry=True),f)
    state=dict(stage=action,active=True,passed=False,complete=False,GPU_executed=False,commands=[])
    def persist():write(ROOT/(action+'.json'),state)
    persist()
    try:
        plan=json.loads(PLAN.read_text());verify(plan);scope=hot.owner.Owner(ROOT/'owned',persist);state['commands']=scope.commands;state.update(source_commit=plan['source_commit'],plan_sha256=sha(PLAN))
        def run(label,argv,wall,env=None):
            e,out,err=scope.run(action+'-'+label,[hot.PRLIMIT,'--as=8589934592','--',*argv],plan['environment'] if env is None else env,P,wall=wall,text_cap=plan['budgets']['output'],rss_cap=plan['budgets']['rss'],total_cap=plan['budgets']['total'],cpu=plan['budgets']['cpu'],file_cap=plan['budgets']['total'])
            need(hot.owner.closed(e) and e['exit_code']==0,'failed/unclosed:'+label);return out.read_text(),err.read_text()
        if action=='build':
            need(not (ROOT/'build').exists(),'fresh coherent build only');need(sha(plan['qualified_fallbacks']['CPU_binary']['path'])==plan['qualified_fallbacks']['CPU_binary']['sha256'],'CPUfallback continuity')
            run('configure',plan['configure'],plan['budgets']['configure_wall'])
            for n,h in plan['generated_inputs'].items():need(sha(ROOT/'build'/n)==h,'generated header')
            text,_=run('compile',plan['build'],plan['budgets']['build_wall']);run('dependencies',[inputs.NJ,'-C',str(ROOT/'build'),'-t','deps'],20);actual=base.dependencies(plan)
            known={str(Path(x).resolve()):h for x,h in json.loads(Path(plan['external_inputs']['path']).read_text()).items()};links={}
            for line in text.splitlines():
                p=Path(line.strip())
                if not p.is_absolute() or not p.is_file() or p.is_relative_to(ROOT):continue
                n=str(p.resolve());need(n in known and sha(p)==known[n],'uncovered link:'+n);links[n]=known[n]
            need(len(links)==22,'external links');path=ROOT/'build/cpu_native_schedule';text,error=run('ELF',[hot.READELF,'--file-header','--program-headers','--dynamic',str(path)],10)
            libs=re.findall(r'Shared library: \[([^\]]+)\]',text);need(not error and set(libs)=={'libm.so.6','libstdc++.so.6','libgcc_s.so.1','libc.so.6'},'CPU-only ELF libraries')
            def field(name):
                values=re.findall(r'^\s*'+re.escape(name)+r':\s*(.+)$',text,re.M);need(len(values)==1,'ELF field');return values[0].strip()
            interp=re.findall(r'\[Requesting program interpreter: ([^\]]+)\]',text);need(len(interp)==1,'ELF interpreter');elf=dict(elf_class=field('Class'),data=field('Data'),machine=field('Machine'),elf_type=field('Type'),interpreter=interp[0]);need(elf==plan['expected_ELF'],'ELF identity')
            state.update(actual_inputs=actual,links=links,libraries=libs,ELF=elf,binaries=dict(pool=dict(path=str(path),sha256=sha(path))))
        else:
            build=base.qualified_build(plan);results=[]
            if action!='check':
                checked=json.loads((ROOT/'check.json').read_text());need(base.closed(checked,plan) and checked['binaries']==build['binaries'],'numeric gate first');index=int(action.split('-')[1])
                for i in range(6):
                    if i<index:need(base.closed(json.loads((ROOT/f'measure-{i}.json').read_text()),plan),'prior arm gate')
                    elif i>index:need(not (ROOT/f'measure-{i}-attempt.json').exists(),'later arm already attempted')
            for row in plan['formats']:
                if action!='check' and row['down']!=42:continue
                arm='check' if action=='check' else ORDER[int(action.split('-')[1])];args=[row['shard'],str(row['down']),*map(str,row['offsets']),str(row['blob_bytes']),arm]+([] if action=='check' else ['mixed-only']);text,error=run('down'+str(row['down']),[build['binaries']['pool']['path'],*args],plan['budgets']['check_wall'] if action=='check' else plan['budgets']['measure_wall'],plan['native_environment'] if action=='check' else plan['speed_environment']);v=json.loads(text);need(not error and v['passed'],'fixture output')
                if action=='check':need(v['cases']==180 and v['whole_calls']==40 and v['selected_cases']==360 and v['selected_whole_calls']==(2 if row['down']==42 else 0) and v['lease_controls']==4 and v['wrong_activation_controls']==2 and v['guards'] and v['inputs_immutable'] and not v['performance_qualified'],'numeric coverage')
                else:
                    need(v['scope']=='CPU_schedule_only' and v['arm']==arm and v['down']==row['down'] and len(v['cells'])==4 and v['inputs_immutable'] and not v['model_qualified'],'timed scope');seen=set()
                    for c in v['cells']:
                        key=(c['jobs_per_call'],c['NT_policy'],c['pattern']);need(key not in seen and c['workers']==5 and c['host_works'],'cell/topology');seen.add(key);need(c['rounds']>=64 and c['rounds']%64==0 and c['calibration'][-1]['wall_s']>=.5 and c['calibration'][-1]['rounds']==c['rounds'] and len(c['samples'])==3,'calibration/count/complete workload cycles')
                        for s in c['samples']:need(s['wall_s']>=.25 and v['timer_pair_s']/s['wall_s']<.001 and s['calls']==c['rounds'] and s['jobs']==s['calls']*c['jobs_per_call'] and s['whole_calls']==(s['calls'] if arm=='selected' and row['down']==42 and c['jobs_per_call'] in (24,36) and c['NT_policy']=='observed_mix' else 0) and s['jobs']<=s['token_expert_rows']<=4*s['jobs'] and 0<=sum(s[n] for n in ['pool_gu_s','pool_quant_s','pool_down_s','pool_whole_s'])<=s['wall_s']+1e-6,'actual dispatch/timing eligibility')
                    need(seen=={(n,t,p) for n in [24,36] for t in ['observed_mix'] for p in ['reuse','stream64']},'shape coverage')
                results.append(dict(down=row['down'],result=v))
            state.update(results=results,binaries=build['binaries'],numerical_qualified=action=='check',performance_qualified=False)
        verify(plan);state.update(passed=True,complete=True,ownership_closed=True)
    except BaseException as e:state.update(error=repr(e),ownership_closed=all(hot.owner.closed(c) for c in state['commands']));raise
    finally:state['active']=False;persist()

if __name__=='__main__':
    need(sys.executable==hot.PYTHON and len(sys.argv)==2 and sys.argv[1] in ['prepare','build','check',*[f'measure-{i}' for i in range(6)]],'qualified Python/action')
    with (B/'owned-v0141-measurement.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);hot.guards.no_model_owners([]);prepare() if sys.argv[1]=='prepare' else execute(sys.argv[1])
