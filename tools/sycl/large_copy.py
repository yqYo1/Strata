#!/usr/bin/env python3
"""Main-only source-kind/copy-granularity diagnostic; no inference adoption."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode=True
import query_profile as q
import generation_engine as gen
import cpu_capacity as assets
import cpu_census_engine as faults
import measurement_guards as guards
import measurement_owner as owner_lib
from check_verifier_retirement import DELTA

P,B,A=gen.P,gen.B,gen.A
ROOT=B/'large-h2d-source-v1'
RESULT=A/'large-copy-current.json'
BYTES=1538035200
MIB=1<<20
CONFIGS=[(s,c)for c,ss in [(0,('vector','host-usm','stage')),(64*MIB,('stage','host-usm','vector')),
    (256*MIB,('vector','host-usm','stage'))]for s in ss]
ORDER=[CONFIGS,list(reversed(CONFIGS)),CONFIGS[3:]+CONFIGS[:3]]


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def parse(text,source,chunk,count,n,timed):
    rows=[json.loads(s)for s in text.splitlines()]
    guards.require(len(rows)==count+2 and rows[0]['kind']=='identity' and rows[-1]==
        dict(kind='closed',passed=True,owners_retired=True),'incomplete copy output/owner footer')
    identity=rows[0];samples=rows[1:-1]
    guards.require(identity['source']==source and identity['bytes']==n and identity['chunk']==chunk and
        identity['samples']==count and identity['unpressured'] is True and
        identity['physical_bus_traffic_measured'] is False and identity['GPU_service_measured'] is False,
        'copy identity/measurement scope differs')
    guards.require(all(type(identity.get(k)) is int and 0<=identity[k]<65536 for k in
        ('source_mod_65536','destination_mod_65536')),'missing/invalid pointer geometry')
    for field in ('setup_wall_s','setup_cpu_s','timer_max_s'):
        guards.require(type(identity[field]) in (int,float) and math.isfinite(identity[field]) and identity[field]>=0,'invalid setup/timer clock')
    for i,s in enumerate(samples):
        guards.require(s['kind']=='sample' and s['index']==i and s['first_copy'] is (i==0) and
            s['copies']==(n+(chunk or n)-1)//(chunk or n) and s['logical_bytes']==n and
            s['payload_match'] is True and s['guards_match'] is True,'copy sample/payload/guard/count mismatch')
        for key in ('wall_s','cpu_s','stage_wall_s','stage_cpu_s','submit_wall_s','submit_cpu_s','drain_wall_s','drain_cpu_s'):
            guards.require(type(s[key]) in (int,float) and math.isfinite(s[key]) and s[key]>=0,'invalid copy phase clock')
        for clock in ('wall','cpu'):
            guards.require(abs(s[clock+'_s']-sum(s[k+'_'+clock+'_s']for k in ('stage','submit','drain')))<=8e-9,'copy phase conservation failed')
        guards.require(s['wall_s']>0 and s['cpu_s']<=s['wall_s']+.001,'copy CPU/wall clock mismatch')
        if timed:guards.require(s['wall_s']>=.1 and identity['timer_max_s']<=min(.001,.01*s['wall_s']),'ineligible timing floor/timer overhead')
    return dict(identity=identity,samples=samples,
        subsequent_median_s=statistics.median(s['wall_s']for s in samples[1:])if len(samples)>1 else None,
        timing_eligible=timed)


def decision(runs):
    guards.require(len(runs)==3 and all(r['passed'] and len(r['processes'])==9 for r in runs),'three closed copy cohorts required')
    candidates=[]
    for source,chunk in CONFIGS:
        if (source,chunk)==('vector',0):continue
        ratios=[]
        for cohort in runs:
            mapping={(r['source'],r['chunk']):r['subsequent_median_s']for r in cohort['processes']}
            guards.require(len(mapping)==9 and all(math.isfinite(x) and x>0 for x in mapping.values()),'incomplete copy matrix')
            ratios.append(mapping['vector',0]/mapping[source,chunk])
        mean=math.exp(sum(math.log(x)for x in ratios)/3)
        candidates.append(dict(source=source,chunk=chunk,ratios=ratios,geomean=mean,
            eligible=min(ratios)>=1.02 and mean>=1.05))
    eligible=[x for x in candidates if x['eligible']]
    return dict(candidates=candidates,selected=max(eligible,key=lambda x:x['geomean'])if eligible else None,
        scope='Component screen only; unpressured host-pointer reuse after full readback,not model speed,physical bus saturation or inference adoption.')


def pins(revision):
    p=gen.benchmark_pins(revision);name='tests/core/test_large_copy.py';raw=gen.git('show',revision+':'+name)
    guards.require(raw==(P/name).read_bytes(),'uncommitted copy host test');p[name]=hashlib.sha256(raw).hexdigest();return p


def environment(root,mode):
    env=dict(gen.committed(gen.SETTINGS)['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
    if mode!='small':
        env={k:v for k,v in env.items()if k!='UR_ENABLE_LAYERS' and not k.startswith('UR_LOG_')}
        if mode=='large':
            env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
            for k in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):env[k]='level:warning;flush:warning;output:stderr'
    guards.require(all(env[k]==v for k,v in gen.committed(gen.SETTINGS)['environment'].items()
        if k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE')),
        'qualified copy safety environment changed')
    return env


def current(record):
    full,count=q.current_build()
    guards.require(full['source_commit']==record['native_source'] and count==record['native_inputs_recovered'] and
        full['fixtures']['sycl_large_h2d_bench']==record['fixture'] and
        gen.common.digest(q.BUILD_RECORD)==record['build_sha256'] and gen.common.digest(q.COMPONENT)==record['component_sha256'] and
        gen.common.digest(gen.SETTINGS)==record['settings_sha256'] and assets.machine_identity()==record['machine'] and
        assets.fingerprint(pins(record['source_commit']))==record['controller_fingerprint'],'copy qualification/source drift')
    settings=gen.committed(gen.SETTINGS)
    for name,item in settings['runtime_elfs'].items():
        if name in DELTA:guards.require(DELTA[name][0]==item['sha256'],'original runtime baseline differs')
        expected=DELTA[name][1]if name in DELTA else item['sha256']
        guards.require(gen.common.digest(name)==expected,'copy runtime changed: '+name)
    return full


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh single-use copy experiment only')
    full,count=q.current_build();revision=gen.git('rev-parse','HEAD').decode().strip()
    host=gen.committed(A/'prefill-recovery-current.json')['large_copy_parser_host']
    guards.require(host['passed'] and host['ownership_closed'] and host['tests']==4 and
        host['controller_sha256']==gen.common.digest(P/'tools/sycl/large_copy.py') and
        host['test_sha256']==gen.common.digest(P/'tests/core/test_large_copy.py'), 'matching passed copy parser/selection host tests required')
    record=dict(stage='prepared',source_commit=revision,native_source=full['source_commit'],native_inputs_recovered=count,
        fixture=full['fixtures']['sycl_large_h2d_bench'],build_sha256=gen.common.digest(q.BUILD_RECORD),
        component_sha256=gen.common.digest(q.COMPONENT),settings_sha256=gen.common.digest(gen.SETTINGS),
        machine=assets.machine_identity(),controller_fingerprint=assets.fingerprint(pins(revision)),
        parser_host=host,
        plan=dict(bytes=BYTES,source_chunk_order=ORDER,samples_per_process=6,
            qualification='Six traced2MiB children (three source kinds,whole/1MiB),then nine parameter-validated full-size children (one sample/configuration). Stop before clean timing on any failure.',
            comparison='Three fixed clean cohorts,9 independent native processes each. Record first copy separately and five subsequent samples/process. Reset and full guard/payload readback outside each timed copy. Staging includes its full std::memcpy. Allocations/source construction outside timing; setup clocks retained. First-copy means post-initialization,not driver cold start. Full readback may perturb later samples; source kind also bundles pointer alignment/import behavior. No model or device-memory-pressure simulation.',
            timing='Every full timed sample>=100ms;1000-pair maximum stamp overhead<=1ms and<=1% of each sample. No retries/replaced samples. Failure preserves extracted samples and closes the experiment.',
            selection='Compare each alternative to vector whole within each cohort. Require every cohort throughput ratio>=1.02 and geometric mean>=1.05. Select largest eligible geomean for a separate model-specific plan only. No clean speed-based early stopping.',
            budgets=dict(wall_s=180,cpu_s=600,rss_bytes=12<<30,text_bytes=64<<20,total_bytes=64<<20),
            retention='Keep individual source/shape/phase samples,setup/timer/alignment,source/runtime and fault/owner outcomes. Retire successful full traces and frozen metadata after verified committed closure. Preserve6e40 fallback; overwrite only the recoverable comparison cache.'),qualification=None,runs=[],decision=None)
    current(record);ROOT.mkdir(mode=0o700);write(ROOT/'frozen.json',record);write(RESULT,record)


def execute(qualify):
    record=gen.committed(RESULT);current(record);frozen=json.loads((ROOT/'frozen.json').read_text());mutable={'stage','qualification','runs','decision'}
    guards.require({k:v for k,v in record.items()if k not in mutable}=={k:v for k,v in frozen.items()if k not in mutable},'frozen copy plan differs')
    guards.require(record['stage']==('prepared'if qualify else'qualified') and
        (qualify or record['qualification']['passed'] and len(record['runs'])<3 and all(r['passed']for r in record['runs'])),
        'closed/failed copy cohort; never retry')
    root=ROOT/('checks'if qualify else'cohort-'+str(len(record['runs'])))
    guards.require(not root.exists(),'existing attempt; inspect exact owner');root.mkdir()
    for name in ('tmp','cache'):(root/name).mkdir()
    boot,cursor=faults.start_fault_scope();row=dict(passed=False,complete=False,processes=[]);supervisor=None
    def persist():row['active']=bool(supervisor and supervisor.active is not None);write(root/'record.json',dict(row,commands=supervisor.commands if supervisor else []))
    supervisor=owner_lib.Owner(root/'owned',persist)
    cases=([('small',s,c,2,2*MIB)for c in (0,MIB)for s in ('vector','host-usm','stage')]+
           [('large',s,c,1,BYTES)for s,c in CONFIGS])if qualify else[('clean',s,c,6,BYTES)for s,c in ORDER[len(record['runs'])]]
    try:
        for i,(mode,source,chunk,count,n)in enumerate(cases):
            item=dict(mode=mode,source=source,chunk=chunk,bytes=n,passed=False);row['processes'].append(item)
            command=[record['fixture']['path'],source,str(chunk),str(count),str(n)]
            entry,out,err=supervisor.run(str(i),command,environment(root,mode),P,wall=180,cpu=600,rss_cap=12<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
            # Preserve all recoverable sample lines BEFORE any rejecting validator.
            item['samples']=[]
            malformed=0
            for line in out.read_text().splitlines():
                if not line.startswith('{"kind":"sample"'):continue
                try:item['samples'].append(json.loads(line))
                except json.JSONDecodeError:malformed+=1
            guards.require(not malformed,'malformed copy sample line; earlier samples preserved')
            guards.require(owner_lib.closed(entry) and entry['exit_code']==0 and not any(x in err.read_text()for x in ('UR_RESULT_ERROR_','ZE_RESULT_ERROR_','terminate called','large_h2d failed:')),'copy process failed')
            item.update(parse(out.read_text(),source,chunk,count,n,not qualify));item.update(passed=True,exit_code=0,ownership_closed=True)
            current(record);persist()
        row.update(passed=True,complete=True)
    except BaseException as error:row['error']=str(error)
    finally:
        row['ownership_closed']=supervisor.active is None and all(owner_lib.closed(c)for c in supervisor.commands)
        row.update(fault_probe_successful=False,fresh_faults=None,boot_unchanged=False)
        try:row.update(faults.fault_evidence(cursor,boot))
        except Exception as error:row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and row['fresh_faults']==[] and row['boot_unchanged']);persist()
        if qualify:record['qualification']=row
        else:record['runs'].append(row)
        record['stage']='qualified'if row['passed'] and (qualify or len(record['runs'])<3)else'closed'
        if record['stage']=='closed' and row['passed']:
            try:record['decision']=decision(record['runs'])
            except Exception as error:record['decision_error']=str(error)
        write(RESULT,record)
    print(json.dumps({k:row.get(k)for k in ('passed','complete','ownership_closed','error')}));return 0 if row['passed'] and not record.get('decision_error') else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','qualify','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare()if args.action=='prepare'else execute(args.action=='qualify')
if __name__=='__main__':sys.exit(main())
