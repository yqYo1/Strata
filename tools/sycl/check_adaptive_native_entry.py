#!/usr/bin/env python3
"""Exercise real positive replacements and a subsequent occupied prefill loan.

Main-authored; repository LICENSE. Two synthetic requests,not a speed or text
quality benchmark. No custom Interactive.poll closure or full API trace.
"""
import copy
import faulthandler
import fcntl
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode=True
import adaptive_native_admission as admission
import check_mtp_width_model as width
import cpu_capacity as assets
import engine_once
import native_request_protocol as protocol
import profile_split
import run_q6_columns as validation
import runtime_probe
import ur_validation

P,B,RESULT,base=admission.P,admission.B,admission.RESULT,admission.base
ROOT=B/'adaptive-audit-entry-v1'
KEY='adaptive_residency_native_entry_v1'
SELF=Path(__file__).resolve()
require,digest,write=admission.require,admission.digest,admission.write
CASES=((1024,256),(32769,256))


def compact(row):
    value={k:v for k,v in row.items() if k not in ('sources','commands')}
    value['source_inputs']=dict(files=len(row['sources']),sha256=width.signature(row['sources']))
    return value


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def ids(n):return [248045]+[13]*(n-1)


def configuration():
    cohort=json.loads((width.SPEED_ROOT/'record.json').read_text());plan=cohort['plans'][0]
    env=width.generation_context(0).environment(cohort,plan)
    require(width.signature(env)==plan['environment_sha256'],'recover exact original environment')
    env={k:v.replace(plan['root'],str(ROOT)) for k,v in env.items()}
    env.update(STRATA_TRACE='1',STRATA_TRACE_ADAPT='1',STRATA_SYCL_ADAPT_AUDIT='1',
        STRATA_SYCL_CACHE_LOAN_AUDIT='1',STRATA_ADAPT_NOWAIT='0')
    config=copy.deepcopy(cohort['configuration']);args=config['args']
    require(args.count('--adapt-swaps')==1 and args[args.index('--adapt-swaps')+1]=='0', 'original static cache config differs')
    args[args.index('--adapt-swaps')+1]='8'
    require('--adapt-every' not in args and '--eos-ids' not in args,'explicit new diagnostic selectors only')
    args+=['--adapt-every','8','--eos-ids','248319']
    config['log']=str(ROOT/'owned/native.stderr')
    return config,env,cohort['model_assets']


def startup(line):
    require(line.startswith('INFO '),'native INFO missing')
    values={}
    for item in line.split()[1:]:
        require(item.count('=')==1,'INFO syntax');key,value=item.split('=',1)
        require(key and key not in values,'duplicate INFO field');values[key]=value
    wanted=dict(context='131072',kv='int8',kv_resident='0',expert_slots='1256',spec='4',mtp_max='4',
        pool_workers='5',pcie_frac='0.00',spec_min_p='0.00')
    require(all(values.get(k)==v for k,v in wanted.items()) and int(values.get('arena_mib','0'))>0,
            'actual startup placement/configuration differs')
    return {k:values[k] for k in (*wanted,'arena_mib')}


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT),'one fresh positive entry')
    host_controls=controls()
    native,device,providers=admission.native();config,env,expected=configuration()
    models=assets.model_pins(config,expected,full=True)
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    sources=base.controls(revision,controller=SELF)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    row=dict(stage='prepared',source_commit=revision,sources=sources,host_controls=host_controls,
        native_source_commit=native['source_commit'],binary=native['binaries']['strata'],
        build_record=dict(path=str(admission.build.ROOT/'record.json'),sha256=digest(admission.build.ROOT/'record.json')),
        native_validation_probe=device['native_validation_probe'],runtime_manifest=device['runtime_manifest'],
        boot=device['boot'],driver=device['driver'],configuration=config,environment=env,model_assets=models,
        python=dict(path=str(Path(sys.executable).resolve()),sha256=digest(sys.executable),version=sys.version),
        requests=[dict(request=i+1,input_tokens=n,output_tokens=m,input_sha256=width.signature(ids(n)))
                  for i,(n,m) in enumerate(CASES)],
        budgets=dict(wall=1500,cpu=7200,rss_cap=104<<30,text_cap=8<<20,file_cap=32<<20,total_cap=256<<20),
        whole_wall_s=2100,passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,GPU_execution_attempted=False,model_executed=False,
        plan='One fresh owned direct serving engine,two synthetic requests1024/256 then32769/256. Commit exact inputs/source/native/environment first. Same original cap4/half1/INT8resident131072/1256slots/5workers/8192chunk/throughput BF16 HC/GDN16/MTP16/20ms pool spin/runtime safety;only positive8swaps/every8,rare EOS and opt-in trace/payload/map/dispatch/loan audits are added before exec. Require INFO full-resident0,READY,complete ordered RESUME/PP/REUSED/T/DONE for both,normalQUIT0,strict pre-main validation,exact current PID/environment/providers,completed actual T1/T4 Q6 windows per request,real prepared/masked/published positive generations with exact tuple/count conservation,incoming GPU dispatch observed and valid CPU/GPU job checks,second-request occupied post-positive loan/refill with complete payload/maps/ready checks,unchanged final15asset hashes/boot/driver,no fresh GPU fault and closed owners. Rare EOS controls extent,not quality. Keep original failure status;no retry,hash refresh,full trace,profiler or runtime safety change. Native and post-processing progress persist separately for any host fault.',
        limits='Synthetic native correctness only;outgoing replacement CPU observations are recorded separately. No thinking/task quality,clean/sustained PP/TG,physical resource saturation,ordinary provider teardown or262144/RESTORE qualification. Checked record-failure injection still needs a separate native control before clean adaptive comparisons.',
        retention='Keep compact per-request extent/hash,Q6 and replacement/dispatch/loan conservation,source/configuration and validation/fault outcome. Retire successful full output/JIT after its exact next comparison/qualification consumer;retain one minimal distinct failure diagnostic only.')
    save(row);print('Prepared two-request positive-swap and post-positive32K loan check;commit before execution.',flush=True)


def current(frozen,full=False):
    require(base.committed(RESULT)[KEY]==compact(frozen) and
            base.controls(frozen['source_commit'],controller=SELF)==frozen['sources'],'native entry source/plan changed')
    native,device,providers=admission.native()
    require(native['binaries']['strata']==frozen['binary'] and
            digest(frozen['build_record']['path'])==frozen['build_record']['sha256'] and
            device['native_validation_probe']==frozen['native_validation_probe'] and
            device['runtime_manifest']==frozen['runtime_manifest'],'native/runtime qualification changed')
    config,env,models=configuration()
    require(config==frozen['configuration'] and env==frozen['environment'] and models==frozen['model_assets'] and
            frozen['python']==dict(path=str(Path(sys.executable).resolve()),sha256=digest(sys.executable),version=sys.version),
            'configuration/assets/interpreter changed')
    assets.model_pins(config,models,full=full)
    return providers


def live(entry,row,expected,providers):
    p=Path('/proc',str(entry['pid']))
    require((p/'exe').resolve()==Path(row['binary']['path']).resolve(),'native PID/ELF changed')
    env=dict(v.split('=',1) for v in (p/'environ').read_text().split('\0') if '=' in v)
    require(all(env.get(k)==v for k,v in expected.items()) and set(env)-set(expected)<={'PWD','OLDPWD','SHLVL','_'},
            'native startup environment changed')
    require(runtime_probe.executable_mappings((p/'maps').read_text())<=
        set(providers)|{row['binary']['path'],row['native_validation_probe']['library']['path']},'unexpected provider')


def audit(text):
    prepared={};masked={};published={};footers=[];loans=[]
    patterns=[re.compile(v) for v in (
        r'strata serve: ADAPT_AUDIT phase=prepared generation=(\d+) incoming=(\d+) outgoing=(\d+) slot=(\d+)',
        r'strata serve: ADAPT_AUDIT phase=masked generation=(\d+) swaps=(\d+) identities=3',
        r'strata serve: ADAPT_AUDIT phase=published generation=(\d+) swaps=(\d+) identities=3 payload_match=1',
        r'strata serve: ADAPT_AUDIT phase=dispatch request=(\d+) published=(\d+) calls=(\d+) incoming_vram_entries=(\d+) outgoing_cpu_entries=(\d+) post_swap_refills=(\d+)',
        r'strata serve: ADAPT_AUDIT phase=loan-after-positive completed=(\d+) lent=(\d+) identities=3 payload_match=1')]
    for line in text.splitlines():
        if not line.startswith('strata serve: ADAPT_AUDIT '):continue
        matches=[(i,p.fullmatch(line)) for i,p in enumerate(patterns) if p.fullmatch(line)]
        require(len(matches)==1,'malformed audit progress')
        kind,m=matches[0];v=tuple(map(int,m.groups()));require(all(0<=n<2**64 for n in v),'audit counter bounds')
        if kind==0:
            require(v[0]==len(published)+1 and v[0] not in masked,'prepare generation/order')
            prepared.setdefault(v[0],[]).append(v[1:])
        elif kind in (1,2):
            target=masked if kind==1 else published
            require(v[0] not in target and v[0] in prepared and
                    (kind==1 or v[0] in masked),'duplicate or unordered generation stage')
            target[v[0]]=v[1]
        elif kind==3:
            require(v[0]==len(footers)+1 and v[1]==len(published),'footer generation/order')
            footers.append(v)
        else:
            require(v[0]==len(published)>0,'loan before positive publication');loans.append(v)
    require(prepared and set(prepared)==set(masked)==set(published)==set(range(1,len(prepared)+1)),
            'incomplete positive replacement generations')
    for generation,rows in prepared.items():
        require(0<len(rows)<=8 and len(rows)==masked[generation]==published[generation] and
                len(set(v[0] for v in rows))==len(rows) and len(set(v[1] for v in rows))==len(rows) and
                len(set(v[2] for v in rows))==len(rows) and all(
                    i!=o and i//512==o//512 and i<24576 and o<24576 and s<1256 for i,o,s in rows),
                'replacement tuple/count conservation')
    require(len(footers)==2 and [v[0] for v in footers]==[1,2] and
            0<footers[0][1]<=footers[1][1]==len(prepared) and footers[0][2]<footers[1][2] and
            footers[0][3]>0 and footers[1][3]>footers[0][3] and footers[0][5]==0 and footers[1][5]>0 and
            loans and len(loans)==footers[1][5] and all(g>=footers[0][1] and n>0 for g,n in loans),
            'actual positive dispatch or post-positive occupied refill not exercised')
    return dict(generations=len(prepared),swaps=sum(map(len,prepared.values())),
        request_coverage=[dict(request=r,published=g,dispatch_calls=c,incoming_GPU_entries=i,
            outgoing_CPU_entries=o,post_swap_refills=l) for r,g,c,i,o,l in footers],
        post_positive_loan_groups=len(loans),outgoing_CPU_observed=footers[-1][4]>0,
        all_observed_payload_map_dispatch_checks_passed=True)


def controls():
    p='strata serve: ADAPT_AUDIT '
    good=[p+'phase=prepared generation=1 incoming=10 outgoing=3 slot=4',
        p+'phase=prepared generation=1 incoming=11 outgoing=4 slot=5',
        p+'phase=masked generation=1 swaps=2 identities=3',
        p+'phase=published generation=1 swaps=2 identities=3 payload_match=1',
        p+'phase=dispatch request=1 published=1 calls=48 incoming_vram_entries=3 outgoing_cpu_entries=1 post_swap_refills=0',
        p+'phase=loan-after-positive completed=1 lent=188 identities=3 payload_match=1',
        p+'phase=dispatch request=2 published=1 calls=96 incoming_vram_entries=8 outgoing_cpu_entries=2 post_swap_refills=1']
    audit('\n'.join(good))
    corrupt=[good[:3]+good[4:],good[:2]+good[3:4]+good[2:3]+good[4:],
        [good[0].replace('generation=1','generation=2'),*good[1:]],
        [good[0],good[0],*good[2:]],
        [good[0].replace('slot=4','slot=1256'),*good[1:]],
        [*good[:2],good[2].replace('swaps=2','swaps=1'),*good[3:]],
        [*good[:3],good[3].replace('swaps=2','swaps=1'),*good[4:]],
        [*good[:4],good[4].replace('request=1','request=2'),*good[5:]],
        [good[5],*good[:5],good[6]],
        [*good[:5],good[5].replace('lent=188','lent=0'),good[6]],
        [*good[:4],good[4].replace('incoming_vram_entries=3','incoming_vram_entries=0'),*good[5:]],
        [*good[:6],good[6].replace('calls=96','calls=48')],
        [*good[:6],good[6].replace('incoming_vram_entries=8','incoming_vram_entries=3')],
        good[:-1],good+[good[-1]],[*good[:6],good[6]+' unknown=1']]
    for lines in corrupt:
        try:audit('\n'.join(lines))
        except RuntimeError:continue
        raise RuntimeError('corrupt adaptive audit control accepted')
    return dict(passed=True,complete_positive_fixture=True,negative=len(corrupt),GPU_executed=False,
        scope='Ordered native diagnostic reducer only;no actual copies/dispatch/loan or GPU execution.')


def run():
    row=json.loads((ROOT/'record.json').read_text());frozen=copy.deepcopy(row)
    require(row['stage']=='prepared' and not (ROOT/'execution.json').exists() and faulthandler.is_enabled(),
            'one fresh entry with -X faulthandler required')
    outcome=dict(passed=False,complete=False,host_probe_accepted=False,requests=[]);owner=None;cursor=None
    def persist():write(ROOT/'execution.json',dict(outcome,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    with (ROOT/'execution.json').open('x') as stream:stream.write('{}\n')
    started=time.monotonic()
    try:
        cursor=profile_split.cursor();providers=current(frozen)
        env=engine_once.native_environment(row,row['environment']);owner=base.supervision.Owner(ROOT/'owned',persist)
        def interact(io,native):
            outcome['startup']=startup(io.line(600));require(io.line(600)=='READY 131072 stop','native READY differs')
            live(native,row,env,providers);outcome['native_pid']=native['pid'];persist()
            for ordinal,(n,m) in enumerate(CASES,1):
                io.send((f'GEN {m} '+','.join(map(str,ids(n)))+'\n').encode())
                reader=protocol.RequestProtocol(n,m)
                while reader.done is None:reader.feed(io.line(600))
                live(native,row,env,providers)
                outcome['requests'].append(dict(request=ordinal,input_tokens=n,output_tokens=m,
                    input_sha256=width.signature(ids(n)),output_sha256=width.signature(reader.tokens),
                    progress_notifications=reader.progress_count,progress_reached=reader.position,finish='length'))
                persist()
            io.send(b'QUIT\n');io.proc.stdin.close();io.exit(90)
        row.update(active=True,ownership_closed=False,GPU_executed=None,GPU_execution_attempted=True,model_executed=None)
        write(ROOT/'record.json',row)
        native,out,err=owner.run('native',[row['binary']['path'],'--serve',*row['configuration']['args']],
            env,P,interaction=interact,**row['budgets'])
        outcome['native_exit_code']=native['exit_code'];outcome['native_normal_closed']=base.supervision.closed(native)
        persist();require(outcome['native_normal_closed'] and native['exit_code']==0,'native positive entry did not close normally')
        text=err.read_text();outcome['validation']=validation.validation_log(text,native['pid'],row['native_validation_probe']['loader']['path'],False)
        require(not re.search(r'ADAPT_AUDIT_FAILURE|terminate called|retirement failed|Strata SYCL cleanup:',text), 'native audit/runtime failure')
        outcome['adaptive']=audit(text);outcome['Q6']=[]
        for request in (1,2):
            lines=[]
            for line in text.splitlines():
                if not line.startswith(('strata Q6_ROUTE ','strata Q6_SHAPE ')):continue
                match=re.match(r'strata Q6_(?:ROUTE|SHAPE) request=(\d+) ',line)
                require(match and int(match[1]) in (1,2),'unexpected Q6 request identity')
                if int(match[1])==request:lines.append(line)
            snap=width.short.snapshots('\n'.join(lines),1,request)
            require({v['tokens'] for v in snap['windows'] if v['phase']=='decode'}=={1,4} and
                    sum(v['replay_completed'] for v in snap['windows'] if v['phase']=='decode')>=24,'actual T1/T4 extent differs')
            outcome['Q6'].append(dict(request=request,completed_heads=len(snap['completed_head_capture_ids'])))
        outcome.update(host_probe_accepted=True,complete=True);persist()
        current(frozen,full=True);require(time.monotonic()-started<row['whole_wall_s'],'whole entry deadline')
    except BaseException as error:outcome['error']=repr(error)
    finally:
        if owner is not None and cursor is not None:ur_validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed']=bool(outcome.get('passed') and outcome['complete'] and len(outcome['requests'])==2 and
            outcome.get('driver_after')==row['driver'] and 'error' not in outcome)
        row.update(stage='closed' if outcome['passed'] else 'failed',passed=outcome['passed'],complete=outcome['complete'],
            active=bool(owner and owner.active),ownership_closed=outcome.get('ownership_closed',owner is None),
            outcome=outcome,commands=owner.commands if owner else [],
            GPU_executed=True if outcome['requests'] else None if row['GPU_execution_attempted'] else False,
            model_executed=True if outcome['requests'] else None if row['GPU_execution_attempted'] else False)
        persist();save(row)
    require(row['passed'],str(outcome));print('Positive replacement/actual dispatch and post-positive32K loan close normally;no speed claim.',flush=True)


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run'],['controls']),'use prepare|run|controls')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1]=='controls':print(json.dumps(controls()))
        else:(prepare if sys.argv[1]=='prepare' else run)()
