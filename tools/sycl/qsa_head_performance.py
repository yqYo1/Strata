#!/usr/bin/env python3
"""Bounded full-path normalizer comparison,after lossless numerical qualification.

Own project code,repository LICENSE. Synthetic varied operands and selected-ID
patterns are not an actual-model throughput or hardware saturation measurement.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
sys.dont_write_bytecode=True
import qsa_head_native as n
import qsa_normalizer_check as numeric
q,r,P,B=n.q,n.r,n.P,n.B
ROOT=n.ROOT/'performance-v1'
PERF='performance'
RESULT=n.RESULT
CPP='bench/sycl/qsa_head_bench.cpp'
require,digest,write=n.require,n.digest,n.write


def sources(revision):return q.sources(revision,Path(__file__).resolve(),[CPP,n.FIXTURE,n.ORACLE])


def qualification():
    record=r.validation.committed(RESULT);component=record[n.KEY]
    require(component['properties4']['outcome']['passed'],'compiled native properties first')
    for key in ['tiny4','component4','component16']:
        o=component[numeric.KEY][key]['outcome'];require(o['passed'] and o['complete'] and o['ownership_closed'] and not o['fresh_faults'],'finish all lossless numerical gates first:'+key)
    numeric.built();numeric.contract()
    row=json.loads((n.ROOT/'record.json').read_text());require(n.compact(row)==component['build'] and row['passed'] and row['ownership_closed'],'exact native build required')
    allowed,rt=n.admitted(row);return record,row,allowed,rt


def compact(row):return {k:v for k,v in row.items() if k not in ['source_sha256','commands_executed']}


def admitted(row):
    record,native,allowed,rt=qualification()
    require(sources(row['source_commit'])==row['source_sha256'],'committed performance inputs changed')
    require(digest(row['input_inventory']['path'])==row['input_inventory']['sha256'],'prospective input inventory changed')
    inputs=json.loads(Path(row['input_inventory']['path']).read_text())
    for p,h in inputs.items():require(digest(p)==h,'performance compiler/object/source input changed:'+p)
    if 'binary' in row:require(digest(row['binary']['path'])==row['binary']['sha256'],'benchmark ELF changed')
    return record,native,inputs,rt


def prepare():
    record,native,allowed,rt=qualification();require(not ROOT.exists() and PERF not in record[n.KEY],'fresh performance build only')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip();pins=sources(revision)
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir();(ROOT/'cache').mkdir()
    compile_base=next(argv for label,argv in native['commands'] if label=='original');flags=compile_base[1:compile_base.index('-MD')]
    compile_command=[native['compiler']['path'],*flags,'-MD','-MF',str(ROOT/'bench.d'),'-c',str(P/CPP),'-o',str(ROOT/'bench.o')]
    link=list(next(argv for label,argv in native['commands'] if label=='link4'))
    link[link.index(str(n.ROOT/'oracle4.o'))]=str(ROOT/'bench.o');link[link.index('-o')+1]=str(ROOT/'bench')
    for path in [Path(v) for v in link if v.endswith('.o') and Path(v)!=ROOT/'bench.o']:
        require(path.is_file() and path.is_relative_to(n.ROOT),'unexpected kernel/runtime object');allowed[str(path)]=digest(path)
    allowed.update({str(P/name):h for name,h in pins.items()});write(ROOT/'input-inventory.json',allowed)
    row=dict(stage='prepared',source_commit=revision,source_sha256=pins,native_build_source=native['source_commit'],native_geometry=native['geometry_proof'],runtime_sha256=digest(r.RESULT),boot=rt['boot'],commands=[['compile',compile_command],['link',link]],environment=dict(native['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),input_inventory=dict(path=str(ROOT/'input-inventory.json'),sha256=digest(ROOT/'input-inventory.json'),files=len(allowed)),budgets=dict(wall=600,cpu=1800,rss_cap=8<<30,text_cap=32<<20,file_cap=256<<20,total_cap=2<<30),passed=False,active=False,ownership_closed=True,criteria='CPU fixture build only from qualified exact native kernel objects. Actual compiler and linker inputs frozen;no GPU trial before this build closes. Synthetic fourcase Page4query8192/cap2051 performance is separate from model evidence.',retention='Same imminent normalizer comparison generation. Keep only compact source/recipe,samples and decision;retire ELF/objects/normal raw output at comparisonclosure.')
    admitted(row);write(ROOT/'record.json',row);record[n.KEY][PERF]=dict(build=compact(row));write(RESULT,record);print('Performance build prepared;commit before compile.',flush=True)


def build():
    record=r.validation.committed(RESULT);row=json.loads((ROOT/'record.json').read_text());require(row['stage']=='prepared' and compact(row)==record[n.KEY][PERF]['build'],'fresh committed build plan');_,native,allowed,_=admitted(row);owner=None
    def persist():write(ROOT/'record.json',dict(row,active=bool(owner and owner.active is not None),commands_executed=owner.commands if owner else []))
    owner=r.supervision.Owner(ROOT/'build-owned',persist)
    try:
        for label,argv in row['commands']:
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,**row['budgets']);require(r.supervision.closed(entry) and entry['exit_code']==0,'performance compile/link failed:'+err.read_text()[-3000:])
        observed={}
        for dep in shlex.split((ROOT/'bench.d').read_text().replace('\\\n',' ').split(':',1)[1]):
            path=Path(dep).resolve();sha=digest(path);require(allowed.get(str(path))==sha,'unfrozen compiler dependency:'+str(path));observed[str(path)]=sha
        require(str(P/CPP) in observed and len(observed)>200,'incomplete compiler closure');links=q.linking.traced_inputs(out.read_text(),err.read_text(),allowed,ROOT,ROOT)
        write(ROOT/'compiler-inputs.json',observed);write(ROOT/'link-inputs.json',links);binary=ROOT/'bench'
        row.update(stage='built',passed=True,binary=dict(path=str(binary),sha256=digest(binary),bytes=binary.stat().st_size),actual_compiler_inputs=dict(files=len(observed),path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),uncovered=0),actual_link_inputs=dict(frozen=len(links['frozen_inputs']),path=str(ROOT/'link-inputs.json'),sha256=digest(ROOT/'link-inputs.json'),uncovered=0));admitted(row)
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=False,ownership_closed=owner.active is None and all(r.supervision.closed(c) for c in owner.commands),commands_executed=owner.commands);row['passed']=bool(row['passed'] and row['ownership_closed']);write(ROOT/'record.json',row);record[n.KEY][PERF]['build']=compact(row);write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','ownership_closed','error']}),flush=True);require(row['passed'],'performance fixture build failed')


def prepare_run():
    record=r.validation.committed(RESULT);built=json.loads((ROOT/'record.json').read_text());require(built['stage']=='built' and built['passed'] and built['ownership_closed'],'compiled fixture first');_,_,_,rt=admitted(built);scope=record[n.KEY][PERF];require('comparison' not in scope,'no repeat/overwrite of prior comparison')
    plans=[]
    for index in range(3):
        folder=ROOT/str(index);folder.mkdir();(folder/'tmp').mkdir();(folder/'cache').mkdir()
        env=dict(rt['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
        plans.append(dict(index=index,root=str(folder),command=[built['binary']['path'],str(index)],environment=env))
    row=dict(stage='prepared',source_commit=built['source_commit'],source_sha256=built['source_sha256'],binary=built['binary'],plans=plans,budgets=dict(wall=900,cpu=1800,rss_cap=8<<30,text_cap=16<<20,file_cap=128<<20,total_cap=2<<30),boot=rt['boot'],processes=[],passed=False,complete=False,ownership_closed=True,criteria='All3fresh processes use exact native objects already qualified by compiled properties,traced tiny and Page4/Page16 lossless operand/normalizer/output interval gates.4cases(context32768/117488 bycontiguous/strided sortedunique causalIDs),Page4,8192queries,cap2051,varied nonzero Q/K/V/scales;syntheticnotmodelcapture.2warmups+3alternating paired rounds percase,batch8,floor100ms. Exact within-arm fulloutputrepeat and finite payload outside timing;paired changedbits are descriptive only and separate numerical gate remains mandatory. Inputimmutability/guards,knownproviders/immutableenv/pre-mainvalidation,clockmedian<=1000ns/p99<=10000ns/resolution<=1000ns,actualpackandattentioneventdurationenclosure,closedowner/no newfault. Stop first invalid/numerical/fault,incomplete isnotperformancefail. Keep all72samples. Advanceonlyifeachprocessmedian candidate/original<1 forwall+attention+pack/attentionservice,and eachcase3processGM<=0.99 forall3metrics. No retries/exclusions/adoption fromfixture. No PCIe/HW saturation claim.')
    write(ROOT/'comparison.json',row);scope['comparison']=compact(row);write(RESULT,record);print('Three-process comparison prepared;commit frozen order/criteria before run.',flush=True)


def parse(stdout,index):
    clocks=[dict(v.split('=',1) for v in line.split()[1:]) for line in stdout.splitlines() if line.startswith('HEAD_CLOCK ')]
    require(len(clocks)==1,'clock control absent');clock={k:float(v) for k,v in clocks[0].items()};require(clock['resolution_ns']<=1000 and clock['median_ns']<=1000 and clock['p99_ns']<=10000 and clock['floor_ms']==100,'clock control failed')
    rows=[dict(v.split('=',1) for v in line.split()[1:]) for line in stdout.splitlines() if line.startswith('HEAD_SAMPLE ')];require(len(rows)==24,'sample coverage differs')
    samples=[];seen=set()
    for row in rows:
        sample={k:(float(v) if k.endswith('_ms') else v if k=='arm' else int(v)) for k,v in row.items()}
        case=sample['scenario'];require(case in range(4) and sample['process']==index and sample['context']==[32768,32768,117488,117488][case] and sample['pattern']==case%2 and sample['queries']==8192 and sample['cap']==2051 and sample['page']==4 and sample['batch']==8 and sample['round'] in range(3) and sample['arm'] in ['original','candidate'] and sample['arm_repeat']==1,'sample workload differs')
        key=(case,sample['round'],sample['arm']);require(key not in seen,'duplicate sample');seen.add(key)
        require(all(math.isfinite(sample[k]) and sample[k]>0 for k in ['wall_ms','pack_ms','attention_ms']) and sample['wall_ms']>=100 and sample['pack_ms']+sample['attention_ms']<=sample['wall_ms']*1.01,'invalid/ineligible clocks');samples.append(sample)
    require(stdout.count('HEAD_CASE scenario=')==4 and all('HEAD_CASE scenario='+str(k)+' all_inputs_immutable=1 guards=1 arm_repeat=1' in stdout for k in range(4)) and 'HEAD_BENCH_DONE passed=1 scenarios=4 samples=24' in stdout,'posttiming checks incomplete')
    outputs=[dict(v.split('=',1) for v in line.split()[1:]) for line in stdout.splitlines() if line.startswith('HEAD_OUTPUT ')]
    require(len(outputs)==4 and {int(v['scenario']) for v in outputs}==set(range(4)) and all(int(v['elements'])==8192*24*256 and 0<=int(v['changed_bits'])<=int(v['elements']) and v['finite']=='1' and v['numerical_gate']=='SEPARATE' and v['model_quality']=='UNRUN' for v in outputs),'paired output classification incomplete')
    medians=[]
    for case in range(4):
        values={arm:[s for s in samples if s['scenario']==case and s['arm']==arm] for arm in ['original','candidate']}
        metrics={}
        for metric in ['wall_ms','attention_ms','service_ms']:
            def value(s):return s['pack_ms']+s['attention_ms'] if metric=='service_ms' else s[metric]
            a=statistics.median(value(s) for s in values['original']);b=statistics.median(value(s) for s in values['candidate']);metrics[metric]=dict(original=a,candidate=b,ratio=b/a)
        medians.append(dict(scenario=case,metrics=metrics))
    return dict(clock=clock,samples=samples,medians=medians,output_calibration=outputs)


def run():
    record=r.validation.committed(RESULT);scope=record[n.KEY][PERF];built=json.loads((ROOT/'record.json').read_text());row=json.loads((ROOT/'comparison.json').read_text());require(compact(row)==scope['comparison'] and row['stage']=='prepared','frozen comparison required');_,_,_,rt=admitted(built)
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15);cursor=next(v.split(': ',1)[1] for v in text.splitlines() if v.startswith('-- cursor: '));owner=None
    def persist():write(ROOT/'comparison.json',dict(row,active=bool(owner and owner.active is not None),commands_executed=owner.commands if owner else []))
    owner=r.supervision.Owner(ROOT/'run-owned',persist)
    try:
        for plan in row['plans'][len(row['processes']):]:
            _,_,_,rt=admitted(built);folder=Path(plan['root']);lines=[]
            def interact(io,entry):
                require(io.line(30)=='HEAD_BENCH_GATE before-runtime','entry gate');io.send(b'X');require(io.line(90)=='HEAD_BENCH_GATE runtime-ready','runtime gate');q.observed_native(io.proc.pid,plan['environment'],rt,row['binary']);io.send(b'X')
                while (line:=io.line(850))!='HEAD_BENCH_GATE before-exit':
                    lines.append(line)
                    if line.startswith('HEAD_CASE '):print('Performance process'+str(plan['index'])+' '+line,flush=True)
                q.observed_native(io.proc.pid,plan['environment'],rt,row['binary']);io.send(b'X');io.proc.stdin.close()
                require(io.line(20)=='HEAD_BENCH_DONE passed=1 scenarios=4 samples=24','final marker');io.exit(20)
            entry,out,err=owner.run('process'+str(plan['index']),plan['command'],plan['environment'],P,interaction=interact,**row['budgets']);require(r.supervision.closed(entry) and entry['exit_code']==0,'native process failed:'+err.read_text()[-2000:]);proof=r.validation.preload_evidence(err.read_text(),entry['pid']);require(proof['loader']==rt['preload']['loader'],'pre-main provider differs');warning=folder/'validation.log';require(not warning.exists() or not warning.stat().st_size,'validation warning');require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|HEAD_BENCH_ERROR|terminate called|cleanup failed',err.read_text()),'native runtime error')
            outcome=dict(host_probe_accepted=True,**parse(out.read_text(),plan['index']));r.validation.finish(outcome,owner,cursor,row['boot']);require(outcome['passed'],'fault/ownership check failed');row['processes'].append(dict(index=plan['index'],**outcome));persist()
        decision=[]
        for case in range(4):
            metrics={}
            for name in ['wall_ms','attention_ms','service_ms']:
                ratios=[p['medians'][case]['metrics'][name]['ratio'] for p in row['processes']];gm=math.exp(sum(map(math.log,ratios))/3);metrics[name]=dict(process_ratios=ratios,geometric_mean_ratio=gm,passed=all(v<1 for v in ratios) and gm<=.99)
            decision.append(dict(scenario=case,metrics=metrics))
        row.update(complete=True,host_probe_accepted=True,decision=decision,advance_to_model=all(v['passed'] for d in decision for v in d['metrics'].values()),model_adopted=False,model_PP_measured=False)
    except BaseException as error:row['error']=repr(error)
    finally:
        r.validation.finish(row,owner,cursor,row['boot']);row['stage']='closed' if row['passed'] else 'failed';persist();scope['comparison']=compact(row);write(RESULT,record)
    print(json.dumps({k:v for k,v in row.items() if k in ['stage','passed','complete','ownership_closed','advance_to_model','decision','error']}),flush=True);require(row['passed'],'performance evidence incomplete/invalid')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'prepare':prepare,'build':build,'prepare-run':prepare_run,'run':run}[sys.argv[1]]()
