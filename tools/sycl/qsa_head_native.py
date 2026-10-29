#!/usr/bin/env python3
"""Own bounded M4 component build/qualification; repository LICENSE applies.

Selected 6e40 is untouched. Generated M8/M4 kernels have independent entrypoints;
all native work remains serialized under the shared measurement lock.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
import matrix_shapes as q
import qsa_head_sources as g
r,P,B=q.r,q.P,q.B
ROOT=B/'qsa-head-native-v2'
RESULT=q.RESULT
NATIVE=['src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp','src/kernels/sycl/prompt_attention.cpp']
FIXTURE='bench/sycl/qsa_head_oracle.cpp'
ORACLE='tests/sycl/qsa_bf16_prefix.cpp'
require,digest,write=q.require,q.digest,q.write


def sources(revision):return q.sources(revision,Path(__file__).resolve(),[FIXTURE,ORACLE])


def native_sources(revision):
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',g.BASE,'--','include'],cwd=P,text=True).splitlines()
    out=dict(r.build.blobs(P,g.BASE,[*NATIVE,*names]))
    for name in [FIXTURE,ORACLE]:out[name]=r.build.git(P,'show',revision+':'+name)
    out.update({'bench/sycl/'+name:raw for name,raw in g.generated().items()})
    return out


def compact(row):
    out={k:v for k,v in row.items() if k not in ['source_sha256','commands_executed']}
    out['source_inputs']=dict(files=len(row['source_sha256']),aggregate_sha256=hashlib.sha256(json.dumps(row['source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return out


def admitted(row):
    require(sources(row['source_commit'])==row['source_sha256'],'controller/fixture source changed')
    require(q.closure.sdk_inputs()==row['sdk'],'SDK changed')
    rt=r.validation.committed(r.RESULT);require(digest(r.RESULT)==row['runtime_sha256'],'runtime qualification changed')
    require(r.current(rt)['binary']==row['selected_binary'],'selected fallback changed')
    data=native_sources(row['source_commit'])
    require(g.proof()==row['geometry_proof'] and len(data)==row['native_source_files'],'native source mapping changed')
    require(hashlib.sha256(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in sorted(data.items())},sort_keys=True,separators=(',',':')).encode()).hexdigest()==row['native_source_digest'],'native source digest changed')
    for name,raw in data.items():require((ROOT/'source'/name).read_bytes()==raw,'native source changed:'+name)
    inventory=row['input_inventory'];require(digest(inventory['path'])==inventory['sha256'],'prospective input inventory changed')
    allowed=json.loads(Path(inventory['path']).read_text());require(len(allowed)==inventory['files'],'input count changed')
    for p,h in allowed.items():require(digest(p)==h,'frozen source/tool/provider changed:'+p)
    for asset in row.get('binaries',{}).values():require(digest(asset['path'])==asset['sha256'],'native ELF changed')
    return allowed,rt


def prepare():
    require(not ROOT.exists(),'fresh component only;inspect owner first')
    record=r.validation.committed(RESULT);query=record['qsa_head_geometry']['shape_query']
    require(query['passed'] and query['M4_component_eligible'] and query['retirement']['verified_absent'],'positive capability and ended property generation required')
    prior=record['qsa_head_geometry']['component']['retirement']
    require(prior['verified_absent'] and prior['all_owned_processes_stopped'] and not Path(prior['root']).exists(),'retire closed coordinate prototype first')
    require('component_v2' not in record['qsa_head_geometry'],'do not overwrite original outcome')
    rt=r.validation.committed(r.RESULT);full=r.current(rt);revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()
    pins=sources(revision);sdk=q.closure.sdk_inputs();data=native_sources(revision)
    ROOT.mkdir(mode=0o700)
    for folder in ['source','tmp','cache']:(ROOT/folder).mkdir()
    for name,raw in data.items():
        path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    allowed=q.prospective_inputs(full['compiler'],rt,pins)
    allowed.update({str(ROOT/'source'/name):hashlib.sha256(raw).hexdigest() for name,raw in data.items()})
    write(ROOT/'input-inventory.json',allowed)
    flags=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t','-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(ROOT/'source/include'),'-I'+str(ROOT/'source/include/strata/sycl/compat'),'-I'+str(ROOT/'source/bench/sycl')]
    units=[(Path(n).stem,ROOT/'source'/n,[]) for n in NATIVE]
    units += [(n,ROOT/'source/bench/sycl'/(n+'.cpp'),[]) for n in ['original','candidate']]
    units += [('oracle'+str(page),ROOT/'source'/FIXTURE,['-DSTRATA_QSA_TEST_PAGE_SIZE='+str(page)]) for page in [4,16]]
    commands=[];objects=[]
    for name in ['original','candidate']:
        commands.append(['IR'+name,[full['compiler']['path'],*flags,'-fsycl-device-only','-S','-emit-llvm',str(ROOT/'source/bench/sycl'/(name+'.cpp')),'-o',str(ROOT/(name+'.ll'))]])
    for label,path,extra in units:
        obj=ROOT/(label+'.o');objects.append(str(obj));commands.append([label,[full['compiler']['path'],*flags,*extra,'-MD','-MF',str(ROOT/(label+'.d')),'-c',str(path),'-o',str(obj)]])
    for page in [4,16]:commands.append(['link'+str(page),[full['compiler']['path'],*flags,'-save-temps=obj','-v','-Wl,-t',*objects[:6],str(ROOT/('oracle'+str(page)+'.o')),'-pthread','-ldl','-o',str(ROOT/('qsa_head_p'+str(page)))]])
    row=dict(stage='prepared',source_commit=revision,source_sha256=pins,compiler=full['compiler'],sdk=sdk,geometry_proof=g.proof(),native_source_files=len(data),native_source_digest=hashlib.sha256(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in sorted(data.items())},sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        original_native_source=g.BASE,selected_binary=full['binary'],runtime_sha256=digest(r.RESULT),boot=rt['boot'],
        input_inventory=dict(path=str(ROOT/'input-inventory.json'),sha256=digest(ROOT/'input-inventory.json'),files=len(allowed)),
        commands=commands,environment=dict(full['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        budgets=dict(wall=600,cpu=1800,rss_cap=8<<30,text_cap=32<<20,file_cap=256<<20,total_cap=2<<30),
        active=False,ownership_closed=True,passed=False,GPU_executed=False,production_or_decode_changed=False,
        criteria='CPU build only,zero uncovered actual compiler/link inputs;property-only4normal/state kernels before submit. Reviewed convergence/layout before bounded tiny trace. Independent FP32state+full BF16tensor oracle,guards,admission and physical262144component addresses atPage4/16 follow;not full-model context or performance/adoption.',
        retention='One imminent head-geometry native comparison generation;keep original selected6e40+12fixtures. Retire this extra build/raw output after its gate/decision and exact original source/recipe recovery.')
    admitted(row);write(ROOT/'record.json',row);record['qsa_head_geometry']['component_v2']=dict(build=compact(row));write(RESULT,record);print('Component prepared;commit frozen plan before compile.',flush=True)


def build():
    record=r.validation.committed(RESULT);row=json.loads((ROOT/'record.json').read_text());require(row['stage']=='prepared' and compact(row)==record['qsa_head_geometry']['component_v2']['build'],'fresh committed build plan')
    allowed,_=admitted(row);owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None);write(ROOT/'record.json',dict(row,commands_executed=owner.commands if owner else []))
    owner=r.supervision.Owner(ROOT/'build-owned',persist)
    try:
        links={}
        for label,argv in row['commands']:
            print('CPU compile:'+label,flush=True)
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,**row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code']==0,'compile failed:'+label+':'+err.read_text()[-3500:])
            if label.startswith('link'):links[label]=q.linking.traced_inputs(out.read_text(),err.read_text(),allowed,ROOT,ROOT)
        require('JointMatrixGetElementCoordINTEL' in (ROOT/'original.ll').read_text() and 'JointMatrixGetElementCoordINTEL' not in (ROOT/'candidate.ll').read_text(),'original coordinate control or candidate noncoordinate IR differs')
        observed={}
        for dep in ROOT.glob('*.d'):
            for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
                path=Path(name).resolve();sha=digest(path);require(allowed.get(str(path))==sha,'unfrozen actual compiler input:'+str(path));observed[str(path)]=sha
        require(len(observed)>600 and all(str(ROOT/'source'/n) in observed for n in [*NATIVE,FIXTURE,ORACLE]),'incomplete actual compiler closure')
        write(ROOT/'compiler-inputs.json',observed);write(ROOT/'link-inputs.json',links)
        row.update(stage='built',passed=True,binaries={str(page):dict(path=str(ROOT/('qsa_head_p'+str(page))),sha256=digest(ROOT/('qsa_head_p'+str(page))),bytes=(ROOT/('qsa_head_p'+str(page))).stat().st_size) for page in [4,16]},actual_compiler_inputs=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(observed),uncovered=0),actual_link_inputs=dict(path=str(ROOT/'link-inputs.json'),sha256=digest(ROOT/'link-inputs.json'),frozen_counts={k:len(v['frozen_inputs']) for k,v in links.items()},uncovered=0))
        admitted(row)
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=False,ownership_closed=owner.active is None and all(r.supervision.closed(c) for c in owner.commands),commands_executed=owner.commands);row['passed']=bool(row['passed'] and row['ownership_closed']);write(ROOT/'record.json',row);record['qsa_head_geometry']['component_v2']['build']=compact(row);write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','ownership_closed','error']}),flush=True);require(row['passed'],'component build failed')


def prepare_gate(mode,page):
    record=r.validation.committed(RESULT);component=record['qsa_head_geometry']['component_v2'];row=json.loads((ROOT/'record.json').read_text());require(row['passed'] and row['ownership_closed'] and not row['active'] and compact(row)==component['build'],'closed source-bound component required');_,rt=admitted(row)
    name=mode+str(page);require(name not in component,'never overwrite prior gate');folder=ROOT/name;require(not folder.exists(),'fresh native gate only')
    if mode!='properties':require(component['properties4']['outcome']['passed'],'compiled properties before submissions')
    if mode=='component':require(component['tiny4']['outcome']['passed'],'tiny traced gate before extended components')
    folder.mkdir(mode=0o700);(folder/'tmp').mkdir();(folder/'cache').mkdir()
    env=dict(rt['environment'],TMPDIR=str(folder/'tmp'),XDG_CACHE_HOME=str(folder/'cache'),ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),STRATA_TEST_MAIN_GATES='1',UR_LOG_TRACING='level:info;flush:info;output:file,'+str(folder/'api.trace'),UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(folder/'validation.log'))
    if mode=='component':env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
    args=['--'+mode]
    if mode=='properties':args.append(next(p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')))
    plan=dict(stage='prepared',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),source_sha256=sources(row['source_commit']),binary=row['binaries'][str(page)],command=[row['binaries'][str(page)]['path'],*args],environment=env,mode=mode,page=page,boot=rt['boot'],budgets=dict(wall=600,cpu=1200,rss_cap=8<<30,text_cap=32<<20,file_cap=64<<20,total_cap=512<<20),criteria='One fresh owned process,stop firstfailure,no retry. Properties only beforekernel;source-review before submissions. Tinytrace actual SG/WG/local/ABI,complete output and FP32state parity,independent host oracle/guards,all input immutable,pre-main validation,correct actualproviders,quiet validation and no newfault. Extended physical finaladdresses do not qualify full-model context.')
    write(folder/'plan.json',plan);component[name]=plan;write(RESULT,record);print('Native gate prepared;commit before run.',flush=True)


def dispatches(trace,properties):
    created={};counts={v['name']:0 for v in properties}
    for line in trace.splitlines():
        if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
            match=re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)',line)
            if match:created[match[2]]=match[1]
        if '<--- urEnqueueKernelLaunchWithArgsExp(' not in line or 'UR_RESULT_SUCCESS' not in line:continue
        match=re.search(r'\.hKernel = (0x[0-9a-f]+)',line);require(match and match[1] in created,'launch without kernel create');name=created[match[1]]
        prop=next((p for p in properties if p['name']==name),None)
        if prop is None:continue
        require('.numArgs = '+prop['arguments']+',' in line and re.search(r'\.pLocalWorkSize = 0x[0-9a-f]+ \('+prop['wg']+r'\)',line),'actual launch ABI/WG differs')
        sizes=sorted(int(v) for v in re.findall(r'UR_EXP_KERNEL_ARG_TYPE_LOCAL, \.index = \d+, \.size = (\d+)',line))
        require(sizes==sorted([16384,768,48,48,48,256,48,1584,1536] if prop['arm']=='candidate' else [16384,1024,64,64,64,256,64,2112]),'actual local extents differ');counts[name]+=1
    require(all(v>0 for v in counts.values()),'missing normal/state arm dispatch');return {hashlib.sha256(k.encode()).hexdigest():v for k,v in counts.items()}


def run_gate(mode,page):
    record=r.validation.committed(RESULT);component=record['qsa_head_geometry']['component_v2'];name=mode+str(page);plan=component[name];folder=ROOT/name;row=json.loads((ROOT/'record.json').read_text());require(plan['stage']=='prepared' and plan==json.loads((folder/'plan.json').read_text()) and not (folder/'owned').exists(),'fresh committed native plan');_,rt=admitted(row)
    require(sources(row['source_commit'])==plan['source_sha256'],'gate source differs')
    if mode!='properties':
        review=r.validation.committed(RESULT.with_name('research-index.json'),('qsa_uniform_m4_portable_source_audit_20261029','native_review'));require(review['source_accepted'],'review actual M4 source before submit')
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15);cursor=next(v.split(': ',1)[1] for v in text.splitlines() if v.startswith('-- cursor: '))
    outcome=dict(complete=False,passed=False,host_probe_accepted=False);owner=None;lines=[]
    def persist():write(folder/'record.json',dict(outcome,active=bool(owner and owner.active is not None),commands=owner.commands if owner else []))
    owner=r.supervision.Owner(folder/'owned',persist)
    try:
        def interact(io,entry):
            require(io.line(30)=='QSA_BF16_READY before-runtime','entry gate differs');io.send(b'X')
            while (line:=io.line(90))!='QSA_BF16_READY runtime-ready':lines.append(line)
            q.observed_native(io.proc.pid,plan['environment'],rt,plan['binary']);io.send(b'X')
            while (line:=io.line(550))!='QSA_BF16_READY after-cleanup':lines.append(line)
            q.observed_native(io.proc.pid,plan['environment'],rt,plan['binary']);io.send(b'X')
            while not io.eof or not io.exit_watch.poll(0):
                io.poll()
                if select.select([io.proc.stdout],[],[],.02)[0]:
                    io.read();io.pending.clear()
            io.exit(10)
        entry,out,err=owner.run(name,plan['command'],plan['environment'],P,interaction=interact,**plan['budgets']);require(r.supervision.closed(entry) and entry['exit_code']==0,'native process failed:'+err.read_text()[-2000:])
        proof=r.validation.preload_evidence(err.read_text(),entry['pid']);require(proof['loader']==rt['preload']['loader'],'pre-main provider differs');warning=folder/'validation.log';require(not warning.exists() or not warning.stat().st_size,'validation warning');require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|HEAD_ERROR|terminate called|cleanup failed',err.read_text()),'native runtime error')
        stdout=out.read_text();properties=[];actual=None
        if mode=='properties':
            properties=[dict(part.split('=',1) for part in v[len('HEAD_PROPERTY '):].split()) for v in lines if v.startswith('HEAD_PROPERTY ')];require(len(properties)==4 and 'HEAD_DONE mode=--properties passed=1' in stdout,'property coverage');require(not re.search(r'urEnqueue\w*Kernel',(folder/'api.trace').read_text()),'property query submittedkernel')
        else:
            require(('HEAD_DONE mode=--tiny passed=1' if mode=='tiny' else 'fixture_status=PASS model_quality=UNRUN') in stdout,'oracle completion absent')
            require(stdout.count('case exact_FP32_online_state=PASS')==(1 if mode=='tiny' else 11),'state/case coverage')
            if mode=='tiny':actual=dispatches((folder/'api.trace').read_text(),component['properties4']['outcome']['properties'])
        admitted(row);outcome.update(complete=True,host_probe_accepted=True,properties=properties,dispatch=actual,numerical_passed=mode!='properties',physical262144_component_addresses=mode=='component',model_context_fill_qualified=False,model_speed_qualified=False,parameter_validation=True)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        r.validation.finish(outcome,owner,cursor,plan['boot']);persist();plan['stage']='closed' if outcome['passed'] else 'failed';plan['outcome']=outcome;write(RESULT,record)
    print(json.dumps({k:v for k,v in outcome.items() if k!='properties'}),flush=True);require(outcome['passed'],'native M4 gate failed')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1] in ['prepare','build']:{'prepare':prepare,'build':build}[sys.argv[1]]()
        else:{'prepare-gate':prepare_gate,'run-gate':run_gate}[sys.argv[1]](sys.argv[2],int(sys.argv[3]))
