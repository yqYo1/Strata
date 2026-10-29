#!/usr/bin/env python3
"""Own CPU-only QSA codegen discriminator; repository LICENSE applies.

No binary link, runtime initialization or GPU enqueue. Native qualification is
separate, and int ID-range adoption needs a bounded dispatcher/fallback proof.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
import matrix_shapes as q
import qsa_head_sources as g
r,P,B=q.r,q.P,q.B
ROOT=B/'qsa-compiler-ir-v1'
KEY='qsa_compiler_ir_v1'
RESULT=q.RESULT
require,digest,write=q.require,q.digest,q.write
ARMS=['control','fixed_wg','id_int']
REVIEW='qsa_id_range_int_source_contract_20261029'


def sources(revision):return q.sources(revision,Path(__file__).resolve())


def native_sources():
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',g.BASE,'--','include'],cwd=P,text=True).splitlines()
    out=dict(r.build.blobs(P,g.BASE,names));generated=g.generated()
    out['original.cpp']=generated['original.cpp']
    text=generated['candidate.cpp'].decode().replace('qsa_head_candidate','qsa_head_original')
    control=generated['original.cpp'].decode()
    for before,after,count in g.EDITS:control=g.change(control,before,after,count)
    require(text==control,'WG arm must differ by exactly one annotation')
    out['fixed_wg.cpp']=text.encode();return out


def fingerprint(data):return hashlib.sha256(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in sorted(data.items())},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def compact(row):
    out={k:v for k,v in row.items() if k not in ['source_sha256','commands_executed']}
    out['controller_inputs']=dict(files=len(row['source_sha256']),aggregate=fingerprint({k:v.encode() for k,v in row['source_sha256'].items()}));return out


def admitted(row):
    require(sources(row['source_commit'])==row['source_sha256'],'committed compiler/controller inputs differ')
    require(q.closure.sdk_inputs()==row['sdk'],'SDK changed')
    rt=r.validation.committed(r.RESULT);require(digest(r.RESULT)==row['runtime_sha256'],'runtime identity changed')
    require(r.current(rt)['binary']==row['selected_binary'],'selected fallback changed')
    data=native_sources();require(len(data)==row['native_files'] and fingerprint(data)==row['native_digest'],'native inputs changed')
    for name,raw in data.items():require((ROOT/'source'/name).read_bytes()==raw,'source snapshot changed:'+name)
    require(digest(row['inventory']['path'])==row['inventory']['sha256'],'prospective inventory changed')
    allowed=json.loads(Path(row['inventory']['path']).read_text())
    for name,h in allowed.items():require(digest(name)==h,'compiler/header/provider input changed:'+name)
    return allowed


def prepare():
    record=r.validation.committed(RESULT);prior=record['qsa_identity_rescale']['retirement']
    require(prior['binaries_retired'] and prior['all_owned_processes_stopped'],'close prior extra binaries first')
    require(KEY not in record and not ROOT.exists(),'fresh CPU-only plan')
    rt=r.validation.committed(r.RESULT);full=r.current(rt);rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip();pins=sources(rev);data=native_sources();sdk=q.closure.sdk_inputs()
    ROOT.mkdir(mode=0o700)
    for name in ['source','tmp','cache']:(ROOT/name).mkdir()
    for name,raw in data.items():
        path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    allowed=q.prospective_inputs(full['compiler'],rt,pins);allowed.update({str(ROOT/'source'/n):hashlib.sha256(v).hexdigest() for n,v in data.items()});write(ROOT/'inventory.json',allowed)
    flags=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-device-only','-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1','-I'+str(ROOT/'source/include'),'-I'+str(ROOT/'source/include/strata/sycl/compat')]
    commands={}
    for arm in ARMS:
        source=ROOT/'source'/('fixed_wg.cpp' if arm=='fixed_wg' else 'original.cpp')
        commands[arm]=[full['compiler']['path'],*flags,'-fsycl-id-queries-range='+('int' if arm=='id_int' else 'size_t'),'-S','-emit-llvm','-MD','-MF',str(ROOT/(arm+'.d')),str(source),'-o',str(ROOT/(arm+'.ll'))]
    row=dict(stage='prepared',source_commit=rev,source_sha256=pins,native_source=g.BASE,native_files=len(data),native_digest=fingerprint(data),compiler=full['compiler'],sdk=sdk,selected_binary=full['binary'],runtime_sha256=digest(r.RESULT),boot=rt['boot'],inventory=dict(path=str(ROOT/'inventory.json'),sha256=digest(ROOT/'inventory.json'),files=len(allowed)),commands=commands,environment=dict(full['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),budgets=dict(wall=300,cpu=600,rss_cap=8<<30,text_cap=16<<20,file_cap=64<<20,total_cap=512<<20),results={},GPU_executed=False,model_adopted=False,criteria='Three CPU-only emitted LLVM controls. Exact same selected math;WG adds only exact128attribute;ID-int changes only IDrange flag. Fresh control and allactual compiler dependencies frozen. No GPU/native trial from metadata alone. Attention-kernel integer/address simplification or verified loop movement required;record normalized bodies/counts for Main review. FP32operation counts must remain equal. IDint CPUcompile needs source-review acceptance;broader public admission still blocks native adoption until guarded/fallback dispatch is proved.',retention='Current imminent CPUdiscriminator only;keep selected6e40+12fixtures. Retire normal compile logs and extra generation after useful codegen decision/source recovery;one originalIR textualhold belongs separately to the prior source review.')
    admitted(row);write(ROOT/'record.json',row);record[KEY]=compact(row);write(RESULT,record);print('CPU IR plan prepared;commit before compiling.',flush=True)


def metrics(path):
    text=path.read_text();functions={};current=None;body=[]
    for line in text.splitlines():
        if line.startswith('define '):current=line;body=[]
        elif line=='}' and current is not None:
            joined='\n'.join(body)
            if 'spir_kernel' in current and 'MatrixMulAdd' in joined:
                name=current.split('@',1)[1].split('(',1)[0];normalized=[];ssa={}
                def rename(match):
                    original=match.group(0)
                    if original not in ssa:ssa[original]='%v'+str(len(ssa))
                    return ssa[original]
                # Preserve definition/use identity while ignoring arbitrary SSA numbering.
                re.sub(r'%\d+',rename,current)
                for v in body:
                    v=re.sub(r', ![\w.]+ !\d+','',v);v=re.sub(r'\brange\([^)]*\)\s*','',v);v=re.sub(r'#\d+','',v);v=re.sub(r'%\d+',rename,v)
                    if v.strip() and not v.lstrip().startswith(';'):normalized.append(v.strip())
                i64=sum(bool(re.search(r'\b(?:add|sub|mul|[us]div|[us]rem|shl|lshr|ashr|and|or|xor)\b.*\bi64\b',v)) for v in body)
                cast64=sum(bool(re.search(r'\b(?:zext|sext|trunc)\b.*\bi64\b',v)) for v in body)
                fp={op:sum(bool(re.search(r'\b'+op+r'\b',v)) for v in body) for op in ['fadd','fsub','fmul','fdiv']}
                functions[name]=dict(normalized_body_sha256=hashlib.sha256('\n'.join(normalized).encode()).hexdigest(),instruction_lines=len(normalized),i64_arithmetic=i64,i64_casts=cast64,FP32_ops=fp,required_WG_metadata='reqd_work_group_size' in current)
            current=None
        elif current is not None:body.append(line)
    require(len(functions)==2,'exact two normal/state attention bodies expected')
    return functions


def run(arm):
    require(arm in ARMS,'explicit CPU arm only');record=r.validation.committed(RESULT);row=json.loads((ROOT/'record.json').read_text());require(compact(row)==record[KEY] and arm not in row['results'],'frozen fresh arm,never overwrite')
    if arm=='id_int':
        review=r.validation.committed(RESULT.with_name('research-index.json'),(REVIEW,'native_review'));require(review['source_accepted_for_CPU_IR'],'review integer flag before CPUcomparison')
    allowed=admitted(row);owner=None;outcome=dict(passed=False,GPU_executed=False)
    def persist():write(ROOT/(arm+'-record.json'),dict(outcome,active=bool(owner and owner.active is not None),commands=owner.commands if owner else []))
    owner=r.supervision.Owner(ROOT/(arm+'-owned'),persist)
    try:
        entry,out,err=owner.run(arm,row['commands'][arm],row['environment'],ROOT,**row['budgets']);require(r.supervision.closed(entry) and entry['exit_code']==0,'CPUcompile failed:'+err.read_text()[-2500:]);deps={}
        for name in shlex.split((ROOT/(arm+'.d')).read_text().replace('\\\n',' ').split(':',1)[1]):
            p=Path(name).resolve();h=digest(p);require(allowed.get(str(p))==h,'unfrozen actual compiler dependency:'+str(p));deps[str(p)]=h
        source=row['commands'][arm][-3];require(source in deps and len(deps)>500,'actual compiler source/dependency coverage');write(ROOT/(arm+'-inputs.json'),deps)
        outcome.update(passed=True,IR=dict(path=str(ROOT/(arm+'.ll')),sha256=digest(ROOT/(arm+'.ll')),bytes=(ROOT/(arm+'.ll')).stat().st_size),actual_compiler_inputs=dict(files=len(deps),path=str(ROOT/(arm+'-inputs.json')),sha256=digest(ROOT/(arm+'-inputs.json')),uncovered=0),functions=metrics(ROOT/(arm+'.ll')));admitted(row)
    except BaseException as error:outcome.update(passed=False,error=repr(error))
    finally:
        outcome['ownership_closed']=owner.active is None and all(r.supervision.closed(c) for c in owner.commands);outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed']);persist();row['results'][arm]=outcome;row['stage']='compiled' if len(row['results'])==3 and all(v['passed'] for v in row['results'].values()) else 'prepared';write(ROOT/'record.json',row);record[KEY]=compact(row);write(RESULT,record)
    print(json.dumps(outcome),flush=True);require(outcome['passed'],'CPU IR arm failed')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if sys.argv[1]=='prepare':prepare()
        else:run(sys.argv[2])
