#!/usr/bin/env python3
"""Main-owned,serial,source-pinned pair-union/attention component gate.

Own MIT code. The baseline is recovered from its original Git source, never
from a later direct-K overlay. No production target or model route is added.
"""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import build_engine
from build_fused_profile import SYSTEM_HEADER_UPDATES
import measurement_guards as guards
import measurement_owner as supervision
import runtime_probe as runtime
import ur_validation as validation
from measure_down_scatter import postchecks

P=Path(__file__).resolve().parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT=B/'qsa-pair-probe-v1'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
BASE='6e40afc1beddd059ee518617f6c1efce1da14aca'
PREFIX='src/kernels/sycl/qsa_bf16_prefix.cpp'
NEW=['include/strata/kernels/qsa_pair_union.hpp','src/kernels/sycl/qsa_pair_union.cpp',
     'include/strata/kernels/qsa_pair_attention.hpp','src/kernels/sycl/qsa_pair_attention.cpp',
     'bench/sycl/qsa_pair_probe.hpp','bench/sycl/qsa_pair_probe.cpp']
NATIVE=['src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp',
        'src/kernels/sycl/prompt_attention.cpp']
require,digest=guards.require,supervision.sha
git=validation.git


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def controller_inputs(revision):
    pending=[Path(__file__).resolve()];seen=set();result={}
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path);name=str(path.relative_to(P));raw=git('show',revision+':'+name)
        require(raw==path.read_bytes(),'commit exact controller: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules=[n.name for n in node.names] if isinstance(node,ast.Import) else \
                [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=path.parent/((module or '').split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    return result


def source_inputs(revision):
    selected=git('ls-tree','-r','--name-only',BASE,'--','include').decode().splitlines()+NATIVE+[PREFIX]
    data={name:raw for name,raw in build_engine.blobs(P,BASE,selected)}
    for name in NEW:
        raw=git('show',revision+':'+name)
        require(raw==(P/name).read_bytes(),'commit native/fixture: '+name);data[name]=raw
    return data


def replace_once(text,before,after):
    require(text.count(before)==1,'unique generated source anchor: '+before)
    return text.replace(before,after)


def generated(data):
    original=data[PREFIX].decode()
    require(hashlib.sha256(data[PREFIX]).hexdigest()==
        '4f9f815c9baae709f919f6d5ad1af04677c10d8c13c7597b3857cf81567ae192',
        'not the selected original prefix')
    cut=original.index('bool qsa_bf16_prefix_batch(')
    text=original[:cut]
    text=replace_once(text,'namespace strata::kernels {','namespace strata::kernels::pair_probe_baseline {')
    text=replace_once(text,'QsaPrefixStateProbe probe = {}) {',
        'QsaPrefixStateProbe probe = {}, bool pack_only = false) {')
    text=replace_once(text,'  sycl_backend::finish(stream, prepared);',
        '  pair_probe::events.push_back(prepared);\n  sycl_backend::finish(stream, prepared);\n  if (pack_only) return true;')
    text=replace_once(text,'  sycl_backend::finish(stream, event);',
        '  pair_probe::events.push_back(event);\n  sycl_backend::finish(stream, event);')
    # Only the selected ordered-exp specialization is instantiated. Its pack
    # can be called separately to charge identical prefix work to both arms.
    text+='''} // namespace strata::kernels::pair_probe_baseline
bool pair_probe::selected(const uint16_t *q,const strata::kernels::QsaAttnPools &p,
    const int32_t *ids,const int32_t *steps,int64_t cap,const strata::kernels::QsaShapes &shape,
    uint16_t *out,int64_t nq,const strata::kernels::QsaPromptBf16Bounds &bounds,int64_t live,
    const strata::kernels::QsaBf16PackedScratch &scratch,bool pack_only,void *stream) {
  return strata::kernels::pair_probe_baseline::prefix_batch_impl<33,false,false,true,true,true,true>(
      q,p,ids,steps,cap,shape,out,nq,bounds,live,scratch,stream,{},pack_only);
}
'''
    result={'generated/selected.cpp':text.encode()}
    for name in ['src/kernels/sycl/qsa_pair_union.cpp','src/kernels/sycl/qsa_pair_attention.cpp']:
        source=data[name].decode()
        source=replace_once(source,'  finish(stream,event);',
            '  pair_probe::events.push_back(event);\n  finish(stream,event);')
        result['generated/'+Path(name).name]=source.encode()
    return result


def encoded_pins(data):
    return {name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(data.items())}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def assets(plan):
    require(controller_inputs(plan['source_commit'])==plan['controllers'],'controller inputs changed')
    source=source_inputs(plan['source_commit']);gen=generated(source)
    require(fingerprint(encoded_pins(source))==plan['project_inputs_sha256'] and
        encoded_pins(gen)==plan['generated_inputs'],'native/source mapping changed')
    for name,raw in {**source,**gen}.items():
        require((ROOT/name).read_bytes()==raw,'private source changed: '+name)
    for key in ['compiler_manifest','external_manifest']:
        item=plan[key];require(digest(item['path'])==item['sha256'],'tool/header manifest changed')
    rows=json.loads(Path(plan['compiler_manifest']['path']).read_text())
    files={r['path']:r['sha256'] for r in rows}
    require(plan['prospective_system_header_updates']=={k:list(v) for k,v in SYSTEM_HEADER_UPDATES.items()},
        'prospective SDK/header policy changed')
    for name,(before,after) in SYSTEM_HEADER_UPDATES.items():
        require(files[name]==before,'original SDK header lineage changed');files[name]=after
    files.update(json.loads(Path(plan['external_manifest']['path']).read_text()))
    files.update(plan['additional_headers'])
    for name,sha in files.items():require(digest(name)==sha,'tool/runtime/header changed: '+name)
    require(digest(plan['compiler']['path'])==plan['compiler']['sha256'],'compiler changed')
    return source,gen,files


def prepare_build():
    require(not ROOT.exists(),'inspect existing owner;fresh build only')
    record=validation.committed(RESULT);proposal=record['qsa_pair_proposal']
    require(not proposal['GPU_executed'] and 'native_build' not in proposal,'new source proposal required')
    rt=validation.committed(runtime.RESULT);runtime.current(rt)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'],'qualified runtime required')
    revision=git('rev-parse','HEAD').decode().strip();source=source_inputs(revision);gen=generated(source)
    settings=validation.committed(runtime.SETTINGS)
    sdk=Path(settings['environment']['LD_LIBRARY_PATH'].split(':')[0]).parent
    compiler=sdk/'bin/clang++'
    old=B/'qsa-direct-k-model-v2'
    external=old/'external-inputs.json';compiler_manifest=B/'unitrace-warmed-graph-window-v1/compiler-inputs.json'
    headers={str(p):digest(p) for p in Path('/usr/include/level_zero').glob('*.h')}
    flags=['-std=c++20','-O3','-DNDEBUG','-g0','-fsycl','-fsycl-id-queries-range=size_t',
        '-fno-fast-math','-ffp-contract=off','-DSTRATA_ENABLE_SYCL=1',
        '-I'+str(ROOT/'include'),'-I'+str(ROOT/'include/strata/sycl/compat'),
        '-I'+str(ROOT/'bench/sycl'),'-include',str(ROOT/'bench/sycl/qsa_pair_probe.hpp')]
    names=NATIVE+['generated/selected.cpp','generated/qsa_pair_union.cpp',
                 'generated/qsa_pair_attention.cpp','bench/sycl/qsa_pair_probe.cpp']
    commands=[];objects=[]
    for i,name in enumerate(names):
        obj=ROOT/f'input-{i}.o';objects.append(str(obj))
        commands.append([f'input-{i}',[str(compiler),*flags,'-MD','-MF',str(ROOT/f'input-{i}.d'),
            '-c',str(ROOT/name),'-o',str(obj)]])
    commands.append(['link',[str(compiler),*flags,*objects,'-pthread','-ldl','-o',str(ROOT/'fixture')]])
    environment=dict(PATH=str(sdk/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',
        LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=settings['environment']['LD_LIBRARY_PATH'],
        CPLUS_INCLUDE_PATH=str(sdk/'include'),TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    plan=dict(stage='prepared',source_commit=revision,selected_source=BASE,controllers=controller_inputs(revision),
        project_inputs_sha256=fingerprint(encoded_pins(source)),project_input_count=len(source),
        generated_inputs=encoded_pins(gen),compiler={'path':str(compiler),'sha256':digest(compiler)},
        compiler_manifest={'path':str(compiler_manifest),'sha256':digest(compiler_manifest)},
        prospective_system_header_updates={k:list(v) for k,v in SYSTEM_HEADER_UPDATES.items()},
        external_manifest={'path':str(external),'sha256':digest(external)},additional_headers=headers,
        environment=environment,commands=commands,
        budgets=dict(wall=600,cpu=1800,rss_cap=8<<30,text_cap=16<<20,total_cap=1<<30,file_cap=256<<20),
        criteria='Serial CPU compiler only;all project/generated inputs recover from exact Git sources;every actual dependency is in frozen project/tool/header manifests;complete owned exit0. Source review and native property/oracle gates remain separate.',
        retention='Selected6e40 engine+12fixtures remain untouched. This sole imminent component candidate retires after its dependency-path decision and exact input/recipe recovery;no other closed binaries are renewed.')
    ROOT.mkdir(mode=0o700)
    for folder in ['tmp','cache']:(ROOT/folder).mkdir()
    for name,raw in {**source,**gen}.items():
        path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    assets(plan);write(ROOT/'build-plan.json',plan)
    proposal['native_build']=plan;write(RESULT,record)
    print('Prepared CPU-only build;commit plan before compiling.',flush=True)


def build():
    record=validation.committed(RESULT);plan=record['qsa_pair_proposal']['native_build']
    require(plan==json.loads((ROOT/'build-plan.json').read_text()) and plan['stage']=='prepared' and
        not (ROOT/'build-owned.json').exists(),'fresh committed build plan required')
    source,gen,external=assets(plan);outcome=dict(passed=False,complete=False,active=False,ownership_closed=False,GPU_executed=False)
    owner=None
    def persist():write(ROOT/'build-owned.json',dict(outcome,commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'build-owned',persist)
    try:
        for label,argv in plan['commands']:
            entry,_,_=owner.run(label,argv,plan['environment'],ROOT,**plan['budgets'])
            require(supervision.closed(entry) and entry['exit_code']==0,'compiler failed: '+label)
            print('CPU build closed '+label,flush=True)
        actual={}
        allowed={str(ROOT/name):sha for name,sha in encoded_pins({**source,**gen}).items()};allowed.update(external)
        for dep in ROOT.glob('input-*.d'):
            text=dep.read_text().replace('\\\n',' ');parts=shlex.split(text.split(':',1)[1])
            for name in parts:
                path=str(Path(name).resolve());require(path in allowed,'unfrozen actual compiler input: '+path)
                require(digest(path)==allowed[path],'actual compiler input changed: '+path);actual[path]=allowed[path]
        source,gen,_=assets(plan);binary=ROOT/'fixture'
        write(ROOT/'actual-compiler-inputs.json',actual)
        outcome.update(passed=True,complete=True,binary={'path':str(binary),'sha256':digest(binary),'bytes':binary.stat().st_size},
            actual_compiler_inputs={'path':str(ROOT/'actual-compiler-inputs.json'),'sha256':digest(ROOT/'actual-compiler-inputs.json'),'count':len(actual)})
    except BaseException as error:outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        outcome['active']=False;persist();plan['stage']='closed' if outcome['passed'] else 'failed';plan['outcome']=outcome;write(RESULT,record)
    print(json.dumps(outcome),flush=True);return int(not outcome['passed'])


def admitted_build(record):
    plan=record['qsa_pair_proposal']['native_build'];result=plan['outcome']
    require(plan['stage']=='closed' and result['passed'] and result['complete'] and
        result['ownership_closed'] and not result['active'],'closed successful native build required')
    assets(plan);require(digest(result['binary']['path'])==result['binary']['sha256'],'fixture changed')
    actual=result['actual_compiler_inputs'];require(digest(actual['path'])==actual['sha256'],'compiled input evidence changed')
    for path,sha in json.loads(Path(actual['path']).read_text()).items():require(digest(path)==sha,'compiled dependency changed: '+path)
    rt=validation.committed(runtime.RESULT);runtime.current(rt)
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'],'qualified runtime changed')
    return plan,rt


def prepare_run(mode):
    record=validation.committed(RESULT);proposal=record['qsa_pair_proposal']
    native,rt=admitted_build(record);q=proposal.setdefault('qualification',{})
    require(mode not in q,'no retry/overwrite of a closed or pending gate')
    before={'union':'properties','small':'union','correctness':'small'}.get(mode)
    if before:require(q[before]['outcome']['passed'] and q[before]['outcome']['ownership_closed'],'preceding gate failed/open')
    review=validation.committed(P/'bench/results/2026-10-15-pr6-recovery-audit/research-index.json',
        ('qsa_pair_attention_contract_20261029','native_review'))
    require(review['verdict']=='accepted_with_essential_clarification' and review['clarification_applied'],
        'independent consumer source gate/clarification required')
    folder=ROOT/mode;require(not folder.exists(),'fresh run folder only')
    repeats=1 if mode in ['properties','union'] else 3
    cases=[]
    for i in range(repeats):
        run=folder/f'process-{i}'
        env=dict(rt['environment'],TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),
            ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p),
            UR_LOG_TRACING='level:info;flush:info;output:file,'+str(run/'api.trace'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(run/'validation.log'))
        argv=[native['outcome']['binary']['path'],mode]
        if mode=='properties':argv.append(next(p for p in rt['runtime_files'] if Path(p).name.startswith('libze_loader.so.')))
        cases.append(dict(label=f'process-{i}',command=argv,environment=env))
    plan=dict(stage='prepared',source_commit=git('rev-parse','HEAD').decode().strip(),
        controller_inputs=controller_inputs(git('rev-parse','HEAD').decode().strip()),
        binary=native['outcome']['binary'],runtime_files=rt['runtime_files'],preload=rt['preload'],
        cases=cases,source_contract=review,
        budgets=dict(wall=180,cpu=600,rss_cap=8<<30,text_cap=16<<20,total_cap=256<<20,file_cap=64<<20),
        criteria='Ordered owned children;pre-main UR validation;live exact native ELF/provider/environment/B570DRM checks at runtime-ready and before-exit;silent parameter validation;complete expected case/property markers;independent set/BF16tensor oracle and guards/input immutability;no fresh fault,boot/driver change or survivors. Property-only has zero enqueue;other gates require actual kernel submissions. Stop first failure;no speed/model/context qualification.',
        expected_cases={'properties':3,'union':38,'small':8,'correctness':19}[mode],
        retention='Keep compact property/case oracle samples,source/executable/runtime identity and fault/closure outcomes;retire successful traces and repeated logs after verification. No full trace retention question.')
    folder.mkdir(mode=0o700)
    for case in cases:
        (folder/case['label']).mkdir();(folder/case['label']/'tmp').mkdir();(folder/case['label']/'cache').mkdir()
    write(folder/'plan.json',plan);q[mode]=plan;write(RESULT,record)
    print('Prepared '+mode+';commit plan before native execution.',flush=True)


def live_native(pid,environment,rt,binary):
    proc=Path('/proc',str(pid));actual=dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
    require(all(actual.get(k.encode())==v.encode() for k,v in environment.items()),'native environment mutated')
    mapped=runtime.executable_mappings((proc/'maps').read_text())
    require(mapped<=set(rt['runtime_files'])|{binary['path']},'unknown native executable mapping: '+str(sorted(mapped-set(rt['runtime_files'])-{binary['path']})))
    neo=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p)
    require(neo in mapped and str(Path(environment['UR_ADAPTERS_FORCE_LOAD']).resolve()) in mapped and
        rt['preload']['loader'] in mapped,'native runtime provider absent')
    drm=[]
    for fd in (proc/'fdinfo').iterdir():
        fields=re.findall(r'^(drm-driver|drm-pdev):\s*(\S+)\s*$',fd.read_text(),re.M)
        if fields:
            require(len(fields)==2 and dict(fields)=={'drm-driver':'xe','drm-pdev':'0000:05:00.0'},'wrong native DRM device')
            drm.append(fd.name)
    require(drm,'no native B570 DRM descriptor')
    return dict(mapped_qualified_runtime=True,observed_NEO=neo,DRM_descriptors=len(drm),environment_immutable=True)


def run(mode):
    record=validation.committed(RESULT);plan=record['qsa_pair_proposal']['qualification'][mode]
    folder=ROOT/mode
    require(plan['stage']=='prepared' and plan==json.loads((folder/'plan.json').read_text()) and
        not (folder/'owned.json').exists(),'fresh committed gate required')
    native,rt=admitted_build(record)
    require(controller_inputs(plan['source_commit'])==plan['controller_inputs'] and
        plan['runtime_files']==rt['runtime_files'] and plan['binary']==native['outcome']['binary'],'frozen gate differs')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=re.findall(r'^-- cursor: (.+)$',text,re.M);require(len(cursor)==1,'kernel fault cursor unavailable')
    outcome=dict(passed=False,complete=False,active=False,ownership_closed=False,processes=[],model_qualified=False)
    owner=None
    def persist():
        outcome['active']=bool(owner and owner.active is not None)
        write(folder/'owned.json',dict(outcome,commands=owner.commands if owner else []))
    owner=supervision.Owner(folder/'owned',persist)
    try:
        for case in plan['cases']:
            observed={}
            def interaction(io,entry):
                require(io.line(30)=='PAIR_GATE before-runtime','unexpected native entry');io.send(b'X')
                require(io.line(90)=='PAIR_GATE runtime-ready','runtime gate failed')
                observed['start']=live_native(io.proc.pid,case['environment'],rt,plan['binary']);io.send(b'X')
                while io.line(150)!='PAIR_GATE before-exit':pass
                observed['end']=live_native(io.proc.pid,case['environment'],rt,plan['binary'])
                io.send(b'X');io.proc.stdin.close();io.exit(25)
            entry,out,err=owner.run(case['label'],case['command'],case['environment'],P,interaction=interaction,**plan['budgets'])
            require(supervision.closed(entry) and entry['exit_code']==0,'native failed or owner open')
            stdout,stderr=out.read_text(),err.read_text()
            proof=validation.preload_evidence(stderr,entry['pid']);require(proof['loader']==rt['preload']['loader'],'pre-main provider mismatch')
            body=stderr.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            require(not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|PAIR_ERROR',body),'native runtime error')
            runfolder=folder/case['label'];warnings=runfolder/'validation.log'
            require(not warnings.exists() or warnings.stat().st_size==0,'validation warning/error')
            trace=(runfolder/'api.trace').read_text()
            if mode=='properties':require('urKernelGet' in trace and not re.search(r'urEnqueue\w*Kernel',trace),'property trace missing/unexpected enqueue')
            else:require(re.search(r'urEnqueue\w*Kernel',trace),'actual kernel enqueue missing')
            prefix='PAIR_PROPERTY ' if mode=='properties' else 'PAIR_CASE '
            lines=[line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
            require(len(lines)==plan['expected_cases'] and stdout.count('PAIR_DONE mode='+mode+' passed=1')==1,'case coverage incomplete')
            rows=[]
            for line in lines:
                row=dict(part.split('=',1) for part in line.split())
                if 'name' in row:row['kernel_name_sha256']=hashlib.sha256(row.pop('name').encode()).hexdigest()
                rows.append(row)
            outcome['processes'].append(dict(label=case['label'],passed=True,cases=rows,validation=proof,observed=observed))
            # Close and recheck every process before starting the next.
            post=dict(passed=True,ownership_closed=True);postchecks(post,boot,cursor[0]);require(post['passed'],'fresh fault/boot check failed')
            admitted_build(record);persist();print('Native '+mode+' closed '+case['label'],flush=True)
        outcome.update(passed=True,complete=True)
    except BaseException as error:outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        outcome['active']=False;postchecks(outcome,boot,cursor[0]);persist()
        plan['stage']='closed' if outcome['passed'] else 'failed';plan['outcome']=outcome
        record['qsa_pair_proposal']['GPU_executed']=mode!='properties';write(RESULT,record)
    print(json.dumps({k:v for k,v in outcome.items() if k!='processes'}),flush=True);return int(not outcome['passed'])


def main():
    modes=['properties','union','small','correctness']
    require(len(sys.argv)==2 and sys.argv[1] in ['prepare-build','build',*modes,*['prepare-'+m for m in modes]],
        'use prepare-build|build|prepare-properties|properties|prepare-union|union|prepare-small|small|prepare-correctness|correctness')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        if sys.argv[1]=='prepare-build':prepare_build();return 0
        if sys.argv[1]=='build':return build()
        if sys.argv[1].startswith('prepare-'):prepare_run(sys.argv[1][8:]);return 0
        return run(sys.argv[1])


if __name__=='__main__':raise SystemExit(main())
