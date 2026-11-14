#!/usr/bin/env python3
"""Pin,build and check the exact old/new collector methods on the CPU only."""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode=True
import collector_aggregation as proposal
import unitrace_profile as tool
import query_profile as q

P,B,RESULT=tool.P,tool.B,tool.RESULT
ROOT=B/'unitrace-aggregation-host-v2'
KEY='unitrace_aggregation_host_v2'
CPP=P/'bench/sycl/collector_aggregation.cpp'
INPUTS=('tools/sycl/check_collector_aggregation.py','tools/sycl/collector_aggregation.py',
        'bench/sycl/collector_aggregation.cpp',*tool.INPUTS)


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def pins(revision):
    names=set(INPUTS);pending=[P/name for name in INPUTS if name.endswith('.py')]
    while pending:
        path=pending.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for module in modules:
                local=path.parent/((module or '').split('.')[0]+'.py')
                if local.is_file() and str(local.relative_to(P)) not in names:
                    names.add(str(local.relative_to(P)));pending.append(local)
    rows={}
    for name in sorted(names):
        original=tool.git(P,'show',revision+':'+name)
        tool.guards.require(original==(P/name).read_bytes(),'commit exact host input: '+name)
        rows[name]=tool.sha(original)
    return rows


def plan():
    return dict(question='Does indexing names preserve the actual PTI integer aggregation and remove its quadratic name lookup work?',
        provenance='Original MIT Intel PTI4b6d742 method and proposed method extracted unchanged into the same own C++ harness; no LevelZero,SYCL,device or inference linked.',
        correctness='80 cases=2 deterministic independent seeds*5 sizes(0/1/17/257/513)*2 uniqueness*2 overflow*2 verbose. Include tiles/groups/memory and comma/quote names. Compare every surviving representative key,order and all6 fields; repeat aggregation for idempotence. Controlled naming stub is not production naming or its locks/demangling.',
        performance='N2048 all-unique names: reference must call namingN(N+1)/2,candidateN. One unmeasured warm pair,then7 pairs alternating reference,candidate and candidate,reference; allocation/reset outside clocks. Every batch>=100us and>=1000 times the minimum positive128-pair clock calibration. Require every candidate/reference wall ratio<0.10 and median<0.05. Keep all7 samples; no speed-based stopping,retries or replacements.',
        limits='CPU-only mechanism proxy,not actual collector shutdown,model-stage service,observer overhead or inference throughput. Next require actual collector build and4/10 graph/kernel/owner/fault qualification before another model plan.',
        retention='Keep source/recipe/individual7 samples/equivalence and lookup counts. Retire successful compiler/test logs and this small host fixture after actual collector qualification; keep the original current collector and graph fixture unchanged. No new model generation.')


def admission(record,binary=False):
    tool.guards.require(pins(record['source_commit'])==record['input_sha256'] and record['plan']==plan(), 'host source/plan changed')
    for item in record['generated']:
        tool.guards.require(digest(item['path'])==item['sha256'],'generated host source changed')
    manifest=record['compiler_inputs']
    tool.guards.require(digest(manifest['path'])==manifest['sha256'],'compiler manifest changed')
    rows=json.loads(Path(manifest['path']).read_text())
    tool.guards.require(len(rows)==manifest['files'],'compiler input count differs')
    for item in rows:tool.guards.require(digest(item['path'])==item['sha256'],'compiler/header changed: '+item['path'])
    if binary:tool.guards.require(digest(record['binary']['path'])==record['binary']['sha256'],'host fixture changed')


def prepare(container):
    tool.guards.require(KEY not in container and not ROOT.exists(),'fresh host fixture only; inspect prior ownership')
    previous=container['unitrace_aggregation_host']
    tool.guards.require(previous['stage']=='failed' and not previous['passed'] and previous['ownership_closed']
        and previous['closed_fixture_evidence']['actual_cases']==40 and previous['closed_fixture_evidence']['planned_cases']==80
        and previous['artifact_retirement']['complete'],'preserve/retire diagnosed40-case v1 failure first')
    revision=tool.git(P,'rev-parse','HEAD').decode().strip();inputs=pins(revision)
    full,count=q.current_build()
    retained=json.loads(q.BUILD.read_text())
    review=retained['legacy_provenance_exception']['additional_old_product_review']
    original=tool.git(tool.PTI,'show',proposal.REVISION+':'+proposal.HEADER)
    candidate=proposal.overlay(original)
    ROOT.mkdir(mode=0o700);(ROOT/'tmp').mkdir()
    copyright=original[:original.index(b'#ifndef ')]
    start=original.index(b'  void AggregateDeviceTimeStats() const {\n')
    end=original.index(b'  static std::vector<std::string> ParseFilterList(',start)
    (ROOT/'aggregate-original.inc').write_bytes(copyright+original[start:end])
    (ROOT/'aggregate-candidate.inc').write_bytes(copyright+proposal.REPLACEMENT.encode())
    environment=dict(PATH='/usr/bin:/bin',LANG='C',LC_ALL='C',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'))
    command=['/usr/bin/g++','-std=c++17','-O2','-DNDEBUG','-Wall','-Wextra','-Werror','-I',str(ROOT),str(CPP),'-o',str(ROOT/'aggregation_fixture')]
    owner=None
    record=dict(stage='preparing',source_commit=revision,input_sha256=inputs,plan=plan(),environment=environment,
        command=command,complete=False,passed=False,active=False,ownership_closed=True,
        retained_engine=dict(source_commit=full['source_commit'],binary=full['binary'],inputs_recovered=count,required_artifacts=13),
        legacy_provenance_review=dict(reference=str(q.BUILD.relative_to(P)),unchanged=True,
            review_sha256=tool.sha(json.dumps(review,sort_keys=True).encode()),
            action='No legacy generation overwritten or used as reference. Primary257/262 pins and PR6 final dependency/generated-command/ownership joins remain held explicitly; no new engine build.'),
        official_header_sha256=proposal.ORIGINAL_SHA256,proposed_full_header_sha256=tool.sha(candidate),
        budgets=dict(compile_wall_s=120,check_wall_s=90,cpu_s=120,rss_bytes=2<<30,text_bytes=2<<20,file_bytes=4<<20,total_bytes=16<<20))
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'prepare-record.json',dict(record,commands=owner.commands if owner else []))
    owner=tool.supervision.Owner(ROOT/'prepare-owned',persist)
    entry,_,_=owner.run('preprocess',command[:command.index('-o')]+['-M','-MT','target','-MF',str(ROOT/'dependencies.d')],
        environment,P,wall=60,cpu=60,rss_cap=2<<30,text_cap=2<<20,file_cap=4<<20,total_cap=16<<20)
    tool.guards.require(tool.supervision.closed(entry) and entry['exit_code']==0,'preprocessor failed')
    body=(ROOT/'dependencies.d').read_text().replace('\\\n',' ').split(':',1)[1]
    tool.guards.require('\\' not in body,'unexpected escaped dependency path')
    paths={Path(n).resolve() for n in body.split()}
    paths.update(Path(n).resolve() for n in ('/usr/bin/g++','/usr/bin/as','/usr/bin/ld','/usr/lib/x86_64-linux-gnu/libc.so.6'))
    for name in ('cc1plus','collect2'):
        paths.add(Path(subprocess.check_output(['/usr/bin/g++','-print-prog-name='+name],text=True).strip()).resolve())
    for name in ('crtbegin.o','crtend.o','libstdc++.so','libgcc_s.so'):
        paths.add(Path(subprocess.check_output(['/usr/bin/g++','-print-file-name='+name],text=True).strip()).resolve())
    rows=[dict(path=str(p),sha256=digest(p)) for p in sorted(paths)]
    manifest=ROOT/'compiler-inputs.json';write(manifest,rows)
    record.update(stage='prepared',ownership_closed=True,compiler_inputs=dict(path=str(manifest),files=len(rows),sha256=digest(manifest)),
        generated=[dict(path=str(ROOT/name),sha256=digest(ROOT/name)) for name in ('aggregate-original.inc','aggregate-candidate.inc')])
    admission(record);container[KEY]=record;write(RESULT,container);write(ROOT/'plan.json',record)
    print('Prepared host-only equivalence/scaling fixture; commit exact plan before compile/run.',flush=True)


def run(container):
    record=container[KEY]
    tool.guards.require(record['stage']=='prepared' and record==json.loads((ROOT/'plan.json').read_text()) and
        not (ROOT/'run-record.json').exists(),'fresh committed run only; no retry')
    admission(record)
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'run-record.json',dict(record,commands=owner.commands if owner else []))
    owner=tool.supervision.Owner(ROOT/'run-owned',persist);record['stage']='running';persist()
    try:
        args=dict(rss_cap=2<<30,text_cap=2<<20,file_cap=4<<20,total_cap=16<<20)
        entry,_,_=owner.run('compile',record['command'],record['environment'],P,wall=120,cpu=120,**args)
        tool.guards.require(tool.supervision.closed(entry) and entry['exit_code']==0,'host compile failed')
        binary=ROOT/'aggregation_fixture';record['binary']=dict(path=str(binary),bytes=binary.stat().st_size,sha256=digest(binary))
        admission(record,True)
        entry,out,_=owner.run('check',[str(binary)],record['environment'],ROOT,wall=90,cpu=90,**args)
        tool.guards.require(tool.supervision.closed(entry) and entry['exit_code']==0,'host fixture failed')
        text=out.read_text()
        header=re.fullmatch(r'aggregation_equivalence cases=80 entries=2048 reference_name_calls=2098176 candidate_name_calls=2048 keys=PASS fields=PASS overflow=PASS idempotence=PASS timer_ns=(\d+)',text.splitlines()[0])
        tool.guards.require(header is not None and text.splitlines()[-1]=='aggregation_fixture PASS GPU=NONE inference=UNCHANGED','host proof/footer differs')
        samples=[]
        for line in text.splitlines()[1:-1]:
            m=re.fullmatch(r'aggregation_sample index=(\d+) order=(reference,candidate|candidate,reference) reference_ns=(\d+) candidate_ns=(\d+)',line)
            tool.guards.require(m is not None,'malformed sample')
            i,order,old,new=m.groups();old,new=int(old),int(new)
            tool.guards.require(int(i)==len(samples) and order==('reference,candidate' if int(i)%2==0 else 'candidate,reference')
                and min(old,new)>=100000 and min(old,new)>=int(header[1])*1000,'sample order/eligibility differs')
            samples.append(dict(index=int(i),order=order,reference_ns=old,candidate_ns=new,ratio=new/old))
        tool.guards.require(len(samples)==7,'missing sample')
        record.update(samples=samples,clock_pair_ns=int(header[1]),equivalence_passed=True,name_lookup_counts=dict(reference=2098176,candidate=2048),
            performance_gate_passed=all(r['ratio']<.10 for r in samples) and statistics.median(r['ratio'] for r in samples)<.05,
            complete=True,passed=True,stage='closed')
    except BaseException as error:record.update(stage='failed',error=repr(error))
    finally:
        record['ownership_closed']=owner.active is None and all(tool.supervision.closed(c) for c in owner.commands)
        record['passed']=bool(record['passed'] and record['ownership_closed']);persist();container[KEY]=record;write(RESULT,container)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','performance_gate_passed','error')}),flush=True)
    return 0 if record['passed'] else 1


def main(action):
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tool.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        tool.guards.require(RESULT.read_bytes()==tool.git(P,'show','HEAD:'+str(RESULT.relative_to(P))),'commit current result/plan first')
        return prepare(container) if action=='prepare' else run(container)


if __name__=='__main__':
    tool.guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use check_collector_aggregation.py prepare|run')
    sys.exit(main(sys.argv[1]))
