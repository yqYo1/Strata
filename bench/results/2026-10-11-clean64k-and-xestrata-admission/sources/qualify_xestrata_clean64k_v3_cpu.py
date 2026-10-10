import copy, fcntl, importlib.util, json, sys, tempfile, types
from pathlib import Path
B=Path(__file__).parent;Q=B/'xestrata-clean-64k-comparison-v3';OUT=B/'xestrata-clean64k-protocol-contract-root-v3'
with (B/'owned-v0141-measurement.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir(mode=0o700);sys.path.insert(0,str(Q))
    spec=importlib.util.spec_from_file_location('clean64',Q/'run_clean.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    tests=[];record=dict(passed=False,active=False,complete=False,gpu_executed=False,model_executed=False,tests=tests,controller_sha256=m.sha(Q/'run_clean.py'))
    def check(label,func,rejected=False):
        try:func()
        except (RuntimeError,ValueError,IndexError,TypeError,KeyError):assert rejected,label
        else:assert not rejected,'accepted '+label
        tests.append(dict(case=label,passed=True))
    try:
        admission=Q/'baseline-math-execution-admission.json';a=json.loads(admission.read_text());v=a['requests'][0]
        check('real32K-source-admission',lambda:m.load_gate(admission,m.sha(admission),'baseline'))
        check('real32K-protocol',lambda:m.validate(v,32768))
        full=copy.deepcopy(v);full['protocol']=[x for x in full['protocol']if not x.startswith('PP ')]
        done=next(i for i,x in enumerate(full['protocol'])if x.startswith('DONE '));f=full['protocol'][done].split();f[2]=f[14]='65536';full['protocol'][done]=' '.join(f)
        full['protocol'][:0]=[f'PP {p} 65536 100 100.0'for p in m.EXPECTED64]
        check('full65536-with-terminal65535',lambda:m.validate(full))
        for tag,index,value in [('DONE',2,'65535'),('DONE',14,'65535'),('DONE',8,'1'),('DONE',3,'nan'),('DONE',3,'inf'),('DONE',4,'0'),('DONE',6,'-1'),('DONE',6,'99999'),('DONE',5,'cancelled'),('RESUME',1,'1'),('REUSED',1,'1'),('PP',2,'65535'),('PP',1,'65536')]:
            bad=copy.deepcopy(full);pos=next(i for i,x in enumerate(bad['protocol'])if x.startswith(tag+' '));f=bad['protocol'][pos].split();f[index]=value;bad['protocol'][pos]=' '.join(f)
            check(f'{tag}-{index}-{value}',lambda bad=bad:m.validate(bad),True)
        for tag in ['DONE','RESUME','REUSED']:
            bad=copy.deepcopy(full);bad['protocol'].append(next(x for x in bad['protocol']if x.startswith(tag+' ')));check('duplicate-'+tag,lambda bad=bad:m.validate(bad),True)
        for tag in ['DONE','PP','LP']:
            bad=copy.deepcopy(full);pos=next(i for i,x in enumerate(bad['protocol'])if x.startswith(tag+' '));bad['protocol'].pop(pos);check('missing-'+tag,lambda bad=bad:m.validate(bad),True)
        bad=copy.deepcopy(full);bad['ids'][0]=248320;check('invalid-output-ID',lambda:m.validate(bad),True)
        bad=copy.deepcopy(full);f=bad['logprobs'][0].split();f[1]='nan';bad['logprobs'][0]=' '.join(f);check('LP-nan',lambda:m.validate(bad),True)
        bad=copy.deepcopy(full);f=bad['logprobs'][0].split();f[2]='248320:-1';bad['logprobs'][0]=' '.join(f);check('LP-invalid-ID',lambda:m.validate(bad),True)
        for key,value in [('schema','other'),('event_timing_qualified',True),('engine_normal_exit',False),('owned_closed',False),('math_gate_passed',False),('qualification_scope','overallPASS'),('binary_sha256','0'*64)]:
            bad=copy.deepcopy(a);bad[key]=value;p=OUT/(key+'.json');p.write_text(json.dumps(bad));check('bad-admission-'+key,lambda p=p:m.load_gate(p,m.sha(p),'baseline'),True)
        bad=copy.deepcopy(a);bad['requests'][0]['comparison']['ids_equal']=1;p=OUT/'typed-comparison.json';p.write_text(json.dumps(bad));check('typed-derivative-request-edit',lambda:m.load_gate(p,m.sha(p),'baseline'),True)
        owner=Path(B/'xestrata-clean64k-owner-contract-root-v3/record.json');host=json.loads(owner.read_text());assert host['passed'] and not host['active'] and host['controller_sha256']==m.sha(Q/'run_clean.py')
        opts=types.SimpleNamespace(fixture=Q/'coding-context-65536.tokens.txt',fixture_sha=m.sha(Q/'coding-context-65536.tokens.txt'),arm='baseline',repetition=1,baseline_diagnostic=admission,baseline_diagnostic_sha=m.sha(admission),fork_diagnostic=None,fork_diagnostic_sha=None,owner_qualification=owner)
        check('independent-baseline-preflight-no-fork-gate',lambda:m.preflight(opts,True))
        old=opts.fixture_sha;opts.fixture_sha='0'*64;check('fixture-wronghash',lambda:m.preflight(opts,False),True);opts.fixture_sha=old
        record.update(passed=True,complete=True,owner_receipt_sha256=m.sha(owner),fixture_sha256=opts.fixture_sha,baseline_admission_sha256=m.sha(admission))
    except BaseException as e:record['error']=repr(e);raise
    finally:(OUT/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(tests),gpu_executed=False)))
