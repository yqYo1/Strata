"""Main-owned protocol and CLI qualification; no model or GPU execution."""
import fcntl
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = B/'xe-native-dequant-prefill-model-controller-source-v2'
O = B/'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
ENV = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
HARNESS = '''import json
import protocol, fixtures
import run_root
positive=[]; negative=[]
for n in (64,2049,32768):
    good=fixtures.protocol_case(n)
    assert protocol.validate(good,n=n)['generated_tokens']==16
    positive.append(dict(n=n,kind='complete_protocol'))
    assert protocol.validate(fixtures.protocol_case(n,1,'stop'),n=n)['generated_tokens']==1
    positive.append(dict(n=n,kind='normally_completed_early_EOS_not_strict16'))
    trace=fixtures.trace_case(n)
    assert len(protocol.validate_trace(trace,n=n)['chunk_start_positions'])==len(fixtures.GEOMETRY[n]['offsets'])
    positive.append(dict(n=n,kind='exact_trace_geometry'))
    cases=[good[:-1],good+[good[-1]],good+['ERR failed'],good+['T 123'],
           [x for x in good if not x.startswith('PP ')],
           [x.replace('T 123','T 248320') for x in good],
           [x.replace('LP -0.1','LP nan') for x in good],
           [x.replace('50000','nan') for x in good]]
    for index,bad in enumerate(cases):
        try: protocol.validate(bad,n=n)
        except (ValueError,IndexError): negative.append(dict(n=n,kind='protocol',case=index))
        else: raise AssertionError(('malformed protocol accepted',n,index))
    for index,bad in enumerate([trace+trace,trace.replace('of '+str(n-1),'of '+str(n)),trace.replace('chunk 0','chunk 1')]):
        try: protocol.validate_trace(bad,n=n)
        except ValueError: negative.append(dict(n=n,kind='trace',case=index))
        else: raise AssertionError(('malformed trace accepted',n,index))
for n in (True,65):
    try: protocol.validate(fixtures.protocol_case(64),n=n)
    except ValueError: negative.append(dict(n=n,kind='invalid_explicit_N'))
    else: raise AssertionError('invalid N accepted')
assert len(positive)==9 and len(negative)==35
print(json.dumps(dict(passed=True,positive_cases=positive,negative_cases=negative,controller_imported_without_main=True,
    limitations='Protocol/import cases only. The 27 source rejection descriptions are not executed fault tests; no model numerical/lifetime/performance qualification.')))
'''

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-native-dequant-prefill-model-controller-cpu-root-v2'
    m.require(not out.exists(),'immutable output');out.mkdir()
    r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),
           controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),source_pins={},
           gpu_executed=False,model_executed=False,scope='Actual protocol/trace, CLI and safe module import qualification; unchanged helper identities, not all runtime/fault gates',
           raw_log_budget_bytes=64<<20,whole_wall_budget_seconds=180)
    def save(): (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
    try:
        review_path=B/'xe-native-dequant-prefill-model-controller-main-review-v2.json'
        review=json.loads(review_path.read_text())
        m.require(review['main_reviewed'] and review['full_controller_protocol_fixture_report_read'] and review['syntax_ast_passed'],'full Main source review')
        r['source_pins']=review['source_pins'];r['main_review_sha256']=m.sha(review_path)
        m.require(all(m.sha(A/n)==h for n,h in r['source_pins'].items()),'exact reviewed inputs')
        proof=B/'xe-native-dequant-batch-census-observer-cpu-root-v2/record.json'
        v=json.loads(proof.read_text())
        m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'actual observer qualification')
        r['observer_qualification_sha256']=m.sha(proof)
        def run(label,argv,expected=0):
            e,so,se=owner.run(label,argv,ENV,A,wall=30,cpu=30,text_cap=4<<20,total_cap=64<<20,rss_cap=2<<30)
            m.require(m.closed(e) and e['exit_code']==expected,'normal expected exit '+label)
            return so.read_text(),se.read_text()
        text,_=run('expanded-protocol-fixtures',['/usr/bin/python3','-c',HARNESS])
        r['fixtures']=json.loads(text)
        invalid=[['--input-tokens','64','--group','16','--batch','on'],
                 ['--input-tokens','64','--group','16','--batch','off','--reference','/tmp/unused-no-open'],
                 ['--input-tokens','65','--group','16','--batch','off'],
                 ['--input-tokens','64','--group','65','--batch','off']]
        r['cli_rejections']=[]
        for i,args in enumerate(invalid):
            _,text=run('invalid-cli-'+str(i),['/usr/bin/python3',str(A/'run_root.py'),*args],2)
            m.require('error:' in text,'explicit parser rejection')
            r['cli_rejections'].append(dict(argv=args,exit_code=2,stderr=text))
        r['source_stable']=all(m.sha(A/n)==h for n,h in r['source_pins'].items())
        m.require(r['source_stable'],'source unchanged');r['complete']=True
    except BaseException as error:r['error']=repr(error)
    finally:
        r['active']=owner.active is not None
        r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc']=m.utc();save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','source_stable')}))
    raise SystemExit(0 if r['passed'] else 1)
