"""SOURCE ONLY Main CPU qualification recipe. No execution has been performed."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = B/'xe-native-dequant-prefill-quiet-timing-controller-source-v3'
O = B/'xe-llvm-v711-source-v5/direct_owner.py'
F = B/'xe-native-dequant-prefill-quiet-timing-controller-cpu-fixtures-v3.py'
REVIEW = B/'xe-native-dequant-prefill-quiet-timing-controller-main-review-v3.json'
OWNER_SHA = 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
FIXTURE_SHA = '3eb057bfecab432e0ff36f460f05fc727311d626a1507d5ddbd0934c212fa467'
REVIEW_SHA = '9ac4ab6987523ecf04c832b93261763b59c77f9ed0a855241f345ebeb8ecffdc'
with O.open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!=OWNER_SHA:
        raise ValueError('qualified owner identity before import')
spec=importlib.util.spec_from_file_location('qualified_cpu_owner',O)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(O)==OWNER_SHA,'qualified owner unchanged')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
NAMES={'run_root.py','protocol.py','runtime.py','transfer_ledger.py','route_census.py','batch_census.py','timer_overhead.py','hash_inputs.py','fixtures.py'}

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-native-dequant-prefill-quiet-timing-controller-cpu-root-v3'
    m.require(not out.exists(),'immutable fresh CPU qualification root')
    out.mkdir(mode=0o700)
    fixture_root=out/'fixtures';fixture_root.mkdir(mode=0o700)
    start=time.monotonic()
    r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),
        controller_sha256=m.sha(__file__),owner_sha256=OWNER_SHA,source_pins={},
        fixture_program_sha256=m.sha(F),main_review_sha256=REVIEW_SHA,
        gpu_executed=False,model_executed=False,reset_executed=False,
        synthetic_fixture_receipts_are_model_evidence=False,whole_wall_budget_seconds=180,
        command_wall_budget_seconds=90,command_rss_budget_bytes=2<<30,
        raw_log_budget_bytes=4<<20,total_artifact_budget_bytes=96<<20,
        scope='Main CPU exported-predicate/order/admission, original protocol, actual CPU mapping/readelf and owned CPU QUIT qualification only; no numerical GPU/model or performance adoption.')
    def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
    try:
        m.require(m.sha(F)==FIXTURE_SHA,'exact CPU fixture source recipe')
        m.require(m.sha(REVIEW)==REVIEW_SHA,'immutable Main full review hash')
        review=json.loads(REVIEW.read_text())
        m.require(review['main_reviewed'] is True and review['syntax_ast_passed'] is True and
            review['full_controller_protocol_fixture_report_read'] is True,'actual Main source review, distinct CPU execution')
        r['source_pins']=review['source_pins']
        m.require(set(r['source_pins'])==NAMES and r['source_pins']['run_root.py']=='c1e636e7544fa5ef6a14f6ec294b6454171b19c7b91e241dc22c7fc02a88a5f8' and
            all(m.sha(A/name)==digest for name,digest in r['source_pins'].items()),'exact nine Main-reviewed source pins')
        def run(label,argv,expected=0):
            remaining=180-(time.monotonic()-start)
            m.require(remaining>0,'whole CPU qualification deadline')
            e,so,se=owner.run(label,argv,ENV,A,wall=min(90,remaining),cpu=90,
                text_cap=4<<20,total_cap=96<<20,rss_cap=2<<30)
            m.require(m.closed(e) and e.get('session_anchored') is True and e['exit_code']==expected,
                'normal anchored expected CPU command exit '+label)
            return so.read_text(),se.read_text()
        text,_=run('expanded-protocol-fixtures',['/usr/bin/python3',str(F),str(fixture_root)])
        r['fixtures']=json.loads(text)
        m.require(r['fixtures']['passed'] is True and r['fixtures']['fixture_only'] is True and
            len(r['fixtures']['positive_protocol_cases'])==9 and len(r['fixtures']['negative_protocol_cases'])==35 and
            r['fixtures']['gpu_executed'] is False and r['fixtures']['model_executed'] is False,
            'actual complete CPU fixture report')
        invalid=[['--group','16','--sample-index','0','--batch','on'],
                 ['--input-tokens','32768','--group','16','--sample-index','100','--batch','off'],
                 ['--input-tokens','65','--group','16','--sample-index','0','--batch','off'],
                 ['--input-tokens','32768','--group','65','--sample-index','0','--batch','off']]
        r['cli_rejections']=[]
        for i,args in enumerate(invalid):
            _,err=run('invalid-cli-'+str(i),['/usr/bin/python3',str(A/'run_root.py'),*args],2)
            m.require('error:' in err,'explicit parser rejection '+str(i))
            r['cli_rejections'].append(dict(argv=args,exit_code=2,stderr=err))
        r['source_stable']=all(m.sha(A/name)==digest for name,digest in r['source_pins'].items())
        m.require(r['source_stable'] and m.sha(REVIEW)==REVIEW_SHA and m.sha(F)==r['fixture_program_sha256'] and
            m.sha(O)==OWNER_SHA and m.sha(__file__)==r['controller_sha256'],'all CPU source/review inputs stable')
        m.require(time.monotonic()-start<=180,'final whole CPU deadline')
        r['total_artifact_bytes']=sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
        m.require(r['total_artifact_bytes']<=95<<20,'CPU root artifact reserve below96MiB including final record')
        r['complete']=True
    except BaseException as error:
        r['error']=repr(error)
    finally:
        r['active']=owner.active is not None
        r['passed']=bool(r['complete'] and not r['active'] and not r.get('error') and
            all(m.closed(c) and c.get('session_anchored') is True for c in owner.commands))
        r['finished_utc']=m.utc();save()
    print(json.dumps({key:r.get(key) for key in ('passed','error','source_stable')}))
    raise SystemExit(0 if r['passed'] else 1)
