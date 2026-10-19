#!/usr/bin/env python3
"""Bounded native-event import/copy/graph qualification; no model execution."""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import measurement_guards as guards
import measurement_owner as supervision
from check_verifier_retirement import DELTA, P, B, A, SETTINGS, digest

ROOT=B/'native-event-gate-v1'
RESULT=A/'native-event-gate-current.json'
BUILD=A/'native-gdn-engine-build.json'


def committed(path):
    data=path.read_bytes()
    guards.require(data==subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(P))],cwd=P),
        'commit exact input first: '+str(path))
    return json.loads(data)


def main():
    guards.require(len(sys.argv)==1,'no options or automatic retries')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        compact=committed(BUILD)
        full=json.loads((B/'main-native-gdn-engine/record.json').read_text())
        guards.require(full['passed'] and full['complete'] and full['ownership_closed'] and not full['active'] and
            all(full[k]==compact[k] for k in ('source_commit','binary','fixtures','compiler','configure_argv')) and
            'sycl_host_event_gate_test' in full['fixtures'] and len(full.get('fixture_external_headers',{}))==3,
            'closed exact native-event fixture build required')
        def unchanged():
            for source in [full['project'],*full['dependencies']]:
                for name,expected in source['source_sha256'].items():
                    guards.require(digest(Path(source['root'])/name)==expected,'build input changed: '+name)
            for item in [full['binary'],*full['fixtures'].values()]:
                guards.require(digest(item['path'])==item['sha256'],'compiled artifact changed')
            for name,expected in full['fixture_external_headers'].items():
                guards.require(digest(name)==expected,'external header changed: '+name)
        unchanged()
        guards.require('tools/sycl/check_event_gate.py' in full['project']['source_sha256'],'controller not pinned by build')
        guards.require(not ROOT.exists() and not RESULT.exists(),'inspect existing qualification; no automatic retry')
        settings=committed(SETTINGS)
        def runtime_unchanged():
            guards.require(set(DELTA)<=set(settings['runtime_elfs']),'unknown diagnostic runtime delta')
            for name,item in settings['runtime_elfs'].items():
                expected=item['sha256']
                if name in DELTA:
                    old,expected=DELTA[name]
                    guards.require(old==item['sha256'],'original runtime baseline differs')
                guards.require(digest(name)==expected,'runtime asset changed: '+name)
        record=dict(mode='native-event-gate',source_commit=full['source_commit'],binary=full['binary'],
            fixture=full['fixtures']['sycl_host_event_gate_test'],fixture_external_headers=full['fixture_external_headers'],
            environment_reference=str(SETTINGS.relative_to(P)),environment_sha256=digest(SETTINGS),
            per_case_budgets=dict(wall_seconds=30,cpu_seconds=60,rss_bytes=1<<30,text_bytes=64<<20,total_bytes=64<<20),
            runtime_asset_delta={n:dict(qualified_sha256=v[0],candidate_sha256=v[1]) for n,v in DELTA.items()},
            complete=False,passed=False,active=False,cases=[],model_executed=False,model_quality_qualified=False,
            consumer='Only if all three pass, replace the unsuitable test gate under a separate committed57-case plan. Retire normal traces after compact metrics and closure are committed.',
            gate='Three fresh owned processes in order import,copy,graph. Same native context/device, one regular host-visible event, keep ownership, HOST signal/wait scopes. Arm a2s fallback signaller before import. Import and dependent submits must each return within100ms; require50ms unsignalled/pending observation, independent resident/probe progress, untouched atomic output, checked manual release, exact payload and checked queue/wrapper/native-event/pool teardown. Any timeout/failure stops without retry or model execution.',
            limits='Test-only runtime interoperability/progress qualification, not production projection math, physical wire traffic, model correctness, overlap gain or throughput.')
        runtime_unchanged()
        ROOT.mkdir(mode=0o700)
        for n in ('tmp','cache'):(ROOT/n).mkdir()
        env=dict(settings['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
        cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (ROOT/'record.json').write_text(json.dumps(dict(record,commands=owner.commands if owner else []),indent=2)+'\n')
        owner=supervision.Owner(ROOT/'owned',persist);persist()
        try:
            for mode in ('import','copy','graph'):
                entry,out,err=owner.run(mode,[record['fixture']['path'],mode],env,P,
                    wall=30,cpu=60,rss_cap=1<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20)
                stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                    mode+' failed: '+stderr[-1024:])
                rows=re.findall(r'^event_gate_case=(\w+) import_us=(\d+) submit_us=(\d+) pending_us=(\d+) release=manual payload=PASS status=PASS$',stdout,re.M)
                guards.require(len(rows)==1 and rows[0][0]==mode and
                    int(rows[0][1])<=100000 and int(rows[0][2])<=100000 and int(rows[0][3])>=50000 and
                    not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed',stderr),
                    'native gate result/latency/progress evidence differs')
                record['cases'].append(dict(mode=mode,import_us=int(rows[0][1]),submit_us=int(rows[0][2]),
                    pending_us=int(rows[0][3]),passed=True,stdout_sha256=digest(out),stderr_sha256=digest(err)))
                persist()
            unchanged();runtime_unchanged();record.update(complete=True,passed=True)
        except BaseException as error:record['error']=str(error)
        finally:
            record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
            record['fresh_faults']=[x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:record['driver_after']=guards.driver_guard()
            except Exception as error:record.update(driver_guard_error=str(error),passed=False)
            record['passed']=bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
            persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
        return 0 if record['passed'] else 1


if __name__=='__main__':sys.exit(main())
