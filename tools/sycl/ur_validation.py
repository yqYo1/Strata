#!/usr/bin/env python3
"""Bounded host-only evidence for the pinned UR validation layer; Main only.

This process cannot qualify activation in a later model child. It does not
enumerate a platform/device, create a queue, enqueue work, or load a model.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
import measurement_owner as supervision
import measurement_guards as guards

P=Path(__file__).resolve().parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
INDEX=P/'bench/results/2026-10-15-pr6-recovery-audit/research-index.json'
SETTINGS=P/'tools/sycl/measurement_environment.json'
SCOPE='UR_validation_activation_evidence_scope'
ROOT=B/'ur-validation-canary-v1'
SOURCE=P/'bench/sycl/ur_validation_canary.c'


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=P)


def committed(path):
    raw=path.read_bytes()
    guards.require(raw==git('show','HEAD:'+str(path.relative_to(P))),'commit exact input first: '+str(path))
    return json.loads(raw)


def prepare():
    guards.require(not ROOT.exists(),'fresh canary only; no retry')
    settings=committed(SETTINGS); index=committed(INDEX)
    env=dict(settings['environment'])
    runtime=Path(env['UR_ADAPTERS_FORCE_LOAD']).resolve(); install=runtime.parents[1]
    compiler=Path('/usr/bin/cc').resolve()
    command=[str(compiler),'-std=c11','-Wall','-Wextra','-Werror','-O2',
        '-I'+str(install/'include'),str(SOURCE),'-L'+str(install/'lib'),
        '-Wl,-rpath,'+str(install/'lib'),'-lur_loader','-ldl','-o',str(ROOT/'canary')]
    # Preprocessing resolves the actual installed dependency headers before build.
    dependencies=subprocess.check_output([*command[:7],'-M',str(SOURCE)],text=True,timeout=15)
    headers=shlex.split(dependencies.replace('\\\n',' '))[1:]
    paths={str(Path(p).resolve()) for p in headers}
    paths.update((str(compiler),str(Path(sys.executable).resolve()),str(Path(__file__).resolve())))
    for name in ('cc1','as','ld'):
        found=subprocess.check_output([str(compiler),'-print-prog-name='+name],text=True,timeout=5).strip()
        paths.add(str(Path(shutil.which(found) or found).resolve()))
    paths.update(str(P/'tools/sycl'/n) for n in ('measurement_owner.py','measurement_guards.py'))
    for path in paths:
        local=Path(path)
        if local.is_relative_to(P):
            guards.require(local.read_bytes()==git('show','HEAD:'+str(local.relative_to(P))), 'commit source first: '+path)
    admission_path=INDEX.with_name('cache-long-current.json')
    admission=committed(admission_path)
    guards.require(admission['passed'] and admission['ownership_closed'] and not admission['active'] and
        admission['fresh_faults']==[], 'completed current-runtime admission required')
    reference=admission['original_plan']; raw=git('show',reference['commit']+':'+reference['path'])
    guards.require(hashlib.sha256(raw).hexdigest()==reference['sha256'],'admission original plan changed')
    delta=json.loads(raw)['runtime_asset_delta']
    guards.require(set(delta)<=set(settings['runtime_elfs']),'unknown runtime delta')
    runtime_pins={}
    for p,v in settings['runtime_elfs'].items():
        if p in delta: guards.require(delta[p]['qualified_sha256']==v['sha256'],'original runtime baseline changed')
        runtime_pins[p]=delta[p]['candidate_sha256'] if p in delta else v['sha256']
    for p,h in runtime_pins.items(): guards.require(digest(p)==h,'runtime changed: '+p)
    env.update(UR_LOG_TRACING='level:info;flush:info;output:file,'+str(ROOT/'trace.log'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/'validation.log'),
        TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    plan=dict(source_commit=git('rev-parse','HEAD').decode().strip(),
        inputs={p:digest(p) for p in sorted(paths)},runtime_pins=runtime_pins,
        settings_sha256=digest(SETTINGS),runtime_admission=dict(result_sha256=digest(admission_path),original_plan=reference),
        command=command,environment=env,
        root=str(ROOT),acceptance='Host probe only: successful init/teardown, INVALID_NULL_HANDLE46, unchanged version sentinel, trace of the null-platform API with that result and no device/enqueue/kernel call. Record validation logger bytes without requiring a message: exact installed null-check machine code returns silently. This does not qualify a later model process.',
        budgets=dict(wall=30,cpu=20,rss_cap=2<<30,text_cap=1<<20,file_cap=1<<20,total_cap=8<<20),
        executable_sha256=None,complete=False,passed=False,active=False,ownership_closed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'): (ROOT/name).mkdir()
    index[SCOPE]['canary_plan']=plan
    INDEX.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('Prepared host-only canary; commit the exact index plan before run.')


def run():
    index=committed(INDEX); plan=index[SCOPE]['canary_plan']; record=dict(plan)
    guards.require(plan==json.loads((ROOT/'plan.json').read_text()) and not (ROOT/'record.json').exists(), 'fresh committed plan required')
    for p,h in {**plan['inputs'],**plan['runtime_pins'],str(SETTINGS):plan['settings_sha256']}.items():
        guards.require(digest(p)==h,'frozen input changed: '+p)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    cursor_text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in cursor_text.splitlines() if x.startswith('-- cursor: '))
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,_=owner.run('build',plan['command'],plan['environment'],P,**plan['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'canary build failed')
        record['executable_sha256']=digest(ROOT/'canary')
        entry,_,_=owner.run('canary',[str(ROOT/'canary')],plan['environment'],P,**plan['budgets'])
        record['complete']=supervision.closed(entry)
        guards.require(record['complete'] and entry['exit_code']==0,'canary process failed')
        result=json.loads((ROOT/'owned/canary.stdout').read_text())
        guards.require(result['pid']==entry['direct_identity']['pid'] and result['initialized']==result['closed']==0 and
            result['checked']==result['expected']==46 and result['version']==4294967295,'canary result differs')
        record['canary_result']=result
        trace=(ROOT/'trace.log').read_text(errors='replace')
        record['trace_sha256']=digest(ROOT/'trace.log'); record['trace_bytes']=len(trace.encode())
        guards.require('urPlatformGetApiVersion' in trace and 'UR_RESULT_ERROR_INVALID_NULL_HANDLE' in trace and
            not re.search(r'ur(?:Enqueue|Kernel|Queue|Device|PlatformGet\()',trace),'trace missing or unexpected device work')
        diagnostic=ROOT/'validation.log'
        record['validation_log_bytes']=diagnostic.stat().st_size if diagnostic.exists() else 0
        record['host_probe_accepted']=True
    except BaseException as error: record['error']=str(error)
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        record['fresh_faults']=None
        try:
            text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            record['fresh_faults']=[x for x in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            record['driver_after']=guards.driver_guard()
        except Exception as error: record['final_probe_error']=str(error)
        record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record['passed']=bool(record.get('host_probe_accepted') and record['ownership_closed'] and
            record['boot_unchanged'] and record['fresh_faults']==[] and 'final_probe_error' not in record)
        persist()
        index[SCOPE]['canary_plan']=record
        INDEX.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('complete','passed','error','canary_result','validation_log_bytes')}))
    return 0 if record['passed'] else 1


def main():
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use prepare or run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        return run() if sys.argv[1]=='run' else prepare()


if __name__=='__main__': sys.exit(main())
