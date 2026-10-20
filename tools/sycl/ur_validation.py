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
ROOT=B/'ur-validation-canary-v2'
SOURCE=P/'bench/sycl/ur_validation_canary.c'
PRELOAD_SOURCE=P/'bench/sycl/ur_validation_preload.c'
PRELOAD_MODE=False


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=P)


def committed(path, keys=()):
    """Require the consumed input to be committed, without freezing other research.

    A scoped read returns only the checked subtree. Native source/runtime and
    artifact checks still run in preload_qualification; unrelated research
    lifecycle changes are not inputs to that qualification.
    """
    raw=path.read_bytes()
    original=git('show','HEAD:'+str(path.relative_to(P)))
    message='commit exact input first: '+str(path)
    if not keys:
        guards.require(raw==original,message)
        return json.loads(raw)
    current,reference=json.loads(raw),json.loads(original)
    for key in keys:
        guards.require(isinstance(current,dict) and isinstance(reference,dict) and
            key in current and key in reference,'missing committed input scope: '+str(path))
        current,reference=current[key],reference[key]
    # JSON comparison keeps bool/int/float distinct, unlike Python dict equality.
    canonical=lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    guards.require(canonical(current)==canonical(reference),message+' / '+'.'.join(keys))
    return current


def preload_evidence(text,pid):
    begin=list(re.finditer(r'^strata UR validation: pid=(\d+) phase=begin loader=([^\n]+)$',text,re.M))
    # Pinned ur_api.h: UR_RESULT_ERROR_UNINITIALIZED=37, INVALID_NULL_HANDLE=46.
    proof=list(re.finditer(r'^strata UR validation: pid=(\d+) phase=before-main threads_before=1 preinit=37 config_create=0 layers_query=0 enable=0 init=0 config_release=0 checked=46 untouched=1 retained_ref=1$',text,re.M))
    guards.require(len(begin)==len(proof)==1 and int(begin[0][1])==int(proof[0][1])==pid and
        begin[0].start()<proof[0].start() and
        'strata UR validation failure:' not in text,'native-child config/first-init proof missing')
    return dict(pid=pid,loader=begin[0][2],threads_before=1,preinit=37,config_enabled=True,
        first_init=True,dispatch_result=46,output_untouched=True,retained_reference_until_exit=True,
        runtime_teardown_qualified=False)


def preload_qualification():
    result=committed(INDEX,('UR_silent_validation_native_child_scope','preload_component'))
    guards.require(result['passed'] and result['ownership_closed'] and result['fresh_faults']==[] and
        result['provider_mismatch_stopped_before_body'],'completed preload component required')
    ref=result['original_plan'];raw=git('show',ref['commit']+':'+ref['path'])
    guards.require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'original preload plan changed')
    original=json.loads(raw)['UR_silent_validation_native_child_scope']['canary_plan']
    guards.require(original['source_commit']==result['source_commit'],'preload source identity differs')
    for p,h in original['inputs'].items():
        path=Path(p)
        if path.is_relative_to(P):
            guards.require(hashlib.sha256(git('show',result['source_commit']+':'+str(path.relative_to(P)))).hexdigest()==h,
                'original preload input unrecoverable: '+p)
            if path.suffix!='.c': continue  # A later controller does not relabel native input.
        guards.require(digest(path)==h,'preload native input changed: '+p)
    for p,h in original['runtime_pins'].items():guards.require(digest(p)==h,'preload runtime changed: '+p)
    for artifact in (result['library'],result['canary']):
        guards.require(digest(artifact['path'])==artifact['sha256'],'preload fixture bytes changed')
    loader=result['preload_evidence']['loader']
    return dict(library=result['library'],loader=dict(path=loader,sha256=original['runtime_pins'][loader]),
        component_source_commit=result['source_commit'],component_original_plan=result['original_plan'],
        retained_reference_until_exit=True,runtime_teardown_qualified=False)


def finish(record,owner,cursor,boot):
    record.update(ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands),
        fresh_faults=None,boot_unchanged=False)
    try:
        text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
        record['fresh_faults']=[x for x in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
        record['driver_after']=guards.driver_guard()
        record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except Exception as error: record['final_probe_error']=str(error)
    record['passed']=bool(record.get('host_probe_accepted') and record['ownership_closed'] and
        record['boot_unchanged'] and record['fresh_faults']==[] and 'final_probe_error' not in record)


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
    sources=[SOURCE]+([PRELOAD_SOURCE] if PRELOAD_MODE else [])
    headers=[]
    for source in sources:
        dependencies=subprocess.check_output([*command[:7],'-M',str(source)],text=True,timeout=15)
        headers.extend(shlex.split(dependencies.replace('\\\n',' '))[1:])
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
    library=ROOT/'libstrata_ur_validation.so'
    commands=[command]
    native_env=dict(env)
    if PRELOAD_MODE:
        commands.append([*command[:7],'-fPIC','-shared',str(PRELOAD_SOURCE),
            '-L'+str(install/'lib'),'-Wl,-rpath,'+str(install/'lib'),'-lur_loader','-ldl','-o',str(library)])
        loader,=(p for p in runtime_pins if Path(p).name.startswith('libur_loader.so.'))
        native_env.update(LD_PRELOAD=str(library),STRATA_UR_PROBE_EXPECTED_LOADER=loader)
    plan=dict(source_commit=git('rev-parse','HEAD').decode().strip(),
        inputs={p:digest(p) for p in sorted(paths)},runtime_pins=runtime_pins,
        settings_sha256=digest(SETTINGS),runtime_admission=dict(result_sha256=digest(admission_path),original_plan=reference),
        command=command,build_commands=commands,environment=env,native_environment=native_env,
        preload_mode=PRELOAD_MODE,
        root=str(ROOT),acceptance='Host probe only: successful init/teardown, INVALID_NULL_HANDLE46, unchanged version sentinel, trace of the null-platform API with that result and no device/enqueue/kernel call. Record validation logger bytes without requiring a message: exact installed null-check machine code returns silently. This does not qualify a later model process.',
        budgets=dict(wall=30,cpu=20,rss_cap=2<<30,text_cap=1<<20,file_cap=1<<20,total_cap=8<<20),
        executable_sha256=None,complete=False,passed=False,active=False,ownership_closed=False)
    if PRELOAD_MODE:
        plan['acceptance']='Fresh host-only process, no model/device/queue/enqueue. Preload provider matches pinned loader; single thread and UNINITIALIZED before config, successful config create/available-layer query/enable/first init/release, expected46 with unchanged output, same-PID before-main receipt. Original canary body succeeds after its own additional init/teardown. No exit-handler teardown in preload: one diagnostic reference remains until process exit. A separate wrong-provider process must exit125 before body/UR initialization. No later model activation or ordinary runtime-teardown qualification.'
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
        for i,command in enumerate(plan['build_commands']):
            entry,_,_=owner.run('build'+str(i),command,plan['environment'],P,**plan['budgets'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,'canary build failed')
        record['executable_sha256']=digest(ROOT/'canary')
        if plan['preload_mode']:
            record['library']=dict(path=str(ROOT/'libstrata_ur_validation.so'),sha256=digest(ROOT/'libstrata_ur_validation.so'))
        entry,_,_=owner.run('canary',[str(ROOT/'canary')],plan['native_environment'],P,**plan['budgets'])
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
        if plan['preload_mode']:
            text=(ROOT/'owned/canary.stderr').read_text()
            record['preload_evidence']=preload_evidence(text,result['pid'])
            guards.require(record['preload_evidence']['loader']==plan['native_environment']['STRATA_UR_PROBE_EXPECTED_LOADER'] and
                text.index('phase=before-main')<text.index('strata UR canary begin'), 'proof must precede main body')
            bad_env=dict(plan['native_environment'],STRATA_UR_PROBE_EXPECTED_LOADER='/invalid-strata-probe-provider')
            bad,_,_=owner.run('provider-mismatch',[str(ROOT/'canary')],bad_env,P,**plan['budgets'])
            guards.require(supervision.closed(bad) and bad['exit_code']==125 and
                (ROOT/'owned/provider-mismatch.stdout').stat().st_size==0 and
                'unexpected loader provider' in (ROOT/'owned/provider-mismatch.stderr').read_text(), 'wrong-provider guard did not stop before body')
            record['provider_mismatch_stopped_before_body']=True
        record['host_probe_accepted']=True
    except BaseException as error: record['error']=str(error)
    finally:
        finish(record,owner,cursor,boot)
        persist()
        index[SCOPE]['canary_plan']=record
        INDEX.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('complete','passed','error','canary_result','validation_log_bytes')}))
    return 0 if record['passed'] else 1


def prepare_native():
    import cache_long_compare as comparison
    index=committed(INDEX);probe=preload_qualification()
    old,_,proof,original=comparison.qualification()
    root=B/'ur-validation-retained-help-v1';guards.require(not root.exists(),'fresh native help only')
    source=git('rev-parse','HEAD').decode().strip();pins=comparison.gen.benchmark_pins(source)
    env={k:v.replace(original['root'],str(root)) for k,v in original['environment'].items()}
    env.update(STRATA_MEASUREMENT_ROOT=str(root),CUDA_MODULE_LOADING='EAGER',
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'trace.log'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    guards.require('LD_PRELOAD' not in env,'preload belongs only to the final native child')
    plan=dict(source_commit=source,binary=old['binary'],build_source_commit=old['build_source_commit'],
        retained_build_proof=proof,native_validation_probe=probe,root=str(root),environment=env,
        benchmark_fingerprint=comparison.assets.fingerprint(pins),
        command=[sys.executable,'-B',str(P/'tools/sycl/engine_once.py'),'--help'],
        acceptance='Actual unchanged retained396 binary --help only. Probe receipt must have its native PID, first-init config and provider evidence, before any model stage. No model request/device/queue/enqueue; no speed or ordinary teardown claim.',
        budgets=dict(wall=30,cpu=20,rss_cap=2<<30,text_cap=1<<20,file_cap=1<<20,total_cap=8<<20),
        complete=False,passed=False,active=False,ownership_closed=False)
    root.mkdir(mode=0o700)
    for name in ('tmp','cache'):(root/name).mkdir()
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    index['UR_silent_validation_native_child_scope']['native_plan']=plan
    INDEX.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print('Prepared actual retained --help probe; commit exact plan before run.')


def run_native():
    import cache_long_compare as comparison
    scope='UR_silent_validation_native_child_scope';index=committed(INDEX)
    record=index[scope]['native_plan'];root=Path(record['root'])
    guards.require(record==json.loads((root/'plan.json').read_text()) and not (root/'record.json').exists(),'fresh frozen native-help plan required')
    old,_,proof,_=comparison.qualification()
    guards.require(old['binary']==record['binary'] and proof==record['retained_build_proof'] and
        preload_qualification()==record['native_validation_probe'] and
        comparison.assets.fingerprint(comparison.gen.benchmark_pins(record['source_commit']))==record['benchmark_fingerprint'],
        'native help source/fixtures changed')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in text.splitlines() if x.startswith('-- cursor: '))
    owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(record,commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('help',record['command'],record['environment'],P,**record['budgets'])
        record['complete']=supervision.closed(entry)
        guards.require(record['complete'] and entry['exit_code']==0,'retained native help failed')
        native=json.loads((root/'engine-exit.json').read_text())
        guards.require(native['exit_code']==0 and native['args']==['--help'] and native['binary']==record['binary'] and
            native['native_validation_probe']==record['native_validation_probe'],'retained native identity differs')
        log=(root/'owned/help.stderr').read_text(errors='replace')
        record['native_identity']=native['native'];record['preload_evidence']=preload_evidence(log,native['native']['pid'])
        trace=(root/'trace.log').read_text(errors='replace')
        guards.require(record['preload_evidence']['loader']==record['native_validation_probe']['loader']['path'] and
            not re.search(r'ur(?:Enqueue|Kernel|Queue|Device|PlatformGet\()',trace) and
            'UR_RESULT_ERROR_INVALID_NULL_HANDLE' in trace and
            not re.search(r'^strata (?:generate: expert cache|serve: prompt|trace:)',log,re.M),'unexpected model/device work')
        record.update(trace_bytes=len(trace.encode()),trace_sha256=digest(root/'trace.log'),host_probe_accepted=True)
    except BaseException as error:record['error']=str(error)
    finally:
        finish(record,owner,cursor,boot);persist();index[scope]['native_plan']=record
        INDEX.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('complete','passed','error','preload_evidence')}))
    return 0 if record['passed'] else 1


def main():
    global ROOT,SCOPE,PRELOAD_MODE
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run','prepare-preload','run-preload','prepare-native','run-native'),'use prepare/run with optional -preload or -native')
    if sys.argv[1].endswith('-preload'):
        PRELOAD_MODE=True;ROOT=B/'ur-validation-preload-v2';SCOPE='UR_silent_validation_native_child_scope'
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        if sys.argv[1].endswith('-native'):return run_native() if sys.argv[1].startswith('run') else prepare_native()
        return run() if sys.argv[1].startswith('run') else prepare()


if __name__=='__main__': sys.exit(main())
