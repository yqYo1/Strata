#!/usr/bin/env python3
"""Main-owned, host-only qualification of a bounded UR memory subscriber.

Repository LICENSE. No engine rebuild, GPU workload or speed qualification.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import check_model_restore as model

P, B, RESULT = model.P, model.B, model.RESULT
ROOT = B / 'ur-memory-subscriber-v1'
KEY = 'ur_memory_subscriber_v2'
INPUTS = sorted(set([*model.INPUTS,'tools/sycl/memory_trace.py',
    'tools/sycl/ur_memory_subscriber.cpp','tools/sycl/ur_memory_packet_test.cpp']))
require, digest, write = model.require, model.digest, model.write


def pins(revision):
    result={}
    for name,raw in model.gate.native.build.blobs(P,revision,INPUTS):
        require(raw==(P/name).read_bytes(),'commit subscriber input: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def compact(r):
    return {k:v for k,v in r.items() if k not in ('environment','dependencies','commands')}


def prepare(container):
    require(KEY not in container and ROOT.exists(),'fresh admission-corrected host gate only')
    old=json.loads((ROOT/'record.json').read_text())
    require(compact(old)==container['ur_memory_subscriber_v1'] and
        old['stage']=='closed_for_environment_admission_correction' and not old['active'] and
        old['ownership_closed'] and set(old['outcomes'])=={'build'} and old['outcomes']['build']['passed'],
        'closed original subscriber build required;no host test inherited')
    native=model.gate.engine()
    state=json.loads((model.ROOT/'record.json').read_text())
    require(state['complete'] and state['passed'] and not state['active'] and state['ownership_closed'] and
        state['binary']==native['binary'],'completed same-native v4 state gate required')
    probe=model.gate.validation.preload_qualification()
    qualification=model.gate.validation.committed(model.gate.validation.INDEX,
        ('UR_silent_validation_native_child_scope','preload_component'))
    revision=model.gate.native.build.git(P,'rev-parse','HEAD').decode().strip()
    source=pins(revision)
    env=dict(state['environments']['small'])
    sdk=Path(env['LD_LIBRARY_PATH'].split(':')[0]).parent
    framework=(sdk/'lib/libxptifw.so.1.0.1').resolve()
    require(framework.is_file(),'installed XPTI dispatcher missing')
    options=['-std=c++17','-O2','-fPIC','-Wall','-Wextra','-Werror','-pedantic',
        '-I',str(sdk/'include')]
    compiler='/usr/bin/g++';deps={}
    for name in ('ur_memory_subscriber.cpp','ur_memory_packet_test.cpp'):
        command=[compiler,*options,'-M','-MT','fixture',str(P/'tools/sycl'/name)]
        q=subprocess.run(command,cwd=P,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},
            capture_output=True,text=True,timeout=20)
        require(q.returncode==0 and not q.stderr and len(q.stdout)<1<<20,
            'bounded subscriber header discovery failed: '+q.stderr[:2000])
        for path in shlex.split(q.stdout.replace('\\\n',' ').split(':',1)[1]):
            path=Path(path).resolve();deps[str(path)]=digest(path)
    require(len(deps)>20,'actual header closure missing')
    deps[str(framework)]=digest(framework)
    libraries=['-L',str(sdk/'lib'),'-Wl,-rpath,'+str(sdk/'lib'),'-lxptifw']
    commands={
        'library':[compiler,*options,'-shared',str(P/'tools/sycl/ur_memory_subscriber.cpp'),
            *libraries,'-o',str(ROOT/'libstrata_ur_memory.so')],
        'packets-build':[compiler,*options,str(P/'tools/sycl/ur_memory_packet_test.cpp'),
            *libraries,'-o',str(ROOT/'packet-test')],
        'packets':[str(ROOT/'packet-test')],
        'canary':[qualification['canary']['path']]}
    trace=dict(env,XPTI_TRACE_ENABLE='1',XPTI_FRAMEWORK_DISPATCHER=str(framework),
        XPTI_SUBSCRIBERS=str(ROOT/'libstrata_ur_memory.so'),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:warning;flush:warning;output:stderr')
    host={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']}
    compiler_pin={'path':compiler,'sha256':digest(compiler)}
    environment={'host':host,'trace':trace}
    environment_hashes={k:model.model.fingerprint(v) for k,v in environment.items()}
    require(old['compiler']==compiler_pin and old['dependencies']==deps and
        old['commands_planned']==commands and old['environment_sha256']==environment_hashes and
        old['environment']==environment,'collector native recipe/headers/runtime/environment changed')
    for name,h in old['input_sha256'].items():
        raw=model.gate.native.build.git(P,'show',old['source_commit']+':'+name)
        require(hashlib.sha256(raw).hexdigest()==h,'original subscriber source unrecoverable: '+name)
        if name!='tools/sycl/memory_trace.py':require(source[name]==h,'native/transitive source changed: '+name)
    for asset in old['assets'].values():require(digest(asset['path'])==asset['sha256'],'original artifact changed')
    r=dict(stage='build_closed',source_commit=revision,input_sha256=source,
        compiler=compiler_pin,dependencies=deps,commands_planned=commands,environment=environment,
        environment_sha256=environment_hashes,
        native_source_commit=native['source_commit'],native_binary=native['binary'],
        preload=probe,canary=qualification['canary'],
        assets=old['assets'],outcomes={},active=False,passed=False,complete=False,ownership_closed=True,
        inherited_build=dict(result_key='ur_memory_subscriber_v1',source_commit=old['source_commit'],
            original_record_sha256=digest(ROOT/'record.json'),assets_verified=2,actual_dependency_inputs=len(deps),
            native_source_recipe_environment_unchanged=True,host_tests_inherited=False,
            native_binaries_rebuilt=False,changed_input_only='tools/sycl/memory_trace.py'),
        GPU_executed=False,speed_eligible=False,ordinary_runtime_teardown_qualified=False,
        budgets=dict(wall_s=60,cpu_s=60,rss_bytes=2<<30,text_bytes=8<<20,file_bytes=8<<20,total_bytes=32<<20),
        criteria='Commit before every gate. Build shared subscriber and CPU packet fixture with pinned host compiler/header closure. Packet fixture must emit12 events:3 successful USM domains,failed uninitialized-output refusal,free,buffer create/retain/alias/release,unknown mutating API and missing packet controls;no UR initialization or GPU operation. Real qualified host canary must activate ur.call collector exactly once,preserve parameter-validation first-init receipt and normal exit0. Subscriber only reads documented argument packets and reports requested logical bytes,not physical residency/pool ownership. Unknown memory APIs remain unsupported. Stop first failure;no retry. No model/device/speed/adoption or ordinary-teardown claim.',
        provenance='Main-authored Strata source,repository LICENSE. Installed public XPTI/UR headers and Intel LLVM official urtrace documentation/API examples consulted;no external application source copied or XeStrata application source used.',
        retention='Retain current subscriber for imminent Main device/short-prefill owner trace;retire CPU packet executable and successful logs once qualification is compacted. No third engine generation. Closed v4 state record environment consumer ends after this new committed plan pins it.')
    write(ROOT/'record.json',r);container[KEY]=compact(r);write(RESULT,container)
    print('Prepared host subscriber gates;commit before build.',flush=True)


def admission(r,container):
    require(RESULT.read_bytes()==model.gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        compact(r)==container[KEY] and pins(r['source_commit'])==r['input_sha256'],'commit exact subscriber plan')
    require(digest(r['compiler']['path'])==r['compiler']['sha256'] and
        all(digest(p)==h for p,h in r['dependencies'].items()),'compiler/header/runtime identity changed')
    require(set(r['environment'])==set(r['environment_sha256']) and
        all(model.model.fingerprint(v)==r['environment_sha256'][k] for k,v in r['environment'].items()),
        'effective subscriber environment differs')
    native=model.gate.engine()
    require(native['binary']==r['native_binary'] and native['source_commit']==r['native_source_commit'] and
        model.gate.validation.preload_qualification()==r['preload'],'native/qualified loader changed')
    for asset in [r['canary'],*r['assets'].values()]:
        require(digest(asset['path'])==asset['sha256'],'subscriber/canary asset changed')


def run(container,mode):
    r=json.loads((ROOT/'record.json').read_text());admission(r,container)
    require(r['stage']==('prepared' if mode=='build' else 'build_closed'),'no retry/skipped host gate')
    original=json.loads(json.dumps(r))
    owner=None
    def persist():
        r['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(r,commands=owner.commands if owner else []))
    owner=model.gate.native.supervision.Owner(ROOT/'owned',persist)
    outcome=dict(passed=False,complete=False,ownership_closed=False);r['outcomes'][mode]=outcome
    r['stage']=mode+'_running';persist()
    try:
        for label in (('library','packets-build') if mode=='build' else ('packets','canary')):
            b=r['budgets'];entry,out,err=owner.run(label,r['commands_planned'][label],
                r['environment']['trace' if label=='canary' else 'host'],P,
                wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
                file_cap=b['file_bytes'],total_cap=b['total_bytes'])
            require(model.gate.native.supervision.closed(entry) and entry['exit_code']==0,
                'normal host gate exit required: '+label+': '+entry.get('error',str(entry['exit_code'])))
            text=err.read_text(errors='replace')
            if label in ('library','packets-build'):
                require(not text and not out.stat().st_size,'compiler diagnostics require review')
            if label=='packets':
                rows=re.findall(r'^strata UR memory: seq=.*$',text,re.M)
                require(len(rows)==12 and 'active=0 records=12 invalid=1 unsupported=1 lost=0 ' in text,
                    'packet ABI negative/positive controls incomplete')
                require(sum('result=0 kind=alloc ' in line for line in rows)==4 and
                    all(any('domain='+domain+' ptr=0x1000 ' in line and 'requested_bytes=256 ' in line
                        for line in rows) for domain in ('device','host','shared')),
                    'successful pointer/size outputs differ')
                require(any('kind=alias domain=buffer ptr=0x3000 parent=0x2000 ' in line for line in rows)
                    and any('kind=alloc domain=device ptr=(nil) ' in line and 'result=0 ' not in line for line in rows),
                    'alias or failed-output packet differs')
                outcome['packet_controls']=12
            if label=='canary':
                proof=model.gate.validation.preload_evidence(text,entry['pid'])
                require(text.count('activated stream=ur.call format=1 limit=25000')==1 and
                    'active=1 records=0 invalid=0 unsupported=0 lost=0 ' in text,'host XPTI activation incomplete')
                body=json.loads(out.read_text());require(body['initialized']==body['closed']==0 and body['checked']==46,
                    'qualified host canary changed')
                outcome['activation']=proof;outcome['host_xpti_activation']=True
        admission(original,container)
        if mode=='build':
            for name in ('libstrata_ur_memory.so','packet-test'):
                path=ROOT/name;r['assets'][name]=dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))
        outcome['passed']=True
    except BaseException as error:outcome['error']=repr(error)
    finally:
        outcome['ownership_closed']=owner.active is None and all(model.gate.native.supervision.closed(c) for c in owner.commands)
        outcome['passed']=bool(outcome['passed'] and outcome['ownership_closed']);outcome['complete']=outcome['passed']
        r['ownership_closed']=outcome['ownership_closed'];r['stage']=mode+'_closed' if outcome['passed'] else 'failed'
        r['passed']=r['complete']=bool(mode=='check' and outcome['passed']);persist()
        container[KEY]=compact(r);write(RESULT,container)
    print(json.dumps(outcome),flush=True)
    return 0 if outcome['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['build'],['check']),'prepare|build|check')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model.gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(run(container,sys.argv[1]))
