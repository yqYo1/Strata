#!/usr/bin/env python3
"""Main-owned event-lane safety prerequisite; repository LICENSE.

Commit source -> prepare -> commit recipe -> build -> commit ELF -> next.
Each next launches ONE fresh bounded process. No throughput/model gate follows.
"""
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
import coherent_original_reuse as original
import engine_once
import measurement_guards as guards
import measurement_owner as owner_api
import ur_validation

P,B=original.P,original.B
ROOT=B/'prefill-upload-event-lane-v1'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
KEY='prefill_upload_event_lane_v1'
SOURCE='tests/sycl/upload_event_lane.cpp'
HEADER='include/strata/prefill/upload_event_lane.hpp'
PREFIXES=['tools/sycl',SOURCE,HEADER,'include/strata/sycl/checked_cleanup.hpp','AGENTS.md','docs/ARTIFACT_RETENTION.md']
LOADER='/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0'
VALIDATION='/usr/lib/x86_64-linux-gnu/libze_validation_layer.so.1.32.0'
BUILD=dict(wall=180,cpu=240,rss_cap=4<<30,text_cap=8<<20,file_cap=64<<20,total_cap=128<<20)
RUN=dict(wall=90,cpu=180,rss_cap=4<<30,text_cap=8<<20,file_cap=8<<20,total_cap=128<<20)
require,sha=guards.require,owner_api.sha
git=original.coherent.build.common.source.git
write=original.coherent.build.common.write


def compact(row):
    value={k:v for k,v in row.items() if k not in ['project','foundation','actual_inputs','commands']}
    value['project']={k:v for k,v in row['project'].items() if k!='source_sha256'}
    value['foundation_sha256']=original.fingerprint(row['foundation'])
    if 'actual_inputs' in row:
        value['actual_inputs']={k:dict(files=len(v),sha256=original.fingerprint(v)) for k,v in row['actual_inputs'].items()}
    return value


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def check(row,committed=True):
    if committed:require(compact(row)==ur_validation.committed(RESULT,(KEY,)),'commit exact local event plan/stage')
    current=original.coherent.build.common.source.pin(P,PREFIXES)
    require(current['source_sha256']==row['project']['source_sha256'],'frozen event source/controller changed')
    native,_=original.verify_original(expected=row['foundation'])
    require(sha(VALIDATION)==row['validation_provider']['sha256'],'ZE validation image changed')
    require(sha(row['runtime_manifest']['path'])==row['runtime_manifest']['sha256'],'runtime manifest changed')
    require(guards.driver_guard()==row['driver'] and row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'boot/driver changed')
    if 'binary' in row:
        require(sha(row['binary']['path'])==row['binary']['sha256'],'event fixture ELF changed')
        require(all(c['argv']==[row['binary']['path'],c['mode']] for c in row['cases']),'launch argv differs from guarded executable/case')
        require(all(sha(p)==h for group in row['actual_inputs'].values() for p,h in group.items()),'actual compiler/link inputs changed')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'inspect existing receipt/handle;no implicit retry')
    prior=ur_validation.committed(RESULT,('prefill_copy_engine_v6',))
    require(prior['stage']=='closed' and prior['ownership_closed'] and not prior['active']
        and prior['validation_summary']['normal_processes']==14
        and prior['analysis']['all_three_pair_gain_gate_passed'] is False,
        'preserve closed copy-only replacement failure;new safety question must not inherit its performance gate')
    native,proof=original.verify_original()
    cc='/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    flags=['-std=c++17','-O2','-fsycl','--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-Wall','-Wextra','-Werror','-I'+str(P/'include')]
    recipe=[('compile',[cc,*flags,'-MD','-MF',str(ROOT/'probe.d'),'-c',str(P/SOURCE),'-o',str(ROOT/'probe.o')]),
        ('link',[cc,*flags,str(ROOT/'probe.o'),LOADER,'-Wl,-t','-o',str(ROOT/'probe')])]
    env=dict(prior['observer_environment'])
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['EnableDirectSubmission']=='0'
        and env['ZE_ENABLE_HANDLE_LIFETIME']=='0','qualified runtime/checker envelope required')
    cases=[]
    for mode in ['metadata','forward','reverse','permanent','cancel','refuse']:
        root=ROOT/mode
        child=dict(env,TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,'+str(root/'ur.trace'),
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'),
            ZEL_LOADER_LOG_CONSOLE='1',ZEL_LOADER_LOGGING_LEVEL='trace',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='1')
        cases.append(dict(mode=mode,root=str(root),environment=child,argv=[str(ROOT/'probe'),mode]))
    row=dict(stage='prepared',project=original.coherent.build.common.source.pin(P,PREFIXES),foundation=proof,
        external_inputs=native['external_inputs'],runtime_manifest=prior['runtime_manifest'],
        validation_provider=prior['validation_provider'],native_validation_probe=native['native_validation_probe'],
        native_source_commit=git(P,'rev-parse','HEAD').decode().strip(),recipe=recipe,
        build_environment=native['recipe']['environment'],driver=guards.driver_guard(),boot=prior['boot'],
        cases=cases,results=[],commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        model_trial_allowed=False,performance_qualified=False,
        plan=dict(question='E1safety prerequisite for an independent prefill H2D lane:raw ordinal1copy-ready to SYCLordinal0reader,and exported last-reader fence back to copy before scratch overwrite.',
            measured_opportunity='Original current-route trace has~10.31scopy union,~28.66skernel union,~4msoverlap;requested bank bytes35.2GB. Standalone synchronized host-USM rate~6.3GB/s did not improve from engine replacement. This motivates a DIFFERENT bounded scheduling/dependency question,not a saturation or model-speed claim.',
            minimal_coherent_unit='Per-transfer completion token,queue-bound last-reader token,native/SYCLevent ownership,one scratch-overwrite dependency,source/destination retention,cancellation and dual-queue drain. No changed inference math,placement,bank sizing,source-view copying,decode or general Runtime policy.',
            source='Main-authored header and single fresh test TU,repository LICENSE. Official installed SYCL/LevelZero API declarations are behavior references;no external implementation or XeStrata source. Independent serialized probe is archived at3d801e985.',
            cases='Metadata first,no application GPU work;then held forward copy/reader,held second-reader reverse overwrite,permanent repeated readers,cancel after accepted upload,and bad range/null/foreign/stale token refusals. Each native process is fresh;no event reset/recycling or graph capture.',
            gates='Commit source/recipe/SDK/compiler/ELF;required actual providers before first application enqueue and again after exact-kernel prewarm;private pre-main URactivation;full traces and quiet parameter/thread validation;host hold remains pending>=50ms then manual release;dependency submission/import<100ms where asserted;full payload/source/edge guards;normal0/positive closed ownership;unchanged boot/driver,no fresh fault.',
            stop='Stop first identity,source,build,API,guard,validation,deadline,ownership or GPUfault failure. Independent2s host release runs before any wait;raw drain5s and owned native90s wall. No retry,driver reset,foreign process signals or hash refresh.',
            limits='No timing discriminator,overlap gain,model,decode,quality,agent,physical262144 or ordinary runtime teardown qualification. HostUSM only;ordinary immutable RAMsource views remain a separate required source-kind check before production. No throughput derived from uint reader fixture.',
            next='Only after E1source/safety gates pass,commit a separate E2comparison using representative production consumer work and fixed bytes/math,serialized versus event overlap,then actual32K+model only if its own component gate passes. Prior replacement gate stays failed.',
            retention='One temporary component beside unchanged fast14508/namingAengine references. Retain through E1and its immediate E2decision,then retire ended code/artifacts after exact source/recipe recovery. No new inference image.',
            references=['https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_backend_level_zero.md',
                'https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_list.html',
                'https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/event.html']))
    ROOT.mkdir(mode=0o700)
    for name in ['tmp','cache','owned']:(ROOT/name).mkdir(mode=0o700)
    save(row);print('Prepared E1event safety prerequisite;commit exact recipe before build.',flush=True)


def mappings(pid,row):
    allowed=json.loads(Path(row['runtime_manifest']['path']).read_text())
    for image in [row['binary'],row['validation_provider'],*row['native_validation_probe'].values()]:
        allowed[str(Path(image['path']).resolve())]=image['sha256']
    found={}
    for line in Path('/proc',str(pid),'maps').read_text().splitlines():
        v=line.split(maxsplit=5)
        if len(v)==6 and 'x' in v[1] and v[5].startswith('/'):
            p=str(Path(v[5]).resolve());require(allowed.get(p)==sha(p),'unfrozen initialized provider: '+p);found[p]=allowed[p]
    return found


def execute(action):
    row=json.loads((ROOT/'record.json').read_text());native=check(row)
    require(row['stage'] in (['prepared'] if action=='build' else ['built','testing']),'fresh ordered stage required')
    supervisor=owner_api.Owner(ROOT/'owned',lambda:write(ROOT/'ownership.json',supervisor.commands))
    cursor=None
    try:
        if action=='build':
            allowed=json.loads(Path(row['external_inputs']['path']).read_text());allowed.update(json.loads(Path(row['runtime_manifest']['path']).read_text()))
            for name in [SOURCE,HEADER,'include/strata/sycl/checked_cleanup.hpp']:allowed[str(P/name)]=sha(P/name)
            actual=dict(compiled={},links={},objects={})
            for label,argv in row['recipe']:
                entry,out,err=supervisor.run(label,argv,row['build_environment'],P,**BUILD)
                require(owner_api.closed(entry) and entry['exit_code']==0,'event fixture compiler/link failed: '+str(err))
                if label=='compile':
                    for name in shlex.split((ROOT/'probe.d').read_text().replace('\\\n',' ').split(':',1)[1]):
                        p=str(Path(name).resolve());h=sha(p);require(allowed.get(p)==h,'unfrozen compiler input: '+p);actual['compiled'][p]=h
                    actual['objects'][str(ROOT/'probe.o')]=sha(ROOT/'probe.o')
                else:
                    allowed.update(actual['objects'])
                    for line in out.read_text().splitlines():
                        p=Path(line.strip())
                        if p.is_absolute() and p.is_file():
                            h=sha(p);key=str(p.resolve());require(allowed.get(key)==h,'unfrozen linker input: '+key);actual['links'][key]=h
            require(len(actual['compiled'])>100 and len(actual['links'])>8,'complete compiler/link census required')
            row.update(stage='built',passed=True,actual_inputs=actual,binary=dict(path=str(ROOT/'probe'),sha256=sha(ROOT/'probe')))
        else:
            index=len(row['results']);require(index<len(row['cases']),'all cases already closed')
            case=row['cases'][index];root=Path(case['root']);require(not root.exists(),'inspect existing case/owner;no retry')
            root.mkdir(mode=0o700)
            for name in ['tmp','cache']:(root/name).mkdir(mode=0o700)
            journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
            cursor=journal.split('-- cursor: ',1)[1].strip()
            row.update(stage='testing',active=True,ownership_closed=False,journal_cursor=cursor);save(row)
            expected=['before-runtime','runtime-ready']+(['buffers-ready'] if case['mode']!='metadata' else [])+['before-exit']
            gates=[];records=[];providers={}
            def interaction(channel,_entry):
                while True:
                    item=json.loads(channel.line(30))
                    if item['kind']=='gate':
                        name=item['name'];require(name==expected[len(gates)],'event fixture gate order')
                        providers[name]=mappings(channel.proc.pid,row)
                        if name in ['runtime-ready','buffers-ready']:
                            needed=[VALIDATION,LOADER,case['environment']['ZE_ENABLE_ALT_DRIVERS'],case['environment']['UR_ADAPTERS_FORCE_LOAD'],'/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so']
                            require(all(str(Path(p).resolve()) in providers[name] for p in needed),'missing required initialized provider')
                        gates.append(name);channel.send(b'continue\n')
                    else:
                        records.append(item)
                        if item['kind']=='done':channel.exit(20);break
            entry,out,err=supervisor.run(case['mode'],case['argv'],engine_once.native_environment(row,case['environment']),P,interaction=interaction,**RUN)
            require(owner_api.closed(entry) and entry['exit_code']==0,'event fixture failed: '+str(err))
            validation=ur_validation.preload_evidence(err.read_text(),entry['pid'])
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0,'URvalidation warning/error')
            require(not re.search(r'\[(?:error|critical)\]|upload lane failure:|Strata SYCL cleanup:',err.read_text(),re.I),'ZEvalidation/cleanup failure')
            require(gates==expected and records==[dict(kind='done',passed=True,case=case['mode'])],'complete case result/gates required')
            trace=(root/'ur.trace').read_text();enqueues=re.findall(r'---> (urEnqueue\w+)',trace)
            require(bool(enqueues)==(case['mode']!='metadata'),'actual application enqueue scope differs')
            check(row,False)
            row['results'].append(dict(mode=case['mode'],passed=True,native_exit_code=0,ownership_closed=True,
                validation=validation,gates=gates,application_enqueues=len(enqueues),
                initialized_providers=providers['runtime-ready'],after_work_providers=providers['before-exit']))
            row.update(passed=True,active=False,ownership_closed=True)
            if len(row['results'])==len(row['cases']):row.update(stage='closed',complete=True)
    except BaseException as error:
        row.update(stage='failed',passed=False,complete=False,error=repr(error));raise
    finally:
        row['commands'].extend(supervisor.commands);row['active']=supervisor.active is not None
        row['ownership_closed']=supervisor.active is None and all(c.get('session_empty') and c.get('direct_child_reaped') for c in row['commands'])
        if cursor:
            try:
                after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                row['fresh_faults']=[v for v in after.splitlines() if re.search(r'gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout)',v,re.I)]
                row['boot_unchanged']=row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                row['driver_after']=guards.driver_guard()
                if row['fresh_faults'] or not row['boot_unchanged']:row.update(stage='failed',passed=False,complete=False)
            except BaseException as e:row.update(stage='failed',passed=False,complete=False,final_health_error=repr(e))
        save(row)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','complete','ownership_closed','model_trial_allowed']},sort_keys=True),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in [['prepare'],['build'],['next']],'prepare|build|next')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:raise SystemExit(execute(sys.argv[1]))
