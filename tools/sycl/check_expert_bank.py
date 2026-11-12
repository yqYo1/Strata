#!/usr/bin/env python3
"""Main-owned CPU admission/ticket checks before bounded expert-bank GPU work."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision
import query_profile as runtime

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'moe-expert-bank-component-v5'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/moe-expert-bank-current.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gpu_check(mode):
    packed = mode.startswith('packed-')
    smoke = mode.endswith('smoke')
    field = 'packed_gpu' if packed else 'gpu'
    root = B/('moe-packed-expert-bank-gpu-v2' if packed else 'moe-expert-bank-gpu-v2')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        raw = subprocess.check_output(['git','show','HEAD:'+str(RESULT.relative_to(P))],cwd=P)
        record = json.loads(raw)
        guards.require(raw == RESULT.read_bytes() and record['passed'] and record['complete'] and
            record['ownership_closed'] and not record['active'], 'commit passed CPU gate first')
        for name in ('include/strata/prefill/expert_bank_plan.hpp',
                     'include/strata/core/expert_layer_view.hpp',
                     'include/strata/prefill/layer_major_plan.hpp','tests/expert_bank_plan.cpp'):
            guards.require(digest(P/name)==record['source_sha256'][name], 'CPU-tested planner changed: '+name)
        build = json.loads((B/'main-native-gdn-engine/record.json').read_text())
        guards.require(build['complete'] and build['passed'] and build['ownership_closed'] and
            not build['active'], 'closed current build required')
        for group in [build['project'],*build['dependencies']]:
            for name,expected in group['source_sha256'].items():
                guards.require(digest(Path(group['root'])/name)==expected, 'build input changed: '+name)
        fixture = build['fixtures']['sycl_expert_bank_test']
        guards.require(digest(fixture['path'])==fixture['sha256'], 'expert-bank fixture changed')
        provider_fixture = build['fixtures']['file_expert_source_test']
        guards.require(digest(provider_fixture['path'])==provider_fixture['sha256'], 'provider fixture changed')
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()
        names = ('tools/sycl/check_expert_bank.py','tools/sycl/measurement_guards.py',
            'tools/sycl/query_profile.py',
            'tools/sycl/generation_engine.py','tools/sycl/bench_engine.py',
            'tools/sycl/check_verifier_retirement.py','tools/sycl/build_engine.py',
            'tools/sycl/measurement_owner.py','tools/sycl/measurement_environment.json',
            'include/strata/prefill/expert_bank_upload.hpp','tests/sycl/expert_bank.cpp',
            'include/strata/core/expert_layer_view.hpp','include/strata/core/expert_source.hpp',
            'src/core/expert_source.cpp','tests/core/file_expert_source_test.cpp')
        pins = {}
        for name in names:
            raw = subprocess.check_output(['git','show',revision+':'+name],cwd=P)
            guards.require(raw==(P/name).read_bytes(), 'commit exact GPU check input: '+name)
            pins[name] = hashlib.sha256(raw).hexdigest()
        if smoke:
            guards.require(not root.exists() and field not in record, 'inspect existing bank GPU owner')
            root.mkdir(mode=0o700)
            for name in ('tmp','cache'): (root/name).mkdir()
            gpu = dict(source_commit=revision,source_sha256=pins,build_source_commit=build['source_commit'],
                fixture=fixture,provider_fixture=provider_fixture,CPU_gate_source=record['source_commit'],checks=[],
                limits='Production upload helper/cache, synthetic byte readers, private maps and caller unwind only. No real provider enqueue fault injection, model arithmetic, speed, session rollback or physical262144/RESTORE qualification.')
        else:
            gpu = record[field]
            guards.require(gpu['mode']==('packed-smoke' if packed else 'smoke') and gpu['passed'] and gpu['ownership_closed'] and
                gpu['build_source_commit']==build['source_commit'] and gpu['fixture']==fixture and
                gpu['provider_fixture']==provider_fixture and gpu.get('provider_checks_passed') is True and
                gpu['source_sha256']==pins, 'same-source passed smoke required')
        guards.require(not (root/('owned-'+mode)).exists(), 'inspect existing GPU command')
        settings = json.loads((P/'tools/sycl/measurement_environment.json').read_text())
        runtime_evidence = runtime.gen.committed(runtime.COMPONENT) if packed else None
        runtime_admission = dict(mode='packed-expert-bank-'+('smoke' if smoke else 'validate'),
            runtime_asset_delta=runtime_evidence['runtime_asset_delta']) if packed else None
        if packed:
            runtime.runtime_assets_unchanged(runtime_admission)
        else:
            for name,item in settings['runtime_elfs'].items():
                guards.require(digest(name)==item['sha256'], 'runtime changed: '+name)
        env = dict(settings['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
        if not smoke:
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
            env.pop('UR_LOG_TRACING',None)
        guards.driver_guard()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
        cursor = next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
        gpu.update(mode=mode,complete=False,passed=False,GPU_executed=True,
            environment_reference='tools/sycl/measurement_environment.json',
            environment_overrides={k:v for k,v in env.items() if settings['environment'].get(k)!=v},
            removed_environment_keys=[k for k in settings['environment'] if k not in env])
        if packed:
            gpu['runtime_asset_delta'] = runtime_admission['runtime_asset_delta']
            gpu['runtime_reference'] = dict(path=str(runtime.COMPONENT.relative_to(P)),
                sha256=digest(runtime.COMPONENT), source_commit=runtime_evidence['source_commit'],
                scope='Existing passed component runtime identity; not qualification of this changed native route.')
        owner = None

        def persist():
            (root/'record.json').write_text(json.dumps(dict(gpu,
                active=bool(owner and owner.active is not None),
                owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner = supervision.Owner(root/('owned-'+mode),persist)
        try:
            if smoke:
                entry,out,err = owner.run('arena-provider-smoke',[provider_fixture['path']],env,P,
                    wall=60,cpu=180,rss_cap=2<<30,file_cap=64<<20,text_cap=64<<20,total_cap=128<<20)
                stdout,stderr = out.read_text(),err.read_text()
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0 and
                    'arena_resident_layer_test private=PASS file_fallback=PASS shared_exclusion=PASS blob_identity=PASS bounds=PASS close_reopen=PASS' in stdout and
                    'file_expert_source_test: PASS' in stdout and not re.search(
                    'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed',stderr),
                    'Arena provider fixture failed: '+(stderr+stdout)[-5000:])
                gpu['provider_checks_passed'] = True
            entry,out,err = owner.run('expert-bank-'+mode,[fixture['path'],'--'+mode],env,P,
                wall=180,cpu=600,rss_cap=8<<30,file_cap=64<<20,text_cap=64<<20,total_cap=128<<20)
            stdout,stderr = out.read_text(),err.read_text()
            guards.require(supervision.closed(entry) and entry.get('exit_code')==0 and
                not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed',stderr),
                'GPU fixture failed: '+(stderr+stdout)[-5000:])
            cases = (check_packed_cases if packed else check_legacy_cases)(stdout,mode)
            gpu['checks'].append(dict(mode=mode,cases=cases,passed=True))
            if packed: runtime.runtime_assets_unchanged(runtime_admission)
            guards.require(all(digest(P/n)==h for n,h in pins.items()), 'GPU-check inputs changed')
            gpu.update(complete=True,passed=True)
        except BaseException as error:
            gpu['error'] = str(error)
        finally:
            gpu['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            gpu['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
            gpu['fresh_faults'] = [s for s in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
            try: gpu['driver_after'] = guards.driver_guard()
            except Exception as error:
                gpu['driver_error'] = str(error);gpu['passed'] = False
            gpu['passed'] = bool(gpu['passed'] and gpu['ownership_closed'] and gpu['boot_unchanged'] and not gpu['fresh_faults'])
            persist();record[field] = gpu
            RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:gpu.get(k) for k in ('mode','complete','passed','ownership_closed','error')}))
        return 0 if gpu['passed'] else 1


def check_legacy_cases(stdout,mode):
    cases = [dict(token.split('=',1) for token in line.split()[1:])
        for line in stdout.splitlines() if line.startswith('expert_bank_case ')]
    expected = ([(4,256,6,2,18,6,48,0),(1,256,6,1,12,0,24,0),(4,256,6,1,5,0,10,1),
        (4,512,6,2,18,6,48,0),(1,512,6,1,12,0,24,0),(4,512,6,1,5,0,10,1)]
        if mode=='smoke' else [(383,256,512,1,642,382,2048,0),
        (384,256,512,2,1282,766,4096,0),(512,256,512,1,513,511,2048,0),
        (384,2662400,512,1,641,383,2048,0),
        (384,512,512,2,1282,766,4096,0),(384,2662400,512,1,641,383,2048,0)])
    source_strides = ([256]*6 if mode=='smoke' else [256,256,256,2662400,256,1382400])
    guards.require(len(cases)==len(expected) and 'expert_bank_fixture PASS' in stdout,
        'incomplete bank fixture cases')
    for case,values,source_stride in zip(cases,expected,source_strides):
        keys=('slots','stride','experts','layers','uploads','hits','consumers','unwind')
        guards.require(tuple(int(case[k]) for k in keys)==values and
            int(case['source_stride'])==source_stride and int(case['status'])==0 and
            int(case['admission_rejections'])==7+2*(source_stride!=int(case['stride'])) and all(
            case[k]=='PASS' for k in ('immutable_slab','guards','private_map','checked_drain')),
            'wrong bank case evidence: '+str(case))
    return cases


def check_packed_cases(stdout,mode):
    cases = [dict(token.split('=',1) for token in line.split()[1:])
        for line in stdout.splitlines() if line.startswith('expert_bank_private_case ')]
    expected = ([(4,512,256,10,3,43,17,120,0), (4,512,256,10,2,10,0,20,2)]
        if mode=='packed-smoke' else [(384,512,256,512,4,2434,1662,8192,0),
        (384,2662400,2662400,512,3,1709,1363,6144,0), (384,512,256,512,2,10,0,20,2)])
    guards.require(len(cases)==len(expected) and 'expert_bank_fixture PASS' in stdout,
        'incomplete packed bank fixture cases')
    keys=('backing_slots','backing_stride','source_stride','experts','layers','uploads','hits','consumers','unwind')
    guards.require('expert_bank_private_enqueue_failure attempts=1 published=0 pending_rejected=PASS checked_drain=PASS' in stdout,
        'missing private enqueue failure/unwind case')
    full_copies = [1,0] if mode=='packed-smoke' else [0,1,0]
    for case,values,full_copy in zip(cases,expected,full_copies):
        guards.require(tuple(int(case[k]) for k in keys)==values and
            int(case['status'])==0 and int(case['admission_rejections'])==11 and
            int(case['full_copy'])==full_copy and all(
            case[k]=='PASS' for k in ('immutable_slab','guards','private_map','checked_drain')),
            'wrong private bank case evidence: '+str(case))
    return cases


def query_limits():
    """Reuse the maintained query-only probe; submit no work or allocations."""
    root = B / 'moe-expert-bank-limits-v2'
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.require(not root.exists(), 'inspect existing device-limit owner before retry')
        raw = subprocess.check_output(['git','show','HEAD:'+str(RESULT.relative_to(P))],cwd=P)
        record = json.loads(raw)
        guards.require(raw == RESULT.read_bytes() and record['complete'] and record['passed'] and
            record['ownership_closed'] and not record['active'], 'commit closed CPU plan first')
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()
        pins = {}
        for name in ('tools/sycl/capability_inventory.cpp','tools/sycl/check_expert_bank.py',
                     'tools/sycl/measurement_environment.json','tools/sycl/measurement_guards.py',
                     'tools/sycl/measurement_owner.py'):
            data = subprocess.check_output(['git','show',revision+':'+name],cwd=P)
            guards.require(data == (P/name).read_bytes(), 'commit exact limit-query input: '+name)
            pins[name] = hashlib.sha256(data).hexdigest()
        settings = json.loads((P/'tools/sycl/measurement_environment.json').read_text())
        compiler = Path(settings['environment']['UR_ADAPTERS_FORCE_LOAD']).parents[1]/'bin/clang++'
        guards.require(digest(compiler) == settings['compiler_sha256'], 'compiler changed')
        for name, identity in settings['runtime_elfs'].items():
            guards.require(digest(name) == identity['sha256'], 'runtime changed: '+name)
        before = guards.driver_guard()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
        cursor = next(s.split(': ',1)[1] for s in journal.splitlines() if s.startswith('-- cursor: '))
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir()
        (root/'cache').mkdir()
        env = dict(settings['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
        binary = root/'device-limits'
        query = dict(source_commit=revision,source_sha256=pins,environment_reference=
            'tools/sycl/measurement_environment.json',compiler=dict(path=str(compiler),sha256=digest(compiler)),
            complete=False,passed=False,ownership_closed=False,driver_before=before,
            allocation_and_kernel_submission=False,
            limits='Device properties only. Capacity is not an allocation, owner, performance or physical-context qualification.')
        if 'device_limits' in record:
            query['previous_query'] = dict(commit=revision,path=str(RESULT.relative_to(P)),
                source_commit=record['device_limits']['source_commit'],passed=record['device_limits']['passed'])
        owner = None

        def persist():
            (root/'record.json').write_text(json.dumps(dict(query,
                active=bool(owner and owner.active is not None),
                owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner = supervision.Owner(root/'owned-limits',persist)
        try:
            for label, command in [('compile-limits',[str(compiler),'-std=c++20','-O2','-fsycl',
                    str(P/'tools/sycl/capability_inventory.cpp'),'-o',str(binary)]),
                    ('query-limits',[str(binary)])]:
                entry,out,err = owner.run(label,command,env,P,wall=180,cpu=600,
                    rss_cap=4<<30,file_cap=16<<20,text_cap=8<<20,total_cap=32<<20)
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                    label+': '+(out.read_text()+err.read_text())[-4000:])
                if label == 'query-limits':
                    properties = {}
                    for line in out.read_text().splitlines():
                        key,sep,value = line.partition('=')
                        if sep and key in ('device','driver','global_mem_size','max_mem_alloc_size'):
                            guards.require(key not in properties, 'duplicate device property')
                            properties[key] = int(value) if key.endswith('_size') else value
                    query['properties'] = properties
                    guards.require(set(properties) == {'device','driver','global_mem_size','max_mem_alloc_size'}
                        and properties['device'] in ('Intel Arc B570 Graphics','Intel(R) Arc(TM) B570 Graphics')
                        and 0 < properties['max_mem_alloc_size'] <= properties['global_mem_size'],
                        'unexpected or incomplete device limits')
            guards.require(all(digest(P/n)==h for n,h in pins.items()), 'limit-query inputs changed')
            query.update(complete=True,passed=True,binary=dict(path=str(binary),
                bytes=binary.stat().st_size,sha256=digest(binary)))
        except BaseException as error:
            query['error'] = str(error)
        finally:
            query['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            query['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
            query['fresh_faults'] = [s for s in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
            try:
                query['driver_after'] = guards.driver_guard()
            except Exception as error:
                query['driver_error'] = str(error)
                query['passed'] = False
            query['passed'] = bool(query['passed'] and query['ownership_closed'] and
                query['boot_unchanged'] and not query['fresh_faults'])
            persist()
            record['device_limits'] = query
            RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:query.get(k) for k in ('complete','passed','ownership_closed','properties','error')}))
        return 0 if query['passed'] else 1


def copy_run_startup_errors(text,pid,traced):
    """Only the verified before-main canary may return an API error in trace."""
    import ur_validation as validation
    proof=validation.preload_evidence(text,pid)
    begin=text.index('strata UR validation: pid='+str(pid)+' phase=begin ')
    post=text.index('strata UR validation API: pid='+str(pid)+' operation=postinit result=46')
    end=text.index('strata UR validation: pid='+str(pid)+' phase=before-main ')
    guards.require(begin<post<end,'startup API proof order differs')
    pattern=r'\s*<--- urPlatformGetApiVersion\(\.hPlatform = nullptr, \.pVersion = 0x[0-9a-fA-F]+ \(-1\.65535\)\) -> UR_RESULT_ERROR_INVALID_NULL_HANDLE;'
    count=0
    for match in re.finditer(r'^.*(?:UR_RESULT_ERROR_|ZE_RESULT_ERROR_).*$',text,re.M):
        guards.require(traced and begin<match.start()<post and re.fullmatch(pattern,match[0]),
            'unexpected API error outside the verified startup canary: '+match[0])
        count+=1
    guards.require(count==int(traced),'startup canary trace count differs')
    return proof,count


def copy_run_startup_controls():
    pid=123
    begin='strata UR validation: pid=123 phase=begin loader=/pinned/loader.so\n'
    error='   <--- urPlatformGetApiVersion(.hPlatform = nullptr, .pVersion = 0x1234 (-1.65535)) -> UR_RESULT_ERROR_INVALID_NULL_HANDLE;\n'
    post='strata UR validation API: pid=123 operation=postinit result=46\n'
    end='strata UR validation: pid=123 phase=before-main threads_before=1 preinit=37 config_create=0 layers_query=0 enable=0 init=0 config_release=0 checked=46 untouched=1 retained_ref=1\n'
    valid=begin+error+post+end
    guards.require(copy_run_startup_errors(valid,pid,True)[1]==1 and
        copy_run_startup_errors(begin+post+end,pid,False)[1]==0,'valid startup controls failed')
    cases=[(valid+error,pid,True),(begin+post+end+error,pid,True),
        (valid.replace('urPlatformGetApiVersion(','urUSMDeviceAlloc('),pid,True),
        (valid.replace('UR_RESULT_ERROR_INVALID_NULL_HANDLE','UR_RESULT_ERROR_INVALID_ARGUMENT'),pid,True),
        (error+valid,pid,True),(begin+error+error+post+end,pid,True),(valid,124,True),
        (valid.replace('untouched=1','untouched=0'),pid,True),(valid,pid,False),
        (begin+post+end,pid,True)]
    for text,actual_pid,traced in cases:
        refused=False
        try:copy_run_startup_errors(text,actual_pid,traced)
        except (RuntimeError,ValueError):refused=True
        guards.require(refused,'unexpected/misplaced/missing startup error accepted')
    return dict(valid_traced=1,valid_quiet=1,negative_cases=len(cases),GPU_executed=False)










def copy_size_commands(permanent,uploads,cap):
    return uploads-permanent+(permanent+cap-1)//cap


def copy_size_cases(text,cap,smoke=False):
    cases=[dict(re.findall(r'(\w+)=([^ ]+)',s)) for s in text.splitlines()
        if s.startswith('expert_bank_private_case ')]
    # cap,backing slots,stride,source stride,experts,layers,permanent,unwind
    shapes=([(c,s,256,256,e,1,p,0) for c in (1,4,16,64,512)
        for s,e,p in [(4,10,3),(384,512,383),(512,512,512)]]+
        [(4,1,256,256,10,1,0,0),(4,8,256,256,10,2,7,1)] if smoke else
        [(cap,4,256,256,10,1,3,0),(cap,384,2662400,2662400,512,1,383,0),
         (cap,384,2662400,1510400,512,1,512,0)])
    guards.require(len(cases)==len(shapes),'incomplete bounded-copy shapes')
    for case,(c,slots,stride,source,experts,layers,permanent,unwind) in zip(cases,shapes):
        uploads,hits,consumers=(5,0,10) if unwind else (2*experts-permanent,permanent,4*experts)
        commands=2 if unwind else copy_size_commands(permanent,uploads,c)
        expected=dict(backing_slots=slots,backing_stride=stride,source_stride=source,experts=experts,
            layers=layers,uploads=uploads*layers,hits=hits*layers,consumers=consumers*layers,
            unwind=layers if unwind else 0,admission_rejections=11,status=0,full_copy=1,
            staged=0,custom_queue=1,copy_runs=int(c>1),copy_commands=commands*layers)
        guards.require(all(int(case[k])==v for k,v in expected.items()) and all(case[k]=='PASS'
            for k in ('immutable_slab','guards','private_map','checked_drain')),'bounded-copy case differs')
    return cases


def copy_size_profile_rows(text,cap):
    import math
    def fields(line):
        tokens=[s.split('=',1) for s in line.split()[1:]]
        guards.require(all(len(s)==2 for s in tokens) and len({s[0] for s in tokens})==len(tokens),
            'malformed/duplicate profile fields')
        return dict(tokens)
    timings=[fields(line) for line in text.splitlines()
        if line.startswith('expert_bank_copy_timing candidate=')]
    profiles=[fields(line) for line in text.splitlines()
        if line.startswith('expert_bank_copy_host_profile ')]
    guards.require(len(timings)==len(profiles)==2,'two timing/accounting shapes required')
    result=[]
    phases=('other_submit','upload_wrapper','reader_submit','final_drain')
    for t,p,(payload,uploads,permanent) in zip(timings,profiles,[(2662400,641,383),(1510400,512,512)]):
        integers=dict(candidate=int(cap>1),payload=payload,logical_uploads=uploads,
            requested_H2D_bytes=payload*uploads,commands=copy_size_commands(permanent,uploads,cap),
            preparation_timed=0,guards_timed=0,actual_copy_read_drain_timed=1,model_qualified=0)
        clocks=('wall_ms','queue_ms','thread_ms','empty_ms')
        guards.require(set(t)==set(integers)|set(clocks)|{'eligible'} and
            all(int(t[k])==v for k,v in integers.items()),'wrong copy timing bytes/dispatch')
        timing={k:(float(v) if k in clocks else int(v)) for k,v in t.items()}
        flags=dict(cap=cap,payload=payload,clock_switches=3073,intervals_disjoint=1,
            coverage_conserved=1,exceptions=0,profiling_only=1)
        floats=[phase+'_'+clock+'_ms' for phase in phases for clock in ('wall','thread')]+[
            'clock_change_ns','timer_estimate_ms']
        guards.require(set(p)==set(flags)|set(floats) and all(int(p[k])==v for k,v in flags.items()),
            'wrong host accounting fields/counts')
        profile={k:(float(v) if k in floats else int(v)) for k,v in p.items()}
        guards.require(all(math.isfinite(timing[k]) and timing[k]>=0 for k in clocks) and
            all(math.isfinite(profile[k]) and profile[k]>=0 for k in floats),'invalid timing/accounting number')
        guards.require(abs(sum(profile[phase+'_wall_ms'] for phase in phases)-timing['wall_ms'])<1e-5 and
            abs(sum(profile[phase+'_thread_ms'] for phase in phases)-timing['thread_ms'])<1e-5,
            'disjoint host accounting does not conserve whole interval')
        eligible=timing['wall_ms']>=100 and timing['queue_ms']>=100 and timing['empty_ms']<=.05*timing['wall_ms']
        guards.require(timing['eligible'] in (0,1) and bool(timing['eligible'])==eligible and
            timing['queue_ms']<=timing['wall_ms']+5 and timing['thread_ms']<=timing['wall_ms']+5,
            'eligibility/clock mismatch')
        guards.require(abs(profile['timer_estimate_ms']-profile['clock_change_ns']*3073/1e6)<1e-8,
            'timer estimate/count mismatch')
        result.append(dict(timing=timing,host=profile,eligible=bool(eligible and
            profile['timer_estimate_ms']<=.01*timing['wall_ms'])))
    return result


def copy_size_controls():
    def example(cap):
        lines=[]
        for payload,uploads,permanent in [(2662400,641,383),(1510400,512,512)]:
            t=dict(candidate=int(cap>1),payload=payload,logical_uploads=uploads,
                requested_H2D_bytes=payload*uploads,commands=copy_size_commands(permanent,uploads,cap),
                wall_ms=200,queue_ms=180,thread_ms=100,empty_ms=1,eligible=1,
                preparation_timed=0,guards_timed=0,actual_copy_read_drain_timed=1,model_qualified=0)
            p=dict(cap=cap,payload=payload,other_submit_wall_ms=1,other_submit_thread_ms=1,
                upload_wrapper_wall_ms=98,upload_wrapper_thread_ms=48,reader_submit_wall_ms=99,
                reader_submit_thread_ms=50,final_drain_wall_ms=2,final_drain_thread_ms=1,
                clock_change_ns=100,clock_switches=3073,timer_estimate_ms=.3073,
                intervals_disjoint=1,coverage_conserved=1,exceptions=0,profiling_only=1)
            lines+=['expert_bank_copy_timing '+' '.join(f'{k}={v}' for k,v in t.items()),
                'expert_bank_copy_host_profile '+' '.join(f'{k}={v}' for k,v in p.items())]
        return '\n'.join(lines)
    for cap in (1,4,16,64,512):guards.require(all(v['eligible'] for v in copy_size_profile_rows(example(cap),cap)),
        'valid profile control failed')
    valid=example(4)
    negatives=[valid.replace('upload_wrapper_wall_ms=98','upload_wrapper_wall_ms=99'),
        valid.replace('reader_submit_thread_ms=50','reader_submit_thread_ms=nan'),
        valid.replace('clock_switches=3073','clock_switches=3072'),
        valid.replace('eligible=1','eligible=0'),valid.replace('exceptions=0','exceptions=1'),
        valid.replace('commands=354','commands=259'),valid.replace('cap=4','cap=4 cap=4')]
    for text in negatives:
        refused=False
        try:copy_size_profile_rows(text,4)
        except (RuntimeError,ValueError):refused=True
        guards.require(refused,'invalid accounting/dispatch control admitted')
    return dict(valid_caps=5,negative_cases=len(negatives),GPU_executed=False)


def copy_size_check(action):
    import build_expert_copy_runs as build
    import ur_validation as validation
    root=B/'expert-copy-size-profile-v1';key='expert_copy_size_profile_v1'
    order=[dict(mode='cpu'),dict(mode='smoke')]+[dict(mode='profile',cap=cap,round=r)
        for r,caps in enumerate([(1,4,16,64,512),(512,64,16,4,1),(16,1,512,4,64)]) for cap in caps]
    def save(row):
        (root/'record.json').write_text(json.dumps(row,indent=2)+'\n')
        public=json.loads(build.RESULT.read_text());public[key]=row
        build.RESULT.write_text(json.dumps(public,indent=2)+'\n')
    def current(row,committed=True):
        if committed:guards.require(validation.committed(build.RESULT,(key,))==row,'commit profile stage before run')
        native=json.loads((build.ROOT/'record.json').read_text());build.context().current(native)
        guards.require(native['stage']=='built' and native['passed'] and native['complete'] and
            native['ownership_closed'] and not native['active'] and native['binaries']==row['binaries']
            and native['native_source_commit']==row['native_source_commit'],'same fresh closed fixture build required')
        p=Path(row['runtime_manifest']['path'])
        guards.require(digest(p)==row['runtime_manifest']['sha256'] and all(digest(p)==h
            for p,h in json.loads(p.read_text()).items()),'runtime provider changed')
        guards.require(row['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and row['driver']==guards.driver_guard(),'boot/driver changed')
        return native
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB);guards.no_model_owners([])
        if action=='prepare':
            guards.require(not root.exists() and key not in json.loads(build.RESULT.read_text()),'fresh profile plan only')
            native=json.loads((build.ROOT/'record.json').read_text());build.context().current(native)
            controls=copy_size_controls();startup=copy_run_startup_controls()
            prior=validation.committed(build.RESULT,('expert_copy_runs_timing_v1',))
            guards.require(prior['complete'] and prior['ownership_closed'] and not prior['active']
                and not prior['performance_qualified'] and all(not d['model_followup_permitted']
                    for d in prior['decisions']),'preserve original failed performance gates')
            env=dict(native['recipe']['environment'])
            guards.require({k:env[k] for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys',
                'EnableDirectSubmission')}==dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',
                EnableDirectSubmission='0'),'preserve qualified safety settings')
            row=dict(stage='ready',native_source_commit=native['native_source_commit'],binaries=native['binaries'],
                native_validation_probe=native['native_validation_probe'],runtime_manifest=prior['runtime_manifest'],
                environment=env,boot=prior['boot'],driver=prior['driver'],order=order,results=[],
                cpu_reference=json.loads((build.cpu.ROOT/'record.json').read_text())['checks'],
                controls=controls,startup_controls=startup,active=False,ownership_closed=True,passed=False,complete=False,
                performance_qualified=False,model_followup_permitted=False,
                budgets=dict(wall=240,cpu=1200,rss_cap=12<<30,text_cap=64<<20,file_cap=64<<20,total_cap=256<<20),
                plan='Commit fresh native/source/recipe and this plan before execution;commit each closed stage '
                    'before next. CPU:repeat513/8/22planner fixture. Smoke:trace all5caps at small/scratch/full-fit '
                    'shapes,one-slot and cancellation,failed-enqueue/refusal controls. Then15quiet independent '
                    'profile processes,three per cap in pinned rotated/reversed orders;each process warms small '
                    'helper/marker then ordinary384and packed-full-fit512. Same requested bytes/readers/order '
                    'within each geometry. Require exact caps/cases/copy counts,source/map/guards/drain checks,'
                    'normal0,quiet validation,closed owners,no fresh fault,frozen source/runtime. Profiles need '
                    '100ms wall/queue,empty<=5%wall,disjoint conserved wall/thread accounting,zero exceptions '
                    'and calibrated clock-change estimate<=1%wall. Stop first failure;no retry. Retain individual '
                    'samples and per-cap ranges/medians. This is attribution/size-screening only;no performance '
                    'gate permits a model trial. Any subsequent clean speed comparison needs a new committed plan.',
                limits='Host profile is current-thread CPU only. Upload wrapper may include internal waits;'
                    'do not call it CPU model math. Queue interval includes copy/read/host gaps,not pure DMA. '
                    'Calibration estimates clock bookkeeping,not full observer overhead. No clean gain,physical '
                    'bus traffic,saturation,model PP/TG,decode,physical262144/RESTORE or teardown qualification.',
                retention='One temporary standalone fixture;coherent engine retained,no new engine generation. '
                    'Keep compact per-process accounting/size/counter values and source/recipe identity;retire '
                    'normal traces/cache after screen and before unrelated work. Review fixture need at clean '
                    'comparison closure. Legacy engine reader/provenance holds remain before another engine build.')
            root.mkdir(mode=0o700);current(row,False);save(row)
            print('Prepared CPU/traced-size/15process profile plan;commit before run.',flush=True);return
        row=json.loads((root/'record.json').read_text());current(row)
        index=len(row['results']);guards.require(row['stage'] in ('ready','running') and index<len(order) and
            all(v['passed'] for v in row['results']),'inspect existing owned profile;no retry')
        item=order[index];mode=item['mode'];case=root/(f'{index:02d}-'+mode)
        guards.require(not case.exists(),'existing profile handle must be inspected');case.mkdir(mode=0o700)
        for name in ('tmp','cache'):(case/name).mkdir(mode=0o700)
        env=dict(row['environment'],TMPDIR=str(case/'tmp'),XDG_CACHE_HOME=str(case/'cache'),
            NEO_CACHE_DIR=str(case/'cache/neo'),SYCL_CACHE_DIR=str(case/'cache/sycl'))
        gpu=mode!='cpu';probe=row['native_validation_probe']
        if gpu:
            env.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(case/'validation.log'))
            if mode=='profile':env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
        else:env.pop('LD_PRELOAD',None)
        cursor=next(s.split(': ',1)[1] for s in subprocess.check_output(
            ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
            if s.startswith('-- cursor: '))
        outcome=dict(**item,passed=False,complete=False,host_probe_accepted=False,GPU_executed=gpu)
        scope=None
        def persist():
            (case/'execution.json').write_text(json.dumps(dict(outcome,active=bool(scope and scope.active),
                commands=scope.commands if scope else []),indent=2)+'\n')
        scope=supervision.Owner(case/'owned',persist)
        try:
            argv=([row['binaries']['prefill_expert_bank_plan_test']['path']] if not gpu else
                [row['binaries']['sycl_expert_bank_test']['path']]+(['--copy-runs-size-smoke'] if mode=='smoke'
                    else ['--copy-runs-size-profile',str(item['cap'])]))
            event,out,err=scope.run(mode,argv,env,P,**row['budgets']);text,errors=out.read_text(),err.read_text()
            outcome.update(native_pid=event['pid'],native_exit_code=event['exit_code'])
            guards.require(supervision.closed(event) and event['exit_code']==0 and not re.search(
                'terminate called|cleanup failed|teardown uncertain',errors),'copy-size child failed')
            if not gpu:
                match=re.search(r'expert_bank_upload_runs configurations=513 windows_each=8 logical_uploads=(\d+) '
                    r'commands=(\d+) runs=(\d+) run_experts=(\d+) negative_checks=22 PASS',text)
                guards.require(not errors and match and tuple(map(int,match.groups()))==tuple(
                    row['cpu_reference'][k] for k in ('logical_uploads','copy_commands','runs','run_experts'))
                    and 'expert_bank_plan PASS' in text,'CPU gate incomplete')
            else:
                outcome['validation'],outcome['startup_errors']=copy_run_startup_errors(errors,event['pid'],mode=='smoke')
                guards.require(outcome['validation']['loader']==probe['loader']['path'],'actual loader differs')
                log=case/'validation.log';guards.require(not log.exists() or not log.stat().st_size,'validation diagnostic')
                cases=copy_size_cases(text,item.get('cap'),mode=='smoke')
                outcome['case_checks']=dict(count=len(cases),exact_geometry_counters=True,
                    immutable_source_guards_private_map_and_checked_drain=True)
                if mode=='smoke':
                    guards.require('urEnqueueUSMMemcpy' in errors and 'urEnqueueKernelLaunch' in errors and
                        'expert_bank_copy_size_smoke PASS model=UNRUN' in text and
                        'expert_bank_run_enqueue_failure modes=3 published=0 no_retry=PASS checked_drain=PASS' in text and
                        'expert_bank_run_admissions rejections=10 enqueues=0 mapping_unchanged=PASS' in text,
                        'missing traced size/enqueue/refusal controls')
                else:
                    guards.require(f"expert_bank_copy_size cap={item['cap']} profiling=1" in text and
                        'expert_bank_copy_size PASS model=UNRUN' in text,'actual profile mode differs')
                    outcome['samples']=copy_size_profile_rows(text,item['cap'])
                    guards.require(all(v['eligible'] for v in outcome['samples']),'profile timing/accounting ineligible')
            current(row);outcome.update(complete=True,host_probe_accepted=True)
        except BaseException as error:outcome['error']=repr(error)
        finally:
            validation.finish(outcome,scope,cursor,row['boot'])
            outcome['passed']=bool(outcome['passed'] and outcome['complete'] and outcome.get('driver_after')==row['driver'])
            persist();row['results'].append(outcome);row.update(stage='running' if outcome['passed'] else 'failed',
                active=bool(scope.active),ownership_closed=outcome['ownership_closed'])
            if outcome['passed'] and len(row['results'])==len(order):row.update(stage='closed',passed=True,complete=True)
            save(row)
        print(json.dumps({k:outcome.get(k) for k in ('mode','cap','passed','complete','ownership_closed','samples','error')}),flush=True)
        guards.require(outcome['passed'],'copy-size profile failed')


def main():
    if sys.argv[1:] in (['copy-size-prepare'],['copy-size-next']):
        copy_size_check(sys.argv[1].rsplit('-',1)[1]);return 0
    if sys.argv[1:] in (['smoke'],['validate'],['packed-smoke'],['packed-validate']):
        return gpu_check(sys.argv[1])
    if sys.argv[1:] == ['limits']:
        return query_limits()
    guards.require(sys.argv[1:] == ['plan'], 'use check_expert_bank.py plan')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.require(not ROOT.exists(), 'inspect prior expert-bank job; do not restart')
        prior = None
        device_limits = None
        if RESULT.exists():
            raw = subprocess.check_output(['git','show','HEAD:'+str(RESULT.relative_to(P))],cwd=P)
            old = json.loads(raw)
            guards.require(raw == RESULT.read_bytes() and old['complete'] and old['ownership_closed'] and
                not old['active'], 'commit the closed preceding bank plan before replacement')
            prior = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
                path=str(RESULT.relative_to(P)),source_commit=old['source_commit'],passed=old['passed'])
            device_limits = old.get('device_limits')
        revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()
        pins = {}
        for name in ('include/strata/prefill/expert_bank_plan.hpp',
                     'include/strata/core/expert_layer_view.hpp',
                     'include/strata/prefill/layer_major_plan.hpp', 'tests/expert_bank_plan.cpp',
                     'tools/sycl/check_expert_bank.py', 'tools/sycl/measurement_guards.py',
                     'tools/sycl/measurement_owner.py'):
            raw = subprocess.check_output(['git','show',revision + ':' + name],cwd=P)
            guards.require(raw == (P/name).read_bytes(), 'commit exact bank plan inputs first: ' + name)
            pins[name] = hashlib.sha256(raw).hexdigest()
        compiler = Path('/usr/bin/c++')
        env = dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        version = subprocess.check_output([str(compiler),'--version'],env=env,text=True).splitlines()[0]
        ROOT.mkdir(mode=0o700)
        binary = ROOT/'plan-test'
        steps = [('compile-plan',[str(compiler),'-O2','-std=c++20','-Wall','-Wextra','-Werror',
                   '-I'+str(P/'include'),str(P/'tests/expert_bank_plan.cpp'),'-o',str(binary)]),
                 ('plan',[str(binary)])]
        record = dict(mode='plan',source_commit=revision,source_sha256=pins,
            compiler=dict(path=str(compiler),version=version,sha256=digest(compiler)),
            complete=False,passed=False,active=False,GPU_executed=False,checks=[],
            limits='CPU capacity/state planning only. No owner/queue/DMA completion, GPU staging, model speed, physical262144 or RESTORE qualification.')
        if prior:
            record['previous_result'] = prior
        if device_limits:
            record['device_limits'] = device_limits
        owner = None

        def persist():
            (ROOT/'record.json').write_text(json.dumps(dict(record,
                active=bool(owner and owner.active is not None),
                owned_commands=owner.commands if owner else []),indent=2)+'\n')

        owner = supervision.Owner(ROOT/'owned-plan',persist)
        try:
            for label,command in steps:
                entry,out,err = owner.run(label,command,env,P,wall=120,cpu=240,
                    rss_cap=2<<30,file_cap=16<<20,text_cap=16<<20,total_cap=32<<20)
                guards.require(supervision.closed(entry) and entry.get('exit_code')==0,
                    label+': '+(out.read_text()+err.read_text())[-4000:])
                if label=='plan':
                    text=out.read_text()
                    guards.require('fallback262144=original exact_boundary=PASS' in text and
                        'resident_layer_view valid_cases=4 rejected_extents=10 PASS' in text and
                        'short_tail_fallback_cases=3' in text and
                        'single_alloc_limit_cases=4' in text and
                        'uploads=9 hits=3 negative_checks=18 PASS' in text and
                        'expert_bank_packed formats=7 layers_each=2 full_fit512=PASS boundary=PASS negative_checks=19 PASS' in text and
                        'expert_bank_plan PASS GPU=UNRUN' in text,'incomplete expert-bank coverage')
                    record['checks']=[dict(case='admission_and_tickets',passed=True,
                        result_lines=text.splitlines())]
            guards.require(all(digest(P/n)==h for n,h in pins.items()),'bank plan inputs changed')
            record.update(complete=True,passed=True,binary=dict(path=str(binary),
                bytes=binary.stat().st_size,sha256=digest(binary)))
        except BaseException as error:
            record['error']=str(error)
        finally:
            record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
            record['passed']=bool(record['passed'] and record['ownership_closed'])
            persist()
            RESULT.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
