#!/usr/bin/env python3
"""Exercise all four new native gates with one bounded direct-engine request.

Main-authored; repository LICENSE. Diagnostic only,not a benchmark:1024 fixed
valid tokens,128 outputs and an explicit rare EOS ID. No text-quality claim.
"""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

sys.dont_write_bytecode = True
import check_mtp_width_model as width
import cpu_capacity as assets
import engine_once
import profile_decode_gates as admission
import run_q6_columns as validation
import runtime_probe
import ur_validation

P, B, RESULT, base = admission.P, admission.B, admission.RESULT, admission.base
ROOT = B / 'decode-gate-entry-v3'
KEY = 'decode_gate_model_entry_v3'
SELF = Path(__file__).resolve()
require, digest, write = base.require, base.digest, base.write
IDS = [248045] + [13] * 1023
ROUNDS = [0,8,16,24]
READER = P / 'bench/sycl/client_resources.py'
sys.path.insert(0,str(READER.parent))
import client_resources


def sources(revision):
    pins = base.controls(revision,controller=SELF)
    raw = base.basis.build.git(P,'show',revision+':'+str(READER.relative_to(P)))
    require(READER.read_bytes() == raw, 'commit exact DRM reader source')
    pins[str(READER.relative_to(P))] = hashlib.sha256(raw).hexdigest()
    return pins


def startup_info(line):
    require(line.startswith('INFO '), 'native INFO marker missing')
    info = {}
    for item in line.split()[1:]:
        require(item.count('=') == 1, 'native INFO field syntax')
        key,value = item.split('=',1)
        require(key and key not in info, 'native INFO duplicate key')
        info[key] = value
    # Full residency is mode0. Only streamed mode1 prints a positive limit.
    expected = dict(context='131072',kv='int8',kv_resident='0',expert_slots='1256',
        spec='4',mtp_max='4',pool_workers='5',pcie_frac='0.00',spec_min_p='0.00')
    require(all(info.get(k) == v for k,v in expected.items()), 'native INFO configuration differs')
    require(info.get('vram_free_mib','').isdecimal(), 'invalid reported free-memory quantity')
    return {k:info[k] for k in (*expected,'vram_free_mib')}


class RequestProtocol:
    """Selected serving protocol,including ordered prompt progress notifications."""
    def __init__(self):
        self.resumed = self.reused = False
        self.position = self.progress_count = 0
        self.tokens,self.done = [],None

    def feed(self,line):
        require(self.done is None, 'protocol after DONE')
        fields = line.split()
        require(bool(fields), 'empty native protocol')
        marker = fields[0]
        if marker == 'RESUME':
            require(line == 'RESUME 0' and not self.resumed, 'unexpected or duplicate RESUME')
            self.resumed = True
        elif marker == 'PP':
            require(self.resumed and not self.reused and len(fields) == 5 and
                    fields[1].isdecimal() and fields[2] == '1024' and
                    self.position < int(fields[1]) <= 1023, 'invalid prompt progress/order')
            require(all(math.isfinite(float(v)) and float(v) >= 0 for v in fields[3:]),
                    'invalid prompt progress clock/rate')
            self.position = int(fields[1]); self.progress_count += 1
        elif marker == 'REUSED':
            require(line == 'REUSED 0' and self.resumed and not self.reused and self.position == 1023,
                    'incomplete prompt or unexpected REUSED')
            self.reused = True
        elif marker == 'T':
            require(self.reused and len(fields) == 2 and fields[1].isdecimal() and
                    0 <= int(fields[1]) < 248320 and len(self.tokens) < 128, 'output order/token bounds')
            self.tokens.append(int(fields[1]))
        elif marker == 'DONE':
            require(self.reused and len(self.tokens) == 128 and len(fields) == 15 and
                    fields[1:3] == ['128','1024'] and fields[5] == 'length' and
                    fields[8] == '0' and fields[14] == '1024', 'incomplete native DONE extent')
            require(all(v.isdecimal() for v in fields[6:13]) and int(fields[6]) <= int(fields[7]) and
                    all(math.isfinite(float(fields[i])) and float(fields[i]) >= 0 for i in (3,4,13)),
                    'invalid native DONE counters/clocks')
            self.done = fields
        else:
            raise RuntimeError('unexpected native protocol: '+line[:200])


def controls():
    old = base.committed(RESULT)['decode_gate_model_entry_v1']
    require(old['stage'] == 'failed' and old['diagnosis']['physical_closure']['confirmed'] and
            old['outcome']['gates'] == [] and old['outcome']['fresh_faults'] == [],
            'preserve and physically settle failed V1 first')
    settled = old['diagnosis']['physical_closure']
    now = base.supervision.identity(settled['original_pid'])
    require(now is None or now['start_ticks'] != settled['original_start_ticks'],
            'original V1 native identity is still live')
    facts = dict(old['diagnosis']['initial_info'],pcie_frac='0.00',spec_min_p='0.00')
    line = 'INFO '+' '.join(k+'='+v for k,v in facts.items())
    startup_info(line)
    variants = [line.replace('kv_resident=0','kv_resident=131072'),
        line.replace('context=131072','context=1024'),line.replace('kv=int8','kv=fp16'),
        line.replace('expert_slots=1256','expert_slots=1255'),line.replace('spec=4','spec=8'),
        line.replace('mtp_max=4','mtp_max=2'),line.replace('pool_workers=5','pool_workers=6'),
        line.replace('vram_free_mib=0','vram_free_mib=-1'),line+' kv_resident=0',
        line.replace('kv_resident=0','')]
    for value in variants:
        try: startup_info(value)
        except RuntimeError: continue
        raise RuntimeError('corrupt INFO control accepted')
    failed = base.committed(RESULT)['decode_gate_model_entry_v2']
    require(failed['stage'] == 'failed' and failed['diagnosis']['physical_closure']['confirmed'] and
            failed['outcome']['gates'] == [] and not failed['outcome']['fresh_faults'],
            'preserve and physically settle failed V2 first')
    closed = failed['diagnosis']['physical_closure']
    now = base.supervision.identity(closed['original_pid'])
    require(now is None or now['start_ticks'] != closed['original_start_ticks'], 'V2 native still live')
    done = 'DONE 128 1024 1.0 2.0 length 12 24 0 1 2 3 0 0.0 1024'
    good = ['RESUME 0','PP 512 1024 0 0.0','PP 1023 1024 1 1023000.0','REUSED 0']+['T 13']*128+[done]
    reader = RequestProtocol()
    for value in good: reader.feed(value)
    require(reader.done is not None and reader.progress_count == 2 and len(reader.tokens) == 128,
            'complete native protocol fixture differs')
    corrupt = [good[1:],['RESUME 1'],good[:1]+good,good[:1]+['PP 1024 1024 1 1'],
        good[:1]+['PP 1023 1023 1 1'],good[:1]+['PP 1 1024 nan 1'],
        good[:1]+['PP 1 1024 1 -1'],good[:2]+[good[1]],
        good[:2]+['REUSED 0'],good[:4]+['PP 1023 1024 2 1'],
        ['T 13'],good[:4]+['REUSED 0'],good[:4]+['T 248320'],good[:4]+['T -1'],
        good[:-1]+['T 13'],good[:4]+[done],good[:-1]+[done.replace('128','127',1)],
        good[:-1]+[done.replace('length','cancel')],good[:-1]+[done.replace('1024','1023')],
        good[:-1]+[done.replace('12 24','25 24')],good[:-1]+[done.replace('1.0','nan')],
        good+['T 13'],good+['RESUME 0'],['ERR test'],['unknown']]
    for lines in corrupt:
        try:
            reader = RequestProtocol()
            for value in lines: reader.feed(value)
        except (RuntimeError,ValueError): continue
        raise RuntimeError('corrupt serving protocol control accepted')
    return dict(passed=True,actual_failed_INFO_now_accepted=True,negative=len(variants),
        protocol=dict(passed=True,ordered_progress_and128_outputs=True,negative=len(corrupt)),
        GPU_executed=False,scope='Source-derived INFO and request protocol only;not native execution or resource counters.')


def compact(row):
    value = {n:v for n,v in row.items() if n not in ('sources','commands')}
    value['source_inputs'] = dict(files=len(row['sources']),sha256=width.signature(row['sources']))
    return value


def save(row):
    write(ROOT / 'record.json',row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT,data)


def configuration():
    cohort = json.loads((width.SPEED_ROOT / 'record.json').read_text())
    old_plan = cohort['plans'][0]
    environment = width.generation_context(0).environment(cohort,old_plan)
    require(width.signature(environment) == old_plan['environment_sha256'], 'recover exact closed speed environment')
    environment = {n:v.replace(old_plan['root'],str(ROOT)) for n,v in environment.items()}
    environment.update(STRATA_SYCL_DECODE_PROFILE_DIR=str(ROOT / 'gates'),
        STRATA_SYCL_DECODE_PROFILE_REQUEST='1',STRATA_TRACE='1')
    config = copy.deepcopy(cohort['configuration'])
    config['args'] += ['--eos-ids','248319']
    config['log'] = str(ROOT / 'owned/native.stderr')
    return config,environment,cohort['model_assets']


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh single native entry only')
    info_controls = controls()
    native,device,providers,collector = admission.native()
    config,environment,expected = configuration()
    model_assets = assets.model_pins(config,expected,full=True)
    revision = base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    pins = sources(revision)
    ROOT.mkdir(mode=0o700)
    for n in ('tmp','cache','gates'): (ROOT / n).mkdir(mode=0o700)
    row = dict(stage='prepared',source_commit=revision,sources=pins,INFO_controls=info_controls,
        native_source_commit=native['source_commit'],binary=native['binaries']['strata'],
        build_record=dict(path=str(admission.build.ROOT / 'record.json'),sha256=digest(admission.build.ROOT / 'record.json')),
        native_validation_probe=device['native_validation_probe'],runtime_manifest=device['runtime_manifest'],
        boot=device['boot'],driver=device['driver'],configuration=config,environment=environment,
        controller_python=dict(path=str(Path(sys.executable).resolve()),sha256=digest(sys.executable),version=sys.version),
        model_assets=model_assets,request=dict(input_tokens=1024,output_tokens=128,
            input_sha256=width.signature(IDS),eos_ids=[248319],request_id=1,completed_rounds=ROUNDS),
        budgets=dict(wall=900,cpu=5400,rss_cap=104<<30,text_cap=8<<20,file_cap=16<<20,total_cap=128<<20),
        whole_wall_s=1500,passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,GPU_execution_attempted=False,model_executed=False,
        plan='Preserve failed V1/V2 and their physical closure. V3 handles the selected-source ordered RESUME0,monotonic PP progress to1023of1024,REUSED0,T and DONE protocol;reject malformed/duplicate/out-of-order or trailing output. Commit source,exact effective environment,model inputs and request before launch. One fresh direct serving engine;INFO kv_resident0,READY131072,GEN128 with1024 valid fixed IDs. Same spec4/cap4/half1,131072INT8 context,1256static slots,5workers,8192chunk and qualified safety. Rare EOS248319 is diagnostic extent control,not production sampling. Require exact four gates0/8/16/24,raw-clock/request/sequence identities,checked completion before publication,private atomic acknowledgements,same PID/ELF/environment/providers,128bounded outputs,normalDONE length/QUIT0,quiet strict before-main validation,completed T1/T4 Q6 heads,unchanged final assets/boot/driver,no freshfaults and closed owners. Existing client_resources.snapshot validates PID/start/SID before/after bounded fdinfo/thread reads,deduplicates client IDs per PCI device and keeps counter regressions ineligible. Observe READY,each drained gate and DONE. Optional unavailable proc/DRM data remains explicitly unknown and cannot qualify resource attribution;it does not replace generator correctness. No retry,hash refresh,full API trace or profiler.',
        limits='Short synthetic diagnostic only;not thinking/task quality,clean PP/TG,sustained generation,collector/model-window coverage,observer overhead,physical resource saturation,ordinary provider teardown or physical262144/RESTORE qualification.',
        retention='Keep compact native identity/gate clocks/token hash/counts/live replay conservation and validation/fault outcome. After next collector-window consumer and verified result commit,retire successful full output/JIT;no new engine generation.')
    save(row)
    print('Prepared1024-input/128-output native gate entry;commit before execution.',flush=True)


def current(row,full_models=False):
    require(base.committed(RESULT)[KEY] == compact(row) and
            sources(row['source_commit']) == row['sources'], 'entry plan or source changed')
    native,device,providers,collector = admission.native()
    require(native['binaries']['strata'] == row['binary'] and native['source_commit'] == row['native_source_commit'] and
            digest(row['build_record']['path']) == row['build_record']['sha256'] and
            device['native_validation_probe'] == row['native_validation_probe'] and
            device['runtime_manifest'] == row['runtime_manifest'], 'native/runtime qualification changed')
    config,environment,expected = configuration()
    require(config == row['configuration'] and environment == row['environment'] and expected == row['model_assets'] and
            row['request']['input_sha256'] == width.signature(IDS), 'fixed input/configuration changed')
    require(row['controller_python'] == dict(path=str(Path(sys.executable).resolve()),
            sha256=digest(sys.executable),version=sys.version), 'controller interpreter changed')
    assets.model_pins(config,row['model_assets'],full=full_models)
    return providers


def run():
    row = json.loads((ROOT / 'record.json').read_text()); frozen = copy.deepcopy(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'execution.json').exists(), 'one native entry attempt only')
    outcome = dict(passed=False,complete=False,host_probe_accepted=False,gates=[],resource_observations=[])
    owner = None
    def persist():
        write(ROOT / 'execution.json',dict(outcome,active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    with (ROOT / 'execution.json').open('x') as stream: stream.write('{}\n')
    cursor = None
    started = time.monotonic()
    try:
        cursor = admission.profile.cursor()
        providers = current(frozen)
        target = engine_once.native_environment(row,row['environment'])
        expected = dict(target)
        owner = base.supervision.Owner(ROOT / 'owned',persist)
        def interact(io,entry):
            old_poll = io.poll
            last = None
            target_identity = None
            cycles = client_resources.CycleIntervals()
            def observe(label):
                try:
                    sample = client_resources.snapshot(target_identity,'0000:05:00.0')
                    intervals = cycles.update(sample)
                    outcome['resource_observations'].append(dict(label=label,available=True,
                        begin_monotonic_ns=sample['begin_monotonic_ns'],end_monotonic_ns=sample['end_monotonic_ns'],
                        process={n:sample['process'][n] for n in ('cpu_ticks','minor_faults','major_faults','rss_pages')},
                        thread_cpu_ticks={str(v['tid']):v['cpu_ticks'] for v in sample['threads']},
                        disappeared_threads=sample['disappeared_threads'],clients=sample['clients'],engine_intervals=intervals))
                except (OSError,ValueError,KeyError,client_resources.TargetExited) as error:
                    outcome['resource_observations'].append(dict(label=label,available=False,error=repr(error)))
            def poll():
                nonlocal last
                old_poll()
                path = ROOT / 'gates/event'
                if not path.exists(): return
                info = path.lstat()
                require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and
                        stat.S_IMODE(info.st_mode) == 0o600 and info.st_size <= 192, 'event type/owner/extent')
                event = path.read_text()
                if event == last: return
                fields = event.split()
                require(len(fields) == 5 and fields[0] == 'v1' and
                        all(v.isdecimal() for v in fields[1:]), 'native gate event syntax')
                request,sequence,rounds,raw_ns = map(int,fields[1:])
                index = len(outcome['gates'])
                require(index < 4 and request == 1 and sequence == index+1 and rounds == ROUNDS[index] and
                        raw_ns > (outcome['gates'][-1]['raw_ns'] if index else outcome['begin_raw_ns']) and
                        raw_ns <= time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), 'native gate identity/order/clock')
                proc = Path('/proc',str(entry['pid']))
                require((proc / 'exe').resolve() == Path(row['binary']['path']).resolve(), 'native PID/ELF differs')
                actual = dict(v.split('=',1) for v in (proc / 'environ').read_text().split('\0') if '=' in v)
                admission.profile.same_environment(actual,expected)
                seen = runtime_probe.executable_mappings((proc / 'maps').read_text())
                require(seen <= set(providers) | {row['binary']['path'],row['native_validation_probe']['library']['path']},
                        'unexpected native provider')
                outcome['gates'].append(dict(request=request,sequence=sequence,rounds=rounds,raw_ns=raw_ns))
                observe('rounds-'+str(rounds))
                acknowledgement = ROOT / 'gates/ack'
                require(not acknowledgement.exists(), 'stale gate acknowledgement')
                temporary = ROOT / 'gates/ack.tmp'
                fd = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'w') as stream: stream.write(f'v1 {request} {sequence} {rounds}\n')
                os.replace(temporary,acknowledgement)
                last = event; persist()
            io.poll = poll
            ready = io.line(600)
            if ready.startswith('INFO '):
                outcome['startup_info'] = startup_info(ready)
                ready = io.line(600)
            require(ready == 'READY 131072 stop', 'native ready protocol differs')
            outcome.update(native_pid=entry['pid'],native_ready=True)
            actual_identity = base.supervision.identity(entry['pid'])
            require(actual_identity and actual_identity['sid'] == entry['pid'], 'native identity before resource observation')
            target_identity = dict(pid=entry['pid'],start_ticks=actual_identity['start_ticks'],sid=actual_identity['sid'])
            observe('ready')
            io.send(('GEN 128 '+','.join(map(str,IDS))+'\n').encode())
            protocol = RequestProtocol()
            while protocol.done is None: protocol.feed(io.line(600))
            require(len(outcome['gates']) == 4, 'incomplete native gate extent')
            require(not (ROOT / 'gates/ack').exists(), 'final acknowledgement not consumed')
            outcome.update(output_tokens=128,output_token_sha256=width.signature(protocol.tokens),
                prompt_progress=dict(notifications=protocol.progress_count,reached=protocol.position,total=1024),
                native_pid=entry['pid'])
            observe('done')
            io.send(b'QUIT\n'); io.proc.stdin.close()
            io.exit(90)
        outcome['begin_raw_ns'] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        row.update(active=True,ownership_closed=False,GPU_executed=None,GPU_execution_attempted=True,model_executed=None)
        write(ROOT / 'record.json',row)
        entry,out,err = owner.run('native',[row['binary']['path'],'--serve',*row['configuration']['args']],
            target,P,interaction=interact,**row['budgets'])
        require(base.supervision.closed(entry) and entry['exit_code'] == 0, 'native request did not close normally')
        text = err.read_text()
        proof = validation.validation_log(text,entry['pid'],row['native_validation_probe']['loader']['path'],False)
        snapshot = width.short.snapshots(text,1,1)
        windows = snapshot['windows']
        require({v['tokens'] for v in windows if v['phase'] == 'decode'} == {1,4} and
                sum(v['replay_completed'] for v in windows if v['phase'] == 'decode') >= 24,
                'actual completed T1/T4 window extent differs')
        require(not re.search(r'terminate called|retirement failed|Strata SYCL cleanup:',text), 'native runtime failure')
        outcome.update(host_probe_accepted=True,complete=True,native_exit_code=0,validation=proof,
            Q6=dict(request=1,completed_head_captures=len(snapshot['completed_head_capture_ids']),
                windows=[{n:v[n] for n in ('phase','tokens','capture_id','shapes','replay_started','replay_completed')}
                         for v in windows],independently_selected_routes_passed=True))
        current(frozen,full_models=True)
        require(time.monotonic()-started < row['whole_wall_s'], 'whole model entry deadline')
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner is not None and cursor is not None:
            ur_validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and
            len(outcome['gates']) == 4 and outcome.get('driver_after') == row['driver'] and 'error' not in outcome)
        row.update(stage='closed' if outcome['passed'] else 'failed',passed=outcome['passed'],
            complete=outcome['complete'],active=bool(owner and owner.active),
            ownership_closed=outcome.get('ownership_closed',owner is None),outcome=outcome,
            commands=owner.commands if owner else [],
            GPU_executed=True if outcome['gates'] else None if row['GPU_execution_attempted'] else False,
            model_executed=True if outcome['gates'] else None if row['GPU_execution_attempted'] else False)
        persist(); save(row)
    require(row['passed'],str(outcome))
    print('All four actual native gates and128outputs close normally;no speed or full-context claim.',flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['run'],['controls']), 'use prepare|run|controls')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1] == 'controls': print(json.dumps(controls()))
        else: (prepare if sys.argv[1] == 'prepare' else run)()
