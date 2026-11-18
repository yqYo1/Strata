#!/usr/bin/env python3
"""Correlate host and device events for the retained overlap fixture."""
import copy
import fcntl
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import fullfit_attention_overlap as fixture

P, B, RESULT = fixture.P, fixture.B, fixture.RESULT
ROOT, KEY = B/'fullfit-attention-attribution-v1', 'fullfit_attention_attribution_v1'
READER = 'tools/test_fullfit_attention_attribution.py'
LIMITS = dict(wall=300, cpu=450, rss_cap=8<<30, text_cap=16<<20,
    file_cap=48<<20, total_cap=64<<20)
CLOCK_MARGIN_NS = 1_000_000
ARMS = ('serial_separate', 'deferred_wait')
LAUNCH = ('zeCommandListAppendLaunchKernel', 'zeCommandListAppendLaunchKernelWithArguments')
require, sha, signature = fixture.require, fixture.sha, fixture.signature
write, validation, reader = fixture.write, fixture.validation, fixture.reader
owner = fixture.owner


def source_pins(revision):
    pins = fixture.basis.comparison.build.controls(revision, controller=Path(__file__).resolve())
    require((P/READER).read_bytes() == validation.git('show', f'{revision}:{READER}'),
        'host controls changed')
    pins[READER] = sha(P/READER)
    return pins


def original():
    native, pin = fixture.platform()
    row = json.loads((fixture.ROOT/'record.json').read_text())
    public = validation.committed(RESULT,(fixture.KEY,))
    expected = public.pop('host_submission_followup')
    require(public == fixture.compact(row) and row['stage']=='closed' and row['ownership_closed']
        and not row['active'] and not row['decision']['eligible_for_separate_reader_ownership_test']
        and row['platform']==pin and row['sources']==fixture.pins(row['source_commit'])
        and row['commands']==fixture.compile_commands()
        and row['compile_environment']==fixture.compile_environment(native),
        'closed original comparison and source/recipe required')
    recovered = copy.copy(row)
    link = fixture.COMPILE_LOGS/'owned'/(str(len(row['commands'])-1)+'.stdout')
    fixture.audit_build(recovered,link.read_text(),native)
    require(recovered['actual_inputs']==row['actual_inputs'] and recovered['binary']==row['binary']
        and all(sha(p)==v for group in row['actual_inputs'].values() for p,v in group.items()),
        'retained executable or actual compiler/link inputs differ')
    return row, expected


def case(row):
    env = {k: v.replace(str(fixture.ROOT/'trace'),str(ROOT/'capture'))
           for k,v in fixture.case_environment(row['platform'],'trace').items()}
    command = fixture.case_command(row['platform'],'trace')
    command = command[:1]+['--host-timing','--chrome-call-logging']+command[1:]
    return dict(command=command,environment=env)


def span(event):
    lo = reader.nanoseconds(event['ts'])
    return lo, lo + reader.nanoseconds(event['dur'])


def classify(enter, exit_, copy_end, overlap):
    if overlap > CLOCK_MARGIN_NS:
        return 'device_overlap_observed'
    if enter-copy_end > CLOCK_MARGIN_NS:
        return 'native_append_begins_after_copy'
    if copy_end-exit_ > CLOCK_MARGIN_NS:
        return 'native_append_returns_before_copy_end'
    if copy_end-enter > CLOCK_MARGIN_NS and exit_-copy_end > CLOCK_MARGIN_NS:
        return 'native_append_spans_copy_tail'
    return 'near_clock_margin'


def correlate(events, points, device, pid):
    require(len(points)==len(device) and all((p['arm'],p['round'])==(d['arm'],d['round'])
        for p,d in zip(points,device)), 'device window order differs')
    operations = []
    host, flow = defaultdict(list), defaultdict(list)
    for event in events:
        cat = event.get('cat','')
        phase = event.get('ph')
        if phase == 'X' and cat == 'gpu_op':
            operations.append((span(event),event))
        elif phase == 'X' and cat == 'cpu_op' and event.get('pid') == pid:
            host[(event.get('tid'),event['ts'])].append(event)
        elif phase in ('s','t') and cat.startswith('Flow_H2D_'):
            flow[(cat,str(event.get('id')),phase)].append(event)
    used = set()
    result = []
    for point, observed in zip(points,device):
        if point['arm'] not in ARMS:
            continue
        begin,end = point['raw_start'],point['raw_end']
        measured = dict(copy=[],kernel=[])
        for (low,high),event in operations:
            if not begin <= low <= high <= end:
                continue
            name = event['name']
            if fixture.COPY.fullmatch(name):
                kind, expected = 'copy', ('zeCommandListAppendMemoryCopy',)
            elif 'prefix_batch_impl' in name:
                kind, expected = 'kernel', LAUNCH
            else:
                continue
            ident = str(event.get('args',{}).get('id',''))
            cat = f'Flow_H2D_{ident}_{pid}'
            source = flow[(cat,ident,'s')]
            target = flow[(cat,ident,'t')]
            require(ident and ident not in used and len(source)==len(target)==1
                and source[0]['pid']==pid
                and (target[0]['ts'],target[0]['tid'],target[0]['pid'])==(event['ts'],event['tid'],event['pid']),
                'missing, duplicate or mismatched operation flow')
            endpoints = host[(source[0]['tid'],source[0]['ts'])]
            require(len(endpoints)==1 and endpoints[0]['name'] in expected,
                'host append is ambiguous or has the wrong API')
            enter,exit_ = span(endpoints[0])
            require(begin <= enter <= exit_ <= end and enter <= high,
                'host append lies outside the joined operation window')
            measured[kind].append((enter,exit_,low,high,event['tid'],source[0]['tid']))
            used.add(ident)
        copies,kernels = measured['copy'],measured['kernel']
        require(len(copies)==point['copies'] and len(kernels)==2*point['attention_calls'],
            'joined APIs do not conserve declared copies and QSA operations')
        first = min(kernels,key=lambda k:k[2])
        copy_end = max(c[3] for c in copies)
        last_copy_return = max(c[1] for c in copies)
        label = classify(first[0],first[1],copy_end,observed['overlap_ns'])
        result.append(dict(arm=point['arm'],round=point['round'],classification=label,
            first_qsa_append_enter_minus_copy_end_ns=first[0]-copy_end,
            first_qsa_append_return_minus_copy_end_ns=first[1]-copy_end,
            first_qsa_device_start_minus_copy_end_ns=first[2]-copy_end,
            last_copy_append_return_minus_device_tail_ns=last_copy_return-copy_end,
            copy_host_append_union_ns=reader.union_ns([(c[0],c[1]) for c in copies]),
            copy_first_enter_to_last_return_ns=last_copy_return-min(c[0] for c in copies),
            qsa_append_threads=len({k[5] for k in kernels}),
            copy_append_threads=len({c[5] for c in copies}),
            collector_logical_lanes={k:sorted({x[4] for x in calls}) for k,calls in measured.items()}))
    return result


def decision(windows):
    require(len(windows)==10 and all(
        [w['round'] for w in windows if w['arm']==arm]==list(range(5)) for arm in ARMS),
        'exact serial and deferred windows required')
    serial = [w for w in windows if w['arm']=='serial_separate' and w['round']>0]
    deferred = [w for w in windows if w['arm']=='deferred_wait' and w['round']>0]
    control = all(w['classification']=='native_append_begins_after_copy' for w in serial)
    classes = {w['classification'] for w in deferred}
    direction = 'unresolved'
    if control and classes == {'native_append_begins_after_copy'}:
        direction = 'test_compute_first_submission'
    if control and classes == {'native_append_returns_before_copy_end'}:
        direction = 'inspect_native_queue_and_command_list_scheduling'
    if control and classes == {'native_append_spans_copy_tail'}:
        direction = 'inspect_blocking_or_driver_append_path'
    return dict(serial_control_passed=control,deferred_classes=sorted(classes),next=direction,
        speed_qualified=False,physical_engine_identity_qualified=False,
        margin_ns=CLOCK_MARGIN_NS,view='Collector API entries and returns correlated by operation ID and flow. No independent UR-entry clock or physical engine identity.')


def save(row):
    write(ROOT/'record.json', row)
    compact = {k:v for k,v in row.items() if k not in ('inputs','environment')}
    compact.update(source_inputs=dict(files=len(row['inputs']),sha256=signature(row['inputs'])),
        environment_sha256=signature(row['environment']))
    data = json.loads(RESULT.read_text())
    data[KEY] = compact
    write(RESULT,data)


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),'fresh run only')
    row, followup = original()
    rev = validation.git('rev-parse','HEAD').decode().strip()
    pins = source_pins(rev)
    ROOT.mkdir(mode=0o700)
    capture = ROOT/'capture'
    capture.mkdir(mode=0o700)
    for name in ('tmp','cache','trace'):
        (capture/name).mkdir()
    save(dict(stage='prepared',active=False,ownership_closed=True,passed=False,complete=False,
        source_commit=rev,inputs=pins,executable=row['binary'],native_source_commit=row['source_commit'],
        base_result_commit=followup['original_result_commit'],platform_sha256=signature(row['platform']),
        **case(row),limits=LIMITS,
        plan=dict(workload='Exact retained fixture --component 0. Capture one host+device run, 25 windows.',
            change='Enable PTI Level Zero host timing and Chrome call logging only. Preserve safety settings and observed workload.',
            measurement='Join 512 copies and two QSA kernels in each of ten serial-separate/deferred windows via unique collector IDs and flows. Compare warm rounds on a conservative 1ms clock margin, with serial as a sign control.',
            stopping='Fail for missing, ambiguous or out-of-window joins, or ownership, validation, driver or budget failure. No retry or performance claim.',
            retention='One compact chronology and comparison against the previous device trace. Retire both captures and the temporary native generation at this boundary unless a named imminent consumer proves otherwise.')))
    print('Prepared one separately pinned same-ELF attribution run.')


def run():
    row = json.loads((ROOT/'record.json').read_text())
    ref = validation.committed(RESULT,(KEY,))
    require(row['stage']=='prepared' and not row['active'] and
        ref==dict({k:v for k,v in row.items() if k not in ('inputs','environment')},
            source_inputs=dict(files=len(row['inputs']),sha256=signature(row['inputs'])),
            environment_sha256=signature(row['environment'])), 'commit the exact plan first')
    base, followup = original()
    plan = case(base)
    require(row['inputs']==source_pins(row['source_commit']) and base['binary']==row['executable']
        and row['platform_sha256']==signature(base['platform']) and row['limits']==LIMITS
        and row['base_result_commit']==followup['original_result_commit']
        and all(row[k]==v for k,v in plan.items()),'changed same-image diagnostic input')
    capture = ROOT/'capture'
    scope = owner.Owner(capture/'owned',lambda:write(capture/'execution.json',scope.commands))
    env, pin = row['environment'],base['platform']
    mapping = dict(admission=dict(runtime_manifest=pin['runtime_manifest'],tools=pin['tools']),
                   preload=pin['preload'],fixture=base['binary'])
    maps = []
    result = dict(host_probe_accepted=False)
    def interact(io, entry):
        for phase in ('before-runtime','runtime-ready','after-cleanup'):
            while (line := io.line(240)) != 'QSA_BF16_READY '+phase:
                require(line.startswith(('device=','case ','fullfit_attention_overlap ')), 'unexpected fixture output')
            proc = Path('/proc',str(entry['pid']))
            require(str((proc/'exe').resolve())==row['executable']['path'],'unexpected native executable')
            if phase=='before-runtime':
                actual=(proc/'environ').read_bytes().split(b'\0')
                required=('UR_ENABLE_LAYERS','UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys',
                    'EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE','ONEAPI_DEVICE_SELECTOR',
                    'ZES_ENABLE_SYSMAN','STRATA_FULLFIT_OVERLAP_OBSERVE_CLOCK','UNITRACE_SystemTime')
                require(all(f'{k}={env[k]}'.encode() in actual for k in required), 'native safety or clock setting drift')
            else:
                maps.append(dict(phase=phase,**fixture.pti_probe.mapped(entry['pid'],mapping,'trace')))
            io.send(b'X')
        require(io.line(45).startswith('fixture_status=PASS '),'no completion sentinel')
        io.exit(45)
    log = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
        text=True,timeout=15)
    cursor = next(s.split(': ',1)[1] for s in log.splitlines() if s.startswith('-- cursor: '))
    row.update(active=True,ownership_closed=False)
    save(row)
    try:
        begin = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        entry,out,err = scope.run('capture',row['command'],env,P,interaction=interact,**LIMITS)
        end = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        result.update(native_pid=entry['pid'],native_exit_code=entry['exit_code'],providers=maps)
        require(owner.closed(entry) and entry['exit_code']==0,'owned native execution failed')
        logs = err.read_text()
        probe = validation.preload_evidence(logs,entry['pid'])
        fixture.qsa.diagnostics(logs,entry['pid'])
        require(len(maps)==2 and probe['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER']
            and (not (capture/'validation.log').exists() or not (capture/'validation.log').stat().st_size),
            'mapped providers or validation changed')
        path, = (capture/'trace').glob('*.json')
        require(path.stat().st_size<LIMITS['file_cap'],'timeline exceeded file bound')
        result.update(trace=dict(sha256=sha(path),bytes=path.stat().st_size),validation=probe)
        trace = reader.load_json(path)
        timing = fixture.samples(out.read_text(),'trace')
        dev = fixture.timeline(trace,timing['samples'],(begin,end))
        windows = correlate(trace['traceEvents'],timing['samples'],dev,entry['pid'])
        result.update(windows=windows,decision=decision(windows),host_probe_accepted=True)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        validation.finish(result,scope,cursor,pin['boot'])
        passed = result['passed'] and result['driver_after']==pin['driver']
        row.update(stage='closed' if passed else 'failed',active=bool(scope.active),
            ownership_closed=result['ownership_closed'],passed=passed,complete=passed,result=result)
        save(row)
    require(passed,result.get('error','driver or owner check failed'))
    print(result['decision'],flush=True)


if __name__ == '__main__':
    require(sys.executable==fixture.PYTHON and len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),
        'use pinned Python and prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fixture.guards.no_model_owners([])
        if sys.argv[1]=='prepare': prepare()
        else: run()
