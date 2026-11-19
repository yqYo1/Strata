#!/usr/bin/env python3
"""Matched native server park/restore check for the attention-bank selector."""

import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import check_model_restore as protocol
import prefill_attention_bank_capacity as capacity
import prefill_attention_bank_measure as measure
import ur_validation as validation

build, basis = capacity.build, capacity.basis
P, B, RESULT = capacity.P, capacity.B, capacity.RESULT
require, sha, signature = capacity.require, capacity.sha, capacity.signature
ROOT = B / 'prefill-attention-bank-restore-v2'
KEY = 'prefill_attention_bank_restore_v2'
PREVIOUS = 'prefill_attention_bank_restore_v1'
THIS = 'tools/sycl/prefill_attention_bank_restore.py'
CAPACITY_EVIDENCE = 'e5408e1c27a6fa33ce314cc3b09e265b7d00cf52'
ORDER = ('host', 'short-copy', 'short-direct', 'near-copy', 'near-direct')
INFO = dict(context='262144', kv='int8', kv_resident='20480', expert_slots='1256',
            spec='4', conversation_cache_mib='8192', conversation_cache_slots='2',
            conversation_cache_min_free_mib='4096')
ACTIVE_STATE = ('L', 'gdn', 'ple', 'tail', 'pooled', 'kv', 'mtp', 'ple_prev')
STALE_STATE = ('stale', 'dead', 'pooled_full')
SELECTOR = measure.SELECTOR
owner_module = capacity.owner_module


def reference():
    native, speed = capacity.retained_build()
    physical = build.historic(CAPACITY_EVIDENCE, capacity.KEY)
    require(physical['stage'] == 'closed' and physical['passed'] and physical['ownership_closed']
            and physical['native_source_commit'] == native['source_commit']
            and physical['binary'] == native['binaries']['strata_full_context_diagnostic']
            and all(physical['outcomes'][name]['final_slot_consumed']
                    and physical['outcomes'][name]['fresh_faults'] == []
                    for name in ('full-copy', 'full-direct')), 'successful matched standalone capacity gate required')
    return native, speed


def sources(revision):
    result = basis.comparison.build.controls(revision, controller=P / THIS)
    result.update(build.stage.source.pin(P, list(basis.POLICIES))['source_sha256'])
    return result


def workloads(near):
    n = 253919 if near else 32769
    initial = [i % 4096 for i in range(n)]
    switch = list(range(128, 161))
    resumed = initial + [256 + i % 16 for i in range(8209)]
    return [(initial, 0), (switch, 0), (resumed, n)]


def input_plan(near):
    result = []
    for ids, reused in workloads(near):
        payload = protocol.wire(ids, 1)
        result.append(dict(prompt=len(ids), max_new=1, reused=reused, wire_bytes=len(payload),
                           wire_sha256=signature_bytes(payload)))
    return result


def signature_bytes(payload):
    # Use the original protocol peer's raw-byte hash contract.
    return hashlib.sha256(payload).hexdigest()


def environment(row, case):
    env = measure.context('numerical').environment(row, dict(root=case['root'], index=0, heads=1))
    probe = row['native_validation_probe']
    env.update(TMPDIR=case['root']+'/tmp', XDG_CACHE_HOME=case['root']+'/cache', STRATA_TRACE='1',
               LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_STATE_HASH='1',
               STRATA_SNAPSHOT_VERIFY='1', STRATA_DECODE_TIMING='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1',
               STRATA_SYCL_PREFILL_MOE_CENSUS='0')
    for name in ('UR_LOG_VALIDATION', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO'):
        env[name] = 'level:warning;flush:warning;output:stderr'
    env.pop('UR_LOG_TRACING', None)
    env[SELECTOR] = str(int(case['direct']))
    require(env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == '1'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['EnableDirectSubmission'] == '0', 'production layout or qualified safety configuration changed')
    return env


def argv(row):
    args = list(row['configuration']['args'])
    require('--serve' not in args and '--layer-split' not in args, 'one native server selector required')
    for name, value in (('--max-context', '262144'), ('--expert-cache', '1256'),
                        ('--kv-resident', '20480'), ('--prefill', '8192'), ('--spec', '4')):
        require(args.count(name) == 1 and args[args.index(name)+1] == value, 'fixed decode/shape option differs')
    for name, value in (('--prompt-cache', '6'), ('--prompt-cache-every', '0'),
                        ('--conversation-cache-mib', '8192'), ('--conversation-cache-slots', '2'),
                        ('--conversation-cache-min-free-mib', '4096')):
        if name not in args:
            args.extend((name, value))
        else:
            args[args.index(name)+1] = value
    return [row['binary']['path'], '--serve', *args]


def make_case(row, name):
    near = name.startswith('near-')
    case = dict(name=name, near=near, direct=name.endswith('-direct'), root=str(ROOT / name),
                requests=input_plan(near), command=argv(row))
    case['environment_sha256'] = signature(environment(row, case))
    require(all(s['prompt'] + s['max_new'] + 8 <= 262144 for s in case['requests']),
            'requests must leave the required server output space')
    return case


def interact(io, expected, vocab_size, *, native):
    info = {}
    while True:
        line = io.line(600)
        require(not line.startswith('ERR'), line)
        if line.startswith('INFO '):
            info.update(t.split('=', 1) for t in line.split()[1:] if '=' in t)
        if line.startswith('READY '):
            require(line == 'READY 262144 stop', 'native READY context differs')
            break
    require(all(info.get(k) == v for k, v in INFO.items()), 'native server INFO does not match fixed placement')
    result = []
    for (ids, reused), receipt in zip(workloads(expected['near']), expected['requests'], strict=True):
        payload = protocol.wire(ids, 1)
        require(len(payload) == receipt['wire_bytes'] and signature_bytes(payload) == receipt['wire_sha256'],
                'committed request bytes differ')
        for offset in range(0, len(payload), 256 << 10):
            io.send(payload[offset:offset+(256 << 10)])
        tokens, publications = [], []
        while True:
            line = io.line(1800)
            require(not line.startswith('ERR'), line)
            if line.startswith('PP '):
                publications.append(int(line.split()[1]))
            if line.startswith('T '):
                tokens.append(int(line.split()[1]))
            if not line.startswith('DONE '):
                continue
            fields = line.split()[1:]
            require(len(fields) == 14, 'native DONE protocol differs')
            generated, prompt, accepted, offered, actual_reused, read = map(int, (fields[0], fields[1], *fields[5:8], fields[13]))
            require(generated == len(tokens) == 1 and 0 <= tokens[0] < vocab_size
                    and (prompt, accepted, offered, actual_reused, read) == (len(ids), 0, 0, reused, len(ids)-reused)
                    and fields[4] in ('length', 'stop'), 'unexpected prompt/decode work or output')
            if native:
                require(publications and publications[-1] == len(ids)-1
                        and reused < publications[0]
                        and all(a < b for a,b in zip(publications, publications[1:])), 'incomplete prefill publication')
            result.append(dict(prompt=prompt, read=read, reused=reused, output_hash=signature(tokens),
                               finish=fields[4], publications=publications))
            break
    io.send(b'QUIT\n')
    io.exit(90)
    return result


def require_store(frame, case, request):
    rows = request['read'] - 1
    if rows < capacity.CHUNK:
        require('strata prefill attention bank: ' not in frame, 'short switch unexpectedly used long store')
        return dict(pieces=0, direct=0, copy_bytes=0)
    n, chunk, gpu, host_bytes = map(int, capacity.one(frame,
        r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=bf16 '
        r'residual_element_bytes=2 residual_total=\d+ residual_gpu_rows=(\d+) residual_gpu=\d+ '
        r'residual_host=(\d+) residual_staging=\d+ layer_cache=\d+;', 'server BF16 residual extent'))
    mode, maximum = map(int, capacity.one(frame,
        r'^strata MoE window configuration: requested_mode=10 effective_mode=(\d+) '
        r'maximum_rows=(\d+) mixed_fp16_bytes=\d+$', 'server MoE window'))
    require(n == rows and chunk == capacity.CHUNK and 0 <= gpu <= n
            and host_bytes == (n-gpu)*capacity.ROW_BYTES and maximum >= chunk, 'server BF16 layout differs')
    pieces, total = capacity.expected_store(n, gpu, maximum)
    requested, recorded, direct, copy = map(int, capacity.one(frame,
        r'^strata prefill attention bank: requested=([01]) pieces=(\d+) direct=(\d+) requested_D2D_bytes=(\d+)$',
        'server attention-store execution'))
    enabled = int(case['direct'])
    require((requested, recorded, direct, copy) == (enabled, pieces, enabled*pieces, 0 if enabled else total)
            and pieces > 0, 'copy/direct attention store was not exercised at this extent')
    return dict(pieces=pieces, direct=direct, copy_bytes=copy, gpu_rows=gpu, host_rows=n-gpu,
                window_mode=mode, maximum_rows=maximum)


def inspect(log, row, case, requests, pid):
    proof = validation.preload_evidence(log, pid)
    require(proof['loader'] == row['native_validation_probe']['loader']['path']
            and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', log),
            'unexpected runtime validation')
    found = list(re.finditer(r'^strata serve: STATE_HASH L=.*$', log, re.M))
    require(len(found) == 3, 'three complete state fingerprints required')
    start = 0
    for seen, request in zip(found, requests, strict=True):
        frame = log[start:seen.start()]
        start = seen.end()
        state = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', seen[0]))
        require(set(state) == set(ACTIVE_STATE + STALE_STATE) and int(state['L']) == request['prompt'],
                'committed position or fingerprint domains differ')
        request['active_state'] = {k:state[k] for k in ACTIVE_STATE}
        request['unused_state'] = {k:state[k] for k in STALE_STATE}
        request['attention_store'] = require_store(frame, case, request)
        cells = [int(t) for t in re.findall(r'^strata serve: DRAFT_PREFILL path=(?:batched|token) mode=[012] cells=(\d+)$', frame, re.M)]
        position = request['reused']
        ends = []
        for count in cells:
            require(0 < count <= capacity.CHUNK, 'draft callback exceeds chunk')
            position += count
            ends.append(position)
        require(ends == request['publications'], 'native draft and publication boundaries differ')
        # Cache-lease diagnostics prove the residency table and payload were
        # restored as well as the conversation snapshot itself.
        for phase in ('lend', 'refill'):
            capacity.one(frame, r'^strata loan audit: phase='+phase+
                         r' table_entries=24576 table_match=1 free_bytes=\d+ total_bytes=\d+$', 'expert table '+phase)
        lent = int(capacity.one(frame, r'^strata loan audit: phase=payload rows=1256 bytes=2575206400 '
            r'lent=(\d+) same_slots=1 payload_match=1$', 'decode expert payload'))
        require(0 < lent <= 1256, 'no checked decode cache restoration')
        request['decode_cache_lent'] = lent
    old = requests[0]['prompt']
    restored, kind = capacity.one(log, r'^strata serve: conversation cache: restored (\d+) \((live|checkpoint)\)', 'actual RESTORE')
    require(int(restored) == old and kind == 'live' and 'conversation cache: skip parking' not in log,
            'the exact parked live conversation must be restored')
    line = capacity.one(log, r'^strata serve: SNAPSHOT_VERIFY (.*)$', 'restored payload')
    snapshot = dict(t.split('=',1) for t in line.split())
    require(snapshot['result'] == 'PASS' and snapshot['source'] == 'ram' and int(snapshot['upto']) == old
            and int(snapshot['main_layers']) == 12
            and all(int(snapshot[k]) > 0 for k in ('main_bytes', 'draft_bytes', 'running_bytes')),
            'complete snapshot byte verification required')
    return dict(validation=proof, restored_tokens=old, restore_payload=snapshot, requests=requests)


def compare(result, control):
    shape = ('cells', 'mode', 'source', 'resident', 'main_layers', 'main_bytes', 'draft_bytes', 'running_bytes', 'upto')
    require(result['restored_tokens'] == control['restored_tokens']
            and all(result['restore_payload'][k] == control['restore_payload'][k] for k in shape),
            'copy/direct parked snapshot extent differs')
    for actual, prior in zip(result['requests'], control['requests'], strict=True):
        for key in ('prompt','read','reused','output_hash','finish','publications','active_state','decode_cache_lent'):
            require(actual[key] == prior[key], 'copy/direct committed server evidence differs: '+key)
        fixed = lambda value: {k:v for k,v in value.items() if k not in ('direct','copy_bytes')}
        require(fixed(actual['attention_store']) == fixed(prior['attention_store']), 'copy/direct store geometry differs')


def compact(row):
    return basis.compact(row)


def save(row):
    build.stage.write(ROOT / 'record.json', row)
    data = build.read(RESULT)
    data[KEY] = compact(row)
    build.stage.write(RESULT, data)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh restore plan required')
    previous = validation.committed(RESULT, (PREVIOUS,))
    require(previous['stage'] == 'failed' and set(previous['outcomes']) == {'host'}
            and previous['host_physical_owner_reconciliation']['no_native_gpu_launched']
            and previous['host_physical_owner_reconciliation']['executor_gone'], 'retain the closed original protocol refusal')
    native, old = reference()
    revision = validation.git('rev-parse', 'HEAD').decode().strip()
    vocab = json.loads((Path(old['configuration']['tokenizer']) / 'vocab.json').read_text())
    require(set(range(4096)) <= set(vocab.values()), 'diagnostic fixture IDs invalid')
    row = {k:old[k] for k in ('configuration','environment_template','model_assets','boot','driver',
                              'runtime_manifest','native_validation_probe')}
    row.update(stage='prepared', source_commit=revision, sources=sources(revision), binary=native['binaries']['strata'],
               native_source_commit=native['source_commit'], build_sha256=signature(build.builder.compact(native)),
               physical_evidence=CAPACITY_EVIDENCE, outcomes={}, vocab_size=max(vocab.values())+1, plans=[],
               active=False, ownership_closed=True, passed=False, complete=False, speed_eligible=False,
               plan=dict(unit='Same native server ELF, cache sizes, request sequence and observers; switch only attention bank store.',
                         order='Host peer then independent copy/direct at 32769 and 253919 parked tokens. Each returns after '
                               'an unrelated 33-token request with 8209 fresh input tokens, including an actual 8208-row suffix.',
                         gates='Actual GEN reuse, active STATE_HASH, greedy fixture token, publication sequence, '
                               'snapshot payload, checked expert decode restore, attention store counts and normal ownership.',
                         limits='Native pipe protocol and instrumented correctness, with conversation caching enabled in both arms. '
                                'Public HTTP behavior, task quality, clean PP/TG and production-default selection are separate.',
                         stop='Stop on first mismatch, ownership, validation, fault or finite budget failure.'))
    ROOT.mkdir(mode=0o700)
    (ROOT / 'host').mkdir(mode=0o700)
    for name in ORDER[1:]:
        case = make_case(row, name)
        for folder in ('tmp','cache'):
            (Path(case['root']) / folder).mkdir(parents=True, mode=0o700)
        row['plans'].append(case)
    for left, right in zip(row['plans'][::2], row['plans'][1::2], strict=True):
        envs = [{k:v.replace(a['root'], '<owned>') for k,v in environment(row,a).items() if k != SELECTOR}
                for a in (left,right)]
        require(envs[0] == envs[1] and left['requests'] == right['requests'], 'unmatched restore comparison')
    save(row)


def current(row):
    require(compact(row) == validation.committed(RESULT, (KEY,)) and sources(row['source_commit']) == row['sources'],
            'commit exact restore plan and inputs')
    native, old = reference()
    require(row['build_sha256'] == signature(build.builder.compact(native)) and row['binary'] == native['binaries']['strata']
            and row['native_source_commit'] == native['source_commit']
            and all(row[k] == old[k] for k in ('configuration','environment_template','model_assets','boot','driver',
                                               'runtime_manifest','native_validation_probe'))
            and row['plans'] == [make_case(row, name) for name in ORDER[1:]], 'exact run inputs changed')


def run(name):
    row = build.read(ROOT / 'record.json')
    current(row)
    require(row['ownership_closed'] and not row['active'] and row['stage'] in ('prepared','running')
            and list(row['outcomes']) == list(ORDER[:ORDER.index(name)]), 'fresh ordered owner required')
    root = ROOT / name
    require(not (root / 'execution.json').exists(), 'existing owner receipt')
    cursor = next(line.removeprefix('-- cursor: ') for line in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if line.startswith('-- cursor: '))
    outcome = dict(complete=False, host_probe_accepted=False)
    owner = None

    def persist():
        build.stage.write(root/'execution.json',
                         dict(outcome=outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))

    owner = owner_module.Owner(root/'owned',persist)
    try:
        if name == 'host':
            peer = host_peer()
            for near in (False,True):
                case = dict(near=near, requests=input_plan(near))
                command = [sys.executable,'-B','-c',peer,json.dumps(case['requests'])]
                entry,_,_ = owner.run('mock-'+str(int(near)),command,dict(PATH='/usr/bin:/bin',LANG='C.UTF-8'),P,
                    interaction=lambda io,e:interact(io,case,row['vocab_size'],native=False),
                    wall=50,cpu=40,rss_cap=512<<20,text_cap=128<<10,file_cap=1<<20,total_cap=4<<20)
                require(owner_module.closed(entry) and entry['exit_code']==0,'owned protocol peer failed')
            outcome['GPU_executed'] = False
        else:
            case = next(c for c in row['plans'] if c['name'] == name)
            request_result = []

            def exchange(io, entry):
                request_result.extend(interact(io, case, row['vocab_size'], native=True))

            entry,_,err = owner.run(name,case['command'],environment(row,case),P,interaction=exchange,
                wall=2400 if case['near'] else 1200,cpu=18000,rss_cap=104<<30,text_cap=16<<20,
                file_cap=32<<20,total_cap=256<<20)
            outcome.update(native_exit_code=entry['exit_code'],GPU_executed=True,requests=request_result)
            require(owner_module.closed(entry) and entry['exit_code']==0,'native serve exit or owned teardown failed')
            outcome.update(inspect(err.read_text(),row,case,request_result,entry['pid']))
            if case['direct']:
                compare(outcome,row['outcomes'][name.replace('-direct','-copy')])
        current(row)
        if name == ORDER[-1]:
            basis.comparison.shared.models.model_pins(row['configuration'],row['model_assets'],full=True)
        outcome.update(complete=True,host_probe_accepted=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome,owner,cursor,row['boot'])
        outcome['passed'] = bool(outcome['passed'] and outcome['complete'] and outcome['driver_after']==row['driver'])
        row['outcomes'][name] = outcome
        done = outcome['passed'] and name == ORDER[-1]
        row.update(stage='closed' if done else 'running' if outcome['passed'] else 'failed',
                   passed=done,complete=done,ownership_closed=outcome['ownership_closed'],active=bool(owner.active))
        persist()
        save(row)
    print(json.dumps({k:outcome.get(k) for k in ('passed','complete','native_exit_code','restored_tokens',
        'fresh_faults','ownership_closed','error')}),flush=True)
    require(outcome['passed'],'restore gate stopped; preserve the original outcome')


def host_peer():
    marker = 'kv_resident=131072'
    require(protocol.PROTOCOL_PEER.count(marker) == 1, 'original peer INFO fixture changed')
    return protocol.PROTOCOL_PEER.replace(marker, 'kv_resident=20480 context=262144 spec=4')


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare',*ORDER), 'unplanned restore action')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            run(action)
