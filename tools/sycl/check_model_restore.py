#!/usr/bin/env python3
"""Owned native GEN capacity/restore diagnostics, never speed or task benchmarks.

Main-authored using this project's existing protocol and snapshot interfaces;
repository LICENSE. Public HTTP qualification is a separate gate.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import full_context_model as model

gate = model.gate
P, B, RESULT = model.P, model.B, model.RESULT
ROOT = B / 'full-context-state-v1'
KEY = 'model_state_v1'
ORDER = ('small', 'near-cap', 'restore')
INPUTS = sorted(set([*model.INPUTS, 'tools/sycl/check_model_restore.py']))
require, digest, write = model.require, model.digest, model.write


def compact(record):
    return {k: v for k, v in record.items() if k not in ('environments', 'commands')}


def inputs(revision):
    pins = {}
    for name, data in gate.native.build.blobs(P, revision, INPUTS):
        require(data == (P / name).read_bytes(), 'commit state diagnostic input: ' + name)
        pins[name] = hashlib.sha256(data).hexdigest()
    return pins


def requests(mode):
    n = 16385 if mode == 'small' else 262112
    a = [i % 4096 for i in range(n)]
    if mode == 'near-cap':
        return [([i % 4096 for i in range(262120)], 16, 0)]
    return [(a, 1, 0), ([128 + i for i in range(33)], 1, 0),
            (a + list(range(256, 272)), 1, n)]


def wire(ids, count):
    return ('GEN ' + str(count) + ' ' + ','.join(map(str, ids)) + '\n').encode()


def prepare(container):
    require(KEY not in container and not ROOT.exists(), 'fresh state plan only')
    component = container['snapshot_qsa_components']
    require(component['passed'] and component['complete'] and component['ownership_closed'] and
            not component['active'] and all(v['passed'] and not v['fresh_faults']
                for v in component['outcomes'].values()) and
            set(component['outcomes']) == set(__import__('check_streamed_qsa').ORDER),
            'closed same-build streamed/attention/snapshot component gates required')
    engine = gate.engine()
    require(component['native_source_commit'] == engine['source_commit'] and
            component['binary'] == engine['binary'], 'component/native generation mismatch')
    base = model.setup()
    model.model_pins(base['config'], base['models'], full=True)
    original_args = base['config']['args']
    pack = Path(original_args[original_args.index('--pack') + 1])
    vocab = set(json.loads((pack / 'tokenizer/vocab.json').read_text()).values())
    require(set(range(4096)) <= vocab and len(vocab) == max(vocab) + 1, 'unexpected tokenizer ID domain')
    revision = gate.native.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    args = list(base['config']['args'])
    for name, value in (('--max-context', '262144'), ('--expert-profile', str(P / 'data/expert-profile.bin')),
                        ('--prompt-cache', '6'), ('--conversation-cache-mib', '8192'),
                        ('--conversation-cache-slots', '2'), ('--conversation-cache-min-free-mib', '4096')):
        if name in args:
            args[args.index(name) + 1] = value
        else:
            args += [name, value]
    require(args[args.index('--expert-cache') + 1] == '1256' and
            args[args.index('--kv-resident') + 1] == '131072', 'decode placement changed')
    require('--layer-split' not in args and '--batch' not in args and '--serve' not in args,
            'unsupported diagnostic topology')
    probe = gate.validation.preload_qualification()
    cases, envs = {}, {}
    ROOT.mkdir(mode=0o700)
    for mode in ORDER:
        root = ROOT / mode
        root.mkdir(); (root / 'tmp').mkdir(); (root / 'cache').mkdir()
        oldroot = base['environment']['STRATA_MEASUREMENT_ROOT']
        env = {k: v.replace(oldroot, str(root)) for k, v in base['environment'].items()}
        require(not env.get('LD_PRELOAD'), 'unexpected inherited preload')
        env.update(TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'), CUDA_MODULE_LOADING='EAGER',
                   LD_PRELOAD=probe['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
                   UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1',
                   STRATA_STATE_HASH='1', STRATA_SNAPSHOT_VERIFY='1', STRATA_DECODE_TIMING='1')
        env.pop('UR_LOG_TRACING', None)
        for name in ('UR_LOG_VALIDATION', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO'):
            env[name] = 'level:warning;flush:warning;output:stderr'
        for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
                     'STRATA_SYCL_PREFILL_SAME_QUEUE', 'SYCL_CACHE_PERSISTENT'):
            require(env[name] == engine['environment'][name], 'qualified safety setting differs: ' + name)
        samples = [dict(prompt=len(ids), max_new=count, reused=reused,
                        wire_bytes=len(wire(ids, count)), wire_sha256=hashlib.sha256(wire(ids, count)).hexdigest())
                   for ids, count, reused in requests(mode)]
        require(all(s['prompt'] + s['max_new'] + 8 <= 262144 for s in samples), 'eight-token admission reserve')
        cases[mode] = dict(command=[engine['binary']['path'], '--serve', *args], requests=samples,
                           wall_s=1200 if mode == 'small' else 3600,
                           environment_sha256=model.fingerprint(env))
        envs[mode] = env
    record = dict(stage='prepared', source_commit=revision, input_sha256=pins,
                  native_source_commit=engine['source_commit'], binary=engine['binary'], vocab_size=len(vocab),
                  selected_original=gate.native.source.baseline()[1], cases=cases, environments=envs,
                  setup_reference=base['reference'], model_assets=base['models'], preload=probe,
                  boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  driver_before=gate.native.guards.driver_guard(), outcomes={},
                  active=False, passed=False, complete=False, ownership_closed=True,
                  budgets=dict(cpu_s=7200, rss_bytes=104 << 30, text_bytes=32 << 20,
                               file_bytes=32 << 20, total_bytes=512 << 20),
                  criteria='Stop first failure;no retry. Small A/B/A+16 first,then admitted262120+16 extent,then262112/B/262128 restore. One-token requests must commit exactly prompt extent;general MTP position is prompt-1+actual verifier windows+accepted drafts,not emitted length. Require exact restore-before-reentry payload verification for all12 main QSA layers,draft and running state;actual restored prefix/read counts;valid generated IDs;normal owned exit0;unchanged models,boot,driver and no fresh fault. Do not compare generated text or cold-versus-warm arithmetic bits. Small traced device qualification belongs to the preceding same-build component suite;model checks use warning-level validation and stage progress.',
                  retention='Keep source/recipe,request byte hashes and individual admission/position/copy/ownership/fault outcomes. Retire successful raw logs/JIT after closure. Both retained generations keep their existing named qualification/comparison consumers.',
                  speed_eligible=False, production_selected=False, model_adoption_qualified=False,
                  physical262144_qualified=False, decode_performance_qualified=False,
                  ordinary_runtime_teardown_qualified=False,
                  next_gate='Public API/agent-session/task quality,failure-owner and repeated clean decode preservation remain separate. This native protocol diagnostic is not a bench.sh performance comparison.')
    write(ROOT / 'record.json', record); container[KEY] = compact(record); write(RESULT, container)
    print('Prepared native state/near-cap plan;commit before every case.', flush=True)


def admission(record, container):
    require(RESULT.read_bytes() == gate.native.build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))) and
            compact(record) == container[KEY] and inputs(record['source_commit']) == record['input_sha256'],
            'commit exact state plan and inputs first')
    engine = gate.engine()
    require(engine['binary'] == record['binary'] and engine['source_commit'] == record['native_source_commit'] and
            gate.native.source.baseline()[1] == record['selected_original'], 'native/fallback identity differs')
    base = model.setup()
    require(base['reference'] == record['setup_reference'] and base['models'] == record['model_assets'] and
            gate.validation.preload_qualification() == record['preload'], 'reference/runtime changed')
    model.model_pins(base['config'], record['model_assets'])
    for mode in ORDER:
        require(model.fingerprint(record['environments'][mode]) == record['cases'][mode]['environment_sha256'],
                'effective environment differs')


def interact(io, mode, row, record):
    info = {}
    while True:
        line = io.line(600)
        require(not line.startswith('ERR'), line)
        if line.startswith('INFO '):
            info.update(t.split('=', 1) for t in line.split()[1:] if '=' in t)
        if line.startswith('READY '):
            require(line == 'READY 262144 stop', 'native readiness extent differs')
            break
    row['info'] = {k: info[k] for k in ('context','kv','kv_resident','expert_slots','spec','mtp_max',
        'lookup','pool_workers','conversation_cache_mib','conversation_cache_slots','conversation_cache_min_free_mib') if k in info}
    require(info.get('expert_slots') == '1256' and info.get('kv_resident') == '131072' and
            info.get('conversation_cache_mib') == '8192', 'actual model/cache admission differs')
    row['requests'] = []
    for (ids, count, reused), expected in zip(requests(mode), record['cases'][mode]['requests']):
        payload = wire(ids, count)
        require(len(payload) == expected['wire_bytes'] and hashlib.sha256(payload).hexdigest() == expected['wire_sha256'],
                'committed request bytes differ')
        # The protocol permits this long line. Each bounded write uses the
        # existing Interactive input cap; all chunks form one exact GEN request.
        for offset in range(0, len(payload), 256 << 10):
            io.send(payload[offset:offset + (256 << 10)])
        tokens = []
        while True:
            line = io.line(1800)
            require(not line.startswith('ERR'), line)
            if line.startswith('T '):
                tokens.append(int(line.split()[1]))
            if line.startswith('DONE '):
                fields = line.split()[1:]
                require(len(fields) == 14, 'native completion fields changed')
                actual = dict(generated=int(fields[0]), prompt=int(fields[1]), finish=fields[4],
                              accepted=int(fields[5]), offered=int(fields[6]), reused=int(fields[7]),
                              read=int(fields[13]), ids=tokens)
                row['requests'].append(actual)
                require(actual['generated'] == len(tokens) == count and actual['prompt'] == len(ids) and
                        actual['reused'] == reused and actual['read'] == len(ids) - reused and
                        0 <= actual['accepted'] <= actual['offered'] and all(0 <= t < record['vocab_size'] for t in tokens) and
                        actual['finish'] in (('length',) if count > 1 else ('length', 'stop')),
                        'native extent/position/finish gate failed')
                break
    io.send(b'QUIT\n'); io.exit(90)


def evidence(text, mode, row, record, pid):
    row['activation'] = gate.validation.preload_evidence(text, pid)
    boundary = re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n', text, re.M)
    require(boundary is not None, 'missing validation activation boundary')
    body = text[boundary.end():]
    require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed', body),
            'runtime failure after intentional pre-main canary')
    states = [dict(re.findall(r'(\w+)=([0-9a-f,-]+)', s))
              for s in re.findall(r'^strata serve: STATE_HASH L=.*$', body, re.M)]
    windows = [int(v) for v in re.findall(r'^strata decode timing: (\d+) windows,', body, re.M)]
    require(len(states) == len(windows) == len(row['requests']), 'state/window accounting incomplete')
    for state, nw, actual in zip(states, windows, row['requests']):
        require(set(state) == {'L','gdn','ple','tail','pooled','kv','mtp','stale','dead','pooled_full','ple_prev'},
                'missing main/draft state fingerprint domains')
        n = actual['prompt']; position = int(state['L'])
        require(position == n - 1 + nw + actual['accepted'] and position <= 262144,
                'MTP-aware committed-position conservation failed')
        require(actual['generated'] != 1 or (position == n and nw == 1 and actual['accepted'] == 0),
                'one-token request position changed')
        actual.update(state_L=position, windows=nw, state_sha256=model.fingerprint(state), ple_prev=state['ple_prev'])
    copies = [dict(re.findall(r'(\w+)=([^ ]+)', s))
              for s in re.findall(r'^strata serve: SNAPSHOT_VERIFY .*$', body, re.M)]
    restores = [int(s) for s in re.findall(r'^strata serve: conversation cache: restored (\d+) tokens', body, re.M)]
    row['restored_tokens'] = restores; row['restore_payloads'] = copies
    if mode == 'near-cap':
        require(not copies and not restores, 'fresh near-cap request unexpectedly restored')
    else:
        n = len(requests(mode)[0][0])
        require(restores == [n] and len(copies) == 1, 'actual parking/restore not exercised')
        proof = copies[0]
        require(proof.get('result') == 'PASS' and int(proof.get('upto', -1)) == n and
                int(proof.get('main_layers', -1)) == 12 and
                all(int(proof.get(k, 0)) > 0 for k in ('main_bytes','draft_bytes','running_bytes')),
                'complete restore payload verification missing')
        require('conversation cache: skip parking' not in body, 'cache budget/RAM skipped required parking')
    route = [tuple(map(int, m)) for m in re.findall(
        r'^strata prefill QSA16 streamed: layer=(\d+) chunks=(\d+) rows=(\d+) pos0=(\d+) '
        r'prefix_H2D_requested_bytes=(\d+) host_publication=GPU_Q8 stage=full_logical_layer checked_drain=PASS$', body, re.M)]
    n = row['requests'][0]['prompt']
    require(route == [(layer, (n - 2) // 8192 + 1, n - 1, 0, 0) for layer in range(3, 48, 4)],
            'initial long-prefill route/extent differs')
    row['initial_prefill'] = dict(rows=n-1, layers=12, chunks_per_layer=(n-2)//8192+1,
                                  pos0=0, prefix_H2D_requested_bytes=0)


def run(container, mode):
    record = json.loads((ROOT / 'record.json').read_text()); admission(record, container)
    index = ORDER.index(mode)
    require(mode not in record['outcomes'] and all(record['outcomes'].get(k, {}).get('passed') for k in ORDER[:index])
            and record['stage'] == ('prepared' if index == 0 else ORDER[index - 1] + '_closed'), 'no retry/skipped gate')
    require(record['driver_before'] == gate.native.guards.driver_guard() and
            record['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'driver/boot changed')
    command = ['journalctl','-k','-b','--no-pager']
    cursor_text = subprocess.check_output([*command,'-n','0','--show-cursor'], text=True, timeout=15)
    cursor = next(s.split(': ', 1)[1] for s in cursor_text.splitlines() if s.startswith('-- cursor: '))
    original = json.loads((ROOT / 'record.json').read_text()); owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(record, commands=owner.commands if owner else []))
    owner = gate.native.supervision.Owner(ROOT / mode / 'owned', persist)
    row = dict(passed=False, complete=False, ownership_closed=False)
    record['outcomes'][mode] = row; record['stage'] = mode + '_running'; persist()
    try:
        case, b = record['cases'][mode], record['budgets']
        entry, out, err = owner.run(mode, case['command'], record['environments'][mode], P,
            interaction=lambda io: interact(io, mode, row, record), wall=case['wall_s'], cpu=b['cpu_s'],
            rss_cap=b['rss_bytes'], text_cap=b['text_bytes'], file_cap=b['file_bytes'], total_cap=b['total_bytes'])
        row['exit_code'] = entry['exit_code']
        require(gate.native.supervision.closed(entry) and entry['exit_code'] == 0, 'normal owned model exit required')
        evidence(err.read_text(errors='replace'), mode, row, record, entry['pid'])
        admission(original, container)
        model.model_pins(model.setup()['config'], record['model_assets'], full=mode == ORDER[-1])
        row['passed'] = True
    except BaseException as error:
        row['error'] = repr(error)
    finally:
        row['ownership_closed'] = owner.active is None and all(gate.native.supervision.closed(c) for c in owner.commands)
        try:
            text = subprocess.check_output([*command,'--after-cursor',cursor], text=True, timeout=15)
            row['fresh_faults'] = [s for s in text.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|timeout|fault|reset|wedg)|devcoredump)', s)]
            row['boot_unchanged'] = record['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            row['driver_after'] = gate.native.guards.driver_guard()
        except BaseException as error:
            row['final_probe_error'] = repr(error)
        row['passed'] = bool(row['passed'] and row['ownership_closed'] and not row.get('fresh_faults', ['unknown']) and
                             row.get('boot_unchanged') and row.get('driver_after') == record['driver_before'] and
                             'final_probe_error' not in row)
        row['complete'] = row['passed']; record['ownership_closed'] = row['ownership_closed']
        record['stage'] = mode + '_closed' if row['passed'] else 'failed'
        record['passed'] = record['complete'] = bool(mode == ORDER[-1] and row['passed'])
        persist(); container[KEY] = compact(record); write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('passed','complete','exit_code','ownership_closed','fresh_faults','error')}), flush=True)
    return 0 if row['passed'] else 1


if __name__ == '__main__':
    require(sys.argv[1:] in ([mode] for mode in ('prepare', *ORDER)), 'prepare|' + '|'.join(ORDER))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); gate.native.guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        if sys.argv[1] == 'prepare': prepare(container)
        else: sys.exit(run(container, sys.argv[1]))
