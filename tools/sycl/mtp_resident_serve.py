#!/usr/bin/env python3
"""Bounded private resident-MTP model route check; Main, repository LICENSE.

Synthetic GEN requests qualify wiring and dispatch only, never speed or task
quality. Production selection is unchanged. Use bench.sh for later timing.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import check_mtp_resident_control as gate
import full_context_model as model

P, B = gate.P, gate.B
ROOT = B / 'mtp-resident-serve-v1'
RESULT = gate.RESULT.with_name('mtp-resident-serve-current.json')
INPUTS = sorted(set([*gate.INPUTS, 'tools/sycl/mtp_resident_serve.py', 'tools/sycl/test_mtp_resident_serve.py',
    'tools/sycl/full_context_model.py', 'tools/sycl/check_full_context.py', 'data/expert-profile.bin']))
require, digest, write = gate.native.guards.require, gate.native.build.digest, gate.write


def inputs(revision):
    pins = {}
    for name, raw in gate.native.build.blobs(P, revision, INPUTS):
        require((P / name).read_bytes() == raw, 'commit exact serve input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def requests():
    return [[i % 4096 for i in range(n)] for n in (33, 257)]


def wire(ids):
    return ('GEN 32 ' + ','.join(map(str, ids)) + '\n').encode()


def private_seal(row):
    return gate.native.fingerprint({k: row[k] for k in ('environments', 'model_assets')})


def compact(row):
    return {k: v for k, v in row.items() if k not in ('environments', 'model_assets', 'commands')}


def native_gate():
    full = gate.current_build()
    closed = gate.validation.committed(gate.RESULT)
    require(closed['stage'] == 'closed' and closed['passed'] and closed['complete'] and
        closed['ownership_closed'] and not closed['active'] and closed['fresh_faults'] == [] and
        closed['binary'] == full['assets']['strata'] and closed['refusals']['passed'] == 32 and
        len(closed['fixtures']) == 2 and all(v['passed'] for v in closed['fixtures']),
        'closed same-build native/device check required')
    return full


def prepare():
    require(not ROOT.exists() and not RESULT.exists(), 'fresh serve root only; no duplicate/retry')
    full = native_gate(); base = model.setup()
    model.model_pins(base['config'], base['models'])
    speed = gate.validation.committed(RESULT.with_name('phase-prefill-speed-current.json'))
    require(speed['stage'] == 'closed' and speed['selection']['selected_source'] == gate.native.BASE and
        not speed['selection']['candidate_adopted'], 'original prefill remains selected')
    preload = gate.validation.preload_qualification()
    revision = gate.native.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    args = list(base['config']['args']); args[args.index('--expert-profile') + 1] = str(P / 'data/expert-profile.bin')
    require('--mtp-max-t' not in args, 'reference already overrides MTP cap')
    vocab_path = Path(base['config']['tokenizer']) / 'vocab.json'
    vocab = set(json.loads(vocab_path.read_text()).values())
    require(set(range(4096)) <= vocab and vocab == set(range(len(vocab))), 'actual vocabulary not dense/compatible')
    ROOT.mkdir(mode=0o700)
    environments = {}; plans = []
    for cap in (0, 1):
        private = ROOT / str(cap); private.mkdir()
        for name in ('tmp', 'cache'): (private / name).mkdir()
        env = dict(speed['environment_template'], TMPDIR=str(private / 'tmp'),
            XDG_CACHE_HOME=str(private / 'cache'), STRATA_MEASUREMENT_ROOT=str(private),
            LD_PRELOAD=preload['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=preload['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1',
            STRATA_SNAPSHOT_VERIFY='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1', STRATA_DECODE_TIMING='1',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(private / 'validation.log'))
        for name in ('STRATA_STATE_HASH', 'UR_LOG_TRACING'): env.pop(name, None)
        require('STRATA_SYCL_PREFILL_SEPARATE_LAYOUT' not in env, 'original prefill policy must remain absent')
        for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
                     'STRATA_SYCL_PREFILL_SAME_QUEUE', 'SYCL_CACHE_PERSISTENT'):
            require(env[name] == full['environment'][name], 'qualified safety changed: ' + name)
        require(env['CUDA_MODULE_LOADING'] == 'EAGER' and env['ZES_ENABLE_SYSMAN'] == '0', 'startup safety differs')
        environments[str(cap)] = env
        plans.append(dict(cap=cap, command=[full['assets']['strata']['path'], '--serve', *args, '--mtp-max-t', str(cap)]))
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins, binary=full['assets']['strata'],
        base_native_source=gate.native.BASE, native_controller_source_commit=full['source_commit'],
        native_result_sha256=digest(gate.native.RESULT), native_gate_sha256=digest(gate.RESULT),
        model_reference=base['reference'], model_assets=base['models'],
        prefill_reference_sha256=digest(RESULT.with_name('phase-prefill-speed-current.json')),
        preload=preload, environments=environments, plans=plans, vocab_path=str(vocab_path),
        vocab_sha256=digest(vocab_path), vocab_size=len(vocab),
        requests=[dict(prompt_tokens=len(ids), max_new=32, wire_bytes=len(wire(ids)),
            wire_sha256=hashlib.sha256(wire(ids)).hexdigest()) for ids in requests()],
        driver=gate.native.guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        criteria='Exactly two fresh native processes,cap0 then1,each33/257 synthetic input IDs and32 output IDs per fresh request,caches0. Same original prefill/settings/1256slots. INFO must confirm131072 resident INT8,spec4,selected cap,5workers,PCIe0,caches0. Each request has complete batched mode2 draft prompt callbacks and valid vocabulary IDs. Cap0 first targetT1 thenT4 with positive increasing draft-round counters;cap1 every targetT1,zero offered/accepted drafts and zero draft rounds. Same-child preexec validation,normal owned exit,empty validation warnings,unchanged source/model/runtime/boot/driver and no fresh fault;full15 model hashes after cohort. Stop first failure,no retry. No token-string identity,speed/task/physical-context/adoption claim.',
        interpretation='Resident no-decode-draft retains MTP weights/KV/prompt precompute but omits generation KV catchup and proposals. Target verifier geometry also changesT4 toT1. This is a private attribution control,not full-MTP-off or an adopted decoder. State hashing of future MTP cells is disabled because this arm intentionally does not maintain them.',
        budgets=dict(wall=900, cpu=3600, rss_cap=104 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=256 << 20),
        retention='Keep compact per-request route/count/wire/result hashes and per-process ownership/fault outcomes. Retire successful stdout/stderr/JIT after replay. Reuse only model configuration/assets from full_context_model.setup;never its retired candidate engine qualification.',
        outcomes={}, active=False, ownership_closed=True, passed=False, complete=False,
        speed_eligible=False, model_adoption_qualified=False, production_selected=False,
        next_gate='Only after both fresh model arms pass,freeze matched official bench.sh thinking-on long strong/weak MTP,agent-session and independent prefill-preservation timing with prospective clock screens. Selected original decode remains unchanged.')
    row['private_inputs_sha256'] = private_seal(row)
    write(ROOT / 'record.json', row); write(RESULT, compact(row))
    print('Prepared fresh short serve cap0/1 route cohort;commit before each process.', flush=True)


def admission(row):
    require(RESULT.read_bytes() == gate.native.build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))) and
        json.loads(RESULT.read_text()) == compact(row) and inputs(row['source_commit']) == row['input_sha256'] and
        private_seal(row) == row['private_inputs_sha256'], 'committed serve source/plan/private inputs differ')
    full = native_gate()
    require(full['assets']['strata'] == row['binary'] and digest(gate.native.RESULT) == row['native_result_sha256'] and
        digest(gate.RESULT) == row['native_gate_sha256'] and gate.validation.preload_qualification() == row['preload'],
        'native qualification differs')
    require(model.setup()['reference'] == row['model_reference'] and
        digest(RESULT.with_name('phase-prefill-speed-current.json')) == row['prefill_reference_sha256'] and
        digest(row['vocab_path']) == row['vocab_sha256'] and gate.native.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'model/prefill/platform changed')
    model.model_pins(model.setup()['config'], row['model_assets'])
    reference = gate.generation.committed(RESULT.with_name('client-resources-current.json'))
    gate.generation.runtime_assets_unchanged(reference)


def interact(io, row, outcome, cap):
    info = {}
    while True:
        line = io.line(600); require(not line.startswith('ERR'), line)
        if line.startswith('INFO '): info.update(t.split('=', 1) for t in line.split()[1:] if '=' in t)
        if line.startswith('READY '):
            require(line == 'READY 131072 stop', 'actual context readiness differs'); break
    expected = dict(context='131072', kv='int8', kv_resident='131072', expert_slots='1256', spec='4',
        mtp_max=str(cap), lookup='0', pool_workers='5', pcie_frac='0.00', conversation_cache_mib='0')
    require(all(info.get(k) == v for k, v in expected.items()), 'actual model/cache/MTP configuration differs')
    outcome['admission'] = {k: info[k] for k in expected}; outcome['requests'] = []
    for ids, fixture in zip(requests(), row['requests']):
        payload = wire(ids)
        require(len(payload) == fixture['wire_bytes'] and hashlib.sha256(payload).hexdigest() == fixture['wire_sha256'],
            'generated request wire changed')
        io.send(payload); tokens = []; progress = []
        while True:
            line = io.line(300); require(not line.startswith('ERR'), line)
            if line.startswith('PP '): progress.append(int(line.split()[1]))
            if line.startswith('T '): tokens.append(int(line.split()[1]))
            if line.startswith('DONE '):
                f = line.split()[1:]; require(len(f) == 14, 'native DONE shape changed')
                sample = dict(generated_tokens=int(f[0]), prompt_tokens=int(f[1]), finish=f[4],
                    drafts_accepted=int(f[5]), drafts_offered=int(f[6]), reused_tokens=int(f[7]), read_tokens=int(f[13]))
                require(sample['generated_tokens'] == len(tokens) == 32 and sample['prompt_tokens'] == len(ids) and
                    sample['finish'] == 'length' and sample['reused_tokens'] == 0 and sample['read_tokens'] == len(ids) and
                    0 <= sample['drafts_accepted'] <= sample['drafts_offered'] and
                    all(0 <= t < row['vocab_size'] for t in tokens), 'native output extent/IDs differ')
                require(progress and progress[-1] == len(ids) - 1 and all(0 < p < len(ids) for p in progress) and
                    all(a < b for a, b in zip(progress, progress[1:])), 'prefill callbacks incomplete')
                sample.update(output_ids_sha256=gate.native.fingerprint(tokens), publications=len(progress))
                outcome['requests'].append(sample); break
    io.send(b'QUIT\n'); io.exit(90)


def route_evidence(text, samples, cap):
    prefix = gate.generation.request_evidence(text, samples)
    headers = list(re.finditer(r'^strata serve: prompt \d+ tokens = \d+ reused \+ \d+ read[^\n]*$', text, re.M))
    require(len(headers) == len(samples) == 2, 'native footer count differs')
    marks = re.findall(r'^strata resident MTP control: mtp_max_t=(\d+) cumulative_draft_rounds=(\d+)$', text, re.M)
    require(len(marks) == 2 and all(int(c) == cap for c, _ in marks), 'MTP route marker missing')
    result = []; previous = 0; rounds = 0
    for sample, header, marker, prompt in zip(samples, headers, marks, prefix):
        windows = [tuple(map(int, w)) for w in re.findall(r'^strata trace: window (\d+) (\d+)$', text[previous:header.start()], re.M)]
        require(len(windows) > 1 and windows[0] == (sample['read_tokens'] - 1, 1) and
            all(t == (1 if cap else 4) for _, t in windows[1:]) and
            all(1 <= b[0] - a[0] <= a[1] for a, b in zip(windows, windows[1:])) and
            sum(t - 1 for _, t in windows) == sample['drafts_offered'], 'actual verifier route/work differs')
        accepted_before_tail = sum(b[0] - a[0] - 1 for a, b in zip(windows, windows[1:]))
        committed = len(windows) + sample['drafts_accepted']
        require(0 <= sample['drafts_accepted'] - accepted_before_tail < windows[-1][1] and
            0 <= committed - sample['generated_tokens'] < windows[-1][1],
            'accepted-prefix positions/output count do not conserve windows')
        actual = int(marker[1])
        require((actual == sample['drafts_offered'] == sample['drafts_accepted'] == 0) if cap else actual > rounds,
            'no-draft control still executes MTP or reference does not draft')
        require(cap == 1 or actual - rounds == len(windows) - 1, 'reference draft round/window count differs')
        result.append(dict(target_windows=len(windows), first_T=1, subsequent_T=1 if cap else 4,
            draft_rounds_delta=actual - rounds, cumulative_draft_rounds=actual, prompt_draft=prompt))
        rounds = actual; previous = header.end()
    return result


def run(index):
    row = json.loads((ROOT / 'record.json').read_text()); admission(row)
    require(index in (0, 1) and not row['active'] and str(index) not in row['outcomes'] and
        (row['stage'] == 'prepared' if index == 0 else row['stage'] == 'partial' and row['outcomes']['0']['passed']),
        'ordered fresh process only; no duplicate/retry')
    before = dict(row); owner = None; outcome = dict(passed=False)
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(s.split(': ', 1)[1] for s in cursor_text.splitlines() if s.startswith('-- cursor: '))
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))
    try:
        owner = gate.native.supervision.Owner(ROOT / str(index) / 'owned', persist)
        row['stage'] = 'running'; persist()
        entry, out, err = owner.run('model', row['plans'][index]['command'], row['environments'][str(index)],
            ROOT / str(index), interaction=lambda io, _: interact(io, row, outcome, index), **row['budgets'])
        outcome['exit_code'] = entry.get('exit_code')
        require(gate.native.supervision.closed(entry) and entry['exit_code'] == 0, 'native model exit/ownership failed')
        text = err.read_text(errors='replace')
        outcome['activation'] = gate.validation.preload_evidence(text, entry['pid'])
        outcome['routes'] = route_evidence(text, outcome['requests'], index)
        warning = ROOT / str(index) / 'validation.log'
        outcome['validation_warning_bytes'] = warning.stat().st_size if warning.exists() else 0
        require(not outcome['validation_warning_bytes'], 'validation warnings')
        admission(before)
        if index == 1:
            model.model_pins(model.setup()['config'], row['model_assets'], full=True)
            outcome['final_full_model_hashes'] = 15
        outcome.update(passed=True, stdout_sha256=digest(out), stderr_sha256=digest(err), host_probe_accepted=True)
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner is not None: gate.validation.finish(outcome, owner, cursor, row['boot'])
        else: outcome.update(passed=False, ownership_closed=True)
        outcome['passed'] = bool(outcome.get('passed') and outcome.get('driver_after') == row['driver'])
        row['outcomes'][str(index)] = outcome
        row['ownership_closed'] = outcome['ownership_closed']
        row['stage'] = ('closed' if index else 'partial') if outcome['passed'] else 'failed'
        row['complete'] = row['passed'] = bool(index == 1 and outcome['passed'])
        persist(); write(RESULT, compact(row))
    print(json.dumps({k: outcome.get(k) for k in ('passed', 'ownership_closed', 'exit_code', 'fresh_faults', 'error')}), flush=True)
    return 0 if outcome['passed'] else 1


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['0'], ['1']), 'prepare|0|1')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); gate.native.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: sys.exit(run(int(sys.argv[1])))
