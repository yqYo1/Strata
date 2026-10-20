#!/usr/bin/env python3
"""Validate the existing FP16 draft-KV path after a guarded small warmup."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import prefill_status as status
import production_profile as profiling

c = status.c
gen, assets, guards, supervision = c.gen, c.assets, c.guards, c.supervision
P, B, A = c.P, c.B, c.A
ROOT = B / 'mtp-f16-prefill-v1'
RESULT = A / 'prefill-mtp-current.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def draft_rows(text):
    return [dict(cells=int(n), position=int(pos), first_needed=int(first), batch=int(batch))
        for n, pos, first, batch in re.findall(r'strata draft kv: (\d+) cells from (\d+) \(first needed (\d+)\), batch (\d+):', text)]


def current(record, full=False):
    build, proof, preload = status.qualification()
    guards.require(record['binary'] == build['binary'] and record['qualification'] == proof
        and record['native_validation_probe'] == preload, 'native/preload qualification changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(assets.fingerprint(pins) == record['benchmark_fingerprint'], 'controller/benchmark source changed')
    original, reference = status.original_plan()
    guards.require(reference == record['original_configuration'], 'original model/config changed')
    gen.runtime_assets_unchanged(dict(mode='engine-compare', runtime_asset_delta=original['runtime_asset_delta']))
    assets.model_pins(record['config'], original['model_assets'], full=full)
    for path, expected in record['generated_inputs'].items(): guards.require(digest(path) == expected, 'generated input changed')
    guards.require(assets.machine_identity() == record['machine_identity'], 'machine identity changed')


def prepare():
    guards.require(not RESULT.exists() and not ROOT.exists(), 'inspect existing MTP attempt; never blind retry')
    build, proof, preload = status.qualification()
    original, reference = status.original_plan()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    config = copy.deepcopy(original['config'])
    config['args'][config['args'].index('--expert-profile') + 1] = str(ROOT / 'expert-profile.bin')
    config.update(log=str(ROOT / 'engine.stderr'), env={})
    env = {k: v.replace(original['root'], str(ROOT)) for k, v in original['environment'].items()}
    env.update(STRATA_MEASUREMENT_ROOT=str(ROOT), STRATA_MTP_BATCH_F16='1', STRATA_DRAFT_TIMING='1',
        STRATA_TRACE='1', UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:warning;flush:warning;output:stderr')
    for name in ('STRATA_PREFILL_LAYER_MAJOR_PROFILE', 'STRATA_PREFILL_CPU_WAIT_PROFILE',
                 'STRATA_SYCL_PREFILL_COMMAND_PROFILE', 'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
                 'STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE'):
        env[name] = '0'
    baseline = json.loads((P / 'tools/sycl/measurement_environment.json').read_text())['environment']
    for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
                 'STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[name] == baseline[name], 'qualified safety setting changed')
    guards.require('LD_PRELOAD' not in env and env['ZES_ENABLE_SYSMAN'] == '0', 'private preload/Sysman-off required')
    assets.model_pins(config, original['model_assets'])
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    shutil.copyfile(P / 'data/expert-profile.bin', ROOT / 'expert-profile.bin')
    plan = dict(binary=build['binary'], native_validation_probe=preload)
    generated = {}
    for path, body in ((ROOT / 'config.json', config), (ROOT / 'plan.json', plan)):
        path.write_text(json.dumps(body, indent=2) + '\n'); generated[str(path)] = digest(path)
    generated[str(ROOT / 'expert-profile.bin')] = digest(ROOT / 'expert-profile.bin')
    guards.require(generated[str(ROOT / 'expert-profile.bin')] == original['profile_sha256'], 'expert profile changed')
    with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
        '--prefill-only', '--prefill-tokens', '32768', '--think', '--runs', '1', '--profile-request-gates',
        '--workload-id', 'mtp_f16_32k_20261021', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(ROOT / 'bench')]
    record = dict(stage='prepared', source_commit=revision, build_source_commit=build['source_commit'],
        binary=build['binary'], qualification=proof, native_validation_probe=preload,
        benchmark_fingerprint=assets.fingerprint(gen.benchmark_pins(revision)), original_configuration=reference,
        config=config, environment=env, generated_inputs=generated, command=command,
        machine_identity=assets.machine_identity(),
        source_audit='Existing STRATA_MTP_BATCH_F16 selects Q8_0 weight dequantization to FP16 plus oneMKL f16 GEMM, rather than Q8_1 activation dot-product prompt_product. Main verifier math unchanged; MTP KV numerics may change. Gemm scratch is carved before region and rebound before carve; draft temporary views stay inside separately bounded region. Source spans and ordered same-queue drain protect outputs, weights and host uploads. No native source, runtime/driver/safety setting or engine generation changes.',
        mechanism_reference='Closed39K trace:36 exact Q80Block prompt_product records,4.1735755s. This validates feasibility only; clean matched speed, acceptance, quality and full context remain separate.',
        gates='Before long HTTP request, require complete successful64..255-token thinking warmup, positive batched draft-KV completion, same owned native executable/PID/SID, actual immutable F16/safety environment and no runtime error. Abort without long dispatch on failure. Then one>=32768 fresh/read input,24-or-fewer thinking output, positive draft-KV cell/batch conservation, official journal/sample, exit0, owned closure, model/source/runtime checks and no fault. No automatic retry.',
        limits='Parameter-validated feasibility diagnostic, not a clean A/B comparison, numerical/task quality, sustained decode or physical262144/session/RESTORE qualification. No adoption from this short-output request.',
        retention='Keep useful request and route/closure/fault evidence. Retire normal logs after comparison. Two existing engine generations remain; no new binary.',
        active=False, ownership_closed=True, passed=False, adoption_qualified=False)
    RESULT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(stage='prepared', candidate='existing_MTP_BATCH_F16', new_engine_generation=False)))


def run(record):
    guards.require(RESULT.read_bytes() == gen.git('show', 'HEAD:' + str(RESULT.relative_to(P))), 'commit exact plan before action')
    guards.require(record['stage'] == 'prepared', 'inspect existing run; never retry')
    current(record, full=True)
    env = dict(record['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    boot = c.read_boot(); driver = guards.driver_guard()
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(row.split(': ', 1)[1] for row in before.splitlines() if row.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (ROOT / 'record.json').write_text(json.dumps(dict(record, commands=owner.commands if owner else []), indent=2) + '\n')
    owner = supervision.Owner(ROOT / 'owned', persist)
    def interact(io, leader):
        for expected in ('begin', 'end'):
            while True:
                line = io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '): break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=expected, phase='prefill-only', run=0), 'unexpected request gate')
            if expected == 'begin':
                start = json.loads((ROOT / 'engine-start.json').read_text()); native = start['native']
                identity = supervision.identity(native['pid'])
                guards.require(identity and identity['sid'] == leader['pid'] and identity['start_ticks'] == native['start_ticks']
                    and Path('/proc', str(native['pid']), 'exe').resolve() == Path(record['binary']['path']).resolve(), 'native owned identity differs')
                actual = dict(item.split(b'=', 1) for item in Path('/proc', str(native['pid']), 'environ').read_bytes().split(b'\0') if b'=' in item)
                for key in ('STRATA_MTP_BATCH_F16', 'STRATA_DRAFT_TIMING', 'STRATA_SYCL_PREFILL_SAME_QUEUE',
                            'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission'):
                    guards.require(actual.get(key.encode()) == env[key].encode(), 'native effective environment differs: ' + key)
                journal = next((ROOT / 'bench').glob('*/requests.jsonl'))
                rows = [json.loads(row) for row in journal.read_text().splitlines()]
                warm = rows[2]['sample']
                guards.require(rows[1]['phase'] == rows[2]['phase'] == 'warm-up/chat' and warm['ok'] and warm['stream_complete']
                    and rows[1]['thinking'] is True and 64 <= warm['read_tokens'] <= 255 and warm['generated_tokens'] > 0, 'small thinking warmup failed')
                text = (ROOT / 'engine.stderr').read_text(); batch = draft_rows(text)
                guards.require(batch and all(r['batch'] >= 64 for r in batch), 'small draft-KV batch not reached')
                guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed', text), 'small native error')
                record['small_warmup'] = dict(native=native, request=warm, batches=batch, effective_F16_environment_verified=True)
                persist()
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); guards.require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    record['stage'] = 'running'; persist()
    try:
        entry, _, _ = owner.run('bench', record['command'], env, P, interaction=interact,
            wall=1800, cpu=54000, rss_cap=96 << 30, text_cap=64 << 20, file_cap=64 << 20, total_cap=128 << 20)
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'owned benchmark failed')
        native = json.loads((ROOT / 'engine-exit.json').read_text())
        guards.require(native['exit_code'] == 0 and native['binary'] == record['binary']
            and native['args'] == ['--serve', *record['config']['args']], 'native exit/command failed')
        result = json.loads(next((ROOT / 'bench').glob('*/results.json')).read_text())
        samples = result['results']['prefill_only']['requests']
        guards.require(result['complete'] and len(samples) == 1 and not result['results']['failures'], 'official long request failed')
        sample = samples[0]
        guards.require(all(sample.get(k) is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed'))
            and sample['actual_prompt_tokens'] == sample['actual_read_tokens'] >= 32768 and sample['actual_reused_tokens'] == 0
            and 1 <= sample['generated_tokens'] <= 24, 'actual long request extent differs')
        text = (ROOT / 'engine.stderr').read_text(); rows = draft_rows(text)[len(record['small_warmup']['batches']):]
        guards.require(rows and sum(r['cells'] for r in rows) == sample['read_tokens'] - 1 and all(r['batch'] >= 64 for r in rows), 'long draft-KV cells/batches differ')
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', text), 'native runtime error')
        journal, _ = gen.journal_evidence(ROOT)
        guards.require(journal['valid'] and journal['complete'], 'official journal failed')
        activation = c.validation.preload_evidence(text, native['native']['pid'])
        guards.require(activation['loader'] == record['native_validation_probe']['loader']['path'], 'validation provider changed')
        assets.verify_pool(text, 5)
        record.update(request=sample, batches=rows, journal=journal, activation=activation)
        current(record, full=True)
        record['passed'] = True
    except BaseException as error:
        record.update(error=repr(error), passed=False)
    record['ownership_closed'] = owner.active is None and all(e.get('direct_child_reaped') and e.get('session_empty')
        and e.get('observation_complete') and e.get('ownership_status') == 'observed' for e in owner.commands)
    try:
        probe = profiling.fault_evidence(cursor, boot); record['fault_probe'] = probe
        guards.require(probe['fresh_faults'] == [] and probe['boot_unchanged'] and probe['driver_after'] == driver, 'fresh fault/platform changed')
    except Exception as error:
        record.update(final_probe_error=repr(error), passed=False)
    record['passed'] = bool(record['passed'] and record['ownership_closed'] and 'final_probe_error' not in record)
    record['stage'] = 'closed' if record['passed'] else 'failed'
    persist(); RESULT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(stage=record['stage'], passed=record['passed'], ownership_closed=record['ownership_closed'], error=record.get('error'))))
    return 0 if record['passed'] else 1


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use mtp_prefill.py prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare(); return 0
        return run(json.loads(RESULT.read_text()))


if __name__ == '__main__':
    raise SystemExit(main())
