#!/usr/bin/env python3
"""Main-owned matched MTP draft-KV comparison using the existing qualified engine.

The original one-shot pilot is recoverable from dbea9dd0e. This controller uses
fresh plans/processes; it never changes the pilot's original criteria or outcome.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import shutil
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import prefill_status as status
import production_profile as profiling
import resource_engine as resources

c = status.c
gen, assets, guards, supervision = c.gen, c.assets, c.guards, c.supervision
P, B, A = c.P, c.B, c.A
RESULT = A / 'prefill-mtp-current.json'
ROOT = B / 'mtp-f16-comparison-v1'
ORDER = (0, 1, 1, 0, 0, 1)
PAIRS = ((0, 1), (3, 2), (4, 5))
WORKLOAD = 'regular-code'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def draft_rows(text):
    pattern = (r'strata draft kv: (\d+) cells from (\d+) \(first needed (\d+)\), batch (\d+): '
               r'drafter idle ([\d.]+) ms, the batches ([\d.]+) ms, all ([\d.]+) ms')
    return [dict(cells=int(n), position=int(pos), first_needed=int(first), batch=int(batch),
                 idle_ms=float(idle), batch_ms=float(batch_ms), all_ms=float(all_ms))
            for n, pos, first, batch, idle, batch_ms, all_ms in re.findall(pattern, text)]


def request_evidence(text, samples):
    """Bind draft work to every native request, including all four warmups."""
    headers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', text, re.M))
    guards.require(len(headers) == len(samples), 'missing/extra native request footer')
    previous = 0; evidence = []
    for header, sample in zip(headers, samples):
        guards.require(tuple(map(int, header.groups())) == (sample['prompt_tokens'], sample['reused_tokens'],
            sample['read_tokens'], sample['generated_tokens']) and sample['reused_tokens'] == 0,
            'native/journal request binding differs')
        rows = draft_rows(text[previous:header.start()]); position = 0
        for row in rows:
            guards.require(row['position'] == position and row['cells'] > 0 and row['batch'] >= 64
                and row['position'] <= row['first_needed'] < row['position'] + row['cells']
                and all(math.isfinite(row[k]) and row[k] >= 0 for k in ('idle_ms', 'batch_ms', 'all_ms')),
                'invalid/missing batched draft-KV span')
            position += row['cells']
        guards.require(rows and position == sample['read_tokens'] - 1, 'draft-KV cells do not conserve request')
        evidence.append(dict(read_tokens=sample['read_tokens'], draft_cells=position,
            batch_count=len(rows), batch_ms=round(sum(row['batch_ms'] for row in rows), 4)))
        previous = header.end()
    guards.require(not draft_rows(text[previous:]), 'unbound trailing draft batch')
    return evidence


def comparison_plan():
    return dict(order=list(ORDER), pairs=[list(p) for p in PAIRS], workload=WORKLOAD,
        question='Does existing FP16 GEMM for MTP prefill improve matched long-input PP without hurting sustained decode or draft acceptance?',
        constants='Same current3b native/eight fixtures, model/tokenizer/expert profile,131072 residentINT8 context,8192 prefill,1256 expert slots,5 CPU workers,spec4; official bench.sh long-generation-only regular-code, thinking enabled,16384 output allowance. Four official warmups then one fresh>=32K request per independent process. Only MTP_BATCH_F16 differs. Immutable validation/trace/draft-timing observers and qualified safety settings match.',
        admission='Before each long HTTP dispatch require all four successful official warmups, native PID/SID/executable/effective environment, positive draft batching and exact per-request cell conservation, no native errors. This prospectively repairs the pilot gate. Final request must have>=1024 generated tokens including thought plus>=512 answer characters,>=256 offered drafts, complete journal/native/route/ownership/source/model/runtime/fault checks. Stop on first failure; never retry/discard.',
        decisions='All three prefill candidate/base ratios>1 and geomean>1.02 qualifies faster screen; all<1 and geomean<1/1.02 slower screen; otherwise inconclusive. Decode uses1.05 and1/1.05 with the same all-pair direction rule and generation-count ratios within1/1.05..1.05. Otherwise decode is inconclusive. Report acceptance change as candidate effect, never require equal acceptance or text. Nonregression is geomean>=0.98 and each ratio>=0.95 separately for PP and TG. Keep all samples/spreads. Strong-MTP classification requires all acceptance>=0.60. No automatic extra repeats or adoption.',
        limits='Matched diagnostic with parameter validation, trace and draft timer (entry synchronization), not clean speed or observer-overhead evidence. Each sample is first long request after four warmups, not subsequent-long steady state. Weak-MTP,agent-session,task quality,physical262144/session/RESTORE and70TG remain pending.')


def invariant(plan):
    value = copy.deepcopy(dict(config=plan['config'], environment=plan['environment']))
    value['environment']['STRATA_MTP_BATCH_F16'] = '<arm>'
    return hashlib.sha256(json.dumps(value, sort_keys=True).replace(plan['root'], '<root>').encode()).hexdigest()


def current(record, full=False):
    build, proof, preload = status.qualification()
    guards.require(record['binary'] == build['binary'] and record['qualification'] == proof
        and record['native_validation_probe'] == preload, 'native/preload qualification changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(assets.fingerprint(pins) == record['benchmark_fingerprint'], 'controller/benchmark input changed')
    original, reference = status.original_plan()
    guards.require(reference == record['original_configuration'], 'original configuration changed')
    gen.runtime_assets_unchanged(dict(mode='engine-compare', runtime_asset_delta=original['runtime_asset_delta']))
    assets.model_pins(record['plans'][0]['config'], original['model_assets'], full=full)
    for name, sha in original['tokenizer_assets'].items(): guards.require(digest(name) == sha, 'tokenizer changed')
    for plan in record['plans']:
        guards.require(invariant(plan) == record['invariant_signature'], 'unexpected arm difference')
        for path, expected in plan['generated_inputs'].items(): guards.require(digest(path) == expected, 'generated input changed')
    guards.require(assets.machine_identity() == record['machine_identity'], 'machine changed')


def prepare():
    container = gen.committed(RESULT)
    guards.require(container['stage'] == 'closed' and container['passed'] and container['ownership_closed']
        and not container['active'] and 'comparison' not in container and not ROOT.exists(), 'fresh comparison only')
    build, proof, preload = status.qualification(); original, reference = status.original_plan()
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    # Source/benchmark must already be committed, before generating the frozen plan.
    pins = gen.benchmark_pins(source); ROOT.mkdir(mode=0o700); plans = []
    baseline = json.loads((P / 'tools/sycl/measurement_environment.json').read_text())['environment']
    for cohort, arm in enumerate(ORDER):
        root = ROOT / str(cohort); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        config = copy.deepcopy(original['config'])
        config['args'][config['args'].index('--expert-profile') + 1] = str(root / 'expert-profile.bin')
        config.update(log=str(root / 'engine.stderr'), env={})
        env = {k: v.replace(original['root'], str(root)) for k, v in original['environment'].items()}
        env.update(STRATA_MEASUREMENT_ROOT=str(root), STRATA_MTP_BATCH_F16=str(arm), STRATA_DRAFT_TIMING='1',
            STRATA_TRACE='1', UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:warning;flush:warning;output:stderr', TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'))
        for name in ('STRATA_PREFILL_LAYER_MAJOR_PROFILE', 'STRATA_PREFILL_CPU_WAIT_PROFILE',
                     'STRATA_SYCL_PREFILL_COMMAND_PROFILE', 'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
                     'STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE'): env[name] = '0'
        for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission', 'STRATA_SYCL_PREFILL_SAME_QUEUE'):
            guards.require(env[name] == baseline[name], 'qualified safety changed')
        guards.require('LD_PRELOAD' not in env and env['ZES_ENABLE_SYSMAN'] == '0', 'private validation/Sysman-off required')
        shutil.copyfile(P / 'data/expert-profile.bin', root / 'expert-profile.bin')
        guards.require(digest(root/'expert-profile.bin') == original['profile_sha256'], 'expert profile changed')
        native = dict(binary=build['binary'], native_validation_probe=preload); generated = {}
        for path, body in ((root/'config.json', config), (root/'plan.json', native)):
            path.write_text(json.dumps(body, indent=2)+'\n'); generated[str(path)] = digest(path)
        generated[str(root/'expert-profile.bin')] = original['profile_sha256']
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
            '--long-generation-only', '--long-generation-workload', WORKLOAD, '--think', '--runs', '1',
            '--profile-request-gates', '--workload-id', 'mtp_f16_matched_20261021', '--fail-fast-engine-error',
            '--ready-timeout', '600', '--out', str(root/'bench')]
        plans.append(dict(cohort=cohort, arm=arm, root=str(root), config=config, environment=env,
            generated_inputs=generated, command=command))
    record = dict(stage='prepared', source_commit=source, build_source_commit=build['source_commit'],
        binary=build['binary'], qualification=proof, native_validation_probe=preload,
        benchmark_fingerprint=assets.fingerprint(pins), original_configuration=reference, machine_identity=assets.machine_identity(),
        plan=comparison_plan(), plans=plans, invariant_signature=invariant(plans[0]), processes=[],
        active=False, ownership_closed=True, passed=False, complete=False, adoption_qualified=False,
        budgets=dict(wall=2400, cpu=54000, rss_cap=96<<30, text_cap=64<<20, file_cap=64<<20, total_cap=128<<20),
        retention='Keep individual samples, per-request work/route checks, source/plan/environment and decision. After closure retire normal raw output/caches. Existing native generations/fixtures remain; no rebuild.')
    current(record, full=True); container['comparison'] = record
    RESULT.write_text(json.dumps(container, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(stage='prepared', order=list(ORDER), new_native_generation=False)))


def summarize(rows):
    guards.require(len(rows) == 6 and [r['arm'] for r in rows] == list(ORDER)
        and all(r['passed'] and r['ownership_closed'] for r in rows), 'complete frozen cohort required')
    guards.require(len({(r['native']['pid'], r['native']['start_ticks']) for r in rows}) == 6, 'native process reused')
    guards.require(len({(r['request']['request_sha256'], r['request']['read_tokens']) for r in rows}) == 1, 'input mismatch')
    pp, tg, lengths, acceptance = [], [], [], []
    for left, right in PAIRS:
        base, candidate = rows[left]['request'], rows[right]['request']
        pp.append((candidate['read_tokens']/candidate['prompt_ms'])/(base['read_tokens']/base['prompt_ms']))
        tg.append((candidate['generated_tokens']/candidate['decode_ms'])/(base['generated_tokens']/base['decode_ms']))
        lengths.append(candidate['generated_tokens']/base['generated_tokens'])
        acceptance.append(candidate['acceptance_rate'] - base['acceptance_rate'])
    geo = lambda values: math.exp(statistics.mean(map(math.log, values)))
    def decision(values, threshold):
        return ('candidate_faster_screen' if all(x > 1 for x in values) and geo(values) > threshold else
                'candidate_slower_screen' if all(x < 1 for x in values) and geo(values) < 1/threshold else 'inconclusive')
    spread = {str(arm): {key: dict(median=statistics.median(v), minimum=min(v), maximum=max(v), sample_sd=statistics.stdev(v))
        for key in ('prompt_tps', 'decode_tps', 'acceptance_rate')
        for v in [[r['request'][key] for r in rows if r['arm'] == arm]]} for arm in (0, 1)}
    return dict(prefill_ratios=pp, prefill_geomean=geo(pp), prefill_decision=decision(pp, 1.02),
        decode_ratios=tg, decode_geomean=geo(tg), decode_decision=decision(tg, 1.05) if all(1/1.05 <= x <= 1.05 for x in lengths) else 'inconclusive_generation_length',
        generation_length_ratios=lengths, acceptance_changes=acceptance,
        prefill_nonregression=geo(pp)>=0.98 and min(pp)>=0.95, decode_nonregression=geo(tg)>=0.98 and min(tg)>=0.95,
        strong_MTP_confirmed=all(r['request']['acceptance_rate'] >= .60 for r in rows), spread=spread, adoption_qualified=False)


def run(cohort):
    container = gen.committed(RESULT); record = container['comparison']
    guards.require(record['stage'] in ('prepared', 'collecting') and len(record['processes']) == cohort
        and all(r['passed'] for r in record['processes']) and not record['active'] and record['ownership_closed'], 'ordered fresh processes only')
    current(record, full=cohort == 0); plan = record['plans'][cohort]; root = Path(plan['root'])
    guards.require(not (root/'record.json').exists(), 'native attempt already exists; no retry')
    env = plan['environment']; boot = c.read_boot(); driver = guards.driver_guard()
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15)
    cursor = next(line.split(': ',1)[1] for line in before.splitlines() if line.startswith('-- cursor: '))
    row = dict(cohort=cohort, arm=plan['arm'], passed=False, ownership_closed=False); owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(row, active=record['active'], commands=owner.commands if owner else []), indent=2)+'\n')
    def interact(io, leader):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(1800)
                if line.startswith('STRATA_PROFILE_GATE '): break
            guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=stage, phase='long-generation/'+WORKLOAD, run=0), 'request gate differs')
            if stage == 'begin':
                native = json.loads((root/'engine-start.json').read_text())['native']; identity = supervision.identity(native['pid'])
                guards.require(identity and identity['sid'] == leader['pid'] and identity['start_ticks'] == native['start_ticks']
                    and Path('/proc',str(native['pid']),'exe').resolve() == Path(record['binary']['path']).resolve(), 'native ownership differs')
                actual = dict(part.split(b'=',1) for part in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in part)
                keys = ('STRATA_MTP_BATCH_F16','STRATA_DRAFT_TIMING','STRATA_TRACE','STRATA_SYCL_PREFILL_SAME_QUEUE',
                        'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','UR_ENABLE_LAYERS','ZES_ENABLE_SYSMAN')
                guards.require(all(actual.get(k.encode()) == env[k].encode() for k in keys), 'effective native environment differs')
                journal, = (root/'bench').glob('*/requests.jsonl')
                events = [json.loads(line) for line in journal.read_text().splitlines()]
                guards.require(len(events) == 10 and events[-1]['kind'] == 'start' and events[-1]['phase'] == 'long-generation/'+WORKLOAD, 'warmup/long-start journal extent differs')
                warm = gen.warmup_reconciliation(events[:-1], workloads=())
                samples = [e['sample'] for e in events[:-1] if e['kind'] == 'result']
                guards.require(64 <= samples[0]['read_tokens'] <= 255 and samples[0]['generated_tokens'] > 0, 'small warmup extent differs')
                text = (root/'engine.stderr').read_text(); row['warmup_work'] = request_evidence(text, samples)
                guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', text), 'warmup native error')
                row.update(native=native, successful_warmups=warm['successful_warmups'], effective_environment_verified=True)
                persist()
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); guards.require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    owner = supervision.Owner(root/'owned', persist); record['stage'] = 'collecting'; persist()
    try:
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **record['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code'] == 0, 'owned official benchmark failed')
        result = json.loads(next((root/'bench').glob('*/results.json')).read_text())
        suite = result['results']['long_generation']; samples = gen.collect_samples(suite)
        # Preserve failed/ineligible returned samples before qualification.
        row['request'] = samples[0] if samples else {}
        gen.require_complete_generation(suite, samples, (WORKLOAD,))
        guards.require(samples[0].get('prompt_tps_source') == 'engine' and
            isinstance(samples[0].get('prompt_ms'), (int, float)) and
            not isinstance(samples[0]['prompt_ms'], bool) and math.isfinite(samples[0]['prompt_ms'])
            and samples[0]['prompt_ms'] > 0, 'native prefill clock missing/invalid')
        guards.require(result['complete'] and not result['results']['failures'], 'official result incomplete')
        journal, journal_rows = gen.journal_evidence(root)
        guards.require(journal['valid'] and journal['complete'] and len(journal_rows) == 1
            and journal_rows[0]['request_sha256'] == samples[0]['request_sha256'], 'official request journal differs')
        resources.require_journal_binding(samples, journal_rows)
        warm = gen.warmup_evidence(root, journal)
        native = json.loads((root/'engine-exit.json').read_text())
        guards.require(native['exit_code'] == 0 and native['boot_id'] == boot and native['binary'] == record['binary']
            and native['native'] == row['native'] and native['args'] == ['--serve', *plan['config']['args']], 'native exit/command differs')
        text = (root/'engine.stderr').read_text(); assets.verify_pool(text, 5)
        activation = c.validation.preload_evidence(text, native['native']['pid'])
        guards.require(activation['loader'] == record['native_validation_probe']['loader']['path'], 'validation provider changed')
        path, = (root/'bench').glob('*/requests.jsonl')
        events = [json.loads(line) for line in path.read_text().splitlines()]
        work = request_evidence(text, [e['sample'] for e in events if e['kind'] == 'result'])
        headers = list(re.finditer(r'^strata serve: prompt .*$', text, re.M))
        long_text = text[headers[-2].end():headers[-1].start()]
        cells = samples[0]['read_tokens'] - 1
        route = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 '
            r'residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) '
            r'residual_gpu=(\d+) residual_host=0 residual_staging=0 ', long_text, re.M)
        guards.require(len(route) == 1 and tuple(map(int, route[0])) == (cells, cells*20480, cells, cells*20480)
            and len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='
                + str(cells) + r' mixed_fp16_bytes=0$', long_text, re.M)) == 1,
            'actual all-GPU BF16/mode10 long route differs')
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', text), 'native runtime error')
        if record['processes']:
            first = record['processes'][0]['request']
            guards.require((first['request_sha256'], first['read_tokens']) == (samples[0]['request_sha256'], samples[0]['read_tokens']), 'matched input differs')
        current(record, full=cohort == 5)
        row.update(passed=True, long_work=work[-1], actual_GPU_BF16_rows=cells, actual_MoE_mode=10,
            journal_valid=True, native_exit_code=0, successful_warmups=warm['successful_warmups'])
    except BaseException as error:
        row.update(passed=False, error=repr(error))
        try:
            journal, samples = gen.journal_evidence(root)
            if samples: row['request'] = samples[0]
            row['journal_recovery'] = journal
        except Exception as recovery: row['journal_recovery_error'] = repr(recovery)
    row['ownership_closed'] = owner.active is None and all(supervision.closed(e) for e in owner.commands)
    try:
        row['fault_probe'] = profiling.fault_evidence(cursor, boot)
        guards.require(not row['fault_probe']['fresh_faults'] and row['fault_probe']['boot_unchanged']
            and row['fault_probe']['driver_after'] == driver, 'fresh fault/platform change')
    except Exception as error: row.update(passed=False, fault_probe_error=repr(error))
    row['passed'] = bool(row['passed'] and row['ownership_closed']); record['processes'].append(row)
    record.update(ownership_closed=row['ownership_closed'], passed=all(r['passed'] for r in record['processes']),
        complete=cohort == 5 or not row['passed'], stage='closed' if cohort == 5 else 'collecting' if row['passed'] else 'failed')
    if cohort == 5 and record['passed']:
        try: record['summary'] = summarize(record['processes'])
        except Exception as error: record.update(passed=False, stage='failed', summary_error=repr(error))
    persist(); RESULT.write_text(json.dumps(container, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(cohort=cohort, arm=plan['arm'], passed=row['passed'], ownership_closed=row['ownership_closed'], error=row.get('error'))))
    return 0 if row['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run'))
    parser.add_argument('cohort', nargs='?', type=int, choices=range(6), default=0)
    args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([])
        return prepare() if args.action == 'prepare' else run(args.cohort)


if __name__ == '__main__':
    raise SystemExit(main())
