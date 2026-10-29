#!/usr/bin/env python3
"""Bounded, separately planned QSA attribution after the clean comparison.

Own-fork controller under the repository LICENSE. Reuses the existing checked
observers and qualified binaries; it neither changes inference nor selects a
rejected route. Queue intervals include submission gaps and are not GPU service.
"""
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import qsa_pair_model_compare as comparison

d = comparison.diagnostic
m, P, B, RESULT = d.m, d.P, d.B, d.RESULT
require, digest, write = d.require, d.digest, d.write
ROOT = B / 'qsa-pair-model-profile-v1'
KEY = 'bounded_qsa_attribution'
SECTIONS = ('projections', 'kv_stage', 'kv_append_query', 'indexer_append',
            'selection', 'attention', 'output_projection')
COARSE = ('input', 'QSA', 'GDN', 'MoE', 'upload', 'store')
WALL = ('source', 'upload_submit', 'upload_drain', 'activation', 'input_submit_host',
        'input_submit_device', 'store_wall_host', 'store_wall_device',
        'store_drain_host', 'store_drain_device', 'chunk_drain', 'wrapper_drain')


def fields(line):
    pairs = [part.split('=', 1) for part in line.split()]
    require(all(len(v) == 2 for v in pairs), 'invalid profile field')
    require(len({v[0] for v in pairs}) == len(pairs), 'duplicate profile field')
    return dict(pairs)


def vector(value, width, integer=False):
    values = value.split(',')
    require(len(values) == width, 'profile vector width differs')
    result = [(int if integer else float)(v) for v in values]
    require(all(math.isfinite(v) and v >= 0 for v in result), 'invalid profile number')
    return result


def profile(log, n):
    """Reject partial, repeated and nonconserving observer windows.

    Preserve only aggregate times and coverage in the permanent result; 48-row
    raw output is consumed here rather than copied into Git.
    """
    chunks = (n + 8191) // 8192
    out = {}
    for name, prefix, version in (('coarse', 'strata lm profile: ', '2'),
                                  ('qsa', 'strata lm qsa profile: ', '1')):
        lines = [fields(v[len(prefix):]) for v in log.splitlines() if v.startswith(prefix)]
        require(len(lines) == 50 and 'v' in lines[0], 'missing or repeated profile window')
        head, rows = lines[0], lines[1:]
        require(head['v'] == version and head['complete'] == '1' and
                head['errors'] == head['first_status'] == '0' and
                head['denominator'] == 'started_through_checked_lease_retirement' and
                head['teardown_unmeasured'] == '1' and
                head['queue_intervals'] == 'NONEXCLUSIVE_including_host_submission_gaps',
                'incomplete or differently scoped observer')
        wrapper = vector(head['wrapper_ms'], 1)[0]
        require(wrapper > 0 and [int(r['layer']) for r in rows] == [*range(48), -1],
                'missing, unordered or duplicate observer layer')
        parsed = []
        for raw in rows:
            row = dict(layer=int(raw['layer']))
            for field, width, integer in (
                    ('queue_ms', 7 if name == 'qsa' else 6, False),
                    ('samples', 7 if name == 'qsa' else 6, True),
                    ('host_wall_ms' if name == 'qsa' else 'wall_ms',
                     7 if name == 'qsa' else 12, False)):
                row[field] = vector(raw[field], width, integer)
            scalar = ('expected_chunks', 'cap', 'expected_queries', 'dispatch_chunks',
                      'query_total', 'matrix_chunks', 'fallback_chunks',
                      'matrix_queries', 'fallback_queries', 'staged_chunks') if name == 'qsa' else ('upload_bytes',)
            for key in scalar:
                row[key] = vector(raw[key], 1, True)[0]
            if name == 'coarse':
                for key, integer in (('input_bytes_host_device', True), ('store_bytes_host_device', True),
                                     ('input_queue_host_device_ms', False), ('store_queue_host_device_ms', False)):
                    row[key] = vector(raw[key], 2, integer)
            parsed.append(row)
        layers, total = parsed[:-1], parsed[-1]
        for key in total.keys() - {'layer', 'cap'}:
            value = total[key]
            if isinstance(value, list):
                expected = [sum(r[key][k] for r in layers) for k in range(len(value))]
                require(all(math.isclose(a, b, rel_tol=1e-5, abs_tol=.001)
                            if isinstance(a, float) else a == b for a, b in zip(value, expected)),
                        'observer aggregate vector does not conserve')
            else:
                require(value == sum(r[key] for r in layers), 'observer aggregate count does not conserve')
        if name == 'qsa':
            require(head['sections'].split(',') == list(SECTIONS) and
                    head['granularity'] == 'QSA_LAYER_CHUNK' and head['markers_included'] == '1' and
                    head['kv_stage_noop_sample'] == '1' and head['dispatch'] == 'ACTUAL_matrix_used_result' and
                    head['queries'] == 'SUM_T_not_request_tokens' and
                    head['host_wall'] == 'NONEXCLUSIVE_submission_and_helper_waits_not_CPU_or_GPU_service',
                    'QSA observer semantics differ')
            active = []
            for r in layers:
                if r['expected_chunks']:
                    active.append(r['layer'])
                    require(r['expected_chunks'] == r['dispatch_chunks'] == r['matrix_chunks'] == chunks and
                            r['expected_queries'] == r['query_total'] == r['matrix_queries'] == n and
                            r['fallback_chunks'] == r['fallback_queries'] == 0 and r['cap'] == 2051 and
                            r['staged_chunks'] == 0 and r['samples'] == [chunks] * 7,
                            'QSA actual dispatch/query/sample coverage differs')
                else:
                    require(all(v == 0 if not isinstance(v, list) else all(x == 0 for x in v)
                                for key, v in r.items() if key != 'layer'), 'non-QSA observer row is nonzero')
            require(active == list(range(3, 48, 4)) and total['cap'] == 2051, 'QSA layer/cap coverage differs')
            out[name] = dict(wrapper_ms=wrapper, active_layers=active, aggregate=total)
        else:
            require(head['queue_order'].split(',') == list(COARSE) and
                    head['wall_order'].split(',') == list(WALL) and head['bytes'] == 'REQUESTED_not_wire',
                    'coarse observer semantics differ')
            # This admitted Mode10 route runs two81920-row MoE windows, not
            # one FFN per8192-token chunk. Its private bank writes in place:
            # residual-store callbacks are deliberately absent from this observer.
            for r in layers:
                qsa = r['layer'] % 4 == 3
                require(r['samples'] == [0 if r['layer'] == 0 else chunks,
                        chunks if qsa else 0, 0 if qsa else chunks, (n + 81919) // 81920, 1, 0],
                        'coarse actual chunk/window coverage differs')
                require(all(count or duration == 0 for count, duration in zip(r['samples'], r['queue_ms'])),
                        'coarse unsampled interval is nonzero')
            out[name] = dict(wrapper_ms=wrapper, aggregate=total)
    require(math.isclose(out['coarse']['wrapper_ms'], out['qsa']['wrapper_ms'], rel_tol=1e-5),
            'observer denominators disagree')
    require([r for r in out['qsa']['active_layers']] and
            out['qsa']['aggregate']['expected_chunks'] == 12 * chunks,
            'QSA total chunk coverage differs')
    return out


def admitted():
    record, built, rt, manifest, model = comparison.qualification()
    old = json.loads((comparison.ROOT / 'record.json').read_text())
    require(record[m.KEY][comparison.KEY] == comparison.compact(old) and
            old['complete'] and old['passed'] and old['ownership_closed'] and not old['active'] and
            len(old['processes']) == 6 and all(p['passed'] for p in old['processes']),
            'finish every frozen clean comparison process first')
    review = m.runtime.validation.committed(RESULT.with_name('research-index.json'),
        ('pair_model_mechanism_gap_decision_20261029', 'native_review'))
    require(review['verdict'] == 'accept_with_source_semantics_correction', 'bounded observer review required')
    return record, built, rt, manifest, model, old


def environment(root, rt, arm):
    env = comparison.environment(root, rt, arm)
    env.update(STRATA_PREFILL_LAYER_MAJOR_PROFILE='1', STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1')
    require(all(not env.get(k) or env[k] == '0' for k in
                ('STRATA_SEL_OVERLAP', 'STRATA_IDX_FP16_CHECK', 'STRATA_QSA_DUMP')),
            'selection debug observers must remain disabled')
    return env


def compact(row):
    return {k:v for k, v in row.items() if k not in ('sources', 'model_assets')}


def prepare():
    record, _, rt, _, model, old = admitted()
    require(KEY not in record[m.KEY] and not ROOT.exists(), 'fresh diagnostic only')
    revision = d.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = d.pins(revision)
    assets = d.models.model_pins(model['configuration'], model['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    plans = []
    for index, arm in enumerate(('original', 'candidate')):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        config = copy.deepcopy(model['configuration']); config['log'] = str(root / 'engine.stderr')
        binary = old['plans'][index]['binary']
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=binary, native_validation_probe=old['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
            '--prefill-only', '--prefill-tokens', str(comparison.MINIMUM), '--think', '--runs', '1',
            '--profile-request-gates', '--workload-id', 'native_gdn_20261018',
            '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]
        plans.append(dict(index=index, arm=arm, root=str(root), binary=binary, command=command,
            expected_placement=old['processes'][index]['request_placement'][0],
            generated={n:digest(root / n) for n in ('plan.json', 'config.json')},
            environment_sha256=d.fingerprint(environment(root, rt, arm))))
    row = dict(stage='prepared', source_commit=revision, sources=pins, model_assets=assets,
        runtime_sha256=digest(m.runtime.RESULT), boot=rt['boot'], profile=old['profile'],
        native_validation_probe=old['native_validation_probe'], plans=plans, processes=[],
        request_sha256=old['request_sha256'][0], expected_engine_rows=old['processes'][0]['actual_engine_rows'][0],
        clean_comparison_decision=old['decision'], budgets=old['budgets'],
        plan=dict(question='Localize the next tuning opportunity in QSA composite versus other long-prefill work,after the completed clean screen. Not historical regression archaeology.',
            order='One original then one candidate fresh process;one identical117K first-long request each after the same short warmup. No retries or performance-based exclusion.',
            observers='Both existing coarse v2 and fine QSA v1 flags before exec;fixed2MiB host admission. Exact already qualified binaries,no native rebuild.',
            acceptance='Exact workload/source/binary/runtime/provider/owner/fault/validation/placement and complete,error-free48-layer plus aggregate sample/query/dispatch conservation. Diagnostic closure is not an adoption gate.',
            interpretation='QSA attention interval encloses prefix pack+union+attention. Queue intervals include submission gaps;host wall is not CPU time/GPU service. No exclusive sums,clean speed,physical traffic or hardware saturation claims.',
            next='If QSA composite increment cannot account for the clean loss,stop pair-kernel work. If implicated,only a separately admitted representative real-data Page4 probe can select a tile/union/dispatch change.',
            retention='Keep aggregates,coverage,individual request sample and exact identity. Retire raw per-layer output after verification. Candidate build retained solely through this imminent diagnostic;review retirement at closure.'),
        complete=False, passed=False, active=False, ownership_closed=True)
    write(ROOT / 'record.json', row)
    record[m.KEY][KEY] = compact(row); write(RESULT, record)
    print('Prepared two bounded existing-observer processes;commit before execution.', flush=True)


def run(index):
    record, _, rt, manifest, _, _ = admitted()
    row = json.loads((ROOT / 'record.json').read_text())
    require(record[m.KEY][KEY] == compact(row) and index == len(row['processes']) and index in (0, 1) and
            all(p['passed'] for p in row['processes']), 'committed ordered diagnostic required')
    require(d.pins(row['source_commit']) == row['sources'] and digest(m.runtime.RESULT) == row['runtime_sha256'],
            'diagnostic source/runtime changed')
    plan = row['plans'][index]; root = Path(plan['root']); env = environment(root, rt, plan['arm'])
    config = json.loads((root / 'config.json').read_text())
    d.models.model_pins(config, row['model_assets'])
    require(d.models.file_identity(row['profile']['path']) == row['profile']['identity'] and
            digest(row['profile']['path']) == row['profile']['sha256'], 'placement profile changed')
    require(not (root / 'execution.json').exists() and
            d.fingerprint(env) == plan['environment_sha256'] and
            all(digest(root / n) == sha for n, sha in plan['generated'].items()) and
            digest(plan['binary']['path']) == plan['binary']['sha256'], 'diagnostic plan/binary changed')
    m.runtime.guards.no_model_owners([])
    cursor = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15), re.M)
    require(len(cursor) == 1, 'fresh fault cursor required')
    outcome = dict(index=index, arm=plan['arm'], complete=False, passed=False)
    owner = None
    def persist():
        write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def interact(io, entry):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '): break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='prefill-only', run=0), 'unexpected diagnostic request gate')
            delta = [a-b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
            claim = json.loads((root / 'engine-start.json').read_text()); native = claim['native']
            live = m.runtime.supervision.identity(native['pid'])
            require(live and live['start_ticks'] == native['start_ticks'] and live['sid'] == entry['pid'] and
                    str(Path('/proc', str(native['pid']), 'exe').resolve()) == str(Path(plan['binary']['path']).resolve()),
                    'owned native identity differs')
            expected = comparison.engine_once.native_environment(dict(native_validation_probe=row['native_validation_probe']), env)
            actual = dict(v.split(b'=', 1) for v in Path('/proc', str(native['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
            require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
                    set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                    'native pre-exec environment differs')
            mapped = m.runtime.executable_mappings(Path('/proc', str(native['pid']), 'maps').read_text())
            require(mapped <= set(rt['runtime_files']) | {str(Path(plan['binary']['path']).resolve())} and
                    {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
                     env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native provider mapping differs')
            allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
            d.python_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py', manifest['selected'], env, allowed, plan['command'][2:])
            require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size,
                    'parameter-validation warning/error')
            print(f'Profile {plan["arm"]} request {stage}', flush=True); io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'model teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = m.runtime.supervision.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        require(m.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'diagnostic process failed')
        files = list((root / 'bench').glob('*/results.json')); require(len(files) == 1, 'missing/duplicate official result')
        result = json.loads(files[0].read_text()); samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == 1, 'incomplete diagnostic')
        sample = samples[0]
        require(all(sample[k] for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
                sample['request_sha256'] == row['request_sha256'] and sample['actual_reused_tokens'] == 0 and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == row['expected_engine_rows'] + 1 and
                sample['generated_tokens'] == 24 and sample['prompt_tps_source'] == 'engine' and sample['prompt_ms'] > 10000,
                'diagnostic workload/clock differs')
        require(d.client.verify_request_journal(files[0].parent)['valid'], 'request journal invalid')
        require(d.pins(row['source_commit']) == row['sources'] and digest(m.runtime.RESULT) == row['runtime_sha256'],
                'source/runtime changed during diagnostic')
        end = d.gen.native_end(root, dict(binary=plan['binary'], native_validation_probe=row['native_validation_probe']),
            dict(config=config), row['boot'], entry['pid'])
        log = (root / 'engine.stderr').read_text()
        proof = m.runtime.validation.preload_evidence(log, end['native']['pid'])
        require(proof['loader'] == row['native_validation_probe']['loader']['path'] and
                not any(t in log for t in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')),
                'runtime/validation failure')
        engine_rows = [int(v) for v in re.findall(
            r'strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 residual_dtype=bf16', log)]
        require(engine_rows == [row['expected_engine_rows']], 'actual engine extent differs')
        placement = comparison.pair_model_placement.validate(log, engine_rows, plan['arm'], env)
        require(placement == [plan['expected_placement']], 'profile changed actual clean-arm placement')
        pair = re.findall(r'strata QSA pair complete: calls=(\d+) queries=(\d+) layers=(\d+) chunks=(\d+) metadata_bytes=(\d+) final_drain=checked', log)
        n = engine_rows[0]; chunks = (n + 8191) // 8192
        require(pair == ([(str(12*chunks), str(12*n), '12', str(chunks), '84025344')] if plan['arm'] == 'candidate' else []),
                'actual pair dispatch differs')
        observed = profile(log, n)
        d.models.model_pins(config, row['model_assets'], full=index == 1)
        keep = ('actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens', 'generated_tokens',
                'prompt_ms', 'prompt_tps', 'ttft_s', 'total_s', 'request_sha256')
        outcome.update(complete=True, sample={k:sample[k] for k in keep}, actual_engine_rows=n,
            request_placement=placement[0], observer=observed, validation=proof, native_exit_code=0,
            clean_speed_qualified=False, inference_adopted=False, decode_qualified=False, physical262144_qualified=False)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner: m.runtime.validation.finish(outcome, owner, cursor[0], row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']); persist()
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed', ownership_closed=outcome.get('ownership_closed', False),
            active=bool(owner and owner.active))
        if outcome['passed'] and len(row['processes']) == 2:
            row.update(stage='closed', complete=True, passed=True)
        write(ROOT / 'record.json', row); record[m.KEY][KEY] = compact(row); write(RESULT, record)
    print(json.dumps({k:outcome.get(k) for k in ('arm', 'passed', 'complete', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], 'bounded QSA attribution failed')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if sys.argv[1] == 'prepare' else run(int(sys.argv[2]))
