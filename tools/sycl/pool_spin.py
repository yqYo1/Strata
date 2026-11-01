#!/usr/bin/env python3
"""Bounded same-engine CPU idle-wait discriminator; repository LICENSE.

Main-authored from this fork's owned generation supervision. This changes one
immutable startup setting, not inference math. A code-only diagnostic does not
qualify clean speed, either MTP category, independent phases or model adoption.
"""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import mtp_prefill as generation

model, comparison, shared, build = generation.model, generation.comparison, generation.shared, generation.build
P, B, RESULT = generation.P, generation.B, generation.RESULT
ROOT, KEY = model.gdn16.ROOT / 'pool-spin-diagnostic-v1', 'gdn16_pool_spin_diagnostic_v1'
SOURCE = 'tools/sycl/pool_spin.py'
WORKLOADS = ('regular-code',)
WORKLOAD_ID = 'cpu_idle_wait_code_20261102'
ARMS = (('reference', 20000), ('candidate', 0))
require, digest, write, compact = generation.require, generation.digest, generation.write, generation.compact


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def basis(revision):
    prior = json.loads((generation.ROOT / 'record.json').read_text())
    require(prior['passed'] and prior['complete'] and prior['stage'] == 'closed'
            and prior['ownership_closed'] and not prior['active']
            and len(prior['processes']) == 2 and all(p['passed'] for p in prior['processes']),
            'closed two-arm current-engine long-generation diagnostic required')
    providers, manifest = generation.current(prior)
    recovered = {n: hashlib.sha256(raw).hexdigest()
                 for n, raw in shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = model.pinned_sources(revision)
    require(recovered == prior['sources'] and set(current) - set(recovered) == {SOURCE}
            and not (set(recovered) - set(current))
            and all(current[n] == sha for n, sha in recovered.items()),
            'only the new wait-policy controller may differ from the original diagnostic')
    # generation.current recovers the complete native constructor, actual
    # compiler/link inputs and every selected ELF. Record the actual projected
    # CPU input, which differs from HEAD's observer-enabled source.
    built = json.loads((model.gdn16.ROOT / 'record.json').read_text())
    names = ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')
    pool = {n: built['native_sources'][n] for n in names}
    require(all(digest(model.gdn16.ROOT / 'source' / n) == sha for n, sha in pool.items()),
            'actual compiled pool inputs differ')
    text = (model.gdn16.ROOT / 'source' / names[0]).read_text()
    require('std::getenv("STRATA_POOL_SPIN_US")' in text
            and 'spin_before_sleep_ = std::chrono::microseconds' in text,
            'qualified binary source lacks the immutable startup knob')
    return prior, providers, manifest, pool


def environment(row, plan):
    require((plan['kernel'], plan['spin_us']) == ARMS[plan['index']] and plan['heads'] == 1,
            'unplanned wait-policy arm')
    env = comparison.environment(row, plan)
    env.update(STRATA_SYCL_PREFILL_GDN_BARRIERS3='1', STRATA_SYCL_PREFILL_GDN_FP16='1',
               STRATA_SYCL_PREFILL_GDN_CHUNK64='0', STRATA_MTP_BATCH_F16='1',
               STRATA_SNAPSHOT_VERIFY='1', STRATA_TRACE='1', STRATA_PREFILL_CPU_WAIT_PROFILE='1',
               STRATA_POOL_SPIN_US=str(plan['spin_us']))
    return env


def invariant(row, plan):
    value = dict(config=json.loads((Path(plan['root']) / 'config.json').read_text()),
                 environment=environment(row, plan))
    value['environment']['STRATA_POOL_SPIN_US'] = '<arm>'
    return hashlib.sha256(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>').encode()).hexdigest()


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact wait-policy state')
    prior, files, manifest, pool = basis(row['source_commit'])
    require(row['binary'] == prior['binary'] and row['native_source_commit'] == prior['native_source_commit']
            and row['native_validation_probe'] == prior['native_validation_probe']
            and row['model_assets'] == prior['model_assets'] and row['compiled_pool_sources'] == pool
            and model.pinned_sources(row['source_commit']) == row['sources'], 'frozen source/build differs')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == files,
            'runtime inventory differs')
    require(digest(row['template']['path']) == row['template']['sha256']
            and build.guards.driver_guard() == row['driver'], 'template/driver differs')
    require(build.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole output budget')
    require(all(invariant(row, p) == row['invariant_signature']
                and comparison.fingerprint(environment(row, p)) == p['environment_sha256']
                and all(digest(Path(p['root']) / n) == sha for n, sha in p['generated'].items())
                for p in row['plans']), 'generated arm inputs differ')
    return files, manifest


def proc_fields(text):
    # comm may contain spaces and closing parentheses; field3 starts after its
    # final delimiter. Linux utime/stime are fields14/15, starttime is field22.
    f = text.rsplit(')', 1)[1].split()
    return dict(start_ticks=int(f[19]), cpu_ticks=int(f[11]) + int(f[12]))


def cpu_snapshot(pid):
    begin = time.monotonic_ns()
    root = Path('/proc') / str(pid)
    proc = proc_fields((root / 'stat').read_text())
    threads = {}
    for task in (root / 'task').iterdir():
        try:
            threads[task.name] = proc_fields((task / 'stat').read_text())
        except FileNotFoundError:
            continue  # A changing thread set is explicitly unqualified below.
    return dict(pid=pid, process=proc, threads=threads, begin_ns=begin,
                end_ns=time.monotonic_ns(), ticks_per_s=os.sysconf('SC_CLK_TCK'))


def cpu_delta(before, after):
    require(before['pid'] == after['pid'] and before['ticks_per_s'] == after['ticks_per_s'] > 0
            and before['process']['start_ticks'] == after['process']['start_ticks']
            and before['begin_ns'] <= before['end_ns'] < after['begin_ns'] <= after['end_ns'],
            'CPU identity/clock continuity differs')
    ticks = after['process']['cpu_ticks'] - before['process']['cpu_ticks']
    require(ticks >= 0, 'process CPU counter decreased')
    same = {n for n in before['threads'].keys() & after['threads'].keys()
            if before['threads'][n]['start_ticks'] == after['threads'][n]['start_ticks']}
    deltas = {n: after['threads'][n]['cpu_ticks'] - before['threads'][n]['cpu_ticks'] for n in same}
    require(all(v >= 0 for v in deltas.values()), 'thread CPU counter decreased')
    hz = before['ticks_per_s']; wall = (after['begin_ns'] - before['end_ns']) / 1e9
    complete = set(before['threads']) == set(after['threads']) == same
    return dict(process_cpu_s=ticks / hz, interval_wall_s=wall,
                process_core_equivalent=ticks / hz / wall,
                main_thread_cpu_s=deltas.get(str(before['pid']), 0) / hz,
                common_thread_cpu_s=sum(deltas.values()) / hz,
                stable_threads=len(same), thread_set_unchanged=complete,
                born_or_reused_threads=len(set(after['threads']) - same),
                ended_or_reused_threads=len(set(before['threads']) - same), tick_s=1 / hz,
                snapshot_wall_s=[(v['end_ns'] - v['begin_ns']) / 1e9 for v in (before, after)],
                limits='Native process CPU over the full gated request,including prefill,decode,polling and control. '
                       'Common-thread sum can omit exited threads;no worker-only,decode-only,arithmetic,DRAM or speed claim.')


def prefill_cpu_summary(text):
    cpu = shared.gen.common.extract_cpu_observers(text, 4)
    return dict(cpu_s=float(cpu['wait']['main_cpu_s']), wall_s=float(cpu['wait']['wall_s']),
                known_wait_cpu_s=float(cpu['wait']['wait_cpu_s']), known_wait_wall_s=float(cpu['wait']['wait_wall_s']),
                unclassified_cpu_s=float(cpu['host']['unclassified_cpu_s']), partition_qualified=cpu['partition_qualified'])


def prepare():
    require(not ROOT.exists() and KEY not in build.committed(RESULT), 'fresh bounded diagnostic only')
    rev = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    prior, providers, manifest, pool = basis(rev)
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', providers)
    row = {k: copy.deepcopy(prior[k]) for k in ('binary', 'native_source_commit', 'native_validation_probe',
        'configuration', 'environment_template', 'template', 'boot', 'driver', 'python_contract', 'python_selected')}
    row.update(stage='prepared', source_commit=rev, sources=model.pinned_sources(rev), model_assets=assets,
        compiled_pool_sources=pool, runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        plans=[], processes=[], host_passed=False, passed=False, complete=False, active=False, ownership_closed=True,
        controls=selfcheck(), whole_bytes=192 << 20,
        budgets=dict(wall=1800, cpu=10800, rss_cap=104 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        plan=dict(question='Does reducing the existing CPU worker idle-spin bound warrant repeated phase-specific speed tests?',
            motivation='Older registered-thread accounting showed substantial CPU time outside target pool calls;'
                'it does not prove spinning latency or arithmetic/DRAM saturation. The actual selected pool has a20ms '
                'idle bound and an immutable preexec override. Zero still checks its clock only after1024pause iterations.',
            order='Two fresh processes:explicit20000us reference then0us candidate;four official warmups,then one regular-code request each.',
            constants='Same qualified nine ELFs,providers,safety settings,throughput native16/GDN16/MTP16,heads1,131072INT8resident context,1256decode slots,5workers,spec4,chunk8192,no token graph,caches0. Only POOL_SPIN_US differs.',
            observers='Parameter validation,private activation preload,trace/snapshot checks,prefill CPUv4 partition and two read-only /proc CPU snapshots. No full API trace,profiler or periodic observer. Diagnostic timings include these observers.',
            acceptance='Exact committed source/build/assets/provider/body/native-route/journal/clock/owner gates. '
                '>=32768fresh input,>=1024actual generated tokens,thinking and>=512answer characters,>=256drafts,normal stream/native0,quiet validation,no fresh fault. '
                'Report process CPU across whole request separately from the disjoint main-thread prefill accounting. No exact text requirement.',
            decision='One code-only observation per arm is descriptive. Lower CPU alone is not a speedup. '
                'Freeze clean opposite-order repeated comparisons at both MTP workloads before nonregression or policy adoption. '
                'Judge prefill and decode independently;global startup policy is an experimental control,not a selected shared phase policy.',
            stopping='Stop first source/owner/provider/clock/route/validation/fault/budget/sample failure. No retry,exclusion,speed stop or hash refresh;commit each process before next. Full model hashes before/after.',
            retention='Keep compact samples,CPU partition aggregates,actual native source/configuration and closure decisions. '
                'Retire successful raw journals/logs after verification. No new binary generation;existing engine/fallback consumers unchanged.'))
    for index, (arm, spin) in enumerate(ARMS):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'): (root / part).mkdir()
        config = copy.deepcopy(row['configuration']); config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=index, kernel=arm, spin_us=spin, heads=1, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--long-generation-only', '--long-generation-workload', 'regular-code', '--think', '--runs', '1',
                '--profile-request-gates', '--workload-id', WORKLOAD_ID, '--fail-fast-engine-error',
                '--ready-timeout', '600', '--out', str(root / 'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    row['invariant_signature'] = invariant(row, row['plans'][0])
    require(all(invariant(row, p) == row['invariant_signature'] for p in row['plans']), 'undeclared arm difference')
    save(row); print('Prepared two bounded code-only CPU wait-policy processes;commit before execution.', flush=True)


def official(root, result):
    suite = result['results']['long_generation']; rows = shared.gen.collect_samples(suite)
    shared.gen.require_complete_generation(suite, rows, WORKLOADS)
    require(result['complete'] and not result['results']['failures'] and result['params']['runs'] == 1
            and result['params']['think'] is True and result['params']['long_generation_only'] is True,
            'incomplete official code diagnostic')
    journal, journal_rows = shared.gen.journal_evidence(root)
    require(journal['valid'] and journal['complete']
            and [{k: v for k, v in r.items() if k not in ('controller_eligible', 'reasoning_tokens')} for r in rows] == journal_rows,
            'result/journal differ')
    path, = (root / 'bench').glob('*/requests.jsonl')
    events = [json.loads(v) for v in path.read_text().splitlines()]
    warm = shared.gen.warmup_reconciliation(events, WORKLOADS)
    identity, binding = events[0], result['comparison_identity']
    from strata_bench import client_fingerprint
    workload = dict(workload_id=WORKLOAD_ID, nonce=WORKLOAD_ID, nonce_policy='fixed-id')
    require(identity['mode'] == binding['mode'] == 'suite' and identity['requested_runs'] == 1
            and identity['thinking'] is binding['thinking'] is True
            and identity['source'] == binding['client'] == client_fingerprint()
            and identity['context'] == binding['context'] == result['machine']['context'] == 131072
            and identity['workload'] == binding['workload'] == result['params']['workload'] == workload
            and all(identity[k] == binding[k] == result['machine'][k] for k in ('model', 'quant')),
            'client/model/body binding differs')
    return rows, events, dict(warm, valid=True, journal_sha256=digest(path))


def run(index):
    row = json.loads((ROOT / 'record.json').read_text()); files, manifest = current(row)
    require(row['host_passed'] and row['stage'] in ('ready', 'running') and len(row['processes']) == index < 2
            and all(p['passed'] for p in row['processes']), 'fresh ordered process only;no retry')
    plan = row['plans'][index]; root = Path(plan['root']); env = environment(row, plan)
    require(not (root / 'execution.json').exists() and not (root / 'launch.json').exists(), 'existing attempt')
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome, owner = dict(index=index, kernel=plan['kernel'], spin_us=plan['spin_us'], complete=False,
                          passed=False, host_probe_accepted=False, model_executed=False), None
    offsets, snapshots, identities = {}, {}, []
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def persist(): write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, entry):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(1200)
                if line.startswith('STRATA_PROFILE_GATE '): break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='long-generation/regular-code', run=0), 'request gate differs')
            claim = json.loads((root / 'engine-start.json').read_text())['native']
            snapshots[stage] = cpu_snapshot(claim['pid'])
            actual = build.supervision.identity(claim['pid']); identities.append(claim)
            require(actual and actual['start_ticks'] == claim['start_ticks'] and actual['sid'] == entry['pid']
                    and snapshots[stage]['process']['start_ticks'] == claim['start_ticks']
                    and str(Path('/proc', str(claim['pid']), 'exe').resolve()) == row['binary']['path']
                    and identities[0] == claim, 'native identity differs')
            delta = [a-b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0] - delta[1]) <= 50000000, 'clock/suspend discontinuity')
            expected = model.engine_once.native_environment(row, env)
            actual_env = dict(v.split(b'=', 1) for v in Path('/proc', str(claim['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
            require(all(actual_env.get(k.encode()) == v.encode() for k, v in expected.items())
                    and set(actual_env) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
            mapped = shared.runtime.executable_mappings(Path('/proc', str(claim['pid']), 'maps').read_text())
            require(mapped <= set(files) | {row['binary']['path']} and
                    {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']} <= mapped,
                    'native providers differ')
            shared.python_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py', manifest['selected'], env,
                {v['realpath'] for v in manifest['inventory']['files'].values()}, plan['command'][2:])
            comparison.require_quiet_validation(root); offsets[stage] = (root / 'engine.stderr').stat().st_size
            if stage == 'begin':
                path, = (root / 'bench').glob('*/requests.jsonl')
                events = [json.loads(v) for v in path.read_text().splitlines()]
                require(len(events) == 10 and events[-1]['kind'] == 'start' and events[-1]['phase'] == 'long-generation/regular-code', 'official warmup/start count differs')
                shared.gen.warmup_reconciliation(events[:-1], ())
                shared.gen.request_evidence((root / 'engine.stderr').read_text(), [v['sample'] for v in events[:-1] if v['kind'] == 'result'])
            else:
                text = (root / 'engine.stderr').read_bytes()[offsets['begin']:offsets['end']].decode()
                header, = re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', text, re.M)
                require(header[1] == '0' and header[0] == header[2], 'native request footer differs')
                outcome['route'] = generation.native_route(text, int(header[2]))
                outcome['prefill_main_cpu'] = prefill_cpu_summary(text)
                outcome['request_cpu'] = cpu_delta(snapshots['begin'], snapshots['end'])
            outcome.update(model_executed=True, native_identity=claim); persist()
            print(f'Pool idle process{index} {plan["spin_us"]}us {stage}', flush=True); io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = build.supervision.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        outputs = list((root / 'bench').glob('*/results.json'))
        require(build.supervision.closed(entry) and entry['exit_code'] == 0 and len(outputs) == 1,
                'owned benchmark failed: ' + entry.get('error', 'exit/result/closure gate'))
        samples, events, journal = official(root, json.loads(outputs[0].read_text()))
        comparison.require_valid_client_ttft(samples[0])
        if row['processes']:
            require(journal['request_sha256'] == row['processes'][0]['journal']['request_sha256'], 'paired warmup/request bodies differ')
        comparison.require_quiet_validation(root)
        end = shared.gen.native_end(root, row, dict(config=json.loads((root / 'config.json').read_text())), row['boot'], entry['pid'])
        text = (root / 'engine.stderr').read_text(); generation.gdn.validate_model_text(text, plan)
        binding = shared.gen.request_evidence(text, [e['sample'] for e in events if e['kind'] == 'result'])
        require(not any(v in text for v in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'native error')
        validation = shared.runtime.validation.preload_evidence(text, end['native']['pid'])
        require(validation['loader'] == row['native_validation_probe']['loader']['path'], 'validation loader differs')
        raw = (root / 'engine.stderr').read_bytes()[offsets['begin']:offsets['end']].decode()
        require(generation.native_route(raw, samples[0]['read_tokens']) == outcome['route'], 'live/final route differs')
        current(row)
        if index == 1: shared.models.model_pins(row['configuration'], row['model_assets'], full=True)
        fields = ('workload', 'read_tokens', 'reused_tokens', 'generated_tokens', 'output_chars', 'reasoning_chars',
            'prompt_ms', 'prompt_tps', 'decode_ms', 'decode_tps', 'ttft_s', 'total_s', 'drafts_offered', 'drafts_accepted',
            'acceptance_rate', 'request_sha256', 'output_sha256', 'reasoning_sha256', 'finish_reason', 'controller_eligible')
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=validation, journal=journal,
            native_request_binding=dict(requests=len(binding), draft_cells=sum(v['draft_cells'] for v in binding)),
            samples=[{k: s[k] for k in fields if k in s} for s in samples])
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner: shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        if outcome.get('driver_after') != row['driver']: outcome.update(passed=False, final_driver_mismatch=True)
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        row['processes'].append(outcome); done = index == 1
        row.update(stage='closed' if outcome['passed'] and done else 'running' if outcome['passed'] else 'failed',
            passed=bool(outcome['passed'] and done), complete=bool(outcome['passed'] and done),
            active=bool(owner and owner.active), ownership_closed=outcome.get('ownership_closed', False))
        if row['passed']:
            row['decision'] = dict(diagnostic_complete=True, speed_nonregression_qualified=False, inference_adopted=False,
                physical_resource_saturation_qualified=False, prefill_decode_independence_qualified=False,
                next='Inspect actual CPU/window/output differences;freeze repeated clean phase-specific comparison if promising.')
        save(row)
    require(outcome['passed'], outcome.get('error', 'wait-policy diagnostic failed'))
    print(json.dumps({k: outcome.get(k) for k in ('index', 'kernel', 'passed', 'complete', 'error')}), flush=True)


def selfcheck():
    # Replay the retained real current-build partition, not an invented schema:
    # the main-thread total is named main_cpu_s, not cpu_s.
    profile = build.committed(RESULT)['gdn16_resource_profile_v3']
    require(profile['passed'] and profile['complete'] and profile['ownership_closed'], 'closed CPU partition replay required')
    cpu = profile['observation']['cpu_partition']
    text = '\n'.join(prefix + ' '.join(k + '=' + v for k, v in cpu[name].items())
        for name, prefix in [('wait', 'strata prefill cpu wait: '), ('host', 'strata prefill cpu host: ')])
    replay = prefill_cpu_summary(text)
    require(replay['cpu_s'] == float(cpu['wait']['main_cpu_s']) and replay['partition_qualified'], 'actual CPU partition replay')
    fields = ['S'] + ['0'] * 25; fields[11], fields[12], fields[19] = '40', '10', '123'
    require(proc_fields('7 (comm with ) spaces) ' + ' '.join(fields)) == dict(start_ticks=123, cpu_ticks=50), 'proc comm parsing')
    a = dict(pid=7, process=dict(start_ticks=123, cpu_ticks=50), threads={'7': dict(start_ticks=123, cpu_ticks=50)},
             begin_ns=1, end_ns=2, ticks_per_s=100)
    b = copy.deepcopy(a); b.update(begin_ns=1000000002, end_ns=1000000003)
    b['process']['cpu_ticks'] = b['threads']['7']['cpu_ticks'] = 150
    require(cpu_delta(a, b)['process_cpu_s'] == 1 and cpu_delta(a, b)['thread_set_unchanged'], 'CPU clock accounting')
    born = copy.deepcopy(b); born['threads']['8'] = dict(start_ticks=321, cpu_ticks=0)
    require(not cpu_delta(a, born)['thread_set_unchanged'] and cpu_delta(a, born)['born_or_reused_threads'] == 1, 'thread birth hidden')
    bad = []
    v = copy.deepcopy(b); v['process']['start_ticks'] += 1; bad.append(v)
    v = copy.deepcopy(b); v['process']['cpu_ticks'] = 49; bad.append(v)
    v = copy.deepcopy(b); v['begin_ns'] = 1; bad.append(v)
    v = copy.deepcopy(b); v['threads']['7']['cpu_ticks'] = 49; bad.append(v)
    for value in bad:
        try: cpu_delta(a, value)
        except RuntimeError: pass
        else: raise RuntimeError('invalid CPU window accepted')
    return dict(passed=True, valid_accounting_controls=3, invalid_windows_rejected=len(bad),
                actual_prefill_partition_replay=True, GPU_executed=False)


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.guards.no_model_owners([])
        args = sys.argv[1:]
        if args == ['selfcheck']: print(json.dumps(selfcheck()))
        elif args == ['prepare']: prepare()
        elif args == ['host']: comparison.host(sys.modules[__name__])
        elif len(args) == 1 and args[0] in ('0', '1'): run(int(args[0]))
        else: raise RuntimeError('pool_spin.py selfcheck|prepare|host|INDEX0..1')
