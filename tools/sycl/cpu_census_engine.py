#!/usr/bin/env python3
"""Main-owned final CPU-job census: finite bank gate, official warmups, two long cases."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import cpu_work_census as census
import cpu_capacity as assets
import generation_compare as comparison
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import query_profile as q
from check_expert_bank import check_packed_cases

P, B, A = gen.P, gen.B, gen.A
ROOT = B/'decode-cpu-work-v1'
MODEL = ROOT/'model'
RESULT = A/'cpu-work-census-current.json'
WORKLOADS = ('regular-code', 'varied-prose')


def source_pins(revision):
    pins = gen.benchmark_pins(revision)
    for name in ('data/expert-profile.bin', 'tests/core/test_cpu_work_census.py'):
        raw = gen.git('show', revision+':'+name)
        guards.require((P/name).read_bytes() == raw, 'commit exact census input: '+name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def qualification():
    full, count = q.current_build()
    guards.require('cpu_work_census_host_test' in full['fixtures'], 'census native build required')
    return full, dict(original_inputs_recovered=count, build_receipt_sha256=gen.common.digest(q.BUILD_RECORD),
        shared_component_sha256=gen.common.digest(q.COMPONENT))


def plan():
    return dict(question='What exact GU/Down and actual job-NT mix consumes the current decode CPU pool?',
        order='Fresh same-build packed-smoke then packed-validate; one official bench.sh process, four declared warmups, regular-code then varied-prose.',
        constants='131072 resident INT8 context,1256 per-layer expert slots,8192 prefill,5 pool workers+host,pcie0,adapt0,spec4,suffix0,caches0,greedy,Q8 MTP,packed bank. Thinking on; each long case >=32K fresh input,>=1024 generated tokens and>=512 answer characters.',
        gates='Commit source/recipe/plan first. Match all current native inputs,eleven artifacts,model/runtime/config/controller inputs. Parameter validation and stage trace; preserve all shared safety settings. Before each long request verify the actual owned native executable/environment,four warmups/prior requests,census/clock/work conservation,no fresh fault. Stop on first failure,no retry.',
        interpretation='Direct final CPU jobs, not routing-dump reconstruction. Exact static resident membership,format,NT and native footer conservation. Existing CPU-pool host wall remains distinct from scheduled CPU,physical RAM bandwidth and GPU service. Count diagnostic only; no clean speed,model adoption,quality,agent-session orphysical262144/RESTORE qualification.',
        retention='Keep two samples,compact format/NT bins,membership/profile hash,clock conservation,source/fault/owner identities and decision. Retire successful full logs/JIT after verification. Keep fallback only through this candidate qualification and select one validated generation at closure.')


def current(record, full_models=False):
    full, proof = qualification()
    guards.require(full['source_commit'] == record['build_source_commit'] and full['binary'] == record['binary'] and
        proof == record['qualification'] and record['plan'] == plan(), 'census qualification/plan changed')
    pins = source_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['benchmark_fingerprint'] and
        gen.common.digest(gen.SETTINGS) == record['settings_sha256'], 'census controller/settings changed')
    guards.require(all(gen.common.digest(path) == sha for path, sha in record['generated_inputs'].items()),
        'generated census input changed')
    guards.require(assets.model_pins(record['config'], record['model_assets'], full=full_models) == record['model_assets'] and
        assets.machine_identity() == record['machine_identity'], 'model/machine changed')
    gen.runtime_assets_unchanged(record)
    return full


def fault_evidence(cursor, boot):
    after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'],
        text=True, timeout=15)
    guards.driver_guard()
    return dict(fresh_faults=[s for s in after.splitlines() if re.search(
        r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', s)],
        boot_unchanged=boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), fault_probe_successful=True)


def start_fault_scope():
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15)
    return boot, next(s.split(': ', 1)[1] for s in before.splitlines() if s.startswith('-- cursor: '))


def work_evidence(text, samples, record):
    timing = gen.decode_timing_evidence(text, samples)
    bindings = [dict(prompt_tokens=s['prompt_tokens'], windows=t['windows'], rows=t['work']['target_rows'],
        cpu_jobs=t['work']['cpu_unique_jobs'], cpu_entries=t['work']['cpu_entries'],
        pool_ms=t['ms_per_window']['CPU_pool_run']*t['windows']) for s, t in zip(samples, timing)]
    flags = record['config']['args']
    formats = census.manifest_formats(Path(flags[flags.index('--pack')+1])/'native_experts.txt')
    membership = census.expected_membership(MODEL/'expert-profile.bin')
    return census.extract(text, bindings, formats, membership), timing


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh census only; inspect existing owned run')
    full, proof = qualification()
    closed = gen.committed(comparison.RESULT)
    guards.require(closed['complete'] and closed['passed'] and closed['ownership_closed'] and not closed['active'],
        'closed original workload reference required')
    selected = closed['plans'][1]
    revision = gen.git('rev-parse', 'HEAD').decode().strip(); pins = source_pins(revision)
    models = assets.model_pins(selected['config'], closed['model_assets'], full=True)
    ROOT.mkdir(mode=0o700); MODEL.mkdir()
    for n in ('tmp', 'cache'): (MODEL/n).mkdir()
    config = copy.deepcopy(selected['config'])
    config['args'][config['args'].index('--expert-profile')+1] = str(MODEL/'expert-profile.bin')
    config.update(log=str(MODEL/'engine.stderr'), env={})
    env = {k:v.replace(selected['root'], str(MODEL)) for k,v in selected['environment'].items()}
    env.update(STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1')
    shutil.copyfile(P/'data/expert-profile.bin', MODEL/'expert-profile.bin')
    (MODEL/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (MODEL/'plan.json').write_text(json.dumps(dict(binary=full['binary'], native_validation_probe=None), indent=2)+'\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    record = dict(mode='engine-compare', stage='prepared', source_commit=revision,
        build_source_commit=full['source_commit'], binary=full['binary'], qualification=proof,
        plan=plan(), config=config, environment=env, native_validation_probe=None,
        runtime_asset_delta=closed['runtime_asset_delta'], model_assets=models,
        workload_reference=dict(path=str(comparison.RESULT.relative_to(P)), sha256=gen.common.digest(comparison.RESULT),
            request_sha256=[r['request_sha256'] for r in closed['processes'][1]['requests']]),
        generated_inputs={str(MODEL/n):gen.common.digest(MODEL/n) for n in ('config.json','plan.json','expert-profile.bin')},
        command=['/bin/sh', str(P/'bench.sh'), '--config', str(MODEL/'config.json'), '--port', str(port),
            '--long-generation-only', '--think', '--runs', '1', '--profile-request-gates',
            '--workload-id', comparison.WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(MODEL/'bench')],
        source_count=len(pins), benchmark_fingerprint=assets.fingerprint(pins),
        settings_sha256=gen.common.digest(gen.SETTINGS), machine_identity=assets.machine_identity(),
        complete=False, passed=False, active=False, ownership_closed=True,
        clean_speed_qualified=False, hardware_saturation_qualified=False, model_adoption_qualified=False,
        budgets=dict(wall=3600, cpu=108000, rss_cap=96<<30, text_cap=64<<20, file_cap=64<<20, total_cap=64<<20))
    current(record)
    (ROOT/'frozen-plan.json').write_text(json.dumps(record, indent=2)+'\n')
    RESULT.write_text(json.dumps(record, indent=2)+'\n')
    print('Prepared bank gate then one official two-workload census; commit plan before qualification.')


def execute(action):
    record = gen.committed(RESULT); full = current(record)
    frozen = json.loads((ROOT/'frozen-plan.json').read_text())
    mutable = {'stage','bank','requests','warmups','routes','census','decode_timing','native_identity',
        'effective_environment_verified','native_exit_code','complete','passed','active','ownership_closed',
        'fresh_faults','fault_probe_successful','boot_unchanged','error','fault_probe_error'}
    guards.require({k:v for k,v in record.items() if k not in mutable} ==
        {k:v for k,v in frozen.items() if k not in mutable}, 'frozen census changed')
    bank_run = action == 'qualify'
    guards.require(record['stage'] == ('prepared' if bank_run else 'model_prepared') and
        not record['active'] and (bank_run or record['bank']['passed']), 'wrong census stage; no retries')
    root = ROOT/'bank' if bank_run else MODEL
    guards.require(not (root/'record.json').exists() and not (root/'launch.json').exists(), 'existing census attempt')
    if bank_run:
        root.mkdir(); (root/'tmp').mkdir(); (root/'cache').mkdir()
    boot, cursor = start_fault_scope(); owner = None
    row = dict(passed=False, complete=False)
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(row, commands=owner.commands if owner else []), indent=2)+'\n')
    def interact(io, leader):
        for i, name in enumerate(WORKLOADS):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(3300)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='long-generation/'+name, run=0), 'official request gate differs')
                if stage == 'begin':
                    native = json.loads((MODEL/'engine-start.json').read_text())['native']
                    identity = supervision.identity(native['pid'])
                    guards.require(identity and identity['sid'] == leader['pid'] and identity['start_ticks'] == native['start_ticks'] and
                        Path('/proc', str(native['pid']), 'exe').resolve() == Path(record['binary']['path']).resolve(), 'native owner differs')
                    actual = dict(x.split(b'=', 1) for x in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in x)
                    guards.require(all(actual.get(k.encode()) == v.encode() for k,v in record['environment'].items()) and
                        b'LD_PRELOAD' not in actual, 'native startup environment differs')
                    path, = (MODEL/'bench').glob('*/requests.jsonl')
                    events = [json.loads(x) for x in path.read_text().splitlines()]
                    guards.require(len(events) == 10+2*i and events[-1]['kind'] == 'start' and
                        events[-1]['phase'] == 'long-generation/'+name, 'warmup/prior journal extent differs')
                    gen.warmup_reconciliation(events[:-1], WORKLOADS[:i])
                    previous = [e['sample'] for e in events[:-1] if e['kind'] == 'result']
                    text = (MODEL/'engine.stderr').read_text(errors='replace')
                    gen.request_evidence(text, previous); work_evidence(text, previous, record)
                    guards.require(not re.search(comparison.FAILURE, text), 'small/prior model check failed')
                    fault = fault_evidence(cursor, boot)
                    guards.require(fault['boot_unchanged'] and not fault['fresh_faults'], 'fresh fault before long request')
                    row.update(native_identity=native, effective_environment_verified=True); persist()
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); guards.require(time.monotonic() < deadline, 'benchmark teardown budget')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    owner = supervision.Owner(root/'owned', persist)
    try:
        if bank_run:
            env = dict(json.loads(gen.SETTINGS.read_text())['environment'], TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'))
            row['checks'] = []
            for mode in ('packed-smoke', 'packed-validate'):
                if mode.endswith('validate'):
                    env = dict(env, UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION'); env.pop('UR_LOG_TRACING', None)
                entry, out, err = owner.run(mode, [full['fixtures']['sycl_expert_bank_test']['path'], '--'+mode], env, P,
                    wall=180, cpu=600, rss_cap=8<<30, text_cap=64<<20, file_cap=64<<20, total_cap=64<<20)
                stdout, stderr = out.read_text(), err.read_text()
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0 and
                    not re.search(comparison.FAILURE, stderr), 'packed-bank fixture failed')
                row['checks'].append(dict(mode=mode, cases=check_packed_cases(stdout, mode), passed=True,
                    fixture=full['fixtures']['sycl_expert_bank_test'], exit_code=0))
        else:
            entry, _, _ = owner.run('bench', record['command'], record['environment'], P,
                interaction=interact, **record['budgets'])
            paths = list((MODEL/'bench').glob('*/results.json'))
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0 and len(paths) == 1, 'official benchmark failed')
            result = json.loads(paths[0].read_text())
            samples, journal, events = comparison.official(MODEL, result)
            guards.require([r['request_sha256'] for r in samples] == record['workload_reference']['request_sha256'], 'workload changed')
            native = gen.native_end(MODEL, record, dict(config=record['config']), boot, entry['pid'])
            guards.require(native['native'] == row['native_identity'] and row['effective_environment_verified'], 'native closure differs')
            text = (MODEL/'engine.stderr').read_text(errors='replace')
            all_samples = [e['sample'] for e in events if e['kind'] == 'result']
            work, timing = work_evidence(text, all_samples, record)
            row.update(requests=samples, warmups=journal, routes=comparison.native_routes(text, events, 1),
                census=work[4:], decode_timing=timing[4:], native_exit_code=native['exit_code'])
        current(record, full_models=not bank_run)
        row.update(passed=True, complete=True)
    except BaseException as error:
        row['error'] = str(error)
    finally:
        row.update(ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands))
        try: row.update(fault_evidence(cursor, boot))
        except Exception as error: row.update(fault_probe_error=str(error), passed=False)
        row['passed'] = bool(row['passed'] and row['ownership_closed'] and row.get('boot_unchanged') and
            row.get('fault_probe_successful') and row.get('fresh_faults') == [])
        persist()
        if bank_run:
            record.update(bank=row, stage='model_prepared' if row['passed'] else 'closed')
            if not row['passed']: record.update(error=row.get('error', row.get('fault_probe_error')), ownership_closed=row['ownership_closed'])
        else:
            record.update(row, stage='closed')
        RESULT.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:row.get(k) for k in ('passed','complete','ownership_closed','error')}))
    return 0 if row['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','qualify','run')); args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        return prepare() if args.action == 'prepare' else execute(args.action)


if __name__ == '__main__': sys.exit(main())
