#!/usr/bin/env python3
"""Main-owned official long-generation query-cost diagnostic, not clean speed.

Uses the normal-route component/tiny check in query_profile.py and the existing
Main-authored official benchmark and process ownership helpers. No early route,
queue-property change, external collector or CPU-time observer is enabled.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import sys

sys.dont_write_bytecode = True
import query_profile as query

gen, assets, guards, supervision = query.gen, query.assets, query.guards, query.supervision
P, B, A = query.P, query.B, query.A
ROOT = B / 'decode-query-profile-long-v1'
RESULT = query.RESULT


def tiny_proof(reference):
    raw = gen.git('show', reference['commit'] + ':' + reference['path'])
    guards.require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'tiny check reference changed')
    tiny = json.loads(raw)
    guards.require(tiny['mode'] == 'default-route-query-profile-check' and tiny['passed'] and
        tiny['complete'] and tiny['ownership_closed'] and not tiny['active'] and
        not tiny['fresh_faults'] and tiny.get('query_observer_reached') is True and
        tiny.get('actual_default_route_confirmed') is True and len(tiny['small_requests']) == 2,
        'closed positive default-route tiny qualification required')
    return tiny


def prepare():
    tiny = gen.committed(RESULT)
    reference = dict(commit=gen.git('rev-parse', 'HEAD').decode().strip(),
        path=str(RESULT.relative_to(P)), sha256=gen.common.digest(RESULT))
    guards.require(tiny_proof(reference) == tiny, 'tiny proof differs')
    full, count = query.current_build()
    guards.require(full['binary'] == tiny['binary'] and not ROOT.exists(),
        'same tiny-qualified engine and fresh diagnostic root required')
    guards.require(assets.model_pins(tiny['config'], tiny['model_assets']) == tiny['model_assets'] and
        assets.machine_identity() == tiny['machine_identity'], 'tiny-qualified assets/platform changed')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT/name).mkdir()
    env = {k:v.replace(str(query.ROOT), str(ROOT)) for k,v in tiny['environment'].items()}
    guards.require(env['STRATA_MEASUREMENT_ROOT'] == str(ROOT) and
        env['STRATA_SYCL_QUERY_PROFILE'] == '1' and
        all(env[k] == '0' for k in ('STRATA_SYCL_VERIFY_ARENA_COPY',
            'STRATA_SYCL_VERIFY_PCIE_EARLY', 'STRATA_SYCL_VERIFY_PCIE_EARLY_AUDIT')) and
        not any(k in env for k in ('STRATA_DECODE_CPU_PROFILE', 'STRATA_DECODE_TIMING', 'UR_LOG_TRACING')),
        'unexpected observer, root or early route')
    config = copy.deepcopy(tiny['config'])
    config['args'][config['args'].index('--expert-profile')+1] = str(ROOT/'expert-profile.bin')
    config.update(log=str(ROOT/'engine.stderr'), env={})
    guards.require(config['exe'] == str(P/'tools/sycl/engine_once.py'), 'one-launch wrapper required')
    shutil.copyfile(P/'data/expert-profile.bin', ROOT/'expert-profile.bin')
    guards.require(gen.common.digest(ROOT/'expert-profile.bin') == tiny['profile_sha256'], 'profile changed')
    (ROOT/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = ['/bin/sh', str(P/'bench.sh'), '--config', str(ROOT/'config.json'), '--port', str(port),
        '--long-generation-only', '--think', '--runs', '1', '--workload-id', gen.WORKLOAD_ID,
        '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(ROOT/'bench'),
        '--long-generation-workload', 'regular-code']
    record = dict(mode='default-route-query-profile-long', source_commit=revision,
        build_source_commit=full['source_commit'], binary=full['binary'],
        source_build_record=dict(path=str(query.BUILD_RECORD), sha256=gen.common.digest(query.BUILD_RECORD)),
        component_result_sha256=gen.common.digest(query.COMPONENT), original_build_inputs_recovered=count,
        tiny_check_reference=reference, runtime_asset_delta=tiny['runtime_asset_delta'], root=str(ROOT),
        environment=env, config=config, command=command, model_assets=tiny['model_assets'],
        machine_identity=tiny['machine_identity'], tokenizer_assets=tiny['tokenizer_assets'],
        source_count=len(pins), source_aggregate_sha256=assets.fingerprint(pins),
        config_sha256=gen.common.digest(ROOT/'config.json'), profile_sha256=tiny['profile_sha256'],
        settings_sha256=gen.common.digest(gen.SETTINGS), complete=False, passed=False, active=False,
        clean_speed_qualified=False, observer_overhead_qualified=False,
        hardware_bottleneck_qualified=False, model_quality_qualified=False,
        budgets=dict(wall=1800, cpu=54000, rss_cap=96<<30, text_cap=64<<20,
            file_cap=64<<20, total_cap=64<<20),
        gate='One owned official bench.sh regular-code process after four successful warmups. Thinking on, at least32768 fresh input and1024 output tokens including an answer, journal-bound engine/client clocks and work, native/benchmark exit0. Exactly four ordered query categories per completed request, ready/pending/error conservation, zero errors/clock failures, nonzero resident calls/time, zero inactive early counters and expert-copy bytes/groups. Unchanged source/build/model/runtime/boot; no fresh fault. No automatic retry.',
        limits='Four verifier query points only, not every request query. Wall interval includes runtime checks, event-state locking and completed-event waits, plus clock-reading effects. It is not native-query-only, GPU service, CPU arithmetic or hardware saturation. A diagnostic fraction does not qualify causal overhead, saved time, a production policy change or throughput. Physical262144/session/RESTORE, task quality, strong/weak adoption and70tok/s remain separate.',
        consumer='Choose the next measured synchronization/CPU/transfer discriminator from actual production query cost. Independently qualify observer overhead before adoption attribution; collect clean speed separately. Retire successful raw output after useful request samples and query aggregates are committed.')
    query.runtime_assets_unchanged(record)
    (ROOT/'plan.json').write_text(json.dumps(record, indent=2)+'\n')
    RESULT.write_text(json.dumps(record, indent=2)+'\n')
    print('Prepared official32K+ query diagnostic; commit exact plan before run.')


def run():
    record = gen.committed(RESULT)
    guards.require(record == json.loads((ROOT/'plan.json').read_text()) and not record['complete'] and
        not (ROOT/'record.json').exists(), 'fresh committed plan required; no automatic restart')
    tiny = tiny_proof(record['tiny_check_reference'])
    full, count = query.current_build()
    guards.require(record['binary'] == full['binary'] == tiny['binary'] and
        count == record['original_build_inputs_recovered'] and
        gen.common.digest(query.BUILD_RECORD) == record['source_build_record']['sha256'] and
        gen.common.digest(query.COMPONENT) == record['component_result_sha256'], 'build/component changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) ==
        record['source_aggregate_sha256'], 'controller/benchmark inputs changed')
    for path, sha in [(gen.SETTINGS,record['settings_sha256']),
            (ROOT/'config.json',record['config_sha256']),
            (ROOT/'expert-profile.bin',record['profile_sha256']), *record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path) == sha, 'changed input: '+str(path))
    query.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'],record['model_assets']) == record['model_assets'] and
        assets.machine_identity() == record['machine_identity'], 'model/platform changed')
    import subprocess
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor = next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,
            owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner = supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,_ = owner.run('bench',record['command'],record['environment'],P,**record['budgets'])
        record.update(complete=supervision.closed(entry),exit_code=entry.get('exit_code'))
        result, = (ROOT/'bench').glob('*/results.json')
        generation = json.loads(result.read_text())['results']['long_generation']
        rows = gen.collect_samples(generation); record['requests'] = rows
        gen.require_complete_generation(generation,rows,('regular-code',))
        journal,journal_rows = gen.journal_evidence(ROOT)
        official = [{k:v for k,v in row.items() if k not in ('controller_eligible','reasoning_tokens')}
            for row in rows]
        guards.require(official == journal_rows, 'official work/client clocks differ from journal')
        record['warmups'] = gen.warmup_evidence(ROOT,journal)
        end = json.loads((ROOT/'engine-exit.json').read_text())
        guards.require(entry.get('exit_code') == end['exit_code'] == 0 and end['boot_id'] == boot,
            'benchmark/native failure')
        log = (ROOT/'engine.stderr').read_text(errors='replace')
        assets.verify_pool(log,5)
        headers = re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read'
            r'[^\n]*, (\d+) generated in (\d+) ms[^\n]*$',log,re.M)
        guards.require(len(headers) == 5 and all(int(h[4])>0 for h in headers),
            'expected four warmups and one timed native request')
        guards.require(tuple(map(int,headers[-1][:4])) ==
            (rows[0]['prompt_tokens'],0,rows[0]['read_tokens'],rows[0]['generated_tokens']),
            'native timed-request work differs')
        queries = query.query_evidence(log,len(headers))
        record['query_requests'] = [dict(prompt_tokens=int(h[0]),generated_tokens=int(h[3]),
            native_decode_ms_rounded=int(h[4]),queries=q,
            wrapper_wall_fraction=sum(v['wall_ns'] for v in q.values())/(int(h[4])*1e6))
            for h,q in zip(headers,queries)]
        copies = re.findall(r'^strata serve: SYCL_EXPERT_COPY '+
            r' '.join(k+r'=(\d+)' for k in query.COPY_KEYS)+r'$',log,re.M)
        guards.require(len(copies) == len(headers) and all(
            all(int(x)==0 for x in row[:6]) and all(int(x)>0 for x in row[6:]) for row in copies),
            'unexpected copy route or absent CPU expert work')
        guards.require('ordinary-Arena copy enabled' not in log and
            'early PCIe expert graph enabled' not in log, 'unqualified early route enabled')
        query.current_build()
        guards.require(gen.benchmark_pins(record['source_commit']) == pins, 'source changed during run')
        query.runtime_assets_unchanged(record)
        guards.require(assets.machine_identity() == record['machine_identity'] and
            assets.model_pins(record['config'],record['model_assets']) == record['model_assets'],
            'model/platform changed during run')
        record.update(passed=True,production_query_observation_complete=True)
    except BaseException as error:
        record['error'] = str(error)
        if (ROOT/'engine.stderr').exists():
            record['error_excerpt'] = (ROOT/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
        record['fresh_faults'] = [x for x in after.splitlines() if re.search(
            r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
        record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try: record['driver_after'] = guards.driver_guard()
        except Exception as error: record.update(driver_guard_error=str(error),passed=False)
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and
            record['boot_unchanged'] and not record['fresh_faults'])
        persist(); RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run')); args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); guards.no_model_owners([]); guards.driver_guard()
        return prepare() if args.action == 'prepare' else run()


if __name__ == '__main__': sys.exit(main())
