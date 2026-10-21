#!/usr/bin/env python3
"""Main-owned long-generation reference diagnostic; no native rebuild or speed adoption.

Both official workloads run in each of two independent processes. Commit this
source, prepare once, then commit the generated plan before each ordered run.
"""
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
import generation_engine as gen
import cpu_capacity as assets
import packed_bank as bank
import measurement_guards as guards
import measurement_owner as supervision
from build_engine import (blobs, retained_model_evidence, census_model_evidence,
                          packed_model_evidence, archived_model_observation)

P, B, A = gen.P, gen.B, gen.A
ROOT = B/'generation-reference-v1'
RESULT = A/'generation-reference-current.json'
SPEED = A/'prefill-packed-bank-speed-current.json'
WORKLOADS = ('regular-code', 'varied-prose')
WORKLOAD_ID = 'generation_reference_20261021'
FAILURE = r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:'


def original(reference):
    raw = gen.git('show', reference['commit']+':'+reference['path'])
    guards.require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'original setup cannot be recovered')
    return json.loads(raw)


def retained_model_scope(full, reference):
    raw = {name:gen.git('show', commit+':'+str((A/name).relative_to(P)))
        for name,commit in reference['model_evidence_commits'].items()}
    observations = {name:json.loads(data) for name,data in raw.items()}
    kind = reference.get('model_evidence_kind')
    if kind == 'census':
        selected = reference.get('model_evidence_reference')
        name = 'cpu-work-census-current.json'
        if selected is not None:
            original = archived_model_observation(selected)
            name = Path(selected['path']).name
            guards.require(reference['model_evidence_commits'].get(name) == selected['commit'] and
                observations.get(name) == original, 'retained selected model receipt differs')
        component = 'decode-query-profile-component-current.json'
        return census_model_evidence(full, observations[component], observations[name],
            reference['source_build_record']['sha256'], hashlib.sha256(raw[component]).hexdigest())
    if kind == 'packed':
        return packed_model_evidence(full, observations['decode-query-profile-component-current.json'],
            observations['prefill-packed-bank-speed-current.json'], observations['generation-reference-current.json'])
    guards.require(kind is None, 'unknown retained model evidence kind')
    return retained_model_evidence(full, observations['prefill-status-current.json'],
                                   observations['prefill-mtp-current.json']['comparison'])


def recover_retained(reference):
    receipt = reference['source_build_record']
    guards.require(gen.common.digest(receipt['path']) == receipt['sha256'], 'retained source receipt changed')
    full = json.loads(Path(receipt['path']).read_text())
    guards.require(full['complete'] and full['passed'] and full['ownership_closed'] and not full['active'] and
        full['source_commit'] == reference['source_commit'] and full['project']['revision'] == full['source_commit'] and
        reference['fixtures'].keys() == full['fixtures'].keys(), 'retained source/fixture inventory differs')
    for name, artifact in dict(strata=reference['binary'], **reference['fixtures']).items():
        old = full['binary'] if name == 'strata' else full['fixtures'][name]
        guards.require(artifact['sha256'] == old['sha256'] and
            ('bytes' not in old or artifact['bytes'] == old['bytes']) and
            Path(artifact['path']).stat().st_size == artifact['bytes'] and
            gen.common.digest(artifact['path']) == artifact['sha256'], 'retained artifact changed: '+name)
    count = 0
    for source in [full['project'], *full['dependencies']]:
        recovered = {n:hashlib.sha256(data).hexdigest() for n,data in
            blobs(Path(source['root']), source['revision'], source['source_sha256'])}
        guards.require(recovered == source['source_sha256'], 'retained native/dependency inputs not recoverable')
        count += len(recovered)
    recipe = full['recipe_reference']
    guards.require(hashlib.sha256(gen.git('show', full['source_commit']+':'+recipe['path'])).hexdigest()
        == recipe['sha256'], 'retained recipe not recoverable')
    scope = retained_model_scope(full, reference)
    guards.require(scope == reference['model_evidence_scope'] and
        gen.common.digest(full['compiler']['path']) == full['compiler']['sha256'], 'retained qualification/compiler changed')
    return dict(original_inputs_recovered=count, source_build_sha256=receipt['sha256'],
                model_evidence_scope=scope, model_evidence_commits=reference['model_evidence_commits'])


def qualification():
    full, proof = bank.qualification()
    compact = gen.committed(bank.q.BUILD)
    speed = gen.committed(SPEED)
    guards.require(speed['complete'] and speed['passed'] and speed['ownership_closed'] and not speed['active'] and
        speed['binary'] == full['binary'] and speed['build_source_commit'] == full['source_commit'] and
        speed['qualification']['native'] == proof and len(speed['processes']) == 6 and
        all(r['passed'] and r['complete'] and r['ownership_closed'] and not r['active'] and
            r['native_exit_code'] == 0 and r['fresh_faults'] == [] and r['fault_probe_successful'] and
            r['boot_unchanged'] and r['actual_route_checked'] for r in speed['processes']),
        'closed same-build six-process prefill qualification required')
    retained = compact['retained_validation_reference']
    return full, retained, dict(current=proof, retained=recover_retained(retained),
        closed_prefill_sha256=gen.common.digest(SPEED), retained_reference_sha256=assets.fingerprint(retained)), speed


def environment(setup, root, arm):
    # Keep the qualified arithmetic/placement/safety settings. Remove old heavy
    # timing observers, then add only the prospective diagnostic observers.
    drop = {'STRATA_TRACE', 'STRATA_SNAPSHOT_VERIFY', 'LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER',
            'STRATA_DRAFT_TIMING', 'UR_ENABLE_LAYERS'}
    env = {k:v.replace(setup['root'], str(root)) for k,v in setup['environment'].items()
        if k not in drop and not k.startswith('UR_LOG_') and '_PROFILE' not in k and '_AUDIT' not in k}
    env.update(STRATA_MTP_BATCH_F16='0', STRATA_SYCL_PREFILL_EXPERT_BANK_PACKED=str(arm),
        STRATA_SNAPSHOT_VERIFY='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1', STRATA_TRACE='1',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
    for key in ('UR_LOG_VALIDATION', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO'):
        env[key] = 'level:warning;flush:warning;output:stderr'
    safety = gen.committed(gen.SETTINGS)['environment']
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
                'STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[key] == safety[key], 'qualified safety setting changed: '+key)
    return env


def prospective_plan():
    return dict(order=['retained-3b-packed-off', 'current-e062-packed-on'], workloads=list(WORKLOADS), runs=1,
        question='Do both retained and current engines complete the same two long-generation cases after long prefill, with bound MTP work, actual answer and safe restoration?',
        constants='Official bench.sh --long-generation-only --think --runs1, four warmups then regular-code and varied-prose. Same model/tokenizer/expert profile,131072 INT8 resident context,1256 expert slots,8192 prefill,5 CPU workers,pcie0,adapt0,spec4,suffix0,caches0,greedy,no token graph/no coupled draft,Q8 MTP. Only recorded native source and private packed layout differ.',
        admission='Recover exact native/dependency/recipe and hash every required fixture for both generations. Pin benchmark/helpers/generated configuration/environment/model/runtime. Before each long request verify warmup/prior journal, same native PID/start/SID/executable and effective environment, no errors and complete preceding batched mode2 draft-cell coverage. Each long request must have >=32768 actual fresh input,1024..16384 generated tokens including thought,>=512 answer characters,engine clocks,>=256 offered drafts,normal complete stream/journal/native exit and actual GPU BF16/mode10 route with restored cache/dense state. Stop on source/runtime/owner/fault/request failure; no retries.',
        decision='Two-process diagnostic only; keep each workload sample and measured acceptance separately. One sample per workload/arm cannot qualify strong/weak classification or speed/nonregression. Do not require identical generated text, lengths or draft acceptance. Differences are descriptive and cannot adopt a model route.',
        limits='Parameter validation and stage progress are observers; no clean speed, observer-overhead, full task quality, agent-session,physical262144/session/RESTORE or70tok/s qualification. No private preload activation proof. Requested H2D bytes are not physical traffic or saturation.',
        retention='Keep individual samples, compact route/work/restoration/owner/fault/source checks and decision. Retire successful raw logs/JIT after closure. This is the immediate comparison consumer for the retained3b extra generation; select validated generation and retire extra before any new native build.')


def current(record, full_models=False):
    full, retained, proof, speed = qualification()
    guards.require(proof == record['qualification'] and prospective_plan() == record['plan'] and
        speed['original_configuration'] == record['original_configuration'], 'qualification or comparison plan changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['benchmark_fingerprint'] and
        gen.common.digest(gen.SETTINGS) == record['settings_sha256'], 'benchmark/settings changed')
    setup = original(record['original_configuration'])
    for plan, binary in zip(record['plans'], (retained['binary'], full['binary'])):
        guards.require(plan['binary'] == binary and environment(setup, Path(plan['root']), plan['arm']) == plan['environment'],
            'arm source, binary or immutable environment changed')
        for path, sha in plan['generated_inputs'].items():
            guards.require(gen.common.digest(path) == sha, 'generated comparison input changed: '+path)
    guards.require(assets.machine_identity() == record['machine_identity'] and
        assets.model_pins(record['plans'][0]['config'], record['model_assets'], full=full_models) == record['model_assets'],
        'platform/model identity changed')
    gen.runtime_assets_unchanged(record)
    return setup


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh diagnostic root/result required; never retry')
    full, retained, proof, speed = qualification()
    setup = original(speed['original_configuration'])
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(source)
    models = assets.model_pins(setup['config'], speed['model_assets'], full=True)
    ROOT.mkdir(mode=0o700)
    plans = []
    for arm, build in enumerate((retained, full)):
        root = ROOT/str(arm); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root/name).mkdir()
        config = copy.deepcopy(setup['config'])
        config['args'][config['args'].index('--expert-profile')+1] = str(root/'expert-profile.bin')
        config.update(log=str(root/'engine.stderr'), env={})
        bank.admission.require_batched_warmup(config)
        shutil.copyfile(P/'data/expert-profile.bin', root/'expert-profile.bin')
        guards.require(gen.common.digest(root/'expert-profile.bin') == setup['profile_sha256'], 'expert profile changed')
        (root/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(dict(binary=build['binary'], native_validation_probe=None), indent=2)+'\n')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plans.append(dict(arm=arm, root=str(root), binary=build['binary'], build_source_commit=build['source_commit'],
            config=config, environment=environment(setup, root, arm),
            generated_inputs={str(root/n):gen.common.digest(root/n) for n in ('config.json', 'plan.json', 'expert-profile.bin')},
            command=['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
                '--long-generation-only', '--think', '--runs', '1', '--profile-request-gates',
                '--workload-id', WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root/'bench')]))
    record = dict(mode='engine-compare', stage='prepared', source_commit=source, qualification=proof,
        original_configuration=speed['original_configuration'], plans=plans, processes=[], plan=prospective_plan(),
        runtime_asset_delta=setup['runtime_asset_delta'], settings_sha256=gen.common.digest(gen.SETTINGS),
        model_assets=models, machine_identity=assets.machine_identity(), source_count=len(pins),
        benchmark_fingerprint=assets.fingerprint(pins), complete=False, passed=False, active=False, ownership_closed=True,
        adoption_qualified=False, sustained_decode_qualified=False,
        budgets=dict(wall=3600, cpu=54000, rss_cap=96<<30, text_cap=64<<20, file_cap=64<<20, total_cap=64<<20))
    current(record)
    (ROOT/'plan.json').write_text(json.dumps(record, indent=2)+'\n')
    RESULT.write_text(json.dumps(record, indent=2)+'\n')
    print('Prepared two-process/two-workload diagnostic; commit exact plan before run.')


def official(root, result, workloads=WORKLOADS):
    suite = result['results']['long_generation']
    rows = gen.collect_samples(suite)
    gen.require_complete_generation(suite, rows, workloads)
    guards.require(result['complete'] and not result['results']['failures'] and
        result['params']['runs'] == 1 and result['params']['think'] is True and
        result['params']['long_generation_only'] is True and
        suite['selected_workloads'] == list(workloads) and suite['diagnostic_subset'] is (len(workloads)==1), 'official suite incomplete or subset')
    journal, journal_rows = gen.journal_evidence(root)
    guards.require(journal['valid'] and journal['complete'], 'invalid/incomplete official journal')
    # The suite adds reasoning-token detail, run/order and the controller label.
    normalized = [{k:v for k,v in row.items() if k not in ('controller_eligible', 'reasoning_tokens')} for row in rows]
    guards.require(normalized == journal_rows, 'official samples differ from journal')
    path, = (root/'bench').glob('*/requests.jsonl')
    events = [json.loads(line) for line in path.read_text().splitlines()]
    warm = gen.warmup_reconciliation(events, workloads)
    identity = events[0]; comparison = result['comparison_identity']
    from strata_bench import client_fingerprint
    workload = dict(workload_id=WORKLOAD_ID, nonce=WORKLOAD_ID, nonce_policy='fixed-id')
    guards.require(identity['kind'] == 'identity' and identity['mode'] == comparison['mode'] == 'suite' and
        identity['requested_runs'] == 1 and identity['thinking'] is comparison['thinking'] is True and
        identity['source'] == comparison['client'] == client_fingerprint() and
        identity['context'] == comparison['context'] == result['machine']['context'] == 131072 and
        identity['workload'] == comparison['workload'] == result['params']['workload'] == workload and
        all(identity[k] == comparison[k] == result['machine'][k] for k in ('model', 'quant')), 'client/model/workload binding differs')
    return rows, dict(warm, valid=True, journal_sha256=gen.common.digest(path)), events


def native_routes(text, events, arm, workloads=WORKLOADS):
    assets.verify_pool(text, 5)
    samples = [e['sample'] for e in events if e['kind'] == 'result']
    draft = gen.request_evidence(text, samples)
    headers = list(re.finditer(r'^strata serve: prompt .*$', text, re.M))
    routes = []
    for i, sample in enumerate(samples):
        if i < 4: continue
        cells = sample['read_tokens']-1
        segment = text[headers[i-1].end():headers[i].start()]
        route = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 '
            r'residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) '
            r'residual_gpu=(\d+) residual_host=0 residual_staging=0 ', segment, re.M)
        guards.require(len(route) == 1 and tuple(map(int, route[0])) == (cells, cells*20480, cells, cells*20480) and
            len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='
                +str(min(cells,81920))+r' mixed_fp16_bytes=0$', segment, re.M)) == 1, 'actual long GPU/BF16/mode10 route differs')
        cycle, publications = bank.admission.loan_cycle(segment)
        dense = bank.admission.records(segment, 'strata prefill dense', 3)
        guards.require([(r['phase'], r['residency'], r['activations']) for _,r in dense] ==
            [('retired','Full','0'), ('bank','Bank','0'), ('restored','Full','48')] and
            publications[0][0] < dense[0][0] < dense[1][0] < dense[2][0] < publications[1][0],
            'decode dense/cache state was not restored in the loan')
        routes.append(dict(actual_GPU_BF16_rows=cells, actual_MoE_mode=10, draft_work=draft[i],
            private_bank=bank.geometry(segment, arm), restored_dense_layers=48,
            cache_payload_and_map_match=True, loan_generation=cycle['generation']))
    guards.require(len(routes) == len(workloads), 'missing long route')
    return routes


def run():
    record = gen.committed(RESULT); current(record)
    frozen = json.loads((ROOT/'plan.json').read_text())
    mutable = {'processes', 'stage', 'complete', 'passed', 'active', 'ownership_closed', 'decision'}
    guards.require({k:v for k,v in record.items() if k not in mutable} ==
        {k:v for k,v in frozen.items() if k not in mutable} and not record['complete'] and
        record['stage'] in ('prepared','collecting') and not record['active'] and record['ownership_closed'] and
        all(r['passed'] for r in record['processes']), 'closed or changed diagnostic plan')
    arm = len(record['processes']); guards.require(arm < 2, 'diagnostic complete')
    plan = record['plans'][arm]; root = Path(plan['root']); env = plan['environment']
    guards.require(not (root/'record.json').exists() and not (root/'launch.json').exists(), 'existing native attempt; never retry')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15)
    cursor = next(line.split(': ',1)[1] for line in before.splitlines() if line.startswith('-- cursor: '))
    row = dict(arm=arm, passed=False, complete=False, native_exit_code=None)
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(row, commands=owner.commands if owner else []), indent=2)+'\n')
    def interact(io, leader):
        for i,name in enumerate(WORKLOADS):
            for stage in ('begin','end'):
                while True:
                    line = io.line(3300)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                guards.require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='long-generation/'+name, run=0), 'request gate differs')
                if stage == 'begin':
                    native = json.loads((root/'engine-start.json').read_text())['native']
                    identity = supervision.identity(native['pid'])
                    guards.require(identity and identity['sid'] == leader['pid'] and identity['start_ticks'] == native['start_ticks'] and
                        Path('/proc',str(native['pid']),'exe').resolve() == Path(plan['binary']['path']).resolve(), 'native owner/executable differs')
                    actual = dict(s.split(b'=',1) for s in Path('/proc',str(native['pid']),'environ').read_bytes().split(b'\0') if b'=' in s)
                    guards.require(all(actual.get(k.encode()) == v.encode() for k,v in env.items()) and
                        all(k not in actual for k in (b'LD_PRELOAD',b'STRATA_UR_PROBE_EXPECTED_LOADER')), 'effective native environment differs')
                    journal, = (root/'bench').glob('*/requests.jsonl')
                    events = [json.loads(line) for line in journal.read_text().splitlines()]
                    guards.require(len(events) == 10+2*i and events[-1]['kind'] == 'start' and
                        events[-1]['phase'] == 'long-generation/'+name, 'warmup/prior request journal extent differs')
                    gen.warmup_reconciliation(events[:-1], WORKLOADS[:i])
                    previous = [e['sample'] for e in events[:-1] if e['kind'] == 'result']
                    guards.require(all(gen.sample_eligible(r) for r in previous[4:]), 'prior long request ineligible')
                    text = (root/'engine.stderr').read_text(errors='replace')
                    gen.request_evidence(text, previous)
                    guards.require(not re.search(FAILURE, text), 'warmup/prior native error')
                    row.update(native_identity=native, effective_environment_verified=True)
                    persist()
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); guards.require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    owner = supervision.Owner(root/'owned', persist)
    try:
        entry,_,_ = owner.run('bench', plan['command'], env, P, interaction=interact, **record['budgets'])
        paths = list((root/'bench').glob('*/results.json'))
        if paths:
            result = json.loads(paths[0].read_text())
            row['requests'] = gen.collect_samples(result['results'].get('long_generation', {}))
        guards.require(supervision.closed(entry) and entry.get('exit_code') == 0, 'owned official benchmark failed')
        guards.require(len(paths) == 1, 'missing/ambiguous official result')
        rows, journal, events = official(root, result)
        native = gen.native_end(root, dict(binary=plan['binary'],native_validation_probe=None), plan, boot, entry['pid'])
        text = (root/'engine.stderr').read_text(errors='replace')
        routes = native_routes(text, events, arm)
        if record['processes']:
            guards.require([r['request_sha256'] for r in rows] ==
                [r['request_sha256'] for r in record['processes'][0]['requests']], 'paired input bodies differ')
        current(record, full_models=arm == 1)
        row.update(passed=True, complete=True, requests=rows, journal=journal, routes=routes,
            native_exit_code=native['exit_code'], native_identity=native['native'],
            binary_sha256=plan['binary']['sha256'], build_source_commit=plan['build_source_commit'])
    except BaseException as error:
        row['error'] = str(error)
        if (root/'engine.stderr').exists(): row['error_excerpt'] = (root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        row.update(ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands),
            fresh_faults=None, fault_probe_successful=False, boot_unchanged=False)
        try:
            after = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'], text=True, timeout=15)
            row.update(fresh_faults=[s for s in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)], fault_probe_successful=True)
            row['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            guards.driver_guard()
        except Exception as error: row['fault_probe_error'] = str(error)
        row['passed'] = bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and
            row['fresh_faults'] == [] and row['boot_unchanged'] and 'fault_probe_error' not in row)
        persist(); record['processes'].append(row)
        record.update(stage='collecting' if row['passed'] and arm == 0 else 'closed',
            ownership_closed=row['ownership_closed'], active=row['active'],
            complete=bool(row['passed'] and arm == 1), passed=bool(row['passed'] and arm == 1))
        if record['passed']:
            record['decision'] = dict(pilot_complete=True, repetitions_per_workload_per_arm=1,
                MTP_classification_qualified=False, speed_comparison_qualified=False,
                model_adoption_qualified=False, next='Inspect generated answer/task evidence and measured MTP regimes; freeze repeated clean comparison or next bounded diagnostic before advancement.')
        RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:row.get(k) for k in ('arm','passed','complete','ownership_closed','error')}))
    return 0 if row['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run'))
    args = parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners([]); guards.driver_guard()
        return prepare() if args.action == 'prepare' else run()


if __name__ == '__main__': sys.exit(main())
