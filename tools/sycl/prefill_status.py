#!/usr/bin/env python3
"""Main-owned, two-process 117K status-wrapper attribution; no speed adoption.

The only requested process difference is the UR tracing layer (warning output).
Parameter validation, source, workload, placement and safety settings stay fixed.
Commit this controller, then its generated plan, before executing either process.
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
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_resources as previous

c = previous.c
gen, assets, guards, supervision = c.gen, c.assets, c.guards, c.supervision
P, B, A = c.P, c.B, c.A
ROOT = B / 'prefill-status-wrapper-v1'
RESULT = A / 'prefill-status-current.json'
CPU = A / 'prefill-host-cpu-current.json'


def qualification():
    full, count = c.q.current_build()
    cpu = gen.committed(CPU)
    guards.require(cpu['passed'] and cpu['complete'] and cpu['ownership_closed'] and
        cpu['status_partition_verified']['cpu_wall_conserved'] and
        cpu['source_sha256']['include/strata/sycl/cpu_wait_profile.hpp'] ==
            gen.common.digest(P / 'include/strata/sycl/cpu_wait_profile.hpp'),
        'same-source CPU status qualification required')
    changed = gen.git('diff', '--name-only', 'bb71e78db7b6a4930945a2d564ec6cd42d0ad245',
        full['source_commit'], '--', 'src', 'include', 'CMakeLists.txt', 'cmake').decode().splitlines()
    guards.require(set(changed) == {'include/strata/sycl/cpu_wait_profile.hpp',
        'src/sycl/compat.cpp', 'src/prefill/moe_window.inc'},
        'only the reviewed native status instrumentation may change')
    return full, dict(original_inputs_recovered=count,
        source_build_record_sha256=gen.common.digest(c.q.BUILD_RECORD),
        component_sha256=gen.common.digest(c.q.COMPONENT), cpu_sha256=gen.common.digest(CPU),
        changed_native_inputs=changed), c.validation.preload_qualification()


def original_plan():
    closed = gen.committed(previous.RESULT)
    guards.require(closed['complete'] and closed['ownership_closed'] and not closed['active'] and
        closed['passed'] is False and closed['closed_analysis']['attribution_checks_passed'] and
        closed['closed_analysis']['original_failure_preserved'], 'preserve original failed collection and analysis')
    ref = closed['original_plan']
    raw = gen.git('show', ref['commit'] + ':' + ref['path'])
    guards.require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'original configuration not recoverable')
    return json.loads(raw), ref


def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(), 'fresh status plan required; never retry')
    full, proof, probe = qualification()
    original, ref = original_plan()
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(source)
    config = copy.deepcopy(original['config'])
    c.admission.require_batched_warmup(config)
    models = assets.model_pins(config, original['model_assets'], full=True)
    settings = gen.committed(gen.SETTINGS)
    record = dict(mode='engine-compare', source_commit=source, build_source_commit=full['source_commit'],
        binary=full['binary'], qualification=proof, native_validation_probe=probe,
        original_configuration=ref, source_count=len(pins), benchmark_fingerprint=assets.fingerprint(pins),
        settings_sha256=gen.common.digest(gen.SETTINGS), tokenizer_assets=original['tokenizer_assets'],
        model_assets=models, machine_identity=assets.machine_identity(),
        runtime_asset_delta=original['runtime_asset_delta'], admitted_loan=c.admitted_loan(), plans=[], runs=[],
        gates='Two separate processes, absent/present UR tracing layer at warning logging only. Parameter validation and all runtime safety settings fixed before exec. Each process: one short batched warmup and one fresh117489-input/24-output thinking request. Require same-PID/pre-main validation,96 borrowed-owner retirements, all planned MoE work conserved, complete host/wait accounting,96 status scopes and exact disjoint phase counts, native exit0/owned closure, unchanged inputs and no fresh fault. Stop after any failure; no retry.',
        limits='Attribution only. Requested trace-layer presence is not a measurement of full trace-output cost. No clean throughput/quality/sustained decode/physical262144/session/RESTORE qualification; enqueue can block on device work. Physical PCIe/GDDR traffic and arithmetic ceilings stay unknown.',
        retention='After validation keep per-process status/host aggregates, request samples, work conservation and source/plan identity. Retire normal logs/raw receipts at the status decision; do not archive full profiles.',
        budgets=dict(wall=1800, cpu=54000, rss_cap=96 << 30, text_cap=64 << 20,
                     file_cap=64 << 20, total_cap=64 << 20), complete=False, passed=False)
    ROOT.mkdir(mode=0o700)
    for arm in range(2):
        root = ROOT / str(arm)
        root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        env = {k:v.replace(original['root'], str(root)) for k,v in original['environment'].items()}
        env.update(UR_ENABLE_LAYERS=('UR_LAYER_TRACING,' if arm else '') + 'UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:warning;flush:warning;output:stderr')
        for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys',
                    'EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
            guards.require(env[key] == settings['environment'][key], 'safety setting changed')
        previous.require_observer_environment(env)
        cfg = copy.deepcopy(config)
        cfg['args'][cfg['args'].index('--expert-profile')+1] = str(root / 'expert-profile.bin')
        cfg.update(log=str(root / 'engine.stderr'), env={})
        shutil.copyfile(P / 'data/expert-profile.bin', root / 'expert-profile.bin')
        guards.require(gen.common.digest(root / 'expert-profile.bin') == original['profile_sha256'], 'profile changed')
        (root / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'),
            '--port', str(port), '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '1',
            '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(root / 'bench')]
        native = dict(binary=full['binary'], native_validation_probe=probe)
        (root / 'plan.json').write_text(json.dumps(native, indent=2)+'\n')
        record['plans'].append(dict(arm=arm, root=str(root), environment=env, config=cfg, command=command,
            config_sha256=gen.common.digest(root / 'config.json'), profile_sha256=original['profile_sha256'],
            native_plan_sha256=gen.common.digest(root / 'plan.json')))
    gen.runtime_assets_unchanged(record)
    RESULT.write_text(json.dumps(record, indent=2)+'\n')
    print('Prepared two status diagnostics; commit exact generated inputs before run0.')


def analyze(log, sample, loan):
    attribution = previous.analyze(log, sample, loan)
    part = previous.long_interval(log, sample)
    counts = dict.fromkeys(gen.common.CPU_STATUS_NAMES, 96)
    counts.update(precheck=192, queue_context=192)
    status = gen.common.extract_status_observer(part, 96, counts)
    cpu = gen.common.extract_cpu_observers(part, 4)
    waits = {k:float(cpu['wait'][k]) for k in ('wall_s','main_cpu_s','wait_cpu_s',
        'wait_wall_s','outside_wait_cpu_s')}
    host = {k:float(cpu['host']['moe_status_'+k]) for k in
        ('cpu_s','wall_s','outside_wait_cpu_s','outside_wait_wall_s')}
    # All per-layer/window checks above precede this permanent reduction.
    return dict(status=status, cpu=waits, original_moe_status_region=host,
        work=attribution['window_census'], owners=attribution['borrowed_owner_evidence'],
        native_dense=attribution['native_dense'], refill_ms=attribution['expert_cache_refill_wall_ms'])


def run(arm):
    record = gen.committed(RESULT)
    guards.require(not record['complete'] and len(record['runs']) == arm and
        all(r['passed'] for r in record['runs']), 'ordered diagnostics; stop on failure')
    full, proof, probe = qualification()
    guards.require(full['binary'] == record['binary'] and proof == record['qualification'] and
        probe == record['native_validation_probe'], 'qualified build/helper changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins) == record['source_count'] and assets.fingerprint(pins) == record['benchmark_fingerprint'],
        'benchmark source changed')
    plan = record['plans'][arm]; root = Path(plan['root'])
    guards.require(not (root / 'record.json').exists(), 'never repeat a native launch')
    for path, sha in [(gen.SETTINGS,record['settings_sha256']),
        (root/'config.json',plan['config_sha256']), (root/'expert-profile.bin',plan['profile_sha256']),
        (root/'plan.json',plan['native_plan_sha256']), *record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path) == sha, 'frozen input changed: '+str(path))
    gen.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(plan['config'],record['model_assets']) == record['model_assets'] and
        assets.machine_identity() == record['machine_identity'], 'model/platform changed')
    boot = c.read_boot()
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor = next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    result = dict(arm=arm, complete=False, passed=False, active=False)
    owner = None
    def persist():
        result['active'] = bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(result,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner = supervision.Owner(root/'owned',persist)
    try:
        entry,_,_ = owner.run('bench',plan['command'],plan['environment'],P,**record['budgets'])
        result.update(complete=supervision.closed(entry),exit_code=entry.get('exit_code'))
        guards.require(result['complete'] and entry['exit_code'] == 0,'official benchmark failed')
        path, = (root/'bench').glob('*/results.json')
        samples,_ = c.official(root,json.loads(path.read_text()),1)
        native = json.loads((root/'engine-exit.json').read_text())
        guards.require(native['exit_code'] == 0 and native['boot_id'] == boot and
            native['binary'] == record['binary'] and native['args'] == ['--serve',*plan['config']['args']] and
            native['native_validation_probe'] == probe,'native identity/closure differs')
        log = (root/'engine.stderr').read_text(errors='replace')
        assets.verify_pool(log,5)
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',log),'runtime failure')
        result.update(sample=samples[0],native_identity=native['native'],
            validation_activation=c.activation(log,native['native']['pid'],probe),
            attribution=analyze(log,samples[0],record['admitted_loan']))
        _, after, after_probe = qualification()
        guards.require(after == proof and after_probe == probe and gen.benchmark_pins(record['source_commit']) == pins and
            assets.machine_identity() == record['machine_identity'],'inputs changed during collection')
        gen.runtime_assets_unchanged(record)
        assets.model_pins(plan['config'],record['model_assets'],full=True)
        result['passed'] = True
    except BaseException as error:
        result['error'] = str(error)
        if (root/'engine.stderr').exists(): result['error_excerpt']=(root/'engine.stderr').read_text(errors='replace')[-4096:]
    finally:
        result.update(ownership_closed=owner.active is None and all(supervision.closed(x) for x in owner.commands),
            fresh_faults=None,fault_probe_successful=False,boot_unchanged=False)
        try:
            journal=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            result['fresh_faults']=[x for x in journal.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
            result.update(fault_probe_successful=True,boot_unchanged=c.read_boot()==boot,driver_after=guards.driver_guard())
        except Exception as error: result['final_probe_error']=str(error)
        result['passed']=bool(result['passed'] and result['ownership_closed'] and result['boot_unchanged'] and
            result['fault_probe_successful'] and result['fresh_faults']==[] and 'final_probe_error' not in result)
        persist(); record['runs'].append(result)
        record.update(complete=len(record['runs'])==2 or not result['passed'],
            passed=len(record['runs'])==2 and all(r['passed'] for r in record['runs']))
        RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('arm','complete','passed','ownership_closed','error')}))
    return 0 if result['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('arm',nargs='?',type=int,choices=(0,1))
    args=parser.parse_args()
    guards.require((args.action=='prepare') == (args.arm is None),'use prepare, run0 or run1')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run(args.arm)


if __name__=='__main__':sys.exit(main())
