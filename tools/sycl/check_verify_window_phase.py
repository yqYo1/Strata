#!/usr/bin/env python3
"""Main-owned same-build phase-cache GPU gate; repository LICENSE.

This checks small payloads and real graph/callback owners, not Q6 arithmetic,
whole-model inference, ordinary runtime teardown or performance. No rebuilds,
runtime-setting changes, retries or recovery actions are performed here.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_verify_window_model as build
import measurement_owner as supervision
from run_q6_columns import validation_log

P, B, RESULT, base = build.P, build.B, build.RESULT, build.base
ROOT = B / 'verify-window-phase-device-v2'
KEY = 'verify_window_phase_device_v2'
SELF = Path(__file__).resolve()
require, digest, write = build.require, build.digest, build.write
SUBMISSION = [
    'case stream_copy_ordering=PASS',
    'case explicit_stream_profiling=PASS',
    'case default_stream_profiling=PASS',
    *['case queued_kernels=' + str(n) +
      ' numerical=PASS guards=PASS event_timing=PASS kernel_profiling=PASS'
      for n in (1, 119, 120, 121, 241)],
    'case ordinary_host_roundtrips=3 status=PASS', 'case teardown=PASS']
PHASE = ('phase_cache_fixture_status=PASS windows=16 captures=12 '
         'phase_roundtrip=2 retire_recapture=1 guards=PASS')


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('controller_inputs', 'commands')}
    pins = row['controller_inputs']
    out['controller_inputs'] = dict(files=len(pins), aggregate_sha256=hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode()).hexdigest())
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def built():
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[build.KEY] == build.compact(row) and
            row['stage'] == 'built-not-executed' and row['passed'] and
            row['complete'] and row['ownership_closed'] and not row['GPU_executed'],
            'commit the complete unchanged phase build before GPU preparation')
    allowed = build.inputs(row)
    inv = row['compiled_inputs']
    require(digest(inv['path']) == inv['sha256'], 'compiler inventory changed')
    used = json.loads(Path(inv['path']).read_text())
    require(len(used) == inv['files'] and len(used) > 600 and all(
        allowed.get(p) == h and digest(p) == h for p,h in used.items()),
        'actual compiled source/header input changed')
    links = row['actual_link_inputs']
    require(links['uncovered_external_inputs'] == 0 and all(
        allowed.get(p) == h and digest(p) == h for p,h in links['frozen_inputs'].items()),
        'actual external linker input changed')
    for asset in row['binaries'].values():
        require(digest(asset['path']) == asset['sha256'] and
                Path(asset['path']).stat().st_size == asset['bytes'], 'same-build ELF changed')
    return row


def cursor():
    text = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor',
                                    '--no-pager'], text=True, timeout=15)
    return next(x.split(': ',1)[1] for x in text.splitlines() if x.startswith('-- cursor: '))


def faults(start):
    text = subprocess.check_output(['journalctl','-k','-b','--after-cursor',start,
                                    '-n','2048','--no-pager'], text=True, timeout=15)
    require(len(text.splitlines()) < 2048, 'bounded kernel log is incomplete')
    return [x for x in text.splitlines() if re.search(
        r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]


def prepare():
    require(KEY not in base.committed(RESULT) and not ROOT.exists(), 'one fresh phase gate')
    native = built()
    selected = json.loads((build.selected.model.ROOT / 'record.json').read_text())
    revision = base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    controls = base.controls(revision,controller=SELF)
    environment = dict(selected['environment'], TMPDIR=str(ROOT/'tmp'),
        XDG_CACHE_HOME=str(ROOT/'cache'),
        LD_PRELOAD=selected['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=selected['native_validation_probe']['loader']['path'])
    require(environment['UR_ENABLE_LAYERS'] == 'UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' and
            environment['UR_LOG_TRACING'] == 'level:info;flush:info;output:stderr' and
            environment['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
            environment['NEOReadDebugKeys'] == '1' and environment['EnableDirectSubmission'] == '0' and
            environment['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0',
            'preserve qualified full tracing and immutable runtime safety')
    row = dict(stage='prepared',source_commit=revision,controller_inputs=controls,
        native_source_commit=native['source_commit'],
        build_record=dict(path=str(build.ROOT/'record.json'),sha256=digest(build.ROOT/'record.json')),
        binaries=native['binaries'],native_validation_probe=selected['native_validation_probe'],
        runtime_manifest=selected['runtime_manifest'],boot=selected['boot'],driver=selected['driver'],
        environment=environment,
        cases=[dict(name='submission',argv=[native['binaries']['sycl_submission_test']['path'],'--submission']),
               dict(name='phase-cache',argv=[native['binaries']['sycl_verifier_retirement_test']['path'],'--phase-cache'])],
        budgets=dict(wall=180,cpu=240,rss_cap=4<<30,text_cap=8<<20,file_cap=8<<20,total_cap=64<<20),
        plan='Stop on first failure. Same-build submission10 then direct no-fork phase16windows/12captures/48graphs;require exact payload/guards,phase/T roundtrip,clear/recapture,copy conservation and destructor. Exit0,strict pre-main validation proof,closed owners,unchanged source/ELFs/boot/driver/provider,no fresh fault. Full traces allow only the one precisely matched pre-main null-platform validation probe already qualified in Q6 V4.',
        limits='Synthetic payload only;does not execute Q6 math or full record_window/run,dense/head handoff,model,physical262144 or RESTORE. The validation preload holds a runtime reference until exit;ordinary runtime teardown is not qualified. No speed/adoption evidence.',
        retention=dict(owner='Main',byte_budget=64<<20,
            consumer='Immediate phase-separated decode model prerequisite',
            review_point='After recording/committing compact outcomes,retire successful full traces and duplicate runtime cache;keep one diagnostic only for a distinct unresolved failure.'),
        results=[],commands=[],passed=False,complete=False,active=False,ownership_closed=True,
        GPU_executed=False,model_executed=False,inference_adopted=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'): (ROOT/name).mkdir()
    save(row)
    print('Prepared same-build submission/phase-cache GPU plan;commit before run.',flush=True)


def unchanged(row):
    require(base.committed(RESULT)[KEY] == compact(row) and
        base.controls(row['source_commit'],controller=SELF) == row['controller_inputs'],
        'commit and freeze exact phase GPU inputs')
    require(digest(row['build_record']['path']) == row['build_record']['sha256'], 'build receipt changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'] and
        base.guards.driver_guard() == row['driver'], 'boot/driver changed')
    for asset in [*row['binaries'].values(),*row['native_validation_probe'].values()]:
        require(digest(asset['path']) == asset['sha256'], 'native/validation asset changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'], 'runtime inventory changed')
    runtime = json.loads(Path(inv['path']).read_text())
    require(len(runtime) == inv['files'] and all(digest(p) == h for p,h in runtime.items()),
        'runtime provider changed')


def run():
    row = json.loads((ROOT/'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT/'owned').exists(), 'one frozen phase GPU attempt')
    owner, start = None, None
    try:
        unchanged(row);built();start=cursor()
        owner = supervision.Owner(ROOT/'owned')
        owner.persist = lambda: write(ROOT/'execution.json',dict(commands=owner.commands))
        for case in row['cases']:
            print('Phase GPU:',case['name'],flush=True)
            row['GPU_executed'] = True
            entry,out,err = owner.run(case['name'],case['argv'],row['environment'],ROOT,**row['budgets'])
            text,errors=out.read_text(),err.read_text()
            require(supervision.closed(entry) and entry['exit_code'] == 0,
                    'phase fixture failed: '+(text+errors)[-4000:])
            proof=validation_log(errors,entry['pid'],row['native_validation_probe']['loader']['path'],True)
            if case['name'] == 'submission':
                require([x for x in text.splitlines() if x.startswith('case ')] == SUBMISSION and
                        text.count('fixture_status=PASS mode=submission') == 1,
                        'same-build submission10 completion mismatch')
                count=10
            else:
                require(text.splitlines() == [PHASE,'phase_fixture_status=PASS cases=1'],
                        'phase-cache payload/owner completion mismatch')
                count=16
            require(faults(start) == [], 'new GPU fault')
            row['results'].append(dict(name=case['name'],native_exit=0,checks=count,
                ownership_closed=True,validation=proof,stdout_sha256=digest(out),stderr_sha256=digest(err)))
        # This uses the unchanged prepared row; publish results only after all
        # transitive committed-evidence readers have finished.
        prepared = json.loads((ROOT/'record.json').read_text())
        unchanged(prepared);built()
        row.update(stage='closed',passed=True,complete=True)
    except BaseException as error:
        row.update(stage='failed' if row['GPU_executed'] else 'failed-admission',error=repr(error));raise
    finally:
        row.update(commands=owner.commands if owner else [],active=bool(owner and owner.active),
            ownership_closed=owner is None or (owner.active is None and
                all(supervision.closed(x) for x in owner.commands)))
        try:
            row.update(fresh_faults=faults(start) if start is not None else None,
                driver_after=base.guards.driver_guard(),
                boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot'])
            require((start is None or row['fresh_faults']==[]) and row['boot_unchanged'] and row['ownership_closed'],
                    'final fault/boot/owner gate')
        except BaseException as error:
            row.update(stage='failed',passed=False,complete=False,final_probe_error=repr(error))
        save(row)
    require(row['passed'] and row['complete'],row.get('error',row.get('final_probe_error','phase gate incomplete')))
    print('Small same-build phase GPU gate passed;Q6 arithmetic/model/speed remain unqualified.',flush=True)


if __name__ == '__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','run'),'use prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        {'prepare':prepare,'run':run}[sys.argv[1]]()
