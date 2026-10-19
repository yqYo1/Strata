#!/usr/bin/env python3
"""Main-owned Arena planning and production-verifier copy gate; no model/speed claim."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ENGINE = B / 'main-native-gdn-engine'
ROOT = B / 'verifier-copy-safety-v7'
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
RESULT = A / 'verifier-copy-safety-current.json'
# Separately admitted diagnostic assets, unchanged from the closed CPU cohort.
# This does not refresh the qualified shared runtime or permit long model work.
DELTA = {
    '/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39395': (
        'dfa844e5afd8e8f74bf5ce9da3ea2837706d9116bc8efeaed90efba5acaf7b1c',
        '50adb6b727d027309039803033d41efeae4b248589ee363f321c0d538c7de3ff'),
    '/usr/lib/x86_64-linux-gnu/libudev.so.1.7.8': (
        '4298228175fa62a36af88b1afd4406cdd8b1bf166621f957d0dbbde45a698fe5',
        '0d6934adafb2ef3c640382440c14d6bd60f391e688c000796ffcea0c72849269'),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    profile = sys.argv[1:] == ['--query-profile']
    guards.require(len(sys.argv) == 1 or profile, 'use no options or --query-profile; never retry a closed root')
    root = B / 'decode-query-profile-component-v2' if profile else ROOT
    result = A / 'decode-query-profile-component-current.json' if profile else RESULT
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        guards.driver_guard()
        build_path = A / 'native-gdn-engine-build.json'
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(build_path.relative_to(P))], cwd=P)
        guards.require(committed == build_path.read_bytes(), 'commit the exact build result first')
        compact = json.loads(committed)
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and
            not build['active'] and all(build[k] == compact[k] for k in
            ('source_commit', 'binary', 'fixtures', 'targets', 'configure_argv', 'compiler')),
            'closed matching committed build required')
        sources = [build['project'], *build['dependencies']]

        def unchanged():
            for source in sources:
                for name, expected in source['source_sha256'].items():
                    guards.require(digest(Path(source['root']) / name) == expected, 'build input changed: ' + name)
            for item in [build['binary'], *build['fixtures'].values()]:
                guards.require(digest(item['path']) == item['sha256'], 'build output changed')

        unchanged()
        guards.require('tools/sycl/check_verifier_retirement.py' in build['project']['source_sha256'],
            'the build must pin this controller')
        settings = json.loads(SETTINGS.read_text())
        guards.require(set(DELTA) <= set(settings['runtime_elfs']), 'unknown diagnostic runtime delta')

        def runtime_unchanged():
            for name, item in settings['runtime_elfs'].items():
                expected = item['sha256']
                if name in DELTA:
                    old, expected = DELTA[name]
                    guards.require(old == item['sha256'], 'original runtime baseline differs')
                guards.require(digest(name) == expected, 'runtime asset changed: ' + name)

        runtime_unchanged()
        previous = subprocess.check_output(['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P)
        guards.require(previous == RESULT.read_bytes(), 'commit the prior component outcome first')
        previous_reference = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            path=str(RESULT.relative_to(P)),sha256=hashlib.sha256(previous).hexdigest(),
            outcome='Original v6 host-task gate failed after51 existing passes; this new finite-DMA plan does not change that outcome.')
        guards.require(not root.exists(), 'inspect/close the existing owned check; no automatic retry')
        root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        env = dict(settings['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        record = dict(source_commit=build['source_commit'], binary=build['binary'], fixtures=build['fixtures'],
            environment_reference=str(SETTINGS.relative_to(P)), environment_sha256=digest(SETTINGS),
            runtime_asset_delta={n: dict(qualified_sha256=v[0], candidate_sha256=v[1]) for n,v in DELTA.items()},
            checks=[], complete=False, passed=False, active=False, model_performance_measured=False,
            previous_result_reference=previous_reference,
            per_fixture_budgets=dict(wall_seconds=180,cpu_seconds=600,rss_bytes=4<<30,text_bytes=64<<20,total_bytes=64<<20),
            model_quality_qualified=False, model_route_executed=False, ordinary_arena_copy_default_enabled=False,
            consumer='Prerequisite for the default-off event-gated early copied-expert model check. Retire normal traces after compact results are verified and committed.',
            gate='Same-build native provider17, submission10, verifier8, staged grouped-arithmetic16 and early copied-graph6 cases. The first early case uses32 warmed64MiB host-USM-to-device copies on the same in-order copy queue (2GiB total,128MiB scratch). Gate enqueue and production fetch must each return within100ms; independent resident work must finish, copied completion stay NotReady for50ms and an atomic probe see untouched output while the finite gate remains pending. Then require natural completion and full payload/guard/retirement checks. No resize/retry on a short gate. Require exact unchanged-Q8_1 arithmetic against separate GPU projections, finite/nonzero reference outputs, full-window/row mapping, checked owners and unchanged boot/runtime/source, no fresh GPU fault. Any failure blocks advancement.',
            limits='The six early cases check real-queue synthetic ordering, replay, source retirement and refusal, not native projection equivalence or model overlap. Grouped component covers all seven current model GU/Down pairs plus supported IQ4_XS/Q2_0 at H2560/FF640, windows1/4, groups10/16 and entries10/40, using two distinct blobs repeated in metadata. The oracle shares native scalar projection primitives; it isolates copy/grouping from separate execution, not common-mode projection defects or model quality. No invalid GPU address, stalled kernel, model, physical262144/session/RESTORE or throughput qualification.')
        if profile:
            record.update(mode='normal-route-query-profile-components', model_route_executed=False,
                gate='Same-build provider17, submission10, normal verifier retirement8, grouped arithmetic16, one completed-event query observer case and9 host residency-publication cases. Lend/refill failure at each fanout target must stop publication and withhold reader admission; success preserves exact snapshots. Observer must preserve results, separate all four categories, conserve counts/times and leave counters zero when disabled/reset. Original failed early51+6 result stays unchanged. No early route, queue configuration or model/speed qualification.',
                consumer='Prerequisite for the checked-residency engine small model check; normal51+observer+host9, never admission of early PCIe scheduling.')
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(dict(record, commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(root / 'owned', persist)
        persist()
        try:
            fixtures = [
                    ('provider', 'file_expert_source_test', []),
                    ('submission', 'sycl_submission_test', ['--submission']),
                    ('retirement', 'sycl_verifier_retirement_test', []),
                    ('grouped', 'sycl_mmvq_test', ['--arena-grouped']),
                    (('query-profile', 'sycl_verifier_retirement_test', ['--query-profile']) if profile else
                     ('early', 'sycl_verifier_retirement_test', ['--early']))]
            if profile:
                fixtures.append(('residency-publication', 'residency_publication_host_test', []))
            for label, name, args in fixtures:
                fixture = build['fixtures'][name]
                command = [fixture['path'], *args]
                entry, out, err = owner.run(label, command, env, P, wall=180, cpu=600,
                    rss_cap=4 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                record['last_fixture'] = dict(name=name,command=command,exit_code=entry.get('exit_code'),
                    ownership_closed=supervision.closed(entry),stdout_sha256=digest(out),stderr_sha256=digest(err))
                guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                    'fixture failed: ' + (stdout + stderr)[-8192:])
                if name == 'file_expert_source_test':
                    expected = 'arena_copy_dispatch_test cases=17 native_variable_stride=PASS source_bounds=PASS grouping=PASS limits=PASS cpu_fallback=PASS alias_exclusion=PASS callbacks=PASS'
                    guards.require(expected in stdout and 'file_expert_source_test: PASS' in stdout and
                        not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                        'Arena provider/planner contracts failed: ' + (stdout + stderr)[-8192:])
                    cases = [expected]
                elif name == 'residency_publication_host_test':
                    cases = [x for x in stdout.splitlines() if x.startswith('residency_publication ')]
                    expected = [f'residency_publication phase={phase} fail_at={fail_at} status=PASS'
                                for phase in ('lend','refill') for fail_at in range(-1,3)]
                    expected.append('residency_publication phase=empty status=PASS')
                    guards.require(cases == expected and 'residency_publication_fixture cases=9 status=PASS' in stdout,
                        'residency publication failure injection failed: '+(stdout+stderr)[-8192:])
                    record['residency_publication_contract_passed'] = True
                elif name == 'sycl_submission_test':
                    cases = [x for x in stdout.splitlines() if x.startswith('case ')]
                    guards.require(len(cases) == 10 and 'fixture_status=PASS mode=submission' in stdout and
                        not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                        'submission contracts failed: ' + (stdout + stderr)[-8192:])
                elif name == 'sycl_verifier_retirement_test' and args == ['--query-profile']:
                    cases = [x for x in stdout.splitlines() if x.startswith('case=')]
                    guards.require(cases == ['case=profile expected_exit=0 markers=R PASS'] and
                        'query_profile_fixture_status=PASS cases=1' in stdout and
                        not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                        'query observer contract failed: ' + (stdout + stderr)[-8192:])
                    record['query_profile_contract_passed'] = True
                elif name == 'sycl_verifier_retirement_test' and args == ['--early']:
                    cases = [x for x in stdout.splitlines() if x.startswith('case=')]
                    expected = [('early','F'), ('early-large','G'), ('early-empty','H'),
                                ('early-replay','I'), ('early-plan','J'), ('early-unrecorded','K')]
                    guards.require(cases == [f'case={n} expected_exit=0 markers={m} PASS' for n,m in expected]
                        and 'early_fixture_status=PASS cases=6' in stdout
                        and not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                        'early scheduling contracts failed: ' + (stdout + stderr)[-8192:])
                    gate_rows=re.findall(r'^early_dma_gate bytes=(\d+) copies=(\d+) submit_us=(\d+) fetch_us=(\d+) pending_us=(\d+) resident=PASS output=PASS natural_completion=PASS$',stdout,re.M)
                    guards.require(len(gate_rows)==1, 'missing or ambiguous finite DMA gate evidence')
                    evidence=dict(zip(('bytes','copies','submit_us','fetch_us','pending_us'),map(int,gate_rows[0])))
                    guards.require(evidence['bytes']==2<<30 and evidence['copies']==32 and
                        evidence['submit_us']<=100000 and evidence['fetch_us']<=100000 and evidence['pending_us']>=50000,
                        'finite DMA observation/submit budget failed')
                    record['early_gate_evidence']=dict(evidence,kind='finite_dma',resident_progress=True,
                        output_untouched=True,natural_completion=True)
                    record['early_pcie_copy_contract_passed'] = True
                elif name == 'sycl_verifier_retirement_test':
                    cases = [x for x in stdout.splitlines() if x.startswith('case=')]
                    expected = [('success',0,'S'), ('large',0,'L'), ('empty',0,'Z'), ('partial',0,'P'), ('bounds',0,'B'),
                        ('query',0,'Q'), ('destructor',1,'D'), ('copy-error',1,'E')]
                    guards.require(cases == [f'case={n} expected_exit={e} markers={m} PASS' for n,e,m in expected]
                        and 'fixture_status=PASS cases=8' in stdout, 'verifier contracts failed: ' + (stdout + stderr)[-8192:])
                else:
                    cases = [x for x in stdout.splitlines() if x.startswith('arena_grouped gu=')]
                    expected = [f'arena_grouped gu={gu} down={down} window={window} '
                                f'groups={10 if window==1 else 16} entries={window*10} '
                                'separate_projection=PASS finite=PASS payload=PASS guards=PASS'
                                for gu in (18,21,22,23) for down in (20,42) for window in (1,4)]
                    guards.require(cases == expected and 'arena_grouped_fixture cases=16 status=PASS' in stdout
                        and not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                        'staged grouped arithmetic failed: ' + (stdout + stderr)[-8192:])
                    record['grouped_copy_projection_contract_passed'] = True
                record['checks'].append(dict(command=command, cases=cases, passed=True,
                    stdout_sha256=digest(out), stderr_sha256=digest(err)))
            unchanged(); runtime_unchanged()
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [x for x in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try: record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record.update(driver_guard_error=str(error), passed=False)
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                record['boot_unchanged'] and not record['fresh_faults'])
            persist(); result.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('complete', 'passed', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
