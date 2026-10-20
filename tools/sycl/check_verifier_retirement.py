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
    root = B / 'decode-query-profile-component-v6' if profile else ROOT
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
                gate='Same-build provider17, submission10, normal verifier retirement8, grouped arithmetic16, query observer1, residency publication9, cache tail10, scratch host22 and actual Prefill device1+15+retirement1. Run smoke before full and retirement. Require exact init/moved/shrunk layouts, growth refusal, allocation and GEMM failure poison, reset recovery, metadata/USM/queue-domain refusal, untouched parent bytes and guards. Retirement must exit1 at the failed synchronization with owners retained. Original failed early51+6 stays unchanged. Any failure stops without retry or model execution.',
                consumer='Prerequisite for checked parent-loan implementation; these110 cases never admit early PCIe scheduling or borrowed mode10.',
                limits='Actual Prefill init/relayout and reset failure paths, without model weights or inference. The reset case changes a fully drained external queue to another context, exercising host-domain synchronization rejection; it does not inject a GPU fault or asynchronous device exception. Parent-loan generation/global graph-reader retirement remain unimplemented. No model speed, task quality, physical262144/session/RESTORE qualification. Existing grouped arithmetic oracle shares scalar projection primitives.')
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
                fixtures.append(('cache-tail', 'cache_tail_host_test', []))
                fixtures.append(('prefill-scratch', 'prefill_scratch_host_test', []))
                fixtures.extend(('prefill-device-'+mode, 'sycl_prefill_scratch_test', ['--case',mode])
                    for mode in ('smoke','full','retirement'))
            for label, name, args in fixtures:
                fixture = build['fixtures'][name]
                command = [fixture['path'], *args]
                child_env = env
                delta = {}
                retirement = name == 'sycl_prefill_scratch_test' and args == ['--case','retirement']
                if name == 'sycl_prefill_scratch_test':
                    delta = dict(STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1',STRATA_PREFILL_MMQ='0',
                        STRATA_PREFILL_MOE_COMPACT_FP16='1',STRATA_SYCL_PREFILL_NATIVE16='bf16',
                        STRATA_SYCL_PREFILL_NATIVE16_QSA='1',STRATA_SYCL_PREFILL_GDN_CHUNK64='0',
                        STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm')
                    child_env = dict(env,**delta)
                entry, out, err = owner.run(label, command, child_env, P, wall=180, cpu=600,
                    rss_cap=4 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
                stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
                record['last_fixture'] = dict(name=name,command=command,exit_code=entry.get('exit_code'),
                    ownership_closed=supervision.closed(entry),stdout_sha256=digest(out),stderr_sha256=digest(err))
                guards.require(supervision.closed(entry) and entry.get('exit_code') == (1 if retirement else 0),
                    'fixture failed: ' + (stdout + stderr)[-8192:])
                if name == 'file_expert_source_test':
                    expected = 'arena_copy_dispatch_test cases=17 native_variable_stride=PASS source_bounds=PASS grouping=PASS limits=PASS cpu_fallback=PASS alias_exclusion=PASS callbacks=PASS'
                    guards.require(expected in stdout and 'file_expert_source_test: PASS' in stdout and
                        not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed', stderr),
                        'Arena provider/planner contracts failed: ' + (stdout + stderr)[-8192:])
                    cases = [expected]
                elif name == 'prefill_scratch_host_test':
                    cases = [x for x in stdout.splitlines() if x.startswith('prefill_scratch ')]
                    expected = [f'prefill_scratch case={name} status=PASS' for name in
                        ('guard_alignment','shape_bounds','gemm_64mib','exact_capacity','payload_overflow',
                         'guard_overflow','used_overflow','capacity_bounds','slack_overflow',
                         'initial_layout','relayout_generation','failed_init','failed_relayout',
                         'nested_update','failed_take_and_reset','qsa_global_ordinal','qsa_resident_plan',
                         'qsa_stream_ring_formats','qsa_bad_metadata','qsa_product_overflow',
                         'qsa_block_overflow','qsa_tail_rounding')]
                    guards.require(cases == expected and 'prefill_scratch_fixture cases=22 status=PASS' in stdout,
                        'scratch arithmetic/layout-state contract failed: '+(stdout+stderr)[-8192:])
                    record['scratch_layout_contract_passed'] = True
                elif name == 'sycl_prefill_scratch_test':
                    if retirement:
                        marker = 'prefill_scratch_device retirement=reset queue_domain_changed=1 owners_live=1 old_queue_drained=1 foreign_in_order=1'
                        guards.require(stdout.count(marker)==1 and
                            'prefill reset/destruction completion uncertain; retaining owners until process exit' in stderr and
                            'Strata SYCL engine currently uses runtime device 0 context' in stderr and
                            'UNEXPECTED_RETURN' not in stdout and 'prefill_scratch_device_complete=' not in stdout and
                            'prefill scratch fixture:' not in stderr and
                            not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called',stderr),
                            'Prefill reset did not fail before owner release: '+(stdout+stderr)[-8192:])
                        cases = [marker+' expected_exit=1 status=PASS']
                        record['prefill_reset_retirement_contract_passed'] = True
                    else:
                        mode = args[1]
                        names = ('first_take_refusal',) if mode=='smoke' else (
                            'exact_init','moved_relayout','smaller_relayout','growth_refusal',
                            'one_byte_short_poison','poison_entries','reset_recovery','relayout_short_poison',
                            'gemm_init_refusal','gemm_rebind_refusal','malformed_qsa_refusal',
                            'parent_bounds_refusal','host_usm_refusal','foreign_queue_refusal','canary_and_retirement')
                        cases = [x for x in stdout.splitlines() if x.startswith('prefill_scratch_device case=')]
                        guards.require(cases == [f'prefill_scratch_device case={n} status=PASS' for n in names] and
                            f'prefill_scratch_device_fixture mode={mode} cases={len(names)} status=PASS' in stdout and
                            stdout.count('prefill_scratch_device_complete=1 model_PP=UNRUN')==1 and
                            not re.search('UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed|Strata SYCL cleanup:',stderr),
                            'actual Prefill layout/failure contract failed: '+(stdout+stderr)[-8192:])
                        if mode=='full': record['prefill_device_layout_contract_passed'] = True
                elif name == 'cache_tail_host_test':
                    cases = [x for x in stdout.splitlines() if x.startswith('cache_tail ')]
                    expected = [f'cache_tail case={name} status=PASS' for name in
                        ('sized_boundary','sized_cross_boundary','sized_all','zero_need','no_fit',
                         'uniform_boundary','layer_tail_floor','bounds','invalid_layout','request_rounding')]
                    guards.require(cases == expected and 'cache_tail_fixture cases=10 status=PASS' in stdout,
                        'cache tail contract failed: '+(stdout+stderr)[-8192:])
                    record['cache_tail_contract_passed'] = True
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
                    expected_exit=1 if retirement else 0,exit_code=entry.get('exit_code'),
                    environment_delta=delta,
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
