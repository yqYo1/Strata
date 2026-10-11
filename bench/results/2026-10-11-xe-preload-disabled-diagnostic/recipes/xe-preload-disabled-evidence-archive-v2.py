import base64
import collections
import fcntl
import hashlib
import importlib.util
import json
import re
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-preload-disabled-diagnostic'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def closed_tree(value):
    if isinstance(value, dict):
        if 'direct_child_reaped' in value:
            m.require(value['direct_child_reaped'] and value['session_empty'] and not value['survivors'], 'closed session proof')
            for identity in value.get('owners', []):
                actual = m.identity(identity['pid'])
                m.require(actual is None or actual['start_ticks'] != identity['start_ticks'], 'original owned process absent')
        for child in value.values():
            closed_tree(child)
    elif isinstance(value, list):
        for child in value:
            closed_tree(child)


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    m.require(R.is_dir() and not (R / 'archive-manifest.json').exists(), 'verified incomplete archive namespace')
    manifest = dict(created_utc=m.utc(), byte_exact_copies=[], lossless_encoded_copies=[],
                    source_only_candidate_qualified=False, cleanup_executed=False,
                    gpu_executed=False, model_executed=False, owner_sha256=m.sha(O))

    def copy(source, relative):
        source = Path(source)
        target = R / relative
        m.require(source.is_file() and not source.is_symlink() , 'finite exact input')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            m.require(target.read_bytes() == source.read_bytes(), 'existing partial copy unchanged')
        else:
            target.write_bytes(source.read_bytes())
        m.require(m.sha(source) == m.sha(target), 'byte exact archived copy')
        manifest['byte_exact_copies'].append(dict(source=str(source), archive=str(target.relative_to(W)),
                                                 bytes=target.stat().st_size, sha256=m.sha(target)))
        return target

    def encoded(source, relative):
        source = Path(source)
        raw = source.read_bytes()
        target = R / relative
        write(target, dict(encoding='base64', original_path=str(source), decoded_bytes=len(raw),
                           decoded_sha256=m.sha(source), data=base64.b64encode(raw).decode('ascii')))
        decoded = base64.b64decode(json.loads(target.read_text())['data'], validate=True)
        m.require(decoded == raw, 'lossless evidence')
        manifest['lossless_encoded_copies'].append(dict(source=str(source), archive=str(target.relative_to(W)),
                                                       decoded_bytes=len(raw), decoded_sha256=m.sha(source),
                                                       archive_sha256=m.sha(target)))

    receipts = [f'xe-e32-{mode}-full-build-root-v22/record.json' for mode in ['free', 'contrib-llvm']]
    receipts += [f'xe-e32-{mode}-asan-full-build-root-v19/record.json' for mode in ['free', 'contrib-llvm']]
    receipts += ['xe-e32-lints-baseline-compare-root-v17/record.json']
    receipts += [f'xe-native-dequant-prefill-model-controller-cpu-root-v{n}/record.json' for n in [4, 5, 6]]
    receipts += [f'xe-native-dequant-prefill-model-root-v{n}/n64-g16-off/record.json' for n in [4, 5]]
    receipts += ['xe-native-dequant-prefill-post-logcap-health-root-v1/record.json']
    receipts += [f'xe-native-dequant-prefill-model-root-v6/n{n}-g16-{arm}/record.json'
                 for n in [64, 2049, 32768] for arm in ['off', 'on']]
    receipts += ['xe-preload-control-source-publish-root-v1/record.json']
    roots = []
    archived_receipts = {}
    for relative in receipts:
        source = B / relative
        data = json.loads(source.read_bytes())
        m.require(not data['active'], 'terminal receipt')
        closed_tree(data)
        archived_receipts[relative] = copy(source, 'receipts/' + relative.replace('/', '__'))
        roots.append(source.parent)
    for number in [4, 5]:
        data = json.loads((B / f'xe-native-dequant-prefill-model-root-v{number}/n64-g16-off/record.json').read_bytes())
        m.require(not data['passed'], 'original model failure preserved')
    for relative in receipts:
        if 'model-root-v4/' not in relative and 'model-root-v5/' not in relative:
            m.require(json.loads((B / relative).read_bytes())['passed'], 'actual required PASS')

    recipe_paths = ['xe-e32-full-build-v22/run_root.py', 'xe-e32-host-asan-build-v19/run_root.py',
                    'xe-e32-lints-baseline-compare-v17/run_root.py',
                    'xe-native-dequant-prefill-post-logcap-health-v1.py',
                    'xe-preload-control-source-publish-v1.py',
                    'xe-native-dequant-prefill-model-controller-main-cpu-v6.py']
    for relative in recipe_paths:
        copy(B / relative, 'recipes/' + relative.replace('/', '__'))
    copy(Path(__file__), 'recipes/' + Path(__file__).name)
    copy(B / 'xe-preload-disabled-evidence-archive-v1.py', 'recipes/xe-preload-disabled-evidence-archive-v1.py')
    for n in [4, 5, 6, 7]:
        copy(B / f'xe-native-dequant-prefill-model-controller-main-review-v{n}.json',
             f'reviews/controller-main-review-v{n}.json')
    copy(B / 'xe-optional-preload-control-applied-main-review-v1.json', 'reviews/preload-applied-main-review-v1.json')
    for n in [1, 2, 3]:
        copy(B / f'xe-native-dequant-prefill-model-qualification-v{n}.json', f'reviews/model-qualification-v{n}.json')
    for version in [4, 8]:
        root = B / f'xe-native-dequant-prefill-model-controller-source-v{version}'
        for p in sorted(root.glob('*.py')):
            if version == 4 and p.name != 'run_root.py':
                continue
            copy(p, f'diagnostic-controller-v{version}/' + p.name)
    # Intermediate executed revisions are losslessly represented, not resume backups.
    for version in [5, 6, 7]:
        encoded(B / f'xe-native-dequant-prefill-model-controller-source-v{version}/run_root.py',
                f'historical-controller-encoded/v{version}.json')
    for relative, target in [('xe-optional-preload-control-source-v1/candidate.patch', 'patches/preload-candidate.json'),
                             ('xe-e32-free-full-build-root-v22/source.patch', 'patches/preload-actual-working-diff.json'),
                             ('xe-native-dequant-prefill-model-controller-source-v4/candidate.patch', 'patches/controller-agent-v4.json'),
                             ('xe-native-dequant-prefill-quiet-timing-controller-source-v1/candidate.patch', 'patches/quiet-controller-agent-v1.json')]:
        encoded(B / relative, target)
    for p in sorted((B / 'xe-native-dequant-prefill-quiet-timing-controller-source-v1').iterdir()):
        if p.suffix in ['.py', '.json']:
            copy(p, 'pending-quiet-controller/' + p.name)

    report_names = '''research-xe-joint-matrix-unresolved-provider-mechanisms-next-v1.txt
research-xe-production-mma-minimal-probe-eligibility-next-v1.txt
implementation-xe-model-preload-disabled-controller-v4.txt
research-xe-controller-v5-mapping-provenance-review-next-v1.txt
research-xe-controller-v5-quit-protocol-review-next-v1.txt
research-xe-model-free-packed-a-reachability-and-small-prefill-next-v1.txt
research-zen3-decode-pool-next-decision-measurement-v1.txt
research-xe-packed-a-provider-boundary-mapped-tuple-next-v1.txt
research-xe-native-dequant-n64-api-volume-next-v1.txt
research-xe-preload0-pass-scope-and-2049-kernel-boundary-v1.txt
research-xe-native-dequant-n64-group-join-next-v1.txt
research-xe-native-dequant-ring-capacity-next-v1.txt
research-xe-native-dequant-v2-model-width-addendum-v1.txt
research-xe-native-dequant-v6-n64-group-census-review-next-v1.txt
research-xe-32k-exposed-prefill-phase-next-v1.txt
implementation-xe-quiet32k-paired-timing-controller-v1.txt
research-xe-native-dequant-stream-credit-and-tail-boundary-next-v1.txt
research-xe-six-pass-tail-avoiding-chunk-feasibility-next-v1.txt'''.splitlines()
    reports = []
    for name in report_names:
        original = B / 'research-20261009' / name
        archive = copy(original, name)
        implementation = name.startswith('implementation-')
        reports.append(dict(path=str(original), archive=str(archive.relative_to(W)), sha256=m.sha(archive),
                            bytes=archive.stat().st_size, model='gpt-6.1-sol' if implementation else 'gpt-6-luna',
                            read_only=not implementation, status='completed', main_full_report_review=True,
                            qualification='Source-only; no qualification inferred from report' if implementation
                            else 'Read-only research; no GPU/test/build execution'))
    previous = W / 'bench/results/2026-10-11-xe-native-dequant-prefill-batch/report-registry-v130.json'
    m.require(m.sha(previous) == '38bcb49566f207bb4a6d626802ab983c831851451f6360b3b68417978c49f2ef', 'previous registry pin')
    prior = json.loads(previous.read_bytes())
    m.require(prior['research_completed'] == 527 and len(reports) == 18, 'incremental count')
    write(R / 'report-registry-v131.json', dict(registry_version=131, research_completed=545,
          created_utc=m.utc(), registry_format='incremental; follow previous_committed_registry',
          previous_committed_registry=dict(path=str(previous.relative_to(W)), sha256=m.sha(previous), reports=527),
          new_reports=reports, reports=reports, live_agent_snapshot=dict(source='actual collaboration.list_agents and acknowledged followup_task',
          observation='2026-10-11: previous three agents completed; two Luna agents reassigned after Main report review',
          agents=[dict(agent='/root/research_native_batch_driver_lowering', model='gpt-6-luna',
                        scope='Six-pass tail-avoiding chunk feasibility; readonly', status='completed; fully read and included in this registry'),
                   dict(agent='/root/research_decode_avx2_pool_delta', model='gpt-6-luna',
                        scope='Quiet32K sample ordering and publication gates; readonly', status='running at assignment')],
          note='An assignment snapshot; inspect live state again on resume. Candidate/report existence does not prove an agent is running.'),
          next_research_scopes=['six-pass tail avoidance', 'independent quiet sample order and source/build pins'],
          dependency='Main archives and retires closed logs before any new qualification or GPU run'))

    comparisons = []
    for n in [64, 2049, 32768]:
        paths = [B / f'xe-native-dequant-prefill-model-root-v6/n{n}-g16-{arm}/record.json' for arm in ['off', 'on']]
        off, on = [json.loads(p.read_bytes()) for p in paths]
        select = lambda v: [s for s in v['request_protocol'] if s.startswith(('T ', 'LP '))]
        m.require(select(off) == select(on) and len(select(off)) == 32, 'strict sixteen IDs and LP5 equality')
        def routes(v):
            return [{k: x for k, x in row.items() if k not in ['native_dequant_gu_calls', 'native_dequant_down_calls']}
                    for row in v['route_census']['rows']]
        m.require(routes(off) == routes(on), 'every joined route identity and count')
        m.require(off['transfer_ledger']['rows'] == on['transfer_ledger']['rows'], 'every joined H2D row')
        comparisons.append(dict(input_tokens=n, off_sha256=m.sha(paths[0]), on_sha256=m.sha(paths[1]),
          strict16_token_and_lp5_equal=True, all_route_rows_equal_except_legacy_dequant_wrapper_counts=True,
          all_h2d_rows_equal=True, route_rows=len(routes(off)), h2d_rows=len(off['transfer_ledger']['rows']),
          off_diagnostic_request_timing=off['diagnostic_request_timing'],
          on_diagnostic_request_timing=on['diagnostic_request_timing'],
          h2d_total=off['transfer_ledger']['total'], on_group_rows=on['route_census']['group_rows'],
          performance_eligible=False, adoption=False, full262144_qualified=False, restore_measured=False))
    write(R / 'main-paired-comparison.json', dict(created_utc=m.utc(), comparisons=comparisons,
          scope='Exact finite request equivalence and census joins, no speed or full-model qualification'))

    failure = B / 'xe-native-dequant-prefill-model-root-v4/n64-g16-off/commands/model.stderr'
    m.require(failure.stat().st_size == 64623852 and m.sha(failure) == '1ede5e0e9c4fc9049eaa56c24ecc333cd4a7586d8bb3716980a2fe02584f3154', 'original API budget failure log')
    severe = re.compile(r'UR_RESULT_ERROR|ZE_RESULT_ERROR|\[(?:ERROR|FATAL)\]|(?:Assertion|assertion).*failed|terminate called|longjmp')
    progress, matches, tail = [], [], collections.deque(maxlen=8)
    count = 0
    with failure.open(errors='strict') as stream:
        for count, line in enumerate(stream, 1):
            if severe.search(line):
                matches.append(dict(line=count, text=line.rstrip('\n')))
            if 'strata trace:' in line:
                progress.append(dict(line=count, text=line.rstrip('\n')))
            tail.append(dict(line=count, text=line.rstrip('\n')))
    excerpt = dict(path=str(failure), original_bytes=failure.stat().st_size, original_sha256=m.sha(failure),
                   total_lines=count, error_filter=severe.pattern, error_matches=matches, progress=progress,
                   final_api_window=list(tail), primary_failure="RuntimeError('aggregate owner log cap')",
                   ready_seen=True, gen_sent=True, done_seen=False, original_passed=False,
                   conclusion='Incomplete log-budget diagnostic; no captured severe runtime error. Not proof of absent GPU fault.',
                   consumer='Preserve original budget stop and last observed work; independent fresh health receipt is separate',
                   whole_trace_consumer=None)
    write(R / 'api-log-budget-excerpt.json', excerpt)
    m.require((R / 'api-log-budget-excerpt.json').stat().st_size <= 256 << 10, 'bounded failure evidence')

    plan = dict(status='PLANNED', created_utc=m.utc(), deletion_executed=False, files=[],
                evidence_manifest='archive-manifest.json', original_result_statuses_unchanged=True,
                cleanup_requires_committed_verified_evidence=True, measurement_lock=str(B / 'owned-v0141-measurement.lock'))
    for root in roots:
        receipt_relative = str((root / 'record.json').relative_to(B))
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ['.stdout', '.stderr']:
                parts = path.relative_to(root).parts
                nested_owned_cpu = root.name.startswith('xe-native-dequant-prefill-model-controller-cpu-root-v') and parts[0] in ['owned-quit-after-DONE', 'owned-quit-after-READY']
                m.require('commands' in parts or nested_owned_cpu, 'only supervised closed command logs')
                s = path.stat()
                plan['files'].append(dict(path=str(path), bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                    sha256=m.sha(path), terminal_receipt_sha256=m.sha(root / 'record.json'),
                    evidence=str(archived_receipts[receipt_relative].relative_to(W)),
                    reason='Terminal supervised command; source/recipe/individual outcomes/runtime/cleanup proof retained in immutable receipt. '
                           + ('Budget failure additionally retains complete selected progress and bounded final API window.'
                              if path == failure else 'No named full-history consumer remains.')))
    plan['logical_bytes'] = sum(v['bytes'] for v in plan['files'])
    plan['allocated_bytes'] = sum(v['allocated_bytes'] for v in plan['files'])
    plan['count'] = len(plan['files'])
    write(R / 'retention-v1.json', plan)
    archive_bytes = sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    m.require(archive_bytes < 16 << 20, 'finite compact archive budget')
    manifest['archive_bytes_before_manifest_and_report'] = archive_bytes
    write(R / 'archive-manifest.json', manifest)
    write(R / 'archive-controller-attempts.json', dict(attempts=[dict(controller_sha256=m.sha(B / 'xe-preload-disabled-evidence-archive-v1.py'), status='FAIL', error='RuntimeError: only supervised closed command logs', phase='retention planning', partial_exact_copies_preserved=True, deletion_executed=False, gpu_executed=False), dict(controller_sha256=m.sha(__file__), status='PASS', correction='Include two explicitly owned synthetic CPU child log directories whose closed commands are already embedded in CPU receipt fixtures', comparison_gates_unchanged=True, deletion_executed=False)]))
    print(json.dumps(dict(exact_copies=len(manifest['byte_exact_copies']), encoded=len(manifest['lossless_encoded_copies']),
                         reports=len(reports), comparison='PASS', cleanup='PLANNED', raw_files=plan['count'],
                         raw_logical_bytes=plan['logical_bytes'], archive_bytes=archive_bytes)))
