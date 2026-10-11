import fcntl, hashlib, importlib.util, json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-preload-disabled-diagnostic'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner pin')
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = dict(created_utc=m.utc(), phase='After completed log retirement', byte_exact_copies=[])
    m.require(not (R / 'post-retention-archive-manifest.json').exists(), 'immutable post-retention archive')
    sources = [('xe-preload-disabled-log-retention-v1.py', 'recipes/xe-preload-disabled-log-retention-v1.py'),
               ('xe-preload-disabled-log-retention-root-v1/record.json', 'receipts/xe-preload-disabled-log-retention-root-v1__record.json'),
               ('xe-preload-disabled-evidence-publish-root-v2/record.json', 'receipts/xe-preload-disabled-evidence-publish-root-v2__record.json'),
               ('research-20261009/research-xe-quiet32k-sample-order-and-publication-gates-next-v1.txt', 'research-xe-quiet32k-sample-order-and-publication-gates-next-v1.txt'),
               ('research-20261009/research-xe-prefill-event-timer-queue-boundary-eligibility-next-v1.txt', 'research-xe-prefill-event-timer-queue-boundary-eligibility-next-v1.txt')]
    for source_relative, archive_relative in sources:
        original, target = B / source_relative, R / archive_relative
        m.require(not target.exists() and original.is_file(), 'finite new archived evidence')
        target.write_bytes(original.read_bytes())
        m.require(m.sha(original) == m.sha(target), 'post-retention exact copy')
        manifest['byte_exact_copies'].append(dict(source=str(original), archive=str(target.relative_to(W)),
                                                bytes=target.stat().st_size, sha256=m.sha(target)))
    cleanup = json.loads((B / 'xe-preload-disabled-log-retention-root-v1/record.json').read_bytes())
    m.require(cleanup['passed'] and cleanup['complete'] and not cleanup['active'] and cleanup['removed_count'] == 402,
              'actual terminal successful retirement')
    m.require(cleanup['removed_logical_bytes'] == 93466690 and cleanup['removed_allocated_bytes'] == 95279104,
              'actual measured deletion totals')
    m.require(all(v['unlink_status'] == 'REMOVED' and not Path(v['path']).exists() for v in cleanup['files']), 'all deleted paths absent')
    previous = R / 'report-registry-v131.json'
    registry = json.loads(previous.read_bytes())
    m.require(registry['research_completed'] == 545, 'prior incremental count')
    reports = [dict(path=v['source'], archive=v['archive'], sha256=v['sha256'], bytes=v['bytes'],
                    model='gpt-6-luna', read_only=True, status='completed', main_full_report_review=True,
                    qualification='Read-only source/report review; no execution or performance inference')
               for v in manifest['byte_exact_copies'] if Path(v['source']).name.startswith('research-')]
    (R / 'report-registry-v132.json').write_text(json.dumps(dict(registry_version=132, research_completed=547,
         created_utc=m.utc(), registry_format='incremental; follow previous_committed_registry',
         previous_committed_registry=dict(path=str(previous.relative_to(W)), sha256=m.sha(previous), reports=545),
         new_reports=reports, reports=reports,
         live_agent_snapshot=dict(source='actual collaboration.list_agents immediately before archive',
             observed_utc=m.utc(), agents=[
             dict(agent='/root/implement_optional_kernel_preload_control', model='gpt-6.1-sol', status='running',
                  scope='Separate source-only counterbalanced quiet controller v2; no tests or GPU work'),
             dict(agent='/root/research_native_batch_driver_lowering', model='gpt-6-luna', status='running',
                  scope='SYCL event destruction versus queue/USM lifetime; readonly'),
             dict(agent='/root/research_quiet32k_publication_gates', model='gpt-6-luna', status='running',
                  scope='Prefill-native-batch influence on cold decode; readonly')],
             note='Snapshot only; recheck actual live state before resuming. Old decode agent returned N64 review, not requested quiet audit; new dedicated Luna completed that audit.'),
         next_research_scopes=['SYCL event reference ownership', 'Prefill and decode direct versus indirect behavior'],
         dependency='Main full source review and CPU qualification of counterbalanced controller before any new model launch'),
         indent=2, ensure_ascii=False) + '\n')
    readme = R / 'README.md'
    with readme.open('a') as f:
        f.write('\n## Completed retirement and next qualification\n\n')
        f.write('The [executed manifest](retention-executed-v1.json) records all 402 removals: 93466690 logical bytes and 95279104 allocated bytes. Measured filesystem available space changed by zero bytes between snapshots; the removed file totals do not prove exclusive space reclamation. Original receipts, fixtures, models, binaries and required source remain unchanged.\n\n')
        f.write('The [next registry](report-registry-v132.json) contains two additional fully reviewed Luna reports, 547 total entries. Actual agent state was checked: two researchers and a separate Sol implementer are running new scopes. The quiet v1 audit identifies order bias; its ON preflight requires prior OFF, so simply alternating commands cannot reverse pairs. The separate source-only v2 assignment must support balanced four-pair ordering, a strict immutable diagnostic reference for each first arm, a fresh same-index first-arm comparison for the second, and validated completed predecessors. Pending candidates remain unexecuted and performance-ineligible. Main owns all future tests and GPU runs.\n\n')
        f.write('The event-timer audit confirms adjacent compute-stream timestamps include idle/dependency intervals; they are not isolated dequant or DMA durations. Successful-path event lifetime was reviewed in source. Exceptional destruction semantics require a separate source/specification investigation and do not establish a bug. Its reference to an active 32K root describes a path it excluded; both actual 32K receipts were already terminal when this follow-up was assigned.\n')
    (R / 'post-retention-archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(exact_copies=len(manifest['byte_exact_copies']), reports=len(reports), research_completed=547,
                         cleanup='PASS', model_executed=False)))
