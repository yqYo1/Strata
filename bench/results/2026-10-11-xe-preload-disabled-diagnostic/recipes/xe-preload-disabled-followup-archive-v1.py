import base64, fcntl, importlib.util, json
from pathlib import Path
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-preload-disabled-diagnostic'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner pin')
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = dict(created_utc=m.utc(), byte_exact_copies=[], lossless_encoded_copies=[],
                    pending_controller_main_full_source_review=False, pending_controller_tested=False,
                    model_executed=False, gpu_executed=False)
    m.require(not (R / 'followup-archive-manifest.json').exists(), 'new immutable followup archive')
    def copy(original, relative):
        target = R / relative
        m.require(not target.exists(), 'new exact copy')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(original.read_bytes())
        m.require(m.sha(original) == m.sha(target), 'byte exact followup')
        entry = dict(source=str(original), archive=str(target.relative_to(W)), bytes=target.stat().st_size, sha256=m.sha(target))
        manifest['byte_exact_copies'].append(entry); return entry
    reports = []
    names = ['research-xe-sycl-event-destruction-vs-queue-usm-lifetime-next-v1.txt',
             'research-xe-prefill-native-batch-cold-decode-influence-next-v1.txt',
             'research-xe-prefill-native-batch-cold-decode-source-correction-v1.txt',
             'implementation-xe-quiet32k-counterbalanced-controller-v2.txt']
    for name in names:
        entry = copy(B / 'research-20261009' / name, name)
        impl = name.startswith('implementation-')
        reports.append(dict(path=entry['source'], archive=entry['archive'], sha256=entry['sha256'], bytes=entry['bytes'],
                            model='gpt-6.1-sol' if impl else 'gpt-6-luna', read_only=not impl, status='completed',
                            main_full_report_review=True, qualification='Source-only; full source review and tests pending' if impl
                            else ('Wrong-worktree primary findings rejected; original preserved with separate correction' if 'influence-next' in name
                                  else 'Readonly research; provenance corrections separate, no runtime or performance qualification')))
    source = B / 'xe-native-dequant-prefill-quiet-timing-controller-source-v2'
    m.require(m.sha(source / 'run_root.py') == 'a2c7c348625b880539a7d9ccb7bd7f9100f47e76a841036536a7c322e50beafb', 'returned source-only candidate')
    for p in sorted(source.iterdir()):
        if p.suffix in ['.py', '.json']:
            copy(p, 'pending-counterbalanced-controller/' + p.name)
    raw = (source / 'candidate.patch').read_bytes()
    encoded = R / 'patches/quiet-counterbalanced-controller-v2.json'
    encoded.write_text(json.dumps(dict(encoding='base64', original_path=str(source / 'candidate.patch'),
                       decoded_sha256=m.sha(source / 'candidate.patch'), decoded_bytes=len(raw),
                       data=base64.b64encode(raw).decode('ascii')), indent=2) + '\n')
    manifest['lossless_encoded_copies'].append(dict(source=str(source / 'candidate.patch'), archive=str(encoded.relative_to(W)),
         decoded_sha256=m.sha(source / 'candidate.patch'), decoded_bytes=len(raw), archive_sha256=m.sha(encoded)))
    copy(B / 'xe-preload-disabled-post-retention-publish-root-v1/record.json',
         'receipts/xe-preload-disabled-post-retention-publish-root-v1__record.json')
    copy(Path(__file__), 'recipes/' + Path(__file__).name)
    previous = R / 'report-registry-v132.json'
    m.require(json.loads(previous.read_bytes())['research_completed'] == 547, 'prior registry')
    registry = dict(registry_version=133, research_completed=551, created_utc=m.utc(),
        registry_format='incremental; follow previous_committed_registry', previous_committed_registry=dict(
            path=str(previous.relative_to(W)), sha256=m.sha(previous), reports=547), new_reports=reports, reports=reports,
        live_agent_snapshot=dict(source='actual collaboration.list_agents immediately before archive', observed_utc=m.utc(),
            agents=[dict(agent='/root/implement_optional_kernel_preload_control', model='gpt-6.1-sol', status='completed',
                         scope='Source-only counterbalanced quiet32K v2; report reviewed, code review and tests pending'),
                    dict(agent='/root/research_native_batch_driver_lowering', model='gpt-6-luna', status='running',
                         scope='Actual H2D/pass budget and 1000tps necessary conditions; readonly'),
                    dict(agent='/root/research_quiet32k_publication_gates', model='gpt-6-luna', status='running',
                         scope='Post-prefill owned workspace allocation lifetime/live bytes; readonly')],
            note='Snapshot; check actual status on resume. Each new assignment shares past reports and the then-latest committed registry.'),
        dependency='Main must review full source and qualify both order directions on CPU before any quiet model run',
        next_research_scopes=['H2D/pass necessary performance conditions', 'Actual post-prefill workspace lifetime'])
    (R / 'report-registry-v133.json').write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    (R / 'main-source-provenance-verification.json').write_text(json.dumps(dict(created_utc=m.utc(),
        checked_by='Main, read-only Git show/rev-parse/status and sha256sum at explicit absolute WX',
        source_worktree='/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011',
        head='0740b43488d812117f021ccbe4bd0e15d223f955', tracked_clean=True,
        prefill_content_sha256='c5cfa727302d334e2f6a8aaad8c1bf59e7d581dd8c2cec2af0d031faf807272d',
        prefill_git_blob_sha1='a5971752d742a430e477792bf9e1f807344c0dfe',
        generate_content_sha256='c9b60ef079694302dfa1f8099ddffd5375d7dae503f6edec454c1d7de3d941ee',
        generate_git_blob_sha1='676d525f172609e6d67345b555b70a28eb73c6a9',
        gpu_content_sha256='9792966f6b2c84434a1dddafefb3286c3c5fcfa78e23065b455904f9cf6fcfb8',
        native_flag_line=2099, native_flush_line=2112, final_native_flush_line=2452,
        source_publication_proof='receipts/xe-preload-control-source-publish-root-v1__record.json',
        rejected_claims=['Native flag absent in published WX', 'Prefill source differs from historical build content'],
        cause='Researcher inspected documentation worktree; follow-up Main assignment omitted WX absolute path',
        original_report_preserved=True, correction_report=names[2],
        local_llvm_head='504366f4b82ff00bbac7b0c956635eed8bc599d8',
        local_llvm_original_patch_state_preserved=True,
        verified_llvm_source_facts=['event_impl.hpp:36 EventImplPtr shared_ptr', 'scheduler/commands.hpp:257 EventImplPtr MEvent',
                                    'event_impl.cpp:46 UR event release'], gpu_executed=False, model_executed=False), indent=2) + '\n')
    (R / 'followup-archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (R / 'main-provenance-corrections.md').open('a') as f:
        f.write('\nThe event-lifetime report labels the 40-hex prefill Git blob ID as SHA-256; it is a Git SHA-1 blob ID. Its actual content SHA-256 is c5cfa727302d334e2f6a8aaad8c1bf59e7d581dd8c2cec2af0d031faf807272d. The event destructor source link truncates the LLVM commit; the correct pinned commit is 504366f4b82ff00bbac7b0c956635eed8bc599d8. Main read the local event/shared-command ownership and release sites.\n\n')
        f.write('The original cold-decode report inspected the documentation worktree instead of WX. Main rechecked the explicit immutable WX commit and source hashes; the native batch switch is present at prefill.cpp:2099 and build/publication content matches. Its missing-switch and source-mismatch findings are rejected. The original and separate corrective addendum remain preserved. Corrected scope supports a prefill-only dispatch change; a single cold decode duration still cannot justify a decode-kernel adoption claim.\n')
    with (R / 'README.md').open('a') as f:
        f.write('\nThe [latest registry](report-registry-v133.json) records 551 completed report entries, including the preserved wrong-source finding and correction. The returned counterbalanced controller v2 is archived as source-only; Main has read its report, but has not parsed, imported, tested or fully reviewed its code. Its pending manifest rejects launch. Next action is full Main source review and CPU qualification of both order directions, predecessor rejection and publication schemas. Two Luna investigators continue distinct hardware-transfer and allocation-lifetime questions.\n')
    print(json.dumps(dict(reports=4, research_completed=551, source_only=True, tests_executed=False,
                         exact_copies=len(manifest['byte_exact_copies']))))
