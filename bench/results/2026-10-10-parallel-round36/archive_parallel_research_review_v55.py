"""Preserve fully read numerical, exact-device and compiler-provenance reviews."""
from pathlib import Path
import datetime, fcntl, hashlib, json, shutil

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R = B / 'research-20261009'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A = W / 'bench/results/2026-10-10-parallel-round36'


def ident(p):
    p = Path(p)
    return dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prev = R / 'report-registry-v54.json'
    assert ident(prev)['sha256'] == 'cbcda647cc92d5044d79e66a197ee91a3bb65236fa3d4f833412089cf14a461e'
    d = json.loads(prev.read_text())
    assert len(d['reports']) == 234
    entries = [
        (127, 'gdn-probe-numerical-source-admission', 'research_dense_mirror_phase_ownership', 'Corrected immutable probe numerical/layout source admission', 'da6eaa162f718994760785dfda9e43b64f7de54f7e2ad92ddc15c0a9639d898e'),
        (129, 'b570-exact-pci-admission', 'research_dense_mirror_phase_ownership', 'Installed SYCL/native Level Zero exact selected-device PCI admission', 'dc9d5fba974d34214b33e311e5993804975fa06c5d0e266beabb6ed8cfed75d9'),
        (130, 'installed-native-exp-lowering-provenance', 'research_prefill_algorithms_blogs', 'Installed compiler/IGC native-exp lowering provenance and precise missing artifact', '183711c3fe246919e9b93812453b9e98c5490f54d258d2ea26b25c13218b9bc0'),
    ]
    new = []
    for number, name, agent, scope, digest in entries:
        source = R / f'round{number}-{name}.txt'
        x = ident(source)
        assert x['sha256'] == digest
        target = A / 'research' / source.name
        assert not target.exists()
        shutil.copyfile(source, target)
        entry = dict(path=str(source), **x, agent='/root/' + agent, model='gpt-6-luna', scope=scope, status='completed-read-only', root_full_report_reviewed=True)
        d['reports'].append(entry)
        new.append(entry)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    d.update(
        registry_version=55, created_utc=now, research_completed=len(d['reports']),
        previous_committed_registry=dict(path=str(prev), **ident(prev), storage_commit='0c7d69003d1587958751492de8b16ebfb3b16f2c'),
        new_reports=new,
        live_agent_snapshot=dict(time_utc=now, method='Direct collaboration.list_agents after both eighth-wave reports returned. Completed is a timestamped observation; no stale running claim.', agents=[
            dict(agent='/root/research_dense_mirror_phase_ownership', model='gpt-6-luna', status='completed', round=129),
            dict(agent='/root/research_prefill_algorithms_blogs', model='gpt-6-luna', status='completed', round=130),
            dict(agent='/root/implement_repeat_capture_reader_hardening', model='gpt-6.1-sol', status='completed-source-only'),
        ]),
        current_root_decisions=dict(
            CPU_register_index='REJECT: 12 process/cell paired medians candidate/trait 1.229..1.318; candidate/direct 1.199..1.230. Production unchanged.',
            sign_scratch='Deprioritized without isolated emitted-code evidence for a useful gain',
            GDN_gate_factor='Private necessary numerical discriminator; no production recurrence, valid model domain, speed or adoption claim',
            private_probe_source_commit='cfc549fb491b69b54ad4285da14f301b75bf05a8',
            private_probe_snapshot='b6de67fe89c70fd657ee4d4ec120817281b67289293f5c8869901ff42ce18834',
            cpu_only_build_receipt=dict(path=str(B / 'gdn-gate-factor-probe-cpu-build-v2/record.json'), **ident(B / 'gdn-gate-factor-probe-cpu-build-v2/record.json')),
            numerical_source_review='R127 static bounds/layout accepted. Root renamed unverified finite synthetic input label and added explicit 15-case/7970640-comparison/3-full-repeat assertions, then rebuilt successfully.',
            exact_device_admission='R129 provides selected native Level Zero root-device handle typed PCI query; require 8086:e20c and 0000:05:00.0 before allocations/submission. Not implemented or executed yet.',
            native_exp_provenance='R130 confirms frontend builtin mapping, installed packages and closed flags. Exact IGC v2.41.5 lowering/range/denormal behavior remains unresolved; source contract is not runtime proof.',
            gpu_work_this_batch=False, model_inference_this_batch=False, full_physical_256K_lifecycle=False,
            root_parallel_work='Eight two-agent Luna research waves completed (R115–R130). Separate Sol source handoffs; root exclusively built, validated actual-weight CPU numerical/timing screens, rejected slower candidate, corrected and built private SYCL discriminator, committed artifacts and retired redundant success logs.',
            next_gate='Implement exact selected PCI admission and root-owned bounded diagnostic session controller; actual target primitive verification remains pending. New read-only investigations must address distinct unresolved evidence, not reopen settled candidates.',
        ),
    )
    registry = R / 'report-registry-v55.json'
    assert not registry.exists()
    write(registry, d)
    shutil.copyfile(registry, A / registry.name)
    write(A / 'root-parallel-review-v55.json', dict(created_utc=now, new_reports=len(new), total_reports=len(d['reports']), registry_identity=ident(registry), decisions=d['current_root_decisions'], snapshot=d['live_agent_snapshot']))
    report = A / 'REPORT.md'
    report.write_text(report.read_text() + '''

## Reviewed follow-ups v55

R127, R129 and R130 have now been read in full and preserved unchanged. Eight distinct two-investigator Luna waves completed in parallel with root work this turn (R115–R130), bringing the retained completed-report inventory to237. The snapshot records that both investigators have returned; it does not represent finished agents as running.

R127 found no blocking static bounds/layout issue in the private primitive discriminator, and recommended explicit completeness assertions and correcting the unsupported plausible-domain label. Root implemented both, built successfully in5.931s, and committed source plus compact compiler/build receipts at cfc549fb491b69b54ad4285da14f301b75bf05a8. R129 specifies exact selected-native-handle PCI admission with typed fields rather than ordinal or name-only matching. R130 establishes installed frontend mapping and package/library identities, but cannot prove the exact installed IGC native-exp lowering, valid range or denormal behavior. Neither investigation ran a GPU query.

There was no GPU or model execution in these waves. The private tool remains an unrun necessary copied-expression screen, with actual recurrence/state/output and full physical262144-token lifecycle still required before production adoption. Root's separate selected-real-weight CPU screen rejected the current register-index candidate: its paired median elapsed times were23–32% longer than trait dispatch, and20–23% longer than the direct control. These are caller/dot primitive measurements with synthetic activations, not32K model performance.

The next concrete work is exact device admission and a root-owned bounded diagnostic execution controller; research continues on distinct remaining questions using this committed registry and prior decisions. Tests remain serialized under the root-owned lock, separate from source-only implementation and read-only research agents.
''')
    shutil.copyfile(__file__, A / Path(__file__).name)
    identity = A / 'archive-file-identities.json'
    old = json.loads(identity.read_text())
    updated = {str(p.relative_to(A)): ident(p) for p in sorted(A.rglob('*')) if p.is_file() and p != identity}
    assert set(old) <= set(updated)
    for rel, x in old.items():
        if rel != 'REPORT.md':
            assert updated[rel] == x, rel
    write(identity, updated)
    print(json.dumps(dict(registry=str(registry), **ident(registry), completed=len(d['reports']))))
