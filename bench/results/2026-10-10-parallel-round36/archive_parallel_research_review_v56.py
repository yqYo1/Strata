"""Archive returned capture controls and CPU prefill candidate evidence."""
from pathlib import Path
import datetime, fcntl, hashlib, json, shutil, subprocess

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
    prev = R / 'report-registry-v55.json'
    assert ident(prev)['sha256'] == '65164569ce1d85099b93bb068a2d3437314c3816a74e7d8f6fb8ac6b1e8f1af1'
    d = json.loads(prev.read_text())
    assert len(d['reports']) == 237
    new = []
    for n, name, agent, scope, digest in [
        (131, 'exact-igc-single-kernel-capture-controls', 'research_dense_mirror_phase_ownership', 'Exact-tag IGC Release dump controls and unproven single-kernel/options/byte-cap prerequisites', '37d4839fa5b2f5049394b1b1d0cebbf05c3663cbf78b41fb4a995fd9bf7fda53'),
        (132, 'cpu-prefill-batching-new-candidate', 'research_prefill_algorithms_blogs', 'Other-engine and implementer-blog CPU prefill multi-token/route scheduling candidate screen', '7faee162ac2d9610d320992af262f6c73fe775eb53670c221ee2ae9e0b51d63c'),
    ]:
        source = R / f'round{n}-{name}.txt'
        x = ident(source)
        assert x['sha256'] == digest
        target = A / 'research' / source.name
        assert not target.exists()
        shutil.copyfile(source, target)
        e = dict(path=str(source), **x, agent='/root/' + agent, model='gpt-6-luna', scope=scope, status='completed-read-only', root_full_report_reviewed=True)
        d['reports'].append(e)
        new.append(e)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    d.update(registry_version=56, created_utc=now, research_completed=len(d['reports']), previous_committed_registry=dict(path=str(prev), **ident(prev), storage_commit='205bb5e34aa5250c1f5d02fc18a49582aa878449'), new_reports=new)
    d['live_agent_snapshot'] = dict(time_utc=now, method='Direct collaboration.list_agents: R133 immutable exact-PCI source audit running; R132 candidate investigator returned; separate Sol exact-PCI handoff returned. Historical boundary observation.', agents=[
        dict(agent='/root/research_dense_mirror_phase_ownership', model='gpt-6-luna', status='running', round=133, scope='Immutable actual exact-PCI implementation source/ABI/bounds/refusal admission'),
        dict(agent='/root/research_prefill_algorithms_blogs', model='gpt-6-luna', status='completed', round=132, next_dependency='Confirm complete optimized prefill native-expert caller graph separately from the reviewed general per-token fallback'),
        dict(agent='/root/implement_repeat_capture_reader_hardening', model='gpt-6.1-sol', status='completed-source-only', handoff='implementation-gdn-probe-exact-pci-v1.txt'),
    ])
    decision = d['current_root_decisions']
    decision.update(
        private_probe_source_commit='00ccf742bb0a39a6c6d5d7aba9b43d7a1a9f4518',
        private_probe_snapshot='4b488ae17f57bf791594af87dcd93485f00704d10d0cd3d8c2459125a0585f6f',
        cpu_only_build_receipt=dict(path=str(B / 'gdn-gate-factor-probe-cpu-build-v3/record.json'), **ident(B / 'gdn-gate-factor-probe-cpu-build-v3/record.json')),
        exact_device_admission='Implemented only in private startup gate; native selected device/platform handles, bounded extension discovery, vendor8086/devicee20c/root and typed0000:05:00.0 assertion. Root reviewed source and built in7.238s, verified explicit loader linkage and header/library pins. No runtime query or result yet; independent R133 static audit running.',
        IGC_capture='R131 confirms exact-tag Release dump controls but not one-kernel filename mapping, matching LevelZero options or hard aggregate cap. Hold broad capture; exact native-exp lowering remains unresolved.',
        CPU_prefill_candidates='R132 admits no new candidate: reuse already exists, other leads duplicate prior ideas or require different ISA/formats. General per-token fallback refuses native packs; complete optimized-prefill caller graph needs separate admission before treating verifier-only as universal.',
        root_parallel_work='Nine two-investigator Luna waves R115–R132 fully reviewed while root ran serialized actual-weight numerical/sanitizer/timing CPU work and private SYCL CPU compilation. Current candidate rejected; private exact-PCI interface built and committed. Root prepared bounded diagnostic protocol and remains sole executor.',
        next_gate='Complete static admission of exact-PCI source and prepare bounded owned debug controller; primitive target runtime result, recurrence/state/output and full physical262144 remain pending.',
    )
    handoff = R / 'implementation-gdn-probe-exact-pci-v1.txt'
    d['implementation_handoffs_current_batch'].append(dict(path=str(handoff), **ident(handoff), model='gpt-6.1-sol', status='completed-source-only', root_full_report_reviewed=True, root_build_evidence_commit='00ccf742bb0a39a6c6d5d7aba9b43d7a1a9f4518'))
    registry = R / 'report-registry-v56.json'
    assert not registry.exists()
    write(registry, d)
    shutil.copyfile(registry, A / registry.name)
    write(A / 'root-parallel-review-v56.json', dict(created_utc=now, new_reports=len(new), total_reports=len(d['reports']), registry_identity=ident(registry), decisions=decision, snapshot=d['live_agent_snapshot']))
    report = A / 'REPORT.md'
    report.write_text(report.read_text() + '''

## Reviewed follow-ups v56

R131 and R132 were fully reviewed unchanged, closing the ninth two-investigator Luna wave this turn (R115–R132;239 completed retained reports overall). R131 confirms exact installed-version IGC Release controls but leaves strict single-kernel naming, matching LevelZero options and hard total output cap unresolved; no broad dump is authorized by an inferred guarantee. R132 admits no distinct CPU prefill candidate after checking actual other-engine code and implementer leads. Existing multi-token row kernels already reuse decoded weight blocks. Its reviewed general per-token caller refuses native packs; the complete optimized prefill entry path will be checked separately before interpreting that observation universally.

Separate Sol returned only the three private exact-PCI source files and its report. Root fully reviewed the changed implementation, froze hashes, built normally in7.238s with whole-session finite limits, required precision/JIT flags, explicit loader linkage and before/after header/library pins, and committed source plus compact receipts at00ccf742bb0a39a6c6d5d7aba9b43d7a1a9f4518. The next Luna source-admission audit R133 runs against those immutable bytes while root owns execution preparation. No GPU or model was executed. Startup runtime identity, numerical result, real recurrence/state/output and full physical262144 lifecycle are all pending.
''')
    shutil.copyfile(__file__, A / Path(__file__).name)
    index = A / 'archive-file-identities.json'
    old = json.loads(index.read_text())
    new_index = {str(p.relative_to(A)): ident(p) for p in sorted(A.rglob('*')) if p.is_file() and p != index}
    assert set(old) <= set(new_index)
    for rel, x in old.items():
        if rel != 'REPORT.md':
            assert new_index[rel] == x, rel
    write(index, new_index)
    for args in [('diff', '--check'), ('add', 'bench/results/2026-10-10-parallel-round36'), ('diff', '--cached', '--check'), ('commit', '-m', 'docs(sycl): preserve capture controls and prefill candidate followups'), ('push',), ('rev-parse', 'HEAD')]:
        subprocess.run(['git', *args], cwd=W, check=True)
    print(json.dumps(dict(registry=str(registry), **ident(registry), completed=len(d['reports']))))
