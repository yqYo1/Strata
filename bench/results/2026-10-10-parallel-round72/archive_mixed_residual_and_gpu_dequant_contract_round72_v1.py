from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import shutil
import subprocess

B = Path(__file__).parent
R = B / 'research-20261009'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
C = W / 'bench/results/2026-10-10-hardware-architecture-decisions'
A = W / 'bench/results/2026-10-10-parallel-round72'
COMMIT = '987bad55d75c7fef36152620fb9518bab9d9f94f'


def ident(p):
    p = Path(p)
    return dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')


def copy(p, q):
    shutil.copyfile(p, q)
    assert ident(p) == ident(q)


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
    assert not A.exists()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    estimates = json.loads((C / 'conditional-reuse-estimates.json').read_text())
    src = Path(estimates['production_source']['path'])
    assert ident(src) == {k: estimates['production_source'][k] for k in ('bytes', 'sha256')}
    text = src.read_text()
    assert 'm.R = gpu_rows ? m.residual_gpu_row(c0) : m.residual_scratch;' in text
    assert 'const int64_t reused_tokens = inplace && gpu_tokens == n ? first_len : 0;' in text
    census = W / 'bench/results/2026-10-10-current-native-pack-census/receipt.json'
    assert ident(census)['sha256'] == '902283e324b339127a9eb197954260ab606e32767b4e9acaf1d17e2463d31f84'
    metadata = json.loads(census.read_text())
    assert metadata['relevant_scalar_metadata']['qwen4exp.attention.head_count_kv'] == 2
    assert metadata['relevant_scalar_metadata']['qwen4exp.attention.key_length'] == 256
    samples = []
    for n in (32768, 262144):
        base = next(x for x in estimates['residual_scenarios'] if x['hypothetical_positions'] == n)
        for g in ((0, 8192, 16384, 24576, 32768) if n == 32768 else (0, 8192, 16384, 32768)):
            host_payload = (n - g) * 40960 * 47
            device_payload = g * 40960 * 47
            host_seconds = host_payload / 1e9 / estimates['memory_rates']['H2D_pageable_GBps'] + host_payload / 1e9 / estimates['memory_rates']['D2H_pageable_GBps']
            device_seconds = 4 * device_payload / 1e9 / 330.07315122276316
            samples.append(dict(positions=n, effective_GPU_prefix_positions=g,
                                host_backing_bytes=(n-g)*40960, GPU_plane_extent_bytes=g*40960,
                                H2D_payload_bytes=host_payload, D2H_payload_bytes=host_payload,
                                non_inplace_D2D_payload_bytes_per_handoff_direction=device_payload,
                                inplace_D2D_payload_bytes=0,
                                conditional_host_handoff_seconds=host_seconds,
                                conditional_noninplace_D2D_seconds=device_seconds,
                                conditional_weight_copy_seconds=base['conditional_host_USM_packed_weight_seconds'],
                                conditional_noninplace_serial_copy_seconds=host_seconds+device_seconds+base['conditional_host_USM_packed_weight_seconds'],
                                conditional_inplace_serial_copy_seconds=host_seconds+base['conditional_host_USM_packed_weight_seconds']))
    correction = ('R249 conflates scratch-allocation reuse with execution aliasing. Only first-chunk allocation reuse requires G=n. '
                  'Frozen source sets m.residual_inplace=inplace also for mixed G<n, then m.R aliases residual_gpu_row(c0) for GPU chunks; '
                  'input==m.R and output==residual skip both D2D copies. Mixed in-place therefore eliminates GPU-prefix D2D copies too, '
                  'while retaining existing scratch for host chunks and allocating all G GPU rows. Host suffix copies remain unchanged. '
                  'No runtime numerical or fit qualification follows.')
    write(C / 'mixed-residual-placement-estimates.json', dict(
        created_utc=now, production_source=estimates['production_source'],
        scope='Conditional source arithmetic, no measurement/fit/adoption. G is effective already-rounded prefix. Examples use FIRST0,T8192; other schedules require actual first_len/m.T rounding.',
        bytes_per_position=40960, boundaries=47,
        memory_rate_convention='VRAM copy capacity counts logical read+write: two D2D handoff directions times read/write equals four times each-direction payload. Host rates count one-way payload.',
        VRAM_logical_read_write_copy_GBps=330.07315122276316,
        first_chunk_allocation_reuse='Only inplace && G=n reuses first_len existing scratch rows; subtract first_len*40960 from new GPU allocation, not from total plane extent. Mixed inplace G<n reuses no scratch allocation.',
        root_R249_correction=correction, scenarios=samples))
    layout = src.parents[3] / 'include/strata/core/layout.hpp'
    qsa = src.parents[3] / 'include/strata/kernels/qsa.hpp'
    assert 'int64_t n_head_kv = 2;' in layout.read_text()
    assert 'int64_t head_dim = 256;' in layout.read_text()
    write(C / 'current-pack-memory-geometry-reconciliation.json', dict(
        created_utc=now, scope='Separate metadata/source reconciliation; original R248 and correction receipt remain unchanged.',
        prior_correction=dict(path='residual-dimension-correction.json', **ident(C / 'residual-dimension-correction.json')),
        current_pack_census=dict(path=str(census.relative_to(W)), commit=COMMIT, **ident(census)),
        layout_source=dict(path=str(layout), **ident(layout)), qsa_source=dict(path=str(qsa), **ident(qsa)),
        corrected_QSA_heads=2, corrected_QSA_head_dim=256,
        prior_R248_QSA_heads=4, prior_R248_QSA_head_dim=128,
        explanation='R248 transcribed the head decomposition incorrectly. Current header and source specify2heads*256; product512 matches prior4*128, so all quoted FP16KV byte totals and QSA subtotal bytes remain unchanged. Do not use prior decomposition for attention geometry or other workspace formulas.',
        current_MAXBLOB=metadata['maximum_packed_blob_bytes'],
        current_layer_cache_MAXBLOB_times512_bytes=metadata['layer_cache_MAXBLOB_times512_bytes'],
        GDN_recurrence_bytes=36*128*48*128*4,
        GDN_convolution_history_bytes=36*10240*(4-1)*4,
        limitations='Header/source metadata does not attest actual selected route, residency, free memory, graph/allocator peaks or full-context correctness. Current pack matches historical maximum size arithmetic but memory admission is still unmeasured.'))
    readme = C / 'README.md'
    readme.write_text(readme.read_text() + '''
The [mixed-placement worksheet](mixed-residual-placement-estimates.json) gives the host/device copy tradeoff. With an effective 8K GPU prefix in a 32K call, GPU backing is 320MiB and the conditional host residual time falls from 25.66s to 19.24s. Ordinary placement adds about 0.19s of device copies under the separate attained VRAM-copy rate. Enabling the existing in-place flag also aliases GPU-prefix compute in a mixed layout and skips those device copies; only scratch allocation reuse requires all rows on GPU. This corrects R249's overly narrow all-GPU-only alias claim. Both placements require numerical and memory admission checks.

The [current-pack geometry reconciliation](current-pack-memory-geometry-reconciliation.json) also corrects R248's head decomposition: current metadata/source have two KV heads of width256, rather than four of width128. Their product is the same, so the quoted FP16KV and QSA subtotal bytes remain unchanged; other head-sensitive formulas must use the actual geometry. Source GDN recurrence and convolution history add 113,246,208 and 4,423,680 bytes before other buffers. The current local pack census verifies seven paired formats and MAXBLOB2,662,400; it is a header inspection, not a runtime admission result.
''')
    prev = R / 'report-registry-v94.json'
    assert ident(prev)['sha256'] == '0d435fede6563c949ec411feb1e79ff5295675b9f4a05d1eb4f228aea65c5ccc'
    registry = json.loads(prev.read_text())
    assert len(registry['reports']) == 355
    A.mkdir()
    entries = []
    for name, sha, agent, scope in (
        ('round249-mixed-residual-prefix-traffic-ownership-and-admission.txt', '0236a48d04056bddb6c43a5b6143512e8c3cbb0664f85516fcafa41ea609bddb', '/root/research_bottleneck_evidence_audit_v203', 'Exact mixed residual prefix traffic, rounding and source ownership/admission'),
        ('round250-production-gpu-dequant-paired-format-and-oracle-contract.txt', '023ddf0d751413924ea58450008b9716b94a8df4984c21fc41e8443362148344', '/root/research_native_iq4nl_esimd_k640_v202', 'Actual production GPU dequant role/domain/oracle/finite capacity control contract')):
        p = R / name
        assert ident(p)['sha256'] == sha
        copy(p, A / name)
        e = dict(path=str(p), **ident(p), agent=agent, model='gpt-6-luna', scope=scope, status='completed-read-only', root_full_report_reviewed=True)
        entries.append(e)
        registry['reports'].append(e)
    corrections = dict(R249=correction,
                       R250='Source role/domain/shape recipe accepted, not a capacity result. Root current header census now pins actual seven pairs and all144role descriptors.14*7generic eligibility is broader than actual pack.7/25samples are a predeclared control choice, not an algorithmic requirement. Wrappers returnvoid: timing/marker bounds must not be mislabeled exclusive internal kernel service. New source-only Sol harness assigned; root owns all tests.',
                       R248_metadata='Current header/source KV2heads*256 corrects report4*128; byte product512 and all prior quoted KV totals remain unchanged. Reconciliation is separate from original immutable report/correction receipt.')
    registry.update(registry_version=95, created_utc=now, research_completed=357, new_reports=entries,
                    previous_committed_registry=dict(path=str(prev), **ident(prev), storage_commit=COMMIT),
                    scope='Current actual pack inventory and corrected mixed in-place execution/capacity control plan',
                    live_agent_snapshot=dict(time_utc=now, agents=[dict(agent=e['agent'], model='gpt-6-luna', status='completed') for e in entries] +
                        [dict(agent='/root/implement_prefill_phase_timer_validity', model='gpt-6.1-sol', status='running-source-only', scope='Bounded actual-data GPU IQdequant capacity control; no build/test')]),
                    current_root_decisions=dict(capacity='Broad component attained capacities complete; actual GPU IQdequant remains unmeasured.',
                                                pack='All48layer/144role metadata matched;7actual pairs and50,292,326,400packed bytes, not runtime route/payload qualification.',
                                                mixed='Existing mixed inplace skips GPU-prefix D2D copies; scratch allocation reuse only allGPU. No memory or numerical qualification.',
                                                production='No new model throughput or adopted candidate.', adopted=False))
    registry['root_review_corrections_round72'] = corrections
    reg = R / 'report-registry-v95.json'
    assert not reg.exists()
    write(reg, registry)
    copy(reg, A / reg.name)
    write(A / 'root-review.json', dict(created_utc=now, reports=entries, corrections=corrections, registry_identity=ident(reg), adopted=False))
    (A / 'README.md').write_text('R249 source-derived traffic accepted with a separate root correction: mixed in-place also skips GPU-prefix D2D; only allocation reuse is all-GPU-only. R250 defines role/shape/oracle/lifetime admission for an isolated actual-data GPU dequant control. Main independently validates all144current pack descriptors, computes placement scenarios and corrects R248head decomposition without changing correct KV bytes. Source-only Sol control assigned; root tests pending.\n')
    copy(__file__, A / Path(__file__).name)
    for args in (('diff', '--check'), ('add', str(A.relative_to(W)), str(C.relative_to(W))),
                 ('diff', '--cached', '--check'), ('commit', '-m', 'bench: reconcile mixed residual placement and actual dequant formats'), ('push',)):
        subprocess.run(['git', *args], cwd=W, check=True)
    print(json.dumps(dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip(), registry_identity=ident(reg), reviewed_reports=len(registry['reports']))))
