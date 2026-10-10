from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import math
import shutil
import subprocess

B = Path(__file__).parent
R = B / 'research-20261009'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
C = W / 'bench/results/2026-10-10-hardware-architecture-decisions'
A = W / 'bench/results/2026-10-10-parallel-round71'
OLD_COMMIT = 'ad6193bf8c7d1289941953f1496b6fc15ac2066b'


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
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == OLD_COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
    assert not A.exists()
    p = C / 'conditional-reuse-estimates.json'
    old_identity = ident(p)
    assert old_identity['sha256'] == '38289f144de3f1feeb295b94a53f3142d4cfed0aa238939e852723d80098e1b1'
    old = json.loads(p.read_text())
    src = Path(old['production_source']['path'])
    assert ident(src) == {k: old['production_source'][k] for k in ('bytes', 'sha256')}
    assert 'constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;' in src.read_text()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    fixed = json.loads(p.read_text())
    fixed['scope'] = ('Source/count-derived conditional arithmetic, NOT measured production DMA times, '
                      'minima, current-pack fit or adoption. Frozen unadopted layer-major source requires '
                      'resident QSA K/V. Separate capacity rates use 256MiB transfers; actual residual '
                      'chunks/path/cache/per-layer formats/other work require reconciliation. Corrected '
                      'residual width is D=N*HC=10240 FP32 elements per position; expert N=2560 is distinct.')
    fixed['residual_geometry'] = dict(expert_hidden_N=2560, residual_streams_HC=4,
                                    residual_FP32_elements_D=10240, bytes_per_element=4,
                                    layer_boundaries=47, bytes_per_position=40960,
                                    source_constant_line=113)
    corrected = []
    for original in old['residual_scenarios']:
        n = original['hypothetical_positions']
        plane = n * 10240 * 4
        payload = plane * 47
        rates = fixed['memory_rates']
        seconds = payload / 1e9 / rates['H2D_pageable_GBps'] + payload / 1e9 / rates['D2H_pageable_GBps']
        weights = fixed['all_layer_packed_weights_once_bytes'] / 1e9 / rates['H2D_host_USM_GBps']
        assert plane == 4 * original['residual_buffer_bytes']
        assert payload == 4 * original['all_RAM_residual_H2D_payload_bytes']
        assert math.isclose(seconds, 4 * original['conditional_pageable_residual_seconds'], rel_tol=1e-12)
        assert math.isclose(weights, original['conditional_host_USM_packed_weight_seconds'], rel_tol=1e-12)
        corrected.append(dict(hypothetical_positions=n, residual_buffer_bytes=plane,
                              all_RAM_residual_H2D_payload_bytes=payload,
                              all_RAM_residual_D2H_payload_bytes=payload,
                              conditional_pageable_residual_seconds=seconds,
                              conditional_host_USM_packed_weight_seconds=weights,
                              conditional_serial_residual_and_weight_seconds=seconds + weights))
    fixed['residual_scenarios'] = corrected
    fixed['correction_record'] = 'residual-dimension-correction.json'
    write(p, fixed)
    capacity = 10666115072
    budget = []
    for n in (32768, 262144):
        kv = n * 4 * 128 * 2 * 2 * 12
        indexer = (n // 4 + 2) * 128 * 4 * 12
        rope = 2 * n * (64 // 2) * 4
        budget.append(dict(positions=n, resident_QSA_FP16_KV_bytes=kv,
                           pooled_indexer_bytes=indexer, shared_RoPE_bytes=rope,
                           QSA_subtotal_bytes=kv + indexer + rope,
                           QSA_subtotal_GiB=(kv + indexer + rope) / 2**30,
                           whole_device_capacity_minus_QSA_subtotal_bytes=capacity - kv - indexer - rope,
                           residual_total_bytes=n * 10240 * 4))
    write(C / 'residual-dimension-correction.json', dict(
        created_utc=now, source='Root arithmetic error found by independent completed R248 source audit',
        error='Expert hidden N=2560 was incorrectly used for the four-stream residual D=N*HC=10240.',
        original=dict(commit=OLD_COMMIT, path=str(p.relative_to(W)), **old_identity,
                      residual_scenarios=old['residual_scenarios'],
                      original_32768_conditional_serial_seconds=sum(
                          old['residual_scenarios'][0][k] for k in (
                              'conditional_pageable_residual_seconds', 'conditional_host_USM_packed_weight_seconds'))),
        corrected=dict(path=str(p.relative_to(W)), **ident(p), residual_scenarios=corrected),
        unchanged='Original measurements, expert GEMM arithmetic, historical packed-weight counts, failed results and candidate adoption status remain unchanged.',
        production_source=fixed['production_source'],
        decision='All-RAM residual plus once-per-layer weights is conditionally 33.4603s at 32K before compute/other work, exceeding the 32.768s target budget; placement/bytes must be evaluated in addition to weight reuse.',
        QSA_geometry_scope='R248 frozen-source/prior-report geometry, not newly measured model metadata or free/peak-memory receipt.',
        device_total_capacity_bytes=capacity,
        QSA_geometry=dict(layers=12, kv_heads=4, head_dim=128, indexer_block=4, indexer_dim=128, n_rot=64),
        conditional_memory_budgets=budget,
        full_context_all_GPU_residual_excess_over_whole_device_bytes=262144 * 10240 * 4 - capacity,
        source_eligibility='Current layer-major rejects kv_mode != 0; historical streamed-KV arm is not eligible for this source.',
        limitations='QSA subtotal excludes GDN, convolution, dense/base weights, session/prefill/dequant scratch, layer/expert caches, graph/runtime and allocator peaks. Historical MAXBLOB is not current-pack metadata. No fit, performance or full-context qualification is claimed.'))
    (C / 'README.md').write_text('''# Hardware capacities and the architectural budget

Measured on Ryzen 5 5600X, 128GB RAM and Arc B570 on 2026-10-10. The [capacity matrix](component-capacity-matrix.json) pins individual samples, units and three-process ranges. These are attained rates for specified operations, not absolute hardware maxima or new model throughput.

| Component / operation | Attained rate | What the number covers |
| --- | ---: | --- |
| CPU FP32 FMA | 0.818 TFLOP/s | Six physical cores, register arithmetic |
| RAM read | 36.87 GB/s | Six physical cores, arrays larger than LLC |
| RAM copy | 37.58 GB/s | Non-temporal, logical source read plus destination write |
| Packed host-copy jobs | 13.74 GB/s | One-way payload, three workers; ordinary RAM, source/copy/wakeup/ack control |
| RAM→GPU | 6.447 GB/s | Host USM, one-way 256MiB payload, host-inclusive completion |
| GPU→RAM | 5.643 GB/s | Host USM, one-way 256MiB payload |
| Pageable RAM→GPU / GPU→RAM | 4.605 / 5.274 GB/s | The path used in the residual estimate below |
| VRAM copy | 330.07 GB/s | Logical source read plus destination write |
| GPU GU / Down GEMM, M8192 | 52.69 / 41.61 TFLOP/s | FP16 inputs, FP32 output, hot weights |
| GPU GU / Down GEMM, M80 | 13.16 / 10.08 TFLOP/s | Eight rotating weights, synthetic useful row grouping |
| SSD sequential, 1MiB | 2.266 GB/s | Same-file ZFS path, 16 workers |
| SSD random, 4KiB | 0.05099 GB/s; 12,450 IOPS | Same-file ZFS path, 16 workers |

RAM/VRAM copy counts read and write; PCIe and host packed-copy rates count payload once. They are not directly comparable utilization ratios. Hot large GEMM rates cannot predict small expert service. Register FP32 FMA does not set the native quantized CPU dot-product roof. Device IQ dequantization and actual model service remain gaps.

For 32,768 positions, the 1000 token/s budget is 32.768 seconds. Historical 190.2GB expert payload at the separate attained 6.447GB/s host-USM rate takes 29.50 seconds. Assigning every expert GEMM hypothetically M80/160/640 yields 12.95/6.57/3.85 seconds of GEMM arithmetic respectively. With serial execution and no other work, corresponding target payload budgets are 127.79/168.89/186.43GB. These are conditional scenarios, not measured model times or physical lower bounds. Current route shape/format/source histograms, actual bytes and all remaining work must be measured.

| Change | What can change | Evidence needed |
| --- | --- | --- |
| Reorder independent submissions | Queue overhead and overlap with unchanged bytes/work | Exact operation timeline and dependencies; the safe two-queue control had zero target overlap |
| Keep one layer's packed weights across all chunks | Fewer packed transfers and host copies | Actual route/reuse, simultaneous memory peaks and full-context state/numerical validation |
| Group useful routed rows per weight load | Attained GEMM rate and transfer/dequant amortization | Actual per-expert row histogram, correctness and latency |
| Keep some intermediate rows in VRAM | Fewer residual H2D/D2H transfers | Simultaneous allocations/KV budget and full-context lifetime/numerical validation |
| Change packed representation/device kernels | Packed bytes or dequant/GEMM service | Operation-specific attribution, numerical tests and separate prefill/decode comparisons |
| Group PLE reads by filesystem record | Fewer source calls but potentially more returned bytes | Actual cache/reader geometry and service; the cold census alone does not select I/O policy |

The frozen layer-major source loads every expert's packed blob once per layer and runs all chunks while retaining that layer. Historical layer-specific sizes imply 50,292,326,400 bytes for the explicit 512-expert/layer scenario, versus historical A's 190,240,998,400 bytes: about 3.78 times less. Its largest historical temporary layer-cache allocation was 1,363,148,800 bytes. These are historical pack counts, not current-pack metadata or evidence that the alternative is adopted. Reuse changes the byte budget; submitting the same copies earlier does not.

The residual plane has **D=10240 FP32 elements per position**: expert hidden N=2560 times four streams. The previous worksheet incorrectly used N for this plane. Its preserved original commit/hash and the fourfold correction are in [the correction receipt](residual-dimension-correction.json); original benchmark results are unchanged.

| Positions | Complete residual plane | Payload per PCIe direction over 47 boundaries | Conditional all-RAM residual copies | Once-per-layer packed copies | Serial copy sum |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1,342,177,280 B (1.25GiB) | 63,082,332,160 B | 25.66s | 7.80s | 33.46s |
| 262,144 | 10,737,418,240 B (10GiB) | 504,658,657,280 B | 205.27s | 7.80s | 213.07s |

These combine source counts with separate pageable H2D/D2H and host-USM capacity measurements. Actual chunk-sized service, cache/source path, copy overlap and other work are not measured here. Under this serial scenario, all-RAM residual plus weight reuse alone already exceeds the 32K target budget before computation. Partial GPU placement can lower residual PCIe bytes but consumes VRAM; grouping/reuse must therefore be evaluated with placement, not from weight savings alone.

Current layer-major explicitly rejects streamed QSA K/V (`kv_mode != 0`). For the cited geometry its resident K/V, indexer and RoPE subtotal is 864,038,912 bytes at 32K and 6,912,212,992 bytes at 262,144 positions. The latter leaves 3,753,902,080 bytes below the whole-device capacity of 10,666,115,072 bytes, before GDN, all weights, caches, prefill/dequant/session scratch, graphs and runtime. This is not free memory. The complete 10GiB residual plane alone exceeds that device capacity by 71,303,168 bytes. Full GPU placement cannot fit; mixed placement has no complete admission receipt yet. Historical streamed-KV C is a different, ineligible route for this source. Existing alternative numerical rejection remains unresolved.

The packed host-copy control reaches 13.74GB/s with three workers, with no clear improvement from four or six. This is ordinary aligned RAM with a synthetic acknowledgment, not production host-USM/DMA/source service. Production allocation kind, actual source path and dependencies still need attribution. The current host-copy default is three workers on 12 logical CPUs; source-reading threads are a separate setting.

Next measure actual expert rows, role formats, bytes/source/cache hits and distinct host stager, H2D, GPU IQ dequant, GU/Down, PLE and graph/cache restoration service. Returned oneMKL events need internal-operation coverage qualification before being called complete service spans. Then compare isolated changes with fresh inputs of at least 32K and independent repetitions; qualify all 262,144 physical positions before adoption. GPU safety settings are unchanged and no candidate is adopted by this worksheet.
''')
    prev = R / 'report-registry-v93.json'
    assert ident(prev)['sha256'] == '14ddd7844b7ea8f4044cb765b36cb649b859e7f41ee76a361748b96fd9f53c77'
    registry = json.loads(prev.read_text())
    assert len(registry['reports']) == 353
    A.mkdir()
    entries = []
    for name, sha, agent, scope in (
        ('round247-host-usm-publication-dma-reuse-and-nt-fence-contract.txt',
         'd64d5ac3178f82ff2c17b1dc80b7e61966d56213100aa07f6e6941eea048d631',
         '/root/research_bottleneck_evidence_audit_v203',
         'Production host-USM context/publication/DMA reuse and explicit NT fence contract'),
        ('round248-layer-major-full-context-live-memory-and-eligibility-budget.txt',
         '601da38175f6446e42449c31cb1cc21f8f965be2a5df6b19b4ac222ec6445644',
         '/root/research_native_iq4nl_esimd_k640_v202',
         'Current layer-major resident-KV eligibility and simultaneous full-context memory budget')):
        q = R / name
        assert ident(q)['sha256'] == sha
        copy(q, A / name)
        entry = dict(path=str(q), **ident(q), agent=agent, model='gpt-6-luna', scope=scope,
                     status='completed-read-only', root_full_report_reviewed=True)
        entries.append(entry)
        registry['reports'].append(entry)
    corrections = dict(
        R247='Static allocation/context, ordinary-store release/acquire publication, copy dependency and completion-before-reuse contract accepted. No definite UB found in this scope. Explicit NT must fence before publication; tagged glibc fence does not attest target IFUNC. No unsupported extra fence or weakening adopted; actual host-USM/device consumer remains unqualified.',
        R248='Accepted critical D=10240 residual correction and current resident-KV guard. Root independently recomputed all byte/rate arithmetic. Guard is at frozen source line2507 (report approximate2489-2495). QSA subtotal is6.437511444GiB retaining +2 indexer rows, not exactly6.4375; exact bytes take precedence. Historical MAXBLOB is not current-pack metadata; no memory-fit or inference proof. Original report is preserved unchanged.',
        root='Earlier residual worksheet incorrectly used expert N=2560 instead of four-stream D=10240. Corrected current arithmetic with separate original-commit/hash/values receipt. Previous14.2159s32K serial copy estimate is33.4603s; measurements and expert weight/FLOP counts are unchanged.')
    registry.update(registry_version=94, created_utc=now, research_completed=355, new_reports=entries,
                    previous_committed_registry=dict(path=str(prev), **ident(prev), storage_commit=OLD_COMMIT),
                    scope='Component capacities reconciled with corrected residual transfer and full-context memory budgets',
                    live_agent_snapshot=dict(time_utc=now, source='Direct collaboration.list_agents at turn resume: all three completed',
                                             agents=[dict(agent=e['agent'], model='gpt-6-luna', status='completed') for e in entries] +
                                                    [dict(agent='/root/implement_prefill_phase_timer_validity', model='gpt-6.1-sol', status='completed-source-only')]),
                    current_root_decisions=dict(capacity='All completed attained component rates retained; GPU IQ dequant and actual model route/service remain open.',
                                                residual='Corrected width10240; all-RAM32K copy sum33.4603s conditional before compute. Partial placement requires simultaneous budget and numerical qualification.',
                                                full_context='Current layer-major requires resident K/V; full residual plane10GiB cannot fit on device. Mixed-placement fit/full262144 numerical result unknown.',
                                                production='No new whole-model measurement or adopted optimization.', adopted=False))
    registry['root_review_corrections_round71'] = corrections
    reg = R / 'report-registry-v94.json'
    assert not reg.exists()
    write(reg, registry)
    copy(reg, A / reg.name)
    write(A / 'root-review.json', dict(created_utc=now, reports=entries, corrections=corrections,
                                      registry_identity=ident(reg), residual_correction_identity=ident(C / 'residual-dimension-correction.json'),
                                      adopted=False))
    (A / 'README.md').write_text('R247 accepts the current static host-USM publication/completion contract, with an explicit SFENCE gate for hand-written NT stores. R248 identifies current resident-KV eligibility and the four-stream residual width. Root corrects the earlier residual arithmetic by fourfold with preserved commit/hash/values, not by rewriting measurements. Current architecture worksheet now includes correct PCIe and full-context memory budgets. No new model throughput, memory admission or candidate qualification.\n')
    copy(__file__, A / Path(__file__).name)
    for args in (('diff', '--check'), ('add', str(A.relative_to(W)), str(C.relative_to(W))),
                 ('diff', '--cached', '--check'), ('commit', '-m', 'bench: correct four-stream residual traffic and full-context budgets'), ('push',)):
        subprocess.run(['git', *args], cwd=W, check=True)
    print(json.dumps(dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip(),
                          registry=str(reg), registry_identity=ident(reg),
                          corrected_scenarios=corrected, reviewed_reports=len(registry['reports']))))
