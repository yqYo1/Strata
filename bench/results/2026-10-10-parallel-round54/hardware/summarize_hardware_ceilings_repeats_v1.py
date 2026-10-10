"""Validate every closed hardware cell/sample before combining fresh processes."""
from pathlib import Path
import collections, datetime, fcntl, hashlib, itertools, json, math, statistics

B = Path(__file__).parent
def pin(p):
    with Path(p).open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(p), 'bytes': Path(p).stat().st_size, 'sha256': digest}

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dest = B / 'hardware-ceilings-three-process-summary-v1.json'
    assert not dest.exists()
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'processes': [], 'cells': [], 'scope': 'Three fresh processes, seven non-warmup samples per case per process. Observed sustainable rates, not absolute hardware maxima. Decimal GB/s and TFLOP/s, host-inclusive monotonic submit+wait timing; logical bytes exclude write-allocate traffic. CPU read has a reduction dependency. No inference throughput, simultaneous resource ceiling, measured DRAM traffic or CI claim.', 'performance_adoption': False}
    all_cells = collections.defaultdict(list)
    memory_cases = ['read', 'copy_cached', 'copy_nontemporal', 'triad_cached']
    gpu_transfer = ['H2D_pageable', 'D2H_pageable', 'H2D_host_USM', 'D2H_host_USM']
    gpu_vram = ['VRAM_queue_copy', 'VRAM_kernel_copy', 'VRAM_read_xor', 'VRAM_triad_xor']
    Ms = [1, 4, 8, 32, 80, 160, 320, 640, 1280, 8192]
    boot = None
    for stage in ['cpu', 'measure-gpu']:
        source_identity = binary_identity = environment = None
        for repeat in [1, 2, 3]:
            folder = f'hardware-ceilings-v1-{stage}' if repeat == 1 else f'hardware-ceilings-repeat-v1-{stage}-r{repeat}'
            p = B / folder / 'record.json'
            r = json.loads(p.read_text())
            assert r['passed'] and r['complete'] and not r['active'] and not r.get('error')
            assert all(c['exit_code'] == 0 and c['normal_exit'] and c['session_empty'] and c['observation_complete'] and c['direct_child_reaped'] and not c['errors'] and not c['survivors'] for c in r['commands'])
            if boot is None: boot = r['boot_id']
            assert r['boot_id'] == boot
            if source_identity is None:
                source_identity, binary_identity, environment = r['source'], r['binary'], r['environment']
            assert r['source'] == source_identity and r['binary'] == binary_identity and r['environment'] == environment
            for x in [r['source'], r['binary']]:
                assert pin(x['path']) == x
            assert r['benchmark_stderr']['bytes'] == 0
            samples = collections.defaultdict(dict)
            validations = []
            for row in r['result_rows']:
                kind = row['kind']
                if kind in ['validation', 'gemm_validation']:
                    assert row['passed']
                    if stage == 'cpu': key = ('cpu', row['case'], row['threads'])
                    elif kind == 'gemm_validation': key = ('gemm', row['case'], row['M'], row['weight_mode'])
                    elif row['case'] in gpu_transfer: key = ('transfer', row['case'], row['payload_bytes_per_call'])
                    else: key = ('vram', row['case'], 256 << 20)
                    validations.append(key)
                if kind not in ['sample', 'gemm_sample']: continue
                if row.get('warmup'): continue
                seconds = row['seconds']
                assert math.isfinite(seconds) and seconds > 0
                index = row['sample']
                assert index in range(7)
                if stage == 'cpu':
                    key = ('cpu', row['case'], row['threads'])
                    multiplier = {'read': 1, 'copy_cached': 2, 'copy_nontemporal': 2, 'triad_cached': 3}[row['case']]
                    expected_bytes = (512 << 20) * 4 * multiplier
                    assert row['logical_bytes'] == expected_bytes
                    rate = expected_bytes / seconds / 1e9
                    recorded = row['logical_GBps']
                elif kind == 'gemm_sample':
                    N, K = {'GU': (1280, 2560), 'Down': (2560, 640)}[row['case']]
                    assert row['N'] == N and row['K'] == K
                    assert row['calls'] == (64 if row['M'] < 640 else 16)
                    key = ('gemm', row['case'], row['M'], row['weight_mode'])
                    rate = 2 * row['M'] * N * K * row['calls'] / seconds / 1e12
                    recorded = row['TFLOPps']
                else:
                    if row['case'] in gpu_transfer:
                        assert row['calls'] == 4 and row['logical_bytes'] % 4 == 0
                        payload = row['logical_bytes'] // 4
                        assert payload in [1 << 20, 8 << 20, 64 << 20, 256 << 20]
                        key = ('transfer', row['case'], payload)
                    else:
                        assert row['calls'] == 16
                        multiplier = {'VRAM_queue_copy': 2, 'VRAM_kernel_copy': 2, 'VRAM_read_xor': 1, 'VRAM_triad_xor': 3}[row['case']]
                        assert row['logical_bytes'] == (256 << 20) * 16 * multiplier
                        key = ('vram', row['case'], 256 << 20)
                    rate = row['logical_bytes'] / seconds / 1e9
                    recorded = row['logical_GBps']
                assert math.isclose(rate, recorded, rel_tol=1e-5, abs_tol=0.000002), (key, rate, recorded)
                assert index not in samples[key], (key, index)
                samples[key][index] = {'sample': index, 'seconds': seconds, 'recomputed_rate': rate}
            if stage == 'cpu':
                expected = {('cpu', c, t) for c, t in itertools.product(memory_cases, [1, 2, 4, 6, 12])}
            else:
                assert not r['visible_kernel_GPU_entries'] and not r['devcoredump_after']
                expected = {('transfer', c, n << 20) for c, n in itertools.product(gpu_transfer, [1, 8, 64, 256])}
                expected |= {('vram', c, 256 << 20) for c in gpu_vram}
                expected |= {('gemm', c, m, w) for c, m, w in itertools.product(['GU', 'Down'], Ms, ['hot', 'rotate8'])}
            assert set(samples) == set(validations) == expected
            assert len(validations) == len(expected)
            assert all(set(v) == set(range(7)) for v in samples.values())
            result['processes'].append({'stage': stage, 'repeat': repeat, 'receipt': pin(p), 'source': source_identity, 'binary': binary_identity, 'started_utc': r['started_utc'], 'finished_utc': r['finished_utc'], 'complete_pass': True, 'unique_validated_cells': len(expected), 'measured_samples_per_cell': 7})
            for key, data in samples.items():
                ordered = [data[i] for i in range(7)]
                rates = [x['recomputed_rate'] for x in ordered]
                all_cells[key].append({'repeat': repeat, 'samples': ordered, 'median': statistics.median(rates), 'min': min(rates), 'max': max(rates)})
    result['boot_id'] = boot
    for key, processes in sorted(all_cells.items()):
        medians = [p['median'] for p in processes]
        rates = [x['recomputed_rate'] for p in processes for x in p['samples']]
        result['cells'].append({'cell': list(key), 'unit': 'TFLOP/s' if key[0] == 'gemm' else 'GB/s', 'processes': processes, 'median_of_process_medians': statistics.median(medians), 'process_median_range': [min(medians), max(medians)], 'pooled_sample_range': [min(rates), max(rates)]})
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': str(dest), 'processes': len(result['processes']), 'cells': len(result['cells'])}))
