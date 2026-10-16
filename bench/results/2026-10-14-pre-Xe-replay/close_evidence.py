#!/usr/bin/env python3
"""Archive closed receipts and verify route/source identity; never delete logs."""
import fcntl
import gzip
import hashlib
import json
import re
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
Q = Path(__file__).parent
D = P / 'bench/results/2026-10-14-primary-model-baseline'
ROOTS = ['main-pre-Xe-1eb-build-root-v1', 'main-pre-Xe-1eb-build-root-v2',
         'main-primary-fp16-prefill-short-root-v1', 'main-primary-stock-fp16-root-v1']
HELPER = ('add_executable(sycl_transfer_ceiling EXCLUDE_FROM_ALL\n'
          '  ${PROJECT_SOURCE_DIR}/bench/results/2026-10-14-primary-transfer-ceiling/transfer_ceiling.cpp)\n'
          'target_link_libraries(sycl_transfer_ceiling PRIVATE strata_sycl_runtime)\n'
          'target_compile_features(sycl_transfer_ceiling PRIVATE cxx_std_17)\n\n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def alive_same(pid, ticks):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return int(text[text.rfind(')') + 2:].split()[19]) == ticks


def closed(record):
    assert not record['active']
    for command in record['commands']:
        assert command['direct_child_reaped'] and command['session_empty'] and not command.get('survivors'), command['label']
        for owner in command.get('owners', []):
            assert not alive_same(owner['pid'], owner['start_ticks']), owner


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        optional = [B / f'main-pre-Xe-1eb-gpu-parity-root-v{v}/record.json' for v in [1, 2]]
        optional.append(B / 'main-pre-Xe-1eb-kv-parity-root-v3/record.json')
        roots = ROOTS + [path.parent.name for path in optional if path.exists()]
        records = {name: json.loads((B / name / 'record.json').read_text()) for name in roots}
        for record in records.values():
            closed(record)
        long = records['main-primary-stock-fp16-root-v1']
        short = records['main-primary-fp16-prefill-short-root-v1']
        old = records['main-pre-Xe-1eb-build-root-v2']
        assert long['passed'] and long['complete'] and long['performance_qualified']
        assert short['passed'] and short['complete']
        for record in [short, long]:
            assert not record['fresh_faults'] and not record['driver_dump_exists'] and record['boot_unchanged']
        assert old['passed'] and old['complete'] and not old['GPU_executed']
        pins = long['source_pins']
        normalization = []
        for name, digest in pins.items():
            data = (P / name).read_bytes()
            if name == 'cmake/sycl_backend.cmake':
                assert data.count(HELPER.encode()) == 1
                normalized = data.replace(HELPER.encode(), b'')
                assert sha(normalized) == digest
                normalization.append(dict(path=name, compiled_sha256=digest,
                    current_sha256=sha(data), exact_removed_text=HELPER,
                    scope='Excluded transfer-helper target only; native compiled source unchanged'))
            else:
                assert sha(data) == digest, name
        native = B / 'main-primary-native-build-root-v1/build/strata'
        assert sha(native.read_bytes()) == long['binary_sha256'] == short['binary_sha256']
        source = Path(old['source_path'])
        for item in old['source_manifest']:
            assert sha((source / item['path']).read_bytes()) == item['sha256'], item['path']
        for name, digest in old['binaries'].items():
            assert sha((Path(old['build_path']) / name).read_bytes()) == digest
        ldd = (B / 'main-pre-Xe-1eb-build-root-v2/owned/linked-library-inventory.stdout').read_text()
        assert 'not found' not in ldd
        assert f'libsycl.so.9 => /opt/intel/oneapi/compiler/2026.1/lib/libsycl.so.9' in ldd
        assert f'libur_loader.so.0 => /opt/intel/oneapi/compiler/2026.1/lib/libur_loader.so.0' in ldd
        write_json(Q / 'source-closure.json', dict(passed=True,
            primary_binary_sha256=long['binary_sha256'], primary_source_pin_count=len(pins),
            source_normalization=normalization, historical_source_manifest_count=len(old['source_manifest']),
            historical_source_bytes=old['source_bytes'], historical_source_revision=old['source_revision'],
            historical_binary_sha256=old['binaries']['strata'],
            historical_compiler=(B / 'main-pre-Xe-1eb-build-root-v2/owned/compiler-version.stdout').read_text(),
            historical_link_inventory=ldd,
            historical_compiler_contract='precise FP32 math, global subgroup32, per_kernel; no MMQ/fused TU',
            source_or_math_patches_to_historical_tree=[],
            GPU_reset_or_service_change=False))
        archive_index = []
        for name in roots:
            root = B / name
            paths = [root / 'record.json'] + [root / (label + '.json') for label in
                ['config', 'plan', 'result', 'engine-start', 'engine-exit',
                 'sampler-summary', 'stderr-relay-summary'] if (root / (label + '.json')).exists()]
            for original in paths:
                data = original.read_bytes()
                archive = Q / 'receipts' / name / (original.name + '.gz')
                archive.parent.mkdir(parents=True, exist_ok=True)
                packed = gzip.compress(data, mtime=0)
                assert gzip.decompress(packed) == data
                if archive.exists():
                    assert archive.read_bytes() == packed
                else:
                    archive.write_bytes(packed)
                archive_index.append(dict(path=str(archive.relative_to(Q)), original_path=str(original),
                    bytes=len(data), decoded_sha256=sha(data), archive_bytes=len(packed), archive_sha256=sha(packed)))
        write_json(Q / 'receipts-index.json', archive_index)
        base_path = D / 'receipts/main-primary-stock-bench-accepted-root-v2/record.json.gz'
        base_pin = next(x for x in json.loads((D / 'receipts-index.json').read_text()) if x['path'] == str(base_path.relative_to(D)))
        base_raw = gzip.decompress(base_path.read_bytes())
        assert sha(base_raw) == base_pin['decoded_sha256'] and sha(base_path.read_bytes()) == base_pin['archive_sha256']
        base = json.loads(base_raw)
        assert base['engine_args'] == long['engine_args']
        assert base['binary_sha256'] == long['binary_sha256']
        assert base['standard_client_source'] == long['standard_client_source']
        assert base['pack_metadata_sha256'] == long['pack_metadata_sha256']
        assert base['MTP_metadata_sha256'] == long['MTP_metadata_sha256']
        env_diffs = {key: dict(baseline=base['environment'].get(key), FP16=long['environment'].get(key))
            for key in base['environment'].keys() | long['environment'].keys()
            if base['environment'].get(key) != long['environment'].get(key)}
        housekeeping = {'XDG_CACHE_HOME', 'STRATA_MEASUREMENT_ENGINE_EXIT', 'STRATA_MEASUREMENT_ENGINE_START'}
        assert set(env_diffs) - housekeeping == {'STRATA_PREFILL_MMQ'}, env_diffs
        assert long['environment']['STRATA_PREFILL_MMQ'] == '0'
        def phase_stats(rows):
            import statistics
            return dict(prefill_median_tps=statistics.median(x['prompt_tps'] for x in rows),
                prefill_pooled_tps=sum(x['read_tokens'] for x in rows) * 1000 / sum(x['prompt_ms'] for x in rows),
                decode_median_tps=statistics.median(x['decode_tps'] for x in rows),
                decode_pooled_tps=sum(x['generated_tokens'] for x in rows) * 1000 / sum(x['decode_ms'] for x in rows))
        baseline_rows = base['long_phase_samples']
        candidate_rows = long['long_phase_samples']
        assert len(baseline_rows) == len(candidate_rows) == 3
        assert all(x['ok'] and x['prompt_tokens'] == x['read_tokens'] == 32049 and x['generated_tokens'] == 24 and x['reused_tokens'] == 0
                   for x in baseline_rows + candidate_rows)
        write_json(Q / 'stock-comparison.json', dict(primary_native_binary_sha256=long['binary_sha256'],
            method=long['performance_method'], actual_input_tokens=32049, generated_tokens=24,
            baseline_receipt=str(base_path.relative_to(P)), FP16_receipt='receipts/main-primary-stock-fp16-root-v1/record.json.gz',
            baseline_samples=baseline_rows, FP16_samples=candidate_rows,
            baseline_stats=phase_stats(baseline_rows), FP16_stats=phase_stats(candidate_rows),
            environment_differences=env_diffs, primary_source_postcheck='source-closure.json',
            fresh_faults=long['fresh_faults'], boot_unchanged=long['boot_unchanged'],
            normal_exit=True, adopted=False, quality_qualified=False, physical_262144_qualified=False,
            scope='One process per arm, stock3 repeated fresh32K requests. Prefill route only changed; decoded text/draft acceptance may change as a consequence of numerical route, so decode speed does not prove decode engine improvement. No sustained or restored decode qualification.'))
        r = B / 'research-20261009'
        names = {
            494: 'research-pre-Xe-minimum-runtime-gates-round494-v1.txt',
            495: 'research-pre-Xe-restore-component-priority-round495-v1.txt',
            496: 'research-fp16-grouped-gemm-runtime-quality-round496-v1.txt',
            497: 'research-decode-larger-speculative-window-round497-v1.txt',
            498: 'research-compact-fp16-proposal-review-round498-v1.txt',
            499: 'research-fp16-gemm-effective-contract-round499-v1.txt',
        }
        reports = []
        for round_number, name in names.items():
            original = r / name
            if not original.exists():
                continue
            data = original.read_bytes()
            target = Q / 'research' / name
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.write_bytes(data)
            reports.append(dict(round=round_number, model='gpt-6-luna', read_only=True,
                original_path=str(original), path=str(target.relative_to(Q)), sha256=sha(data)))
        write_json(Q / 'research-index.json', dict(reports=reports,
            previous_registry='../2026-10-14-performance-regression-audit/research-index.json',
            no_intervention_until_final=True, main_controls_builds_and_GPU=True,
            main_corrections=[
                'R490 previously attributed old1eb MoE to MMQ/fused. Actual old build compiles gemm.dp.cpp and fused stub; no MMQ/fused TU. R495 and reconstructed compile database correct this.',
                'R495 cannot assume fused in the canonical current stock arm: STRATA_PF_FUSED is absent; native_supported requires explicit requested(), hence canonical fused mode is off.',
                'R495 cache headroom sign is wrong: 268.60MiB versus325MiB nominal leaves more, not less, headroom. Old occupied expert weights were also268.60MiB; do not infer fewer cached weights from the allocation numbers.',
                'R497 lookup-derived41/61/45 rows are all target-evaluated rows including base/correction, not offered MTP proposals. Use stock drafts_offered/accepted for proposal work.',
                'R494 old physical context proof is cell262143 via262140 input+4 output; it does not qualify262144 input+0 output. New primary source retains its own size_t guards.',
                'R496 grouped GEMM tiny numerical differences are not an automatic quality rejection; realistic numerical/task and latency evaluation remains required.',
                'R499 treats absent MKL_BLAS_COMPUTE_MODE as unknown. For the reconstructed old build and current stock process, Owner uses Popen(env=explicit_whitelist), server config adds no env, and wrappers copy that environment: absence is known. Historical original unrecorded process settings remain separate.',
            ]))
        failure = B / 'main-pre-Xe-1eb-build-root-v1/owned/configure-reference.stderr'
        if failure.exists():
            raw = failure.read_bytes()
            assert len(raw) <= 256 << 10
            write_json(Q / 'export-failure.json', dict(original_status='failed configure, exit1, no GPU or model',
                cause='Initial archive omitted sycl/tools/ple_graph_oracle.cpp; v2 exports sycl/tools from the same commit.',
                original_path=str(failure), bytes=len(raw), sha256=sha(raw), stderr=raw.decode()))
        refusal = B / 'main-pre-Xe-1eb-gpu-parity-root-v1/owned/qsa-xmx-tails.stderr'
        if refusal.exists():
            raw = refusal.read_bytes()
            lines = raw.decode().splitlines()
            at = [i for i, line in enumerate(lines) if line == 'qsa_prompt_attn_batch refused the pools']
            assert len(at) == 1
            errors = sorted(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', raw.decode())))
            assert not errors
            stdout = refusal.with_suffix('.stdout').read_text()
            first = records['main-pre-Xe-1eb-gpu-parity-root-v1']
            write_json(Q / 'qsa-test-refusal.json', dict(original_status='failed v1, qsa exit2; no GPU hang',
                source_revision=first['source_revision'], original_path=str(refusal), bytes=len(raw), sha256=sha(raw),
                original_recipe_sha256=first['recipe_sha256'], partial_stdout=stdout,
                excerpt='\n'.join(lines[max(0, at[0]-4):at[0]+5]), UR_errors=errors,
                cause='Main selected zero fallback_batch XMX multi-format test. Old XMX explicitly refuses Q4 K; test runs Q4 after int8/fp16. Int8/fp16 passed, Q4 refused as declared, not a numerical/GPU failure.',
                relevant_sources=['sycl/src/kernels/qsa_prompt_attn_parity.cpp:254',
                    'sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp:369'],
                fresh_fault_checks=first['fault_checks'],
                correction='v2 uses existing fallback batch128/layout1 case across formats and actual129 rows; no source/engine patch, original failed receipt unchanged.'))
        budget_root = B / 'main-pre-Xe-1eb-gpu-parity-root-v2'
        if (budget_root / 'record.json').exists():
            prior = records[budget_root.name]
            raw = (budget_root / 'owned/kv-stream-restore.stderr').read_bytes()
            text = raw.decode()
            kv = next(x for x in prior['commands'] if x['label'] == 'kv-stream-restore')
            later = records.get('main-pre-Xe-1eb-kv-parity-root-v3')
            assert later and later['passed'] and later['complete']
            assert not prior['passed'] and 'aggregate owner log cap' in prior['error']
            write_json(Q / 'kv-log-budget.json', dict(original_status='v2 incomplete, controller stopped at64MiB aggregate API text cap',
                old_exit_code=kv['exit_code'], error=kv['errors'],
                original_path=str(budget_root / 'owned/kv-stream-restore.stderr'), bytes=len(raw), sha256=sha(raw),
                partial_stdout=(budget_root / 'owned/kv-stream-restore.stdout').read_text(),
                API_success_counts=dict(UR=text.count('UR_RESULT_SUCCESS'), ZE=text.count('ZE_RESULT_SUCCESS')),
                UR_errors=sorted(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', text))),
                validation_errors=[line for line in text.splitlines() if '<VALIDATION>[ERROR]' in line],
                ZE_errors=[line for line in text.splitlines() if '] ERROR (' in line],
                representative_last_lines='\n'.join(text.splitlines()[-24:]),
                post_budget_health=later['fault_checks'][0],
                resolution='Same KV executable/source ran to normal PASS in v3 with parameter validation and warning logging; full API success logs are supplied by earlier small checks. Original incomplete status unchanged.',
                resolved_trace_question=True))
        print(json.dumps(dict(passed=True, archives=len(archive_index), reports=len(reports),
            FP16=phase_stats(candidate_rows))), flush=True)


if __name__ == '__main__':
    main()
