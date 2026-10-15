#!/usr/bin/env python3
"""Mechanics only: same-binary FP32/native16, synthetic129 IDs, all48 layers."""
import array
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import struct
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('v2', 'v3'), 'new fixed build version; v1 failure is immutable')
    version = sys.argv[1]
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build = json.loads((A / f'own-native16-prefill-build-{version}.json').read_text())
        regression_name = f'own-native16-regression-correctness-{version}.json'
        original_regression = json.loads((A / regression_name).read_text())
        if not original_regression['passed']:
            require(version == 'v2' and not original_regression['active'] and
                    not original_regression['fresh_faults'] and not original_regression['driver_dump_exists'],
                    'only known closed separate-queue failure may use a narrower route gate')
            regression_name = f'own-native16-regression-followup2-correctness-{version}.json'
        for name in (f'own-native16-gemm-correctness-{version}.json', regression_name):
            v = json.loads((A / name).read_text())
            require(v['passed'] and v['complete'] and not v['active'] and not v['fresh_faults'], 'component gate')
            for c in v['commands']:
                require(module.closed(c), 'previous command closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'previous owner alive')
        require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'entire source frozen')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'], 'new engine binary')
        old = json.loads((A / 'own-dense-prefill-nll-smoke-v4.json').read_text())
        metadata = json.loads((A / 'selective-iq-clean-bench-full-0-v1.json').read_text())
        require(all(sha(Path(v['path'])) == v['sha256'] for v in old['runtime_files']), 'runtime files')
        for key in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
            require(all(sha(Path(n)) == h for n, h in metadata[key].items()), 'pack/MTP metadata')
        for name, v in old['model_identity'].items():
            st = Path(name).stat()
            require(st.st_size == v['bytes'] and st.st_mtime_ns == v['mtime_ns'], 'model stat identity')
        root = B / f'main-own-native16-model-smoke-root-{version}'
        target = A / f'own-native16-model-smoke-{version}.json'
        require(not root.exists() and not target.exists(), 'fresh model job')
        root.mkdir(mode=0o700)
        tokens, positions = old['tokens'], old['positions']
        require(tokens == list(range(129)) and positions == [0, 63, 127], 'fixed synthetic mechanics input')
        inp = root / 'tokens.txt'; inp.write_text(','.join(map(str, tokens)) + '\n')
        args = list(old['args']); args[args.index('--tokens-file') + 1] = str(inp)
        profile = Path(args[args.index('--expert-profile') + 1])
        local_profile = root / 'expert-profile.bin'; shutil.copyfile(profile, local_profile)
        args[args.index('--expert-profile') + 1] = str(local_profile)
        env = {k: v for k, v in old['environment'].items() if k.startswith('STRATA_') or k in (
            'PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'MKLROOT', 'OMP_NUM_THREADS',
            'CPLUS_INCLUDE_PATH', 'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD', 'SYCL_CACHE_PERSISTENT',
            'NEOReadDebugKeys', 'EnableDirectSubmission', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',
            'UR_ENABLE_LAYERS', 'UR_LOG_VALIDATION', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO')}
        for name in list(env):
            if name.startswith('STRATA_MEASUREMENT_') or name in ('STRATA_PREFILL_BOUNDARY_AUDIT', 'STRATA_PREFILL_PHASE_AUDIT'):
                del env[name]
        env.update(XDG_CACHE_HOME=str(root / 'cache'), STRATA_TRACE='1')
        require(env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'qualified single-queue production route only')
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[], cases={},
            source_commit=build['source_commit'], source_pins=build['source_pins'], binary_sha256=sha(exe),
            model_identity=old['model_identity'], pack_metadata_sha256=metadata['pack_metadata_sha256'],
            MTP_metadata_sha256=metadata['MTP_metadata_sha256'], runtime_files=old['runtime_files'],
            tokens=tokens, positions=positions, input_sha256=sha(inp), args=args, environment=env,
            external_expert_profile_sha256=sha(profile), local_profile_initial_sha256=sha(local_profile),
            recipe_sha256=sha(Path(__file__)), boot_id=boot_id(), raw_tensor_budget_bytes=16 << 20,
            regression_gate=regression_name, regression_gate_sha256=sha(A / regression_name),
            original_full_regression_passed=original_regression['passed'],
            regression_limit='Separate-queue original regression timeout is not a PASS. This model run is restricted to the independently tested single-queue route.',
            scope='Synthetic129 input/MTP4, all48 layers/two64 chunks, GPU64+host64 residuals; FP32 control vs nativeBF16 path, finite all128 NLL and independent selected fullvocabulary NLL. No speed/general-quality/fullcontext/RESTORE qualification.',
            performance_qualified=False, model_quality_qualified=False, physical_262144_qualified=False, restore_qualified=False)
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands; persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, argv, current, wall=900, cpu=18000, rss=88 << 30):
            c, out, err = owner.run(label, argv, current, P, wall=wall, cpu=cpu, rss_cap=rss,
                text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-6000:])
            return out, err
        cursor = None
        try:
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], hostenv, wall=60, cpu=120, rss=2 << 30)
            cursor = re.search(r'-- cursor: (\S+)', out.read_text())[1]
            losses = {}; rows_by_arm = {}
            for arm in ('fp32', 'bf16'):
                r = root / arm; r.mkdir()
                # Identical profile input for both processes, even if an engine
                # later writes a profile. External user's profile stays read-only.
                shutil.copyfile(profile, local_profile)
                current = dict(env, STRATA_SYCL_PREFILL_NATIVE16='0' if arm == 'fp32' else 'bf16',
                    STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=arm, STRATA_PREFILL_NLL=str(r / 'nll.bin'),
                    STRATA_PREFILL_LOGITS=str(r / 'selected.bin'),
                    STRATA_PREFILL_LOGITS_POSITIONS=','.join(map(str, positions)))
                recipe = r / 'recipe.json'
                recipe.write_text(json.dumps(dict(schema='strata-native16-mechanics-recipe-v1', arm=arm, args=args,
                    environment=current, source_commit=build['source_commit'], binary_sha256=sha(exe),
                    input_sha256=sha(inp), raw_consumer='Main mechanics comparison, review at paired closure'), indent=2) + '\n')
                effective = dict(current, STRATA_PREFILL_NLL_RECIPE_SHA256=sha(recipe))
                record['GPU_executed'] = True
                out, err = run('native-' + arm, [str(exe), *args], effective)
                text = err.read_text()
                require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', text), 'native/runtime failure')
                require(list(map(int, re.findall(r'layer-major layer (\d+) of 48: completed 2 chunks', text))) == list(range(48)), 'all48 layers/two chunks')
                banner = [l for l in text.splitlines() if l.startswith('strata prefill: layer-major candidate,')]
                require(len(banner) == 1 and f'residual_dtype={arm} ' in banner[0], 'actual dtype')
                require(('strata prefill: native16 BF16 HC residual/projection IO' in text) == (arm == 'bf16'), 'actual kernel path')
                raw = (r / 'nll.bin').read_bytes(); meta = (r / 'nll.bin.meta').read_text()
                require(raw[:8] == b'STNLLV2\n' and struct.unpack_from('<8I', raw, 8)[:6] == (2, 64, 129, 248320, 128, 0) and raw[40:64] == bytes(24) and len(raw) == 64 + 12 * 128, 'exact dense diagnostic shape')
                values = list(struct.iter_unpack('<Id', raw[64:]))
                require(all(t == tokens[i + 1] and math.isfinite(v) and v >= 0 for i, (t, v) in enumerate(values)), 'aligned finite nonnegative loss')
                require('recipe_sha256=' + sha(recipe) + '\n' in meta and 'complete=1\n' in meta and 'rows=128\n' in meta, 'native recipe/completion tie')
                losses[arm] = [v for _, v in values]
                selected = (r / 'selected.bin').read_bytes(); selected_meta = (r / 'selected.bin.meta').read_text()
                require(struct.unpack_from('=ii', selected) == (248320, 3) and len(selected) == 8 + 3 * 248320 * 4, 'three complete full-vocabulary rows')
                require(all(f'{i}\t{pos}\t{tokens[pos + 1]}\n' in selected_meta for i, pos in enumerate(positions)), 'selected target identities')
                checked = []; rows_by_arm[arm] = []
                for i, pos in enumerate(positions):
                    row = array.array('f'); row.frombytes(selected[8 + i * 248320 * 4:8 + (i + 1) * 248320 * 4])
                    require(all(math.isfinite(v) for v in row), 'every selected vocabulary logit finite')
                    maximum = max(row)
                    nll = maximum - row[tokens[pos + 1]] + math.log(math.fsum(math.exp(float(v) - maximum) for v in row))
                    require(abs(nll - losses[arm][pos]) <= 1.e-10, 'independent FP64 full-vocabulary NLL oracle')
                    checked.append(dict(position=pos, target=tokens[pos + 1], NLL=nll, top1=max(range(len(row)), key=row.__getitem__)))
                    rows_by_arm[arm].append(row)
                record['cases'][arm] = dict(recipe_sha256=sha(recipe), environment=effective, stderr_sha256=sha(err),
                    placement=banner[0], nll_sha256=sha(r / 'nll.bin'), nll_metadata=meta,
                    selected_sha256=sha(r / 'selected.bin'), selected_meta=selected_meta,
                    all128_target_NLLs=[dict(target=t, loss=v) for t, v in values], independent_selected_checks=checked)
                persist()
            delta = [b - a for a, b in zip(losses['fp32'], losses['bf16'])]
            record['difference'] = dict(mean_NLL={k: statistics.mean(v) for k, v in losses.items()},
                mean_delta=statistics.mean(delta), mean_absolute_delta=statistics.mean(map(abs, delta)), max_absolute_delta=max(map(abs, delta)),
                selected_max_absolute_logit_delta=[max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(rows_by_arm['fp32'], rows_by_arm['bf16'])],
                interpretation='Synthetic numerical mechanics only; no candidate quality acceptance or output-text equality criterion')
            require(sha(profile) == record['external_expert_profile_sha256'], 'external expert profile unchanged')
            require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'frozen source')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    out, _ = run('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], hostenv, wall=60, cpu=120, rss=2 << 30)
                    rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))], boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=record['passed'], active=record['active'], difference=record['difference'])), flush=True)


if __name__ == '__main__':
    main()
