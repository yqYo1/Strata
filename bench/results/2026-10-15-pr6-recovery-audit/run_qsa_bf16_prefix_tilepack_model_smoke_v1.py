#!/usr/bin/env python3
"""Opt-in SGLang-design BF16 attention mechanics, synthetic129/all48. No speed or quality acceptance."""
import array
import argparse
import fcntl
import importlib.util
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
    build_path=A/'own-qsa-bf16-prefix-tilepack-engine-build-v1.json'
    component_path=A/'own-qsa-bf16-prefix-tilepack-engine-component-v1.json'
    opts=argparse.Namespace(build=build_path,build_sha256=sha(build_path),prior=component_path,prior_sha256=sha(component_path))
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    package = A/'own-vllm-xe2-qualification-v1'
    manifest = json.loads((package/'manifest.json').read_text())
    require(sha(package/'manifest.json')=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58','reviewed manifest')
    helper = package/'main_controller.py'
    require(sha(helper)==manifest['package_files']['main_controller.py']['sha256'],'reviewed helper')
    spec = importlib.util.spec_from_file_location('model_vllm_helpers',helper)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(sha(opts.build)==opts.build_sha256,'new build receipt hash')
        build=json.loads(opts.build.read_text())
        require(build['passed'] and build['complete'] and build['ownership_closed'] and not build['active'] and not build['fresh_faults'],'complete normal new build')
        # This build schema pins boot in its controller/checked journal cursor,
        # rather than duplicating a top-level boot_id. Derive it explicitly.
        require('b='+boot_id().replace('-','')+';' in build['output']['journal-before']['stdout'],'build journal boot identity')
        for command in build['commands']:
            require(module.closed(command) and command['exit_code']==0,'build normal ownership')
            for identity in command.get('owners',[])+[command['direct_identity']]:
                now=module.identity(identity['pid']);require(now is None or now['start_ticks']!=identity['start_ticks'],'build owner still present')
        build_gate=dict(path=str(opts.build),sha256=opts.build_sha256,boot_id=boot_id(),boot_identity_source='checked build source constant and journal-before cursor',ownership_closed=True)

        require(sha(opts.prior)==opts.prior_sha256,'component receipt hash')
        component=json.loads(opts.prior.read_text())
        require(component['passed'] and component['complete'] and component['ownership_closed'] and not component['active'] and component['boot_id']==boot_id() and component['boot_unchanged'] and not component['fresh_faults'] and component['driver_dump_entries']==[],'normal component closure/boot/fault/dump state')
        for command in component['commands']:
            require(module.closed(command) and command['exit_code']==0,'component normal ownership')
            for identity in command.get('owners',[])+[command['direct_identity']]:
                now=module.identity(identity['pid']);require(now is None or now['start_ticks']!=identity['start_ticks'],'component owner still present')
        component_gate=dict(path=str(opts.prior),sha256=opts.prior_sha256,boot_id=boot_id(),ownership_closed=True,driver_dump_entries=[])

        require(component['mode']=='component' and component['build']['sha256']==opts.build_sha256,'same new engine-linked prefix component gate')
        build['source_pins']={str(Path(n).relative_to(P)):v['sha256'] for n,v in build['frozen_files'].items() if Path(n).is_relative_to(P)}
        build['pins']=build['frozen_files']
        build['binary_sha256']=build['binary']['sha256'];build['binary']=build['binary']['path']
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
        root = B/'main-qsa-bf16-prefix-tilepack-model-smoke-v1'
        target = A/'own-qsa-bf16-prefix-tilepack-model-smoke-v1.json'
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
        (root/'cache').mkdir()
        env.update(XDG_CACHE_HOME=str(root/'cache'),STRATA_TRACE='1',STRATA_ROPE_TABLE='1',ZES_ENABLE_SYSMAN='0')
        require(env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'qualified single-queue production route only')
        record = dict(schema='strata-qsa-bf16-prefix-tilepack-model-smoke-v1',passed=False, complete=False, active=True, GPU_executed=False, commands=[], cases={},
            source_commit=build['source_commit'], source_pins=build['source_pins'], binary_sha256=sha(exe),
            model_identity=old['model_identity'], pack_metadata_sha256=metadata['pack_metadata_sha256'],
            MTP_metadata_sha256=metadata['MTP_metadata_sha256'], runtime_files=old['runtime_files'],
            tokens=tokens, positions=positions, input_sha256=sha(inp), args=args, environment=env,
            external_expert_profile_sha256=sha(profile), local_profile_initial_sha256=sha(local_profile),
            recipe_sha256=sha(Path(__file__)), boot_id=boot_id(), raw_tensor_budget_bytes=16 << 20,
            build_gate=build_gate,component_gate=component_gate,original_full_regression_passed=False,
            regression_limit='Single-queue route only. Original separate-queue regression failure remains failed. New engine-linked prefix component passed; unchanged other operators retain prior scoped evidence, not a new complete engine regression.',
            scope='Synthetic129/MTP4 all48/two64chunks, GPU64+host64 residuals; same new engine and BF16/vLLM projections, row-major score33 versus dense16/VNNI score33 local operands; same numerical/update order/payload; both transient active-prefix BF16 packing and explicit4byte-aligned local matrix operands. ROPE_TABLE=1 residentINT8 4096 for all arms. All128 NLL finite and3 independent fullvocabulary NLL checks. Not speed/general quality/fullcontext/RESTORE.',
            performance_qualified=False, model_quality_qualified=False, physical_262144_qualified=False, restore_qualified=False)
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands; persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, argv, current, wall=900, cpu=18000, rss=88 << 30):
            require(sha(Path(__file__))==record['recipe_sha256'] and boot_id()==record['boot_id'],'controller/boot stable')
            require(sha(opts.build)==opts.build_sha256 and sha(opts.prior)==opts.prior_sha256,'receipt stable')
            require(all(sha(P/n)==value for n,value in build['source_pins'].items()),'source stable')
            require(sha(exe)==build['binary_sha256'],'binary stable')
            for path,value in build['pins'].items(): h.pinned(path,value)
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
            for arm in ('bf16-prefix-optimized','bf16-prefix-tilepack'):
                h.no_model_owners(manifest['known_model_files']); h.driver_guard()
                dtype='bf16'
                r = root / arm; r.mkdir()
                # Identical profile input for both processes, even if an engine
                # later writes a profile. External user's profile stays read-only.
                shutil.copyfile(profile, local_profile)
                current = dict(env, STRATA_SYCL_PREFILL_NATIVE16='bf16',STRATA_SYCL_PREFILL_QSA_MATH=arm,
                    STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=dtype,
                    STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
                    STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
                    STRATA_SYCL_PREFILL_NATIVE16_QSA='1',
                    STRATA_PREFILL_NLL=str(r / 'nll.bin'),
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
                require(len(banner) == 1 and f'residual_dtype={dtype} ' in banner[0], 'actual dtype')
                require(('strata prefill: native16 BF16 HC residual/projection IO' in text) == True, 'actual kernel path')
                require('[native16-route] engine=bf16 backend=vllm' in text,'public Gemm backend selection')
                require(('QSA BF16 operands, FP32 online state/accumulation;' in text)==(arm!='existing'),'actual arithmetic dispatch')
                if arm.startswith('bf16-prefix'):require('KV=active-logical-prefix transient_bytes=' in text,'prefix allocation dispatch')
                require(('operand_layout=dense16-vnni' if arm=='bf16-prefix-tilepack' else 'operand_layout=row-major') in text,'actual operand layout')
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
                out_faults,_=run('journal-after-'+arm,['/usr/bin/journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'],hostenv,wall=60,cpu=120,rss=2<<30)
                rows=[json.loads(v) for v in out_faults.read_text().splitlines() if v.strip()]
                found=[v for v in rows if FAULT.search(str(v.get('MESSAGE','')))]
                record.setdefault('arm_fault_checks',[]).append(dict(arm=arm,faults=found,entries=len(rows)))
                require(not found and boot_id()==record['boot_id'],'fresh GPU fault/boot drift')
                h.driver_guard(); persist()
            record['difference']={}
            for candidate in ('bf16-prefix-tilepack',):
                delta = [b-a for a,b in zip(losses['bf16-prefix-optimized'],losses[candidate])]
                record['difference'][candidate]=dict(mean_NLL={k:statistics.mean(v) for k,v in losses.items()},
                    mean_delta=statistics.mean(delta),mean_absolute_delta=statistics.mean(map(abs,delta)),max_absolute_delta=max(map(abs,delta)),
                    selected_max_absolute_logit_delta=[max(abs(a-b) for a,b in zip(x,y)) for x,y in zip(rows_by_arm['bf16-prefix-optimized'],rows_by_arm[candidate])],
                    interpretation='Synthetic numerical mechanics only; no candidate quality acceptance or output-text equality criterion')
            require(sha(profile) == record['external_expert_profile_sha256'], 'external expert profile unchanged')
            require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'frozen source')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            try:
                if cursor and owner.active is None:
                    out, _ = run('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], hostenv, wall=60, cpu=120, rss=2 << 30)
                    rows = [json.loads(s) for s in out.read_text().splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))], boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc(),ownership_closed=owner.active is None and all(module.closed(c) for c in owner.commands))
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=record['passed'],active=record['active'],difference=record.get('difference'),receipt=str(target),sha256=sha(target))),flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
