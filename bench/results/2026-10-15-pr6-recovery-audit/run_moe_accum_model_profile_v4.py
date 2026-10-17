#!/usr/bin/env python3
"""One actual78K diagnostic, window8 then10, with optimized QSA and explicit capacity."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import socket
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-moe-window')
C = W / 'bench/results/2026-10-15-pr6-recovery-audit'
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('window', type=int, choices=[8, 10])
    ap.add_argument('--prefill-tokens',type=int,choices=[65536],default=65536)
    opts = ap.parse_args()
    tag = 'window' + str(opts.window) + '-tokens' + str(opts.prefill_tokens)
    root = B / ('main-moe-accum-model-profile-' + tag + '-v4')
    out = A / ('moe-accum-model-profile-' + tag + '-v4.json')
    assert not root.exists() and not out.exists()
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('window_model_owner', owner_path)
    reference_path = A / 'own-qsa-bf16-prefix-tilepack-profile-census-bf16-prefix-optimized-r0-v2.json'
    reference = json.loads(reference_path.read_text())
    build_path = A / 'moe-accum-engine-build-v2.json'
    fixture_path = A / 'moe-accum-device-smoke-v2.json'
    health_path = A / 'post-packed-stall-health-v1.json'
    gates = {str(p): json.loads(p.read_text()) for p in (build_path, fixture_path, health_path, reference_path)}
    for p, r in gates.items():
        assert r['passed'] and r['complete'] and r['ownership_closed'] and not r.get('active')
        assert r['fresh_faults'] == [] and not r.get('driver_dump_exists',False) and not r.get('driver_dump_entries',[]) and r['boot_unchanged']
        for command in r['commands']:
            assert owned.closed(command) and command['exit_code'] == 0
            for old in command['owners'] + [command['direct_identity']]:
                now = owned.identity(old['pid'])
                assert now is None or now['start_ticks'] != old['start_ticks'], (p, old)
    build = gates[str(build_path)]
    binary = build['products']['strata']
    assert sha(binary['path']) == binary['sha256']
    assert all(sha(path) == value for path, value in build['source_pins'].items())
    assert all(sha(Path(v['path'])) == v['sha256'] for v in reference['runtime_files'])
    for name, expected in reference['benchmark_source'].items():
        assert sha(P / name) == expected
    for group in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
        assert all(sha(path) == value for path, value in reference[group].items())
    for path, expected in reference['model_identity'].items():
        stat = Path(path).stat()
        assert stat.st_size == expected['bytes'] and stat.st_mtime_ns == expected['mtime_ns']
    if opts.window==10:
        pair_path=A/f'moe-accum-model-profile-window8-tokens{opts.prefill_tokens}-v4.json'
        pair=json.loads(pair_path.read_text())
        assert pair['passed'] and pair['complete'] and pair['ownership_closed'] and not pair['active']
        assert pair['fresh_faults']==[] and not pair['driver_dump_exists'] and pair['binary']==binary
    package = A / 'own-vllm-xe2-qualification-v1'
    meta = json.loads((package / 'manifest.json').read_text())
    assert sha(package / 'manifest.json') == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    helper = package / 'main_controller.py'
    assert sha(helper) == meta['package_files']['main_controller.py']['sha256']
    guard = module('window_model_guard', helper)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.no_model_owners(meta['known_model_files'])
        guard.driver_guard()
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == build['boot_id']
        root.mkdir(mode=0o700)
        env = dict(reference['environment'])
        env.update(XDG_CACHE_HOME=str(root / 'cache'),
                   STRATA_MEASUREMENT_ENGINE_START=str(root / 'engine-start.json'),
                   STRATA_MEASUREMENT_ENGINE_EXIT=str(root / 'engine-exit.json'),
                   STRATA_SYCL_PREFILL_MOE_WINDOW=str(opts.window),
                   STRATA_PREFILL_LAYER_MAJOR_R_GPU='auto', STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
                   STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-optimized',
                   STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS='0')
        assert env['STRATA_PREFILL_MOE_COMPACT_FP16'] == '1' and env['STRATA_PREFILL_MMQ'] == '0'
        assert env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and env['STRATA_SYCL_PREFILL_SCRATCH_ALIAS'] == '1'
        assert env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION'
        assert env['STRATA_TRACE'] == env['STRATA_PREFILL_LAYER_MAJOR_PROFILE'] == env['STRATA_PREFILL_CPU_WAIT_PROFILE'] == '1'
        template_path = A / 'qsa-device-batch-model-wrapper-v1/engine_wrapper.py'
        template = template_path.read_text().replace("assert env['STRATA_SYCL_PROMPT_DIRECT'] == '0'", "assert env['STRATA_SYCL_PROMPT_DIRECT'] == '1'")
        template = re.sub(r'^ENGINE = .*?$', 'ENGINE = ' + repr(binary['path']), template, flags=re.M)
        template = re.sub(r'^ENGINE_SHA256 = .*?$', 'ENGINE_SHA256 = ' + repr(binary['sha256']), template, flags=re.M)
        keys = ['STRATA_SYCL_PREFILL_NATIVE16', 'STRATA_PREFILL_LAYER_MAJOR_R_DTYPE',
                'STRATA_PREFILL_LAYER_MAJOR_R_GPU', 'STRATA_SYCL_PREFILL_NATIVE16_BACKEND',
                'STRATA_SYCL_PREFILL_QSA_MATH', 'STRATA_SYCL_PREFILL_MOE_WINDOW','STRATA_SYCL_PREFILL_COMMAND_PROFILE',
                'STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS', 'STRATA_PREFILL_LAYER_MAJOR_PROFILE',
                'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE', 'STRATA_PREFILL_CPU_WAIT_PROFILE',
                'STRATA_PREFILL_MOE_COMPACT_FP16', 'STRATA_PREFILL_MMQ', 'UR_ENABLE_LAYERS']
        anchor = "    pipe = os.stat('/proc/' + str(child.pid) + '/fd/0')"
        assert template.count(anchor) == 1
        template = template.replace(anchor, "    entry['experiment_environment'] = {k:env[k] for k in " + repr(keys) + "}\n" + anchor)
        wrapper = root / 'engine-wrapper.py'
        wrapper.write_text(template)
        wrapper.chmod(0o700)
        config = dict(reference['config'])
        args = list(config['args'])
        assert args[args.index('--prompt-cache') + 1] == '0'
        for flag in ('--max-context','--kv-resident'):args[args.index(flag)+1]='131072'
        profile = Path(args[args.index('--expert-profile') + 1])
        local = root / 'expert-profile.bin'
        shutil.copyfile(profile, local)
        args[args.index('--expert-profile') + 1] = str(local)
        config.update(exe=str(wrapper), args=args, env={}, log=str(root / 'owned/engine.stderr'))
        options = list(reference['benchmark_options'])
        options[options.index('--prefill-tokens')+1]=str(opts.prefill_tokens)
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        (root / 'plan.json').write_text(json.dumps(dict(port=port, url=f'http://127.0.0.1:{port}',
                runs=1, benchmark_options=options, bench_timeout_seconds=1500), indent=2) + '\n')
        record = dict(schema='strata-moe-accum-real-model-profile-v4', passed=False, complete=False,
                      active=False, GPU_launch_attempted=False, GPU_executed=False, commands=[],
                      window=opts.window, prefill_tokens_requested=opts.prefill_tokens, binary=binary, boot_id=build['boot_id'], environment=env,
                      recipe_sha256=sha(__file__), wrapper_sha256=sha(wrapper),
                      gates={path: sha(path) for path in gates}, build_source_pins=build['source_pins'],
                      runtime_files=reference['runtime_files'], config=config, benchmark_options=options,
                      model_identity=reference['model_identity'], pack_metadata_sha256=reference['pack_metadata_sha256'],
                      MTP_metadata_sha256=reference['MTP_metadata_sha256'], external_profile_sha256=sha(profile),
                      general_quality_qualified=False, physical262144_RESTORE_qualified=False,
                      performance_adopted=False,
                      scope='Official bench.sh minimum64K target, thinking-enabled24-token completion probe; resident128K context, allGPU residual admission,64K window8 versus80K window10 accumulator. TRACE/validation/coarse/CPU profiling, no clean speed/general quality/physical262144/RESTORE adoption.',
                      retention=dict(root=str(root), budget_bytes=1 << 30,
                                     consumer='Main first real FFN window execution and matched diagnostic',
                                     review_point='At closure: preserve request/progress/fault/comparison evidence; retire successful full protocol.'))
        owner = None

        def save():
            record['active'] = bool(owner and owner.active is not None)
            out.write_text(json.dumps(record, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        record['commands'] = owner.commands

        def run(label, argv, child_env=env, wall=60, cpu=120, rss=2 << 30):
            c, so, se = owner.run(label, argv, child_env, P, wall=wall, cpu=cpu,
                                  rss_cap=rss, text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
            assert owned.closed(c) and c['exit_code'] == 0, (label, c.get('exit_code'), str(se))
            return so.read_text(), se.read_text()

        cursor = None
        save()
        try:
            raw, _ = run('journal-before', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', raw)[1]
            record['journal_before_cursor'] = cursor
            record['GPU_launch_attempted'] = True
            record['GPU_executed'] = None
            run('prefill-only', ['/home/yayoi/.local/share/strata-sycl/venv/bin/python',
                                str(P / 'bench/results/2026-10-14-long-generation-benchmark/run_server_job.py'),
                                str(root), 'extended-bench'], wall=1800, cpu=21600, rss=88 << 30)
            result = json.loads((root / 'result.json').read_text())
            record['result'] = result
            assert result['passed'] and result['server_exit_code'] == result['real_engine_exit']['exit_code'] == 0
            assert result['real_engine_exit']['binary_sha256'] == binary['sha256']
            assert result['real_engine_exit']['experiment_environment'] == {k: env[k] for k in keys}
            suites = list((root / 'stock-results').glob('*/results.json'))
            assert len(suites) == 1
            suite = json.loads(suites[0].read_text())
            record['benchmark_result'] = suite
            cohort = suite['results']['prefill_only']
            request, = cohort['requests']
            assert suite['complete'] and not suite['results']['failures'] and cohort['complete']
            assert suite['params']['think'] and request['thinking_requested']
            assert request['ok'] and request['eligible'] and request['actual_prompt_tokens'] >= opts.prefill_tokens and request['actual_prompt_tokens']-1 > 8192 and request['stream_complete'] and request['reasoning_observed']
            assert request['actual_reused_tokens'] == 0 and request['actual_read_tokens'] == request['actual_prompt_tokens']
            if opts.window==10:
                paired=json.loads((A/f'moe-accum-model-profile-window8-tokens{opts.prefill_tokens}-v4.json').read_text())
                assert paired['passed'] and paired['complete'] and paired['ownership_closed']
                original,=paired['benchmark_result']['results']['prefill_only']['requests']
                assert request['request_sha256']==original['request_sha256'] and request['actual_prompt_tokens']==original['actual_prompt_tokens']
                assert paired['binary']==binary
            record['GPU_executed'] = True
            native = (root / 'owned/engine.stderr').read_text()
            assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed', native)
            placements = [line for line in native.splitlines() if line.startswith('strata prefill: layer-major candidate,')]
            assert len(placements) == 1 and 'chunk=8192 residual_dtype=bf16 ' in placements[0] and f"residual_gpu_rows={request['actual_prompt_tokens']-1} " in placements[0] and 'residual_host=0 ' in placements[0]
            record['placement_lines'] = placements
            assert '[native16-route] engine=bf16 backend=vllm' in native
            assert 'operand_layout=row-major' in native and 'operand_layout=dense16-vnni' not in native
            prefix='strata MoE window: ' if opts.window==2 else 'strata MoE accumulator window: '
            window_lines=[line for line in native.splitlines() if line.startswith(prefix)]
            record['window_lines']=window_lines
            record['window_configuration_lines']=[line for line in native.splitlines() if line.startswith('strata MoE window configuration: ')]
            assert len(record['window_configuration_lines'])==1 and f'effective_mode={opts.window} ' in record['window_configuration_lines'][0]
            batched=request['actual_prompt_tokens']-1
            width=min(opts.window*8192,batched)
            assert f'maximum_rows={width} mixed_fp16_bytes={width*2560*2}' in record['window_configuration_lines'][0]
            admission=[line for line in native.splitlines() if line.startswith('strata MoE accumulator admission: ')]
            assert len(admission)==1
            capacity={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',admission[0])}
            assert capacity['requested_mode']==opts.window and capacity['rows']==width and capacity['gpu_prefix_rows']==batched
            assert capacity['mixed_fp16_bytes']==width*2560*2
            assert capacity['free_gpu_bytes']>=capacity['fixed_including_headroom_bytes']+capacity['mixed_fp16_bytes']
            j=8192;w=width
            payloads=[w*2560*4,w*40,w*40,w*40,w*40,w*16,j*2560*2,j*1280*2,j*640*2,j*2560*2,j*2560*2,w*4,j*512*4,j*640*2,j*640*2,j*640*2,4]
            required=sum((v+255)//256*256+256 for v in payloads)
            assert capacity['shared_arena_bytes']>=required
            record['capacity_admission']={**capacity,'required_region_bytes':required,'line':admission[0]}
            expected=48*(batched//width+int(batched%width>8192))
            assert len(window_lines)==1 and f'windows={expected} ' in window_lines[0]
            for key, prefix in [('coarse_profile_lines', 'strata lm profile:'),
                                ('qsa_profile_lines', 'strata lm qsa profile:'),
                                ('cpu_wait_lines', 'strata prefill cpu wait:'),
                                ('cpu_host_lines', 'strata prefill cpu host:')]:
                record[key] = [line for line in native.splitlines() if line.startswith(prefix)]
                assert record[key]
            assert any('complete=1 errors=0' in line for line in record['coarse_profile_lines'])
            assert any('complete=1 errors=0' in line for line in record['qsa_profile_lines'])
            assert sha(profile) == record['external_profile_sha256']
            assert all(sha(path) == value for path, value in build['source_pins'].items())
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            try:
                assert owner.active is None
                if cursor:
                    raw, _ = run('journal-final', ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    record['fresh_faults'] = [row for row in rows if FAULT.search(str(row.get('MESSAGE', '')))]
                record['ownership_closed'] = all(owned.closed(c) for c in owner.commands)
                record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                record['driver_dump_exists'] = (DEV / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
                assert record['ownership_closed'] and record['boot_unchanged'] and not record['driver_dump_exists'] and record['fresh_faults'] == []
                guard.driver_guard()
            except BaseException as error:
                record.update(closure_error=repr(error), passed=False, complete=False)
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(window=opts.window, passed=record['passed'], complete=record['complete'],
                              error=record.get('error'), window_lines=record.get('window_lines'), receipt=str(out))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
