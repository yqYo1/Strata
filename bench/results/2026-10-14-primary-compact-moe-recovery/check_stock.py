#!/usr/bin/env python3
"""Quiet, unchanged Niko bench.sh comparison of one current compact arm."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).parent
D = P / 'bench/results/2026-10-14-primary-model-baseline'
Q = P / 'bench/results/2026-10-14-pre-Xe-replay'
U = Path('/home/yayoi/ghq/github.com/Niko1221/Strata/.worktree/bench-upstream-v0143-pristine-20261012')
BUILD = B / 'main-primary-compact-moe-build-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    value = sys.argv[1]
    assert value in ['0', '1']
    O = B / ('main-primary-stock-compact' + value + '-root-v1')
    build = json.loads((BUILD / 'record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active']
    for mode in ['compact4096', 'compact-tail64']:
        r = json.loads((B / ('main-primary-' + mode + '-pair-root-v2') / 'record.json').read_text())
        assert not r['active'] and r['passed'] and r['complete']
        assert r['binary_sha256'] == build['binaries']['strata']
        assert r['numerical_comparison']['logits_bitwise_equal'] and r['numerical_comparison']['residual_bitwise_equal']
        assert not r['fresh_faults'] and not r['driver_dump_exists'] and r['boot_unchanged']
    prior_path = Q / 'receipts/main-primary-stock-fp16-root-v1/record.json.gz'
    item = next(x for x in json.loads((Q / 'receipts-index.json').read_text())
                if x['path'] == str(prior_path.relative_to(Q)))
    raw = gzip.decompress(prior_path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == item['decoded_sha256'] and sha(prior_path) == item['archive_sha256']
    prior = json.loads(raw)
    assert prior['passed'] and not prior['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        assert (DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists()
        for name, pin in build['source_pins'].items():
            assert sha(P / name) == pin
        engine = Path(build['build_path']) / 'strata'
        assert sha(engine) == build['binaries']['strata']
        subprocess.run(['git', '-C', str(U), 'diff', '--exit-code', 'HEAD', '--', 'bench.sh',
            'tools/strata_bench.py', 'tools/ab_engine.py', 'tools/agent_turns_bench.py', 'tools/serve_load.py'], check=True)
        for name, pin in prior['standard_client_source'].items():
            assert sha(U / name) == pin
        O.mkdir()
        env = dict(prior['environment'])
        env.update(XDG_CACHE_HOME=str(O / 'cache'), STRATA_PREFILL_MOE_COMPACT_FP16=value,
            STRATA_MEASUREMENT_ENGINE_START=str(O / 'engine-start.json'),
            STRATA_MEASUREMENT_ENGINE_EXIT=str(O / 'engine-exit.json'))
        assert not any(k in env for k in ['UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER',
            'STRATA_TRACE', 'STRATA_PREFILL_TIMING', 'STRATA_DECODE_TIMING', 'LD_PRELOAD', 'STRATA_DBG_NAN'])
        assert env['STRATA_PREFILL_MMQ'] == '0' and env['STRATA_PREFILL_RING'] == '8'
        args = prior['engine_args']
        config = dict(exe=str(D / 'engine_wrapper_v3.py'), args=args, cwd=str(P),
            tokenizer=str(Path(args[args.index('--pack') + 1]) / 'tokenizer'),
            model_name='qwen3.8-flash-next-iq3_s-primary', aliases=['strata'], host='127.0.0.1',
            log=str(O / 'owned/engine.stderr'), engine_silence_s=480, parallel=1, open_browser=False)
        (O / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (O / 'plan.json').write_text(json.dumps(dict(port=18096, url='http://127.0.0.1:18096',
            expected_long_prompt_tokens=32049, bench_timeout_seconds=6600), indent=2) + '\n')
        job = D / 'run_server_job_v4.py'
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
            compact=value, engine_args=args, source_pins=build['source_pins'], binary_sha256=sha(engine),
            build_receipt_sha256=sha(BUILD / 'record.json'), source_revision=subprocess.check_output(
                ['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
            source_diff_sha256=build['source_diff_sha256'], recipes={str(f.relative_to(P)): sha(f)
                for f in [Path(__file__), job, D / 'engine_wrapper_v3.py']},
            standard_client_source=prior['standard_client_source'], pack_metadata_sha256=prior['pack_metadata_sha256'],
            MTP_metadata_sha256=prior['MTP_metadata_sha256'], model_identity=prior['model_identity'],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            adopted=False, GPU_reset_or_service_change=False,
            scope='Same new native, quiet current FP16 route, compact0/1 separate owned processes; unchanged Niko stock suite long3. Short numerical/tail gates qualified; full input and RESTORE adoption remain open.')
        for name, pin in record['pack_metadata_sha256'].items():
            assert sha(Path(name)) == pin
        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        def utility(label, argv):
            entry, out, err = owner.run(label, argv, build['environment'], P,
                wall=30, text_cap=64 << 20, total_cap=1 << 30)
            assert module.closed(entry) and entry['exit_code'] == 0
            return out.read_text()
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])).group(1)
            record['journal_before_cursor'] = cursor
            entry, out, err = owner.run('stock-compact' + value,
                ['/home/yayoi/.local/share/strata-sycl/venv/bin/python', str(job), str(O), 'stock-bench'],
                env, P, wall=7200, cpu=7200, text_cap=64 << 20, rss_cap=88 << 30,
                total_cap=1 << 30, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), err.read_text()[-2500:])
            result = json.loads((O / 'result.json').read_text())
            assert result['passed'] and result['real_engine_exit']['exit_code'] == 0
            assert result['real_engine_exit']['binary_sha256'] == record['binary_sha256']
            paths = list((O / 'stock-results').glob('*/results.json'))
            assert len(paths) == 1
            stock = json.loads(paths[0].read_text())
            assert stock['params']['runs'] == 3 and stock['params']['long']
            assert not stock['results']['failures'] and not stock['results']['skipped']
            rows = stock['results']['prompts']['32K']['requests']
            assert len(rows) == 3 and {x['run'] for x in rows} == {0, 1, 2}
            for row in rows:
                assert row['ok'] and row['prompt_tokens'] == row['read_tokens'] == 32049
                assert row['generated_tokens'] == 24 and row['reused_tokens'] == 0
                assert all(math.isfinite(row[k]) and row[k] > 0 for k in ['prompt_tps', 'decode_tps', 'prompt_ms', 'decode_ms'])
            text = (O / 'owned/engine.stderr').read_text()
            effective = re.findall(r'FP16 MoE compact requested=(\d+) effective=(\d+), chunk=(\d+), scratch rows=(\d+), accounted=(\d+), shared region=(\d+) bytes', text)
            assert effective and all(x[1] == value and int(x[2]) == 4096 for x in effective), effective
            record.update(result=result, stock_result=stock, stock_result_sha256=sha(paths[0]),
                effective_allocation=effective, long_phase_samples=rows,
                prefill_median_tps=statistics.median(x['prompt_tps'] for x in rows),
                decode_median_tps=statistics.median(x['decode_tps'] for x in rows))
            for name, pin in build['source_pins'].items():
                assert sha(P / name) == pin
            assert sha(engine) == record['binary_sha256']
            record.update(passed=True, complete=True, performance_qualified=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    text = utility('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(x) for x in text.splitlines() if x.strip()]
                    record['fresh_faults'] = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False
                persist()
        assert record['passed'] and record['complete']
        print(json.dumps(dict(passed=True, compact=value, prefill_median_tps=record['prefill_median_tps'],
            decode_median_tps=record['decode_median_tps'], fresh_faults=0)), flush=True)


if __name__ == '__main__':
    main()
