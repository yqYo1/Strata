#!/usr/bin/env python3
"""Bounded Main-owned two-batch MTP upload check, with validation/progress logs."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
BUILD = B / 'main-primary-draft-upload-fix-build-root-v1'
O = B / 'main-primary-draft-upload-fix-gpu-probe-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    build = json.loads((BUILD / 'record.json').read_text())
    assert build['passed'] and build['complete'] and not build['active']
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists() and not (DEVICE / 'devcoredump').exists()
        assert (DEVICE / 'driver').resolve().name == 'xe'
        assert all(sha(P / n) == h for n, h in build['source_pins'].items())
        engine = Path(build['build_path']) / 'strata'
        assert sha(engine) == build['binaries']['strata']
        O.mkdir()
        old = json.loads(gzip.decompress((D / 'receipts/main-primary-long-generation-suite-root-v4-evidence-v2/record.json.gz').read_bytes()))
        token_path = O / 'tokens.txt'
        token_path.write_text(','.join(str(1000 + i * 7 % 512) for i in range(4097)) + '\n')
        args = old['engine_args'] + ['--tokens-file', str(token_path), '--max-new', '8']
        env = dict(old['environment'])
        for key in ['STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT']:
            env.pop(key, None)
        env.update(XDG_CACHE_HOME=str(O / 'cache'), STRATA_TRACE='1', STRATA_DRAFT_TIMING='1',
            UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
            ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
            ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warning',
            ZEL_LOADER_LOG_CONSOLE='1', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0')
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        record = dict(active=True, passed=False, complete=False, commands=[], environment=env,
            engine_args=args, source_pins=build['source_pins'], binary_sha256=sha(engine),
            build_receipt_sha256=sha(BUILD / 'record.json'), recipe_sha256=sha(Path(__file__)),
            tokens_sha256=sha(token_path), real_input_tokens=4097, batched_input_tokens=4096,
            max_context=262144, max_new=8, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            model_identity=old['model_identity'], pack_metadata_sha256=old['pack_metadata_sha256'],
            MTP_metadata_sha256=old['MTP_metadata_sha256'], performance_qualified=False,
            quality_qualified=False, physical_262144_qualified=False,
            scope='Synthetic-ID two-batch draft source-lifetime smoke test. Validation/progress/timing logging; no speed or text-quality comparison; no deliberate GPU fault.')
        for group in ['pack_metadata_sha256', 'MTP_metadata_sha256']:
            assert all(sha(Path(n)) == h for n, h in record[group].items())
        for name, identity in record['model_identity'].items():
            info = Path(name).stat()
            assert info.st_size == identity['bytes'] and info.st_mtime_ns == identity['mtime_ns']
        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        def utility(label, args):
            entry, out, err = owner.run(label, args, build['environment'], P, wall=30,
                text_cap=64 << 20, total_cap=1 << 30)
            assert module.closed(entry) and entry['exit_code'] == 0
            return out.read_text()
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])).group(1)
            record['journal_before_cursor'] = cursor
            entry, out, err = owner.run('two-batch-MTP', [str(engine)] + args, env, P,
                wall=900, cpu=900 * len(os.sched_getaffinity(0)), text_cap=64 << 20,
                rss_cap=88 << 30, total_cap=1 << 30, file_cap=64 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (entry.get('error'), err.read_text()[-3000:])
            text = err.read_text()
            drafts = re.findall(r'strata draft kv: (\d+) cells from (\d+) \(first needed (\d+)\), batch (\d+):.*', text)
            assert len(drafts) == 1 and tuple(map(int, drafts[0])) == (4096, 0, 0, 2304), drafts
            record['observed_draft_batches'] = [2304, 1792]
            record['draft_timing_lines'] = [line for line in text.splitlines() if line.startswith('strata draft kv:')]
            record['diagnostic_excerpt'] = [line for line in text.splitlines() if
                line.startswith(('strata draft kv:', 'strata generate:', 'strata prefill:', 'strata verify:'))]
            assert 'Strata SYCL cleanup:' not in text and 'aborting with upload storage live' not in text
            assert all(sha(P / n) == h for n, h in build['source_pins'].items())
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor:
                    raw = utility('journal-after', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(x) for x in raw.splitlines() if x.strip()]
                    record['fresh_faults'] = [x for x in entries if FAULT.search(str(x.get('MESSAGE', '')))]
                    record['driver_dump_exists'] = (DEVICE / 'devcoredump').exists()
                    record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False, complete=False)
            finally:
                record['active'] = False
                persist()
                (D / 'draft-upload-gpu-probe-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        assert record['passed'] and record['complete']
        print(json.dumps(dict(passed=True, batches=record['observed_draft_batches'], fresh_faults=len(record['fresh_faults']))), flush=True)


if __name__ == '__main__':
    main()
