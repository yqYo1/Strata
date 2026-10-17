#!/usr/bin/env python3
"""Prepare/commit/run one official bench.sh native-GDN comparison; Main only."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
RESULT = A / 'native-gdn-model-current.json'
BUILD = A / 'native-gdn-engine-build.json'
BASE = A / 'qsa-softmax-measurement-inputs-v1.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    guards.require(len(sys.argv) == 3 and sys.argv[1] in ('prepare', 'run') and
                   sys.argv[2] in ('check', 'base', 'candidate'), 'use prepare|run check|base|candidate')
    action, mode = sys.argv[1:]
    root = B / ('native-gdn-model-' + mode + '-v2')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference = json.loads(BASE.read_text())
        guards.no_model_owners(list(reference['model_identity']))
        guards.driver_guard()
        if action == 'prepare':
            guards.require(not root.exists(), 'use a new committed measurement revision after closing this run')
            build = json.loads(BUILD.read_text())
            guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and not build['active'],
                           'qualified closed engine build required')
            for path, expected in reference['model_identity'].items():
                info = Path(path).stat()
                guards.require(info.st_size == expected['bytes'] and info.st_mtime_ns == expected['mtime_ns'], 'model changed')
            for group in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
                for path, expected in reference[group].items():
                    guards.require(digest(path) == expected, 'model/pack metadata changed: ' + path)
            root.mkdir(mode=0o700)
            for name in ('tmp', 'cache'):
                (root / name).mkdir()
            profile = P / 'data/expert-profile.bin'
            shutil.copyfile(profile, root / 'expert-profile.bin')
            env = dict(reference['environment'])
            for key in list(env):
                if key.startswith('UR_LOG_') or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE',
                                                      'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
                    del env[key]
            settings = json.loads(SETTINGS.read_text())
            if mode == 'check':
                env.update({k: v for k, v in settings['environment'].items() if k.startswith('UR_LOG_')})
                env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1')
            env.update(PATH='/home/yayoi/.local/share/strata-sycl/venv/bin:' + env['PATH'],
                       TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
                       STRATA_MEASUREMENT_ROOT=str(root), ZES_ENABLE_SYSMAN='0',
                       STRATA_SYCL_PREFILL_GDN_CHUNK64='0' if mode == 'base' else '1')
            config = dict(reference['config'])
            flags = list(config['args'])
            flags[flags.index('--expert-profile') + 1] = str(root / 'expert-profile.bin')
            config.update(exe=str(P / 'tools/sycl/engine_once.py'), args=flags,
                          log=str(root / 'engine.stderr'), env={})
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(config_path), '--port', str(port),
                       '--runs', '1' if mode == 'check' else '4', '--prefill-only', '--prefill-tokens',
                       '32768' if mode == 'check' else '65536', '--workload-id', 'native_gdn_20261018',
                       '--think', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
            files = subprocess.check_output(['git', 'ls-files', '-z', '--', 'serve', 'tools', 'bench.sh'], cwd=P).decode().strip('\0').split('\0')
            pins = {}
            for name in files:
                if name != 'bench.sh' and (not name.endswith('.py') or name.startswith('tools/hip/')):
                    continue
                data = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)
                guards.require((P / name).read_bytes() == data, 'uncommitted benchmark input: ' + name)
                pins[name] = hashlib.sha256(data).hexdigest()
            plan = dict(mode=mode, source_commit=revision, build_source_commit=build['source_commit'],
                        binary=build['binary'], environment=env, config=config, command=command,
                        reference=dict(path=str(BASE.relative_to(P)), sha256=digest(BASE)),
                        source_sha256=pins, config_sha256=digest(config_path), profile_sha256=digest(profile),
                        complete=False, passed=False, active=False, model_quality_qualified=False,
                        physical262144_RESTORE_qualified=False, performance_adopted=False)
            (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
            compact = {k: v for k, v in plan.items() if k != 'source_sha256'}
            compact['source_count'] = len(pins)
            compact['source_aggregate_sha256'] = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
            RESULT.write_text(json.dumps(compact, indent=2) + '\n')
            print('Prepared exact inputs. Commit native-gdn-model-current.json before run ' + mode)
            return 0
        record = json.loads(RESULT.read_text())
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P)
        guards.require(committed == RESULT.read_bytes() and record['mode'] == mode and not record['complete'],
                       'commit a fresh matching plan before launch')
        plan = json.loads((root / 'plan.json').read_text())
        guards.require(all(record.get(k) == v for k, v in plan.items() if k != 'source_sha256'),
                       'external plan differs from committed inputs')
        guards.require(hashlib.sha256(json.dumps(plan['source_sha256'], sort_keys=True).encode()).hexdigest() ==
                       record['source_aggregate_sha256'], 'source identity list changed')
        guards.require(digest(BASE) == plan['reference']['sha256'], 'reference inputs changed')
        for path, expected in reference['model_identity'].items():
            info = Path(path).stat()
            guards.require(info.st_size == expected['bytes'] and info.st_mtime_ns == expected['mtime_ns'], 'model changed')
        guards.require(digest(plan['binary']['path']) == plan['binary']['sha256'], 'binary changed')
        guards.require(digest(root / 'config.json') == plan['config_sha256'] and
                       digest(root / 'expert-profile.bin') == plan['profile_sha256'], 'generated input changed')
        for name, expected in plan['source_sha256'].items():
            guards.require(digest(P / name) == expected, 'benchmark source changed: ' + name)
        built = json.loads((B / 'main-native-gdn-engine/record.json').read_text())
        for source in [built['project'], *built['dependencies']]:
            for name, expected in source['source_sha256'].items():
                guards.require(digest(Path(source['root']) / name) == expected, 'engine source changed: ' + name)
        for path, item in json.loads(SETTINGS.read_text())['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime asset changed: ' + path)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(root / 'owned', persist)
        try:
            entry, out, err = owner.run('bench', plan['command'], plan['environment'], P,
                wall=900 if mode == 'check' else 1800, cpu=21600, rss_cap=96 << 30,
                text_cap=64 << 20, file_cap=64 << 20, total_cap=64 << 20)
            record.update(complete=supervision.closed(entry), exit_code=entry.get('exit_code'))
            results = list((root / 'bench').glob('*/results.json'))
            guards.require(len(results) == 1, 'missing official benchmark result')
            suite = json.loads(results[0].read_text())
            requests = suite['results']['prefill_only']['requests']
            fields = ('run','ok','eligible','actual_prompt_tokens','actual_read_tokens','actual_reused_tokens',
                      'generated_tokens','thinking_requested','reasoning_observed','stream_complete',
                      'request_sha256','output_sha256','reasoning_sha256','ttft_s','total_s',
                      'prompt_tps','prompt_ms','decode_tps','decode_ms','prompt_tps_source','decode_tps_source',
                      'drafts_offered','drafts_accepted','finish_reason','err','eligibility_errors')
            record['requests'] = [{k: q[k] for k in fields if k in q} for q in requests]
            end = json.loads((root / 'engine-exit.json').read_text())
            log = (root / 'engine.stderr').read_text(errors='replace')
            record['engine_exit_code'] = end['exit_code']
            record['native_identity'] = end['native']
            record['route_selection'] = [x for x in log.splitlines() if x.startswith('[gdn-chunk64]')]
            guards.require(end['binary'] == plan['binary'] and end['exit_code'] == 0, 'native engine exit')
            guards.require(len(requests) == (1 if mode == 'check' else 4) and suite['complete'] and
                           not suite['results']['failures'], 'incomplete suite')
            guards.require(all(q['ok'] and q['eligible'] and q['stream_complete'] and q['thinking_requested'] and
                               q['actual_reused_tokens'] == 0 for q in requests), 'ineligible samples')
            guards.require(bool(record['route_selection']) == (mode != 'base'), 'wrong GDN route')
            guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', log), 'runtime failure')
            record['passed'] = bool(record['complete'] and entry.get('exit_code') == 0)
        except BaseException as error:
            record['error'] = str(error)
            record['error_excerpt'] = (root / 'engine.stderr').read_text(errors='replace')[-8192:] if (root / 'engine.stderr').exists() else ''
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:
                record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record['driver_guard_error'] = str(error)
                record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'complete', 'passed', 'ownership_closed', 'error', 'route_selection')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
