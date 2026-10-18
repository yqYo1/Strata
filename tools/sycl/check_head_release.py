#!/usr/bin/env python3
"""Main-owned NativeHead lifecycle fixture linked to the current engine libraries."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'native-head-roundtrip-v2'
ENGINE = B / 'main-native-gdn-engine'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/prefill-head-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
SOURCE = P / 'bench/sycl/prefill_head_roundtrip.cpp'
MODEL = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def asset(path):
    st = path.stat()
    return dict(path=str(path), bytes=st.st_size, device=st.st_dev,
                inode=st.st_ino, mtime_ns=st.st_mtime_ns)


def main():
    guards.require(sys.argv[1:] in (['build'], ['check']), 'use check_head_release.py build|check')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        build = json.loads((ENGINE / 'record.json').read_text())
        guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and
                       not build['active'], 'closed current build required')
        for group in [build['project'], *build['dependencies']]:
            for name, want in group['source_sha256'].items():
                guards.require(digest(Path(group['root']) / name) == want, 'engine input changed: ' + name)
        guards.require(digest(build['binary']['path']) == build['binary']['sha256'], 'engine changed')
        names = [str(x.relative_to(P)) for x in (SOURCE, Path(__file__),
                 P / 'tools/sycl/measurement_owner.py', P / 'tools/sycl/measurement_guards.py', SETTINGS)]
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P).decode().strip()
        pins = {}
        for name in names:
            data = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)
            guards.require(data == (P / name).read_bytes(), 'commit fixture/controller inputs first: ' + name)
            pins[name] = hashlib.sha256(data).hexdigest()
        settings = json.loads(SETTINGS.read_text())
        for path, identity in settings['runtime_elfs'].items():
            guards.require(digest(path) == identity['sha256'], 'runtime changed: ' + path)
        compiler = build['compiler']
        guards.require(digest(compiler['path']) == compiler['sha256'], 'compiler changed')
        product = ENGINE / 'build'
        guards.require(digest(product / 'compile_commands.json') == build['compile_database_sha256'],
                       'configured engine inputs changed')
        libraries = [product / name for name in ('libstrata_prefill.a', 'libstrata_spec.a',
            'libstrata_engine.a', 'libstrata_kernels.a', 'libstrata_core.a',
            'libstrata_sycl_runtime.a', 'libstrata_kernels_cpu.a',
            'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a')]
        mkl = Path('/opt/intel/oneapi/mkl/2026.1/lib')
        libraries += [mkl / name for name in ('libmkl_sycl_blas.so', 'libmkl_intel_ilp64.so',
                                             'libmkl_sequential.so', 'libmkl_core.so')]
        linked = {str(path): digest(path) for path in libraries}
        binary = ROOT / 'head-roundtrip'
        if mode == 'build':
            guards.require(not ROOT.exists() and not RESULT.exists(), 'inspect prior head fixture before reuse')
            ROOT.mkdir(mode=0o700)
            for name in ('tmp', 'cache'):
                (ROOT / name).mkdir()
            command = [compiler['path'], '-O3', '-DNDEBUG', '-std=c++20', '-fsycl',
                '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
                '-fsycl-allow-device-image-dependencies', '-DSTRATA_ENABLE_SYCL=1',
                '-Wall', '-Wextra', '-Werror', '-I' + str(P / 'include'),
                '-I' + str(P / 'include/strata/sycl/compat'), str(SOURCE),
                *map(str, libraries), '-Wl,-rpath=' + str(mkl), '-lm', '-ldl', '-lpthread', '-o', str(binary)]
            record = dict(mode=mode, source_commit=revision, fixture_source_sha256=pins,
                build_source_commit=build['source_commit'], engine=build['binary'],
                compiler=compiler, linked_inputs_sha256=linked, complete=False, passed=False,
                active=False, GPU_executed=False, checks=[],
                consumer='Qualify existing output-head detach/restore before a long-prefill capacity trial; retire after the decision.',
                limits='Two exact packed-weight/logit round trips, not shared MTP/verifier model restore, quality, context or speed qualification.')
        else:
            guards.require(RESULT.read_bytes() == subprocess.check_output(
                ['git', 'show', 'HEAD:' + str(RESULT.relative_to(P))], cwd=P), 'commit build gate first')
            record = json.loads((ROOT / 'record.json').read_text())
            guards.require(record['mode'] == 'build' and record['passed'] and record['ownership_closed'] and
                           not record['active'], 'passed closed build required')
            guards.require(record['fixture_source_sha256'] == pins and
                           record['linked_inputs_sha256'] == linked and record['engine'] == build['binary'],
                           'fixture/library/source identity changed')
            guards.require(digest(binary) == record['binary']['sha256'], 'fixture executable changed')
            shards = [MODEL / ('Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000' + str(i) + '-of-00002.gguf')
                      for i in (1, 2)]
            command = [str(binary), *map(str, shards)]
            record.update(mode=mode, complete=False, passed=False, GPU_executed=False,
                          model_assets=[asset(p) for p in shards])
            record['asset_identity_limits'] = 'File identities plus actual loaded Q6_K geometry and exact packed-byte comparisons; not a hash of every GGUF tensor.'
        env = dict(build['environment'] if mode == 'build' else settings['environment'])
        env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
        if mode == 'check':
            env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                       UR_LOG_TRACING='level:info;flush:info;output:stderr')
        record['environment'] = env
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        guards.driver_guard()
        cursor = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = cursor.split('-- cursor: ')[-1].strip()
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            if owner:
                record['owned_commands'] = owner.commands
            (ROOT / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = supervision.Owner(ROOT / ('owned-' + mode), persist)
        persist()
        try:
            entry, out, err = owner.run(mode, command, env, P, wall=300, cpu=1200,
                rss_cap=12 << 30, file_cap=256 << 20, text_cap=64 << 20, total_cap=1 << 30)
            guards.require(supervision.closed(entry) and entry.get('exit_code') == 0,
                           mode + ' failed: ' + (err.read_text(errors='replace') + out.read_text(errors='replace'))[-6000:])
            if mode == 'build':
                record['binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=digest(binary))
            else:
                log = out.read_text()
                lines = [x for x in log.splitlines() if x.startswith('head_cycle ')]
                guards.require(len(lines) == 2 and 'head_component PASS cycles=2 ' in log,
                               'incomplete head lifecycle fixture')
                record['cycles'] = [dict(cycle=int(x.split()[1]), **{k: int(v) for k, v in
                    (field.split('=', 1) for field in x.split()[2:])}) for x in lines]
                guards.require(all(c['cycle'] == i and c['packed_bytes'] == 521472000 and
                    c['weights_equal'] == c['logits_equal'] == 1 for i, c in enumerate(record['cycles'])),
                    'head lifecycle content/geometry mismatch')
                guards.require(all(asset(Path(x['path'])) == x for x in record['model_assets']), 'model asset changed')
                record['GPU_executed'] = True
            guards.require(all(digest(P / name) == h for name, h in pins.items()), 'fixture source changed')
            guards.require(all(digest(name) == h for name, h in linked.items()), 'linked library changed')
            record.update(complete=True, passed=True)
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [line for line in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', line)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:
                record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record['driver_error'] = str(error)
                record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and
                                    record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            compact = {k: v for k, v in record.items() if k not in ('owned_commands', 'linked_inputs_sha256')}
            compact['linked_inputs_count'] = len(linked)
            compact['linked_inputs_aggregate_sha256'] = hashlib.sha256(json.dumps(linked, sort_keys=True).encode()).hexdigest()
            RESULT.write_text(json.dumps(compact, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'complete', 'passed', 'ownership_closed', 'error', 'cycles')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
