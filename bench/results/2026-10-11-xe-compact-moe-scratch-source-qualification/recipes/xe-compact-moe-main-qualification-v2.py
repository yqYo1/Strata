"""Main-owned serialized CPU/build qualification; no device launch or adoption."""
import argparse
import fcntl
import importlib.util
import json
import shutil
import time
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
PARENT = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
WC = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
LLVM = PARENT / '.tools/intel-llvm-v711/install'
GGML = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011')
OWNER = B / 'xe-llvm-v711-source-v5/direct_owner.py'
REVIEW = B / 'xe-compact-moe-applied-main-review-v2.json'
spec = importlib.util.spec_from_file_location('compact_owner', OWNER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner identity')
ENV = dict(PATH=str(LLVM / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8',
           LC_ALL='C.UTF-8', LD_LIBRARY_PATH=str(LLVM / 'lib'))
CPU = ['expert_profile_version_test', 'expert_profile_save_test', 'prefill_publication_test',
       'prefill_compact_moe_test', 'commit_transaction_test', 'full_context_probe_test',
       'mtp_completion_test', 'context_bounds_test', 'pool_tasks_test', 'pool_stress',
       'iq_avx2_parity', 'q8k_quant_parity', 'q2_bitplane_parity', 'expert_multi_test']
DEVICE_TARGETS = ['strata-device', 'bf16_gemv_parity', 'kv_q8_parity', 'kv_q4_parity', 'prefill_dense_gemm_parity']


def strong(c):
    return m.closed(c) and c.get('session_anchored') is True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['host', 'free', 'contrib-llvm', 'free-asan', 'contrib-llvm-asan'], required=True)
    a = parser.parse_args()
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        review = json.loads(REVIEW.read_text())
        m.require(review['main_reviewed'] and review['base_revision'] == '0740b43488d812117f021ccbe4bd0e15d223f955', 'Main applied source review')
        m.require(all(m.sha(WC / p) == h for p, h in review['source_pins'].items()), 'all reviewed source pins')
        compiler_receipt = B / 'xe-llvm-v711-source-root-v5/record.json'
        tc = json.loads(compiler_receipt.read_text())
        m.require(tc['passed'] and tc['complete'] and not tc['active'], 'qualified source compiler')
        m.require(all(m.sha(Path(i['path'])) == i['sha256'] for i in tc['installed_files'].values()), 'compiler installed identities')
        out = B / ('xe-compact-moe-' + a.phase + '-qualification-root-v2')
        m.require(not out.exists(), 'immutable output')
        out.mkdir()
        r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(), phase=a.phase,
                 source_base=review['base_revision'], source_tree=str(WC), source_pins=review['source_pins'],
                 main_review_sha256=m.sha(REVIEW), compiler_receipt_sha256=m.sha(compiler_receipt),
                 controller_sha256=m.sha(__file__), owner_sha256=m.sha(OWNER), model_executed=False,
                 gpu_executed=False, adopted=False, performance_eligible=False, full262144_runtime_qualified=False,
                 raw_log_budget_bytes=64 << 20, total_wall_budget_seconds=7200, build_jobs=4,
                 scope='Source-reviewed compact MoE host fixture or full mode build plus selected CPU CTest; no GPU test')
        def save():
            (out / 'record.json').write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
        owner = m.Owner(out / 'commands', save)
        r['commands'] = owner.commands
        begin = time.monotonic()
        save()
        def cmd(label, argv, env=ENV, cwd=WC, wall=60, rss=40 << 30, expected=0):
            remaining = 7200 - (time.monotonic() - begin)
            m.require(remaining > 0, 'whole phase deadline')
            wall = min(wall, remaining)
            e, so, se = owner.run(label, argv, env, cwd, wall=wall, cpu=max(15, int(wall) * 4),
                                  file_cap=16 << 30, text_cap=64 << 20, total_cap=64 << 20, rss_cap=rss)
            m.require(strong(e) and e['exit_code'] == expected, 'normal anchored closure ' + label)
            r.setdefault('command_evidence', {})[label] = dict(stdout=so.read_text(), stderr=se.read_text())
            save()
            return so.read_text()
        try:
            m.require(shutil.disk_usage(WC).free > 40 << 30, 'fresh build headroom')
            m.require(cmd('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD']).strip() == r['source_base'], 'candidate base')
            paths = cmd('modified', ['/usr/bin/git', 'diff', '--name-only', 'HEAD']).splitlines()
            paths += cmd('new', ['/usr/bin/git', 'ls-files', '--others', '--exclude-standard']).splitlines()
            m.require(set(paths) == set(review['changed_source_pins']), 'only reviewed candidate files')
            r['compiler_version'] = cmd('compiler-version', [str(LLVM / 'bin/clang++'), '--version'])
            if a.phase == 'host':
                for name, extra in [('release', ['-O2']), ('asan-ubsan', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'])]:
                    binary = out / ('compact-moe-host-' + name)
                    cmd('compile-' + name, [str(LLVM / 'bin/clang++'), '-std=c++20', '-Wall', '-Wextra', '-Werror',
                                            '-UNDEBUG', '-I' + str(WC / 'include'), *extra,
                                            str(WC / 'tools/test_prefill_compact_moe.cpp'), '-o', str(binary)], wall=120, rss=4 << 30)
                    testenv = dict(ENV)
                    if name == 'asan-ubsan':
                        testenv.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
                    cmd('test-' + name, [str(binary)], env=testenv, wall=60, rss=4 << 30)
                    r.setdefault('binaries', {})[name] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=m.sha(binary))
            else:
                hostproof = B / 'xe-compact-moe-host-qualification-root-v2/record.json'
                hp = json.loads(hostproof.read_text())
                m.require(hp['passed'] and hp['complete'] and not hp['active'] and all(strong(c) for c in hp['commands']), 'closed release and sanitizer planner fixtures')
                m.require(hp['source_pins'] == r['source_pins'], 'same planner-qualified source')
                r['host_fixture_receipt_sha256'] = m.sha(hostproof)
                asan = a.phase.endswith('-asan')
                mode = a.phase.removesuffix('-asan')
                if asan:
                    rp = B / ('xe-compact-moe-' + mode + '-qualification-root-v2/record.json')
                    rel = json.loads(rp.read_text())
                    m.require(rel['passed'] and rel['complete'] and not rel['active'] and rel['source_pins'] == r['source_pins'], 'matched full release prerequisite')
                    runtime_proof = B / 'xe-llvm-v711-host-sanitizer-runtime-root-v2/record.json'
                    rt = json.loads(runtime_proof.read_text())
                    m.require(rt['passed'] and rt['complete'] and not rt['active'], 'matched sanitizer runtime proof')
                    m.require(all(m.sha(Path(i['path'])) == i['sha256'] for i in rt['installed_runtime_files']), 'sanitizer runtime identity')
                    r['sanitizer_runtime_receipt_sha256'] = m.sha(runtime_proof)
                    r['release_receipt_sha256'] = m.sha(rp)
                m.require(cmd('ggml-head', ['/usr/bin/git', '-C', str(GGML), 'rev-parse', 'HEAD']).strip() == '3cf03257f219afbe7334045ff7c6a06ac68c627d', 'exact ggml')
                m.require(not cmd('ggml-status', ['/usr/bin/git', '-C', str(GGML), 'status', '--porcelain', '--untracked-files=no']).strip(), 'ggml clean')
                build = WC / 'build' / a.phase
                m.require(not build.exists(), 'new candidate build directory, old reference untouched')
                options = ['-DCMAKE_BUILD_TYPE=' + ('RelWithDebInfo' if asan else 'Release'), '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                           '-DSTRATA_ENABLE_XE=ON', '-DSTRATA_LICENSE=' + ('free' if mode == 'free' else 'contrib'),
                           '-DSTRATA_CUDA_ARCHS=', '-DSTRATA_HIP_ARCHS=', '-DSTRATA_ONEMKL=' + ('OFF' if mode == 'free' else 'ON'),
                           '-DSTRATA_NATIVE_EXPERTS=ON', '-DSTRATA_PORTABLE=OFF', '-DSTRATA_BUILD_TESTS=ON', '-DSTRATA_WERROR=ON',
                           '-DSTRATA_GGML_DIR=' + str(GGML), '-DCMAKE_C_COMPILER=' + str(LLVM / 'bin/clang'),
                           '-DCMAKE_CXX_COMPILER=' + str(LLVM / 'bin/clang++')]
                if mode != 'free':
                    options += ['-DMKL_ROOT=/opt/intel/oneapi/mkl/2026.1']
                if asan:
                    options += ['-DSTRATA_SANITIZE=address,undefined']
                cmd('configure', ['/usr/bin/cmake', '-S', str(WC), '-B', str(build), '-G', 'Ninja', *options], wall=900)
                for f in ['CMakeCache.txt', 'compile_commands.json', 'BUILD.json']:
                    if (build / f).is_file():
                        r.setdefault('build_recipe', {})[f] = dict(path=str(build / f), bytes=(build / f).stat().st_size, sha256=m.sha(build / f))
                save()
                cmd('compile-full', ['/usr/bin/nice', '-n', '10', '/usr/bin/cmake', '--build', str(build), '--parallel', '4',
                                     '--target', 'strata', *CPU, *DEVICE_TARGETS], wall=5400, rss=80 << 30)
                r['binary'] = dict(path=str(build / 'strata'), bytes=(build / 'strata').stat().st_size, sha256=m.sha(build / 'strata'))
                r['ninja_compile_link_recipe'] = cmd('ninja-recipe', ['/usr/bin/ninja', '-C', str(build), '-t', 'commands', 'strata'])
                testenv = dict(ENV, LD_LIBRARY_PATH=str(LLVM / 'lib') + (':/opt/intel/oneapi/mkl/2026.1/lib' if mode != 'free' else ''))
                if asan:
                    testenv.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
                r['cpu_test_output'] = cmd('cpu-ctest', ['/usr/bin/ctest', '--test-dir', str(build), '--output-on-failure',
                                                       '--timeout', '180', '-R', '^(' + '|'.join(CPU) + ')$'], env=testenv, wall=900)
                r['device_targets_built_not_run'] = {n: dict(path=str(build / n), bytes=(build / n).stat().st_size, sha256=m.sha(build / n)) for n in DEVICE_TARGETS}
                if not asan:
                    for flag, value in [('--max-new', 'nan'), ('--max-new', '9223372036854775808'), ('--max-context', '262144x')]:
                        label = 'invalid-' + flag[2:] + '-' + value
                        cmd(label, [str(build / 'strata'), flag, value], env=testenv, wall=15, rss=2 << 30, expected=2)
                        m.require('needs an int64 value' in r['command_evidence'][label]['stderr'], 'argument refusal before device setup')
            r['source_stable'] = all(m.sha(WC / p) == h for p, h in r['source_pins'].items())
            m.require(r['source_stable'], 'source stable')
            m.require(all(strong(c) for c in owner.commands), 'all session anchors closed')
            r['complete'] = True
        except BaseException as exc:
            r['error'] = repr(exc)
        finally:
            r['active'] = owner.active is not None
            r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
            r['finished_utc'] = m.utc()
            save()
        print(json.dumps({k: r.get(k) for k in ['phase', 'passed', 'error', 'binary', 'cpu_test_output']}))
        return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
