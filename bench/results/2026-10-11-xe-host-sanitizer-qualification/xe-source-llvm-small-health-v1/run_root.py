import fcntl, importlib.util, json, re, time, types
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE = Path(__file__).resolve().parent
WX = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
LLVM = WX / '.tools/intel-llvm-v711/install'
OWNER = B / 'xe-llvm-v711-source-v3/direct_owner.py'
HELPER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
DUMP = Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
spec = importlib.util.spec_from_file_location('owner', OWNER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
BASE = dict(PATH=str(LLVM / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH=str(LLVM / 'lib'))


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compiler_proof = B / 'xe-llvm-v711-source-root-v5/record.json'
        toolchain = json.loads(compiler_proof.read_text())
        build_proof = B / 'xe-e32-free-full-build-root-v5/record.json'
        build = json.loads(build_proof.read_text())
        m.require(all(d['passed'] and not d['active'] and d['complete'] for d in (toolchain, build)), 'closed compiler and free build')
        asan_proof=B/'xe-e32-free-asan-full-build-root-v4/record.json';asan=json.loads(asan_proof.read_text())
        m.require(asan['passed'] and not asan['active'] and asan['complete'],'closed full host sanitizer build/tests')
        lint_proof=B/'xe-e32-lints-root-v1/record.json';lint=json.loads(lint_proof.read_text())
        m.require(lint['passed'] and not lint['active'] and lint['complete'],'closed mandatory source lints')
        m.require(all(m.sha(WX/f)==h for f,h in build['source_pins'].items()),'same full-build source bytes')
        m.require(all(m.sha(WX/f)==h for f,h in lint['source_pins'].items()),'same lint-qualified source bytes')
        cap_proof=B/'xe-free-host-capabilities-root-v3/record.json';cap=json.loads(cap_proof.read_text())
        m.require(cap['passed'] and not cap['active'] and cap['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'same boot closed source runtime query')
        for f in toolchain['installed_files'].values():
            m.require(m.sha(Path(f['path'])) == f['sha256'], 'compiler identity')
        out = B / 'xe-source-llvm-small-health-root-v1'
        m.require(not out.exists(), 'immutable new output')
        out.mkdir()
        r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 controller_sha256=m.sha(__file__), source_sha256=m.sha(HERE / 'health.cpp'), owner_sha256=m.sha(OWNER), mandatory_lint_receipt_sha256=m.sha(lint_proof),
                 compiler_receipt_sha256=m.sha(compiler_proof), engine_build_receipt_sha256=m.sha(build_proof),
                 mode='free', queue_created=None, allocation_created=None, kernel_launched=None, model_launched=False,full_host_asan_receipt_sha256=m.sha(asan_proof),capability_receipt_sha256=m.sha(cap_proof),
                 reset_executed=False, recovery_executed=False, retry_count=0, scope='First source-built-runtime actual integer kernel/copy health: 64KiB and16384 exact words; no model or concurrent host-USM')
        def save():
            (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
        owner = m.Owner(out / 'commands', save)
        r['commands'] = owner.commands
        helper = types.ModuleType('read_only_helper')
        m.require(m.sha(HELPER) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'health helper identity')
        exec(compile(HELPER.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(HELPER), 'exec'), helper.__dict__)
        begin = time.monotonic()
        cursor = None
        save()
        def command(label, args, env=BASE, wall=20):
            m.require(time.monotonic() - begin < 240, 'whole query deadline')
            e, so, se = owner.run(label, args, env, HERE, wall=wall, cpu=int(wall), text_cap=16 << 20, total_cap=32 << 20, rss_cap=8 << 30)
            m.require(m.closed(e) and e['exit_code'] == 0, 'normal closed ' + label)
            return so.read_text()
        try:
            m.require(not DUMP.exists(), 'no preexisting dump')
            adapters = sorted((LLVM / 'lib').glob('libur_adapter_level_zero*.so*'))
            m.require(bool(adapters), 'source-built Level Zero adapter installed')
            preferred = [p for p in adapters if 'v2' in p.name]
            adapter = (preferred or adapters)[0].resolve()
            runtime = dict(BASE, ONEAPI_DEVICE_SELECTOR='level_zero:gpu', SYCL_CACHE_PERSISTENT='0',
                           UR_ADAPTERS_FORCE_LOAD=str(adapter), UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1', EnableDirectSubmission='0')
            env = helper.diagnostic_environment(runtime)
            r['environment'] = env
            r['adapter'] = dict(path=str(adapter), sha256=m.sha(adapter))
            boot = command('kernel-before', ['/usr/bin/journalctl', '-k', '-b', '--no-pager', '-o', 'json'])
            rows = [json.loads(s) for s in boot.splitlines() if s.startswith('{')]
            rows = [v for v in rows if '0000:05:00.0' in v.get('MESSAGE', '') or re.search(r'\bxe\b', v.get('MESSAGE', ''))]
            r['kernel_preflight_entries'] = rows
            m.require(not any(helper.FAULT.search(v['MESSAGE']) for v in rows), 'no boot GPU fault')
            compiler = LLVM / 'bin/clang++'
            r['compiler_version'] = command('compiler-version', [str(compiler), '--version'])
            binary = out / 'health'
            command('compile', [str(compiler), '-std=c++20', '-fsycl', '-O2', '-Wall', '-Wextra', '-Werror', str(HERE / 'health.cpp'), '-lze_loader', '-o', str(binary)], wall=120)
            r['binary'] = dict(path=str(binary), sha256=m.sha(binary), bytes=binary.stat().st_size)
            text = command('kernel-cursor', ['/usr/bin/journalctl', '-k', '-n', '0', '--show-cursor', '--no-pager'])
            match = re.search(r'^-- cursor: (.+)$', text, re.M)
            m.require(match is not None, 'fresh journal cursor')
            cursor = match[1]
            r['kernel_cursor_before'] = cursor
            save()
            r['gpu_submission_attempted']=True;save()
            text = command('integer-health', [str(binary), '0000:05:00.0'], env=env, wall=30)
            lines = text.splitlines()
            r['query_output'] = lines
            expected=['STATUS 0000:05:00.0 0','QUEUE 0000:05:00.0','H2D 65536 complete','KERNEL 16384 complete','D2H 65536 complete','PASS 0000:05:00.0: 1 round, 16384 exact words']
            m.require([v for v in lines if not v.startswith('LIBRARY ')]==expected,'complete exact integer health protocol')
            r.update(queue_created=True,allocation_created=True,kernel_launched=True,gpu_executed=True,exact_checked_words=16384)
            libraries = sorted(set(s[len('LIBRARY '):] for s in lines if s.startswith('LIBRARY ')))
            m.require(any('libsycl' in s for s in libraries) and str(adapter) in libraries, 'actual runtime mappings present')
            m.require(all(not s.startswith('/opt/intel/oneapi/') for s in libraries), 'no inherited oneAPI SYCL/UR/UMF runtime')
            r['loaded_libraries'] = [dict(path=s, sha256=m.sha(Path(s))) for s in libraries]
            for x in cap['loaded_libraries']:m.require(any(v['path']==x['path'] and v['sha256']==x['sha256'] for v in r['loaded_libraries']),'same actual loaded source/driver library '+x['path'])
            r['complete'] = True
        except BaseException as exc:
            r['error'] = repr(exc)
        finally:
            if cursor and owner.active is None:
                try:
                    text = command('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    rows = [json.loads(s) for s in text.splitlines() if s.startswith('{')]
                    rows = [v for v in rows if '0000:05:00.0' in v.get('MESSAGE', '') or re.search(r'\bxe\b', v.get('MESSAGE', ''))]
                    r['kernel_device_entries'] = rows
                    r['new_fault_messages'] = [v['MESSAGE'] for v in rows if helper.FAULT.search(v['MESSAGE'])]
                    m.require(not r['new_fault_messages'] and not DUMP.exists(), 'no new GPU fault or dump')
                    r['kernel_gate_passed'] = True
                except BaseException as exc:
                    r['kernel_gate_error'] = repr(exc)
            r['source_stable'] = m.sha(HERE / 'health.cpp') == r['source_sha256']
            r['boot_unchanged'] = r['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            r['active'] = owner.active is not None
            r['passed'] = bool(r['complete'] and r.get('kernel_gate_passed') and r['source_stable'] and r['boot_unchanged'] and not r['active'] and not r.get('error') and not r.get('kernel_gate_error'))
            r['finished_utc'] = m.utc()
            save()
        print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'query_output', 'loaded_libraries')}))
        return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
