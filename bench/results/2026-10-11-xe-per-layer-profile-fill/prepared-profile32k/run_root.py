"""Serial first corrected e32 model diagnostic. No timing/adoption claim."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import time
import types

from protocol import selftest, validate, validate_trace

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
HERE = Path(__file__).resolve().parent
LLVM = W / '.tools/intel-llvm-v711/install'
OWNER = B / 'xe-llvm-v711-source-v5/direct_owner.py'
HELPER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
DUMP = Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
FIXTURE = B / 'clean64k-task-fixture-root-r3/coding-review-65536-tokens.txt'
PACK = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
MODEL = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
MTP = Path('/home/yayoi/.local/share/strata-sycl/mtp/rt')
PROFILE = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010/data/expert-profile.bin')
BASE = dict(PATH=str(LLVM / 'bin') + ':/usr/bin:/bin', HOME='/home/yayoi',
            LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH=str(LLVM / 'lib'),
            XDG_RUNTIME_DIR='/run/user/' + str(os.getuid()))
ERROR = re.compile(r'UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\[(?:ERROR|FATAL)\]|GPU completion unconfirmed|terminate called|Assertion .*failed|CAT error|fault response|out of resources', re.I)
s = importlib.util.spec_from_file_location('qualified_owner', OWNER)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')


def stat_identity(path):
    v = path.stat()
    return dict(device=v.st_dev, inode=v.st_ino, bytes=v.st_size,
                mtime_ns=v.st_mtime_ns, ctime_ns=v.st_ctime_ns)


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = B / 'xe-source-llvm-profile32k-root-v1'
        m.require(not out.exists(), 'immutable fresh diagnostic output')
        out.mkdir(mode=0o700)
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        r = dict(active=True, complete=False, passed=False, commands=[],
                 started_utc=m.utc(), boot_id=boot, controller_sha256=m.sha(__file__),
                 protocol_sha256=m.sha(HERE / 'protocol.py'), owner_sha256=m.sha(OWNER),
                 hash_helper_sha256=m.sha(HERE / 'hash_inputs.py'),
                 mode='free', candidate_revision=json.loads((B / 'xe-per-layer-profile-fill-source-commit-v1.json').read_text())['committed_revision'],
                 adopted=False, upstream_pristine=False, performance_eligible=False,
                 full_model_quality_qualified=False, full262144_qualified=False,
                 reset_executed=False, retry_count=0, model_launch_attempted=False,
                 raw_text_log_budget_bytes=64 << 20, total_artifact_budget_bytes=96 << 20,
                 whole_wall_budget_seconds=900, model_wall_budget_seconds=500,
                 model_rss_budget_bytes=88 << 30, observed_libraries={},
                 scope='First32K diagnostic phase profile:32768 task-tail input, eight4096 chunks, synchronized MTP; timeline intervals include device idle/host/transfer waits; no quiet speed or full quality claim')

        def save():
            for c in r['commands']:
                if c['label'] != 'model' or not c.get('direct_identity'):
                    continue
                ident = c['direct_identity']
                now = m.identity(ident['pid'])
                if not now or now['start_ticks'] != ident['start_ticks']:
                    continue
                try:
                    for line in Path('/proc', str(ident['pid']), 'maps').read_text().splitlines():
                        at = line.find('/')
                        if at < 0:
                            continue
                        p = line[at:]
                        if '(deleted)' in p or not any(k in Path(p).name for k in ('libsycl', 'libur_', 'libze_', 'libumf')):
                            continue
                        if p not in r['observed_libraries']:
                            r['observed_libraries'][p] = dict(path=p, sha256=m.sha(Path(p)))
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
            (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

        owner = m.Owner(out / 'commands', save)
        r['commands'] = owner.commands
        save()
        begin = time.monotonic()
        cursor = None
        helper = None

        def run(label, argv, env=BASE, wall=20, interaction=None, rss=2 << 30):
            remaining = 900 - (time.monotonic() - begin)
            m.require(remaining > 0, 'whole diagnostic deadline')
            e, so, se = owner.run(label, argv, env, W, interaction=interaction,
                                  wall=min(wall, remaining), cpu=int(wall) * 2,
                                  text_cap=64 << 20, total_cap=96 << 20,
                                  rss_cap=rss)
            m.require(m.closed(e) and e['exit_code'] == 0, 'normal closed ' + label)
            return so.read_text(), se.read_text()

        def kernel(label, args):
            text, _ = run(label, ['/usr/bin/journalctl', '-k', *args, '--no-pager', '-o', 'json'])
            rows = [json.loads(x) for x in text.splitlines() if x.startswith('{')]
            relevant = [x for x in rows if '0000:05:00.0' in x.get('MESSAGE', '') or re.search(r'\bxe\b', x.get('MESSAGE', ''))]
            r[label + '_relevant_entries'] = relevant
            faults = [x['MESSAGE'] for x in relevant if helper.FAULT.search(x['MESSAGE'])]
            m.require(not faults and not DUMP.exists(), 'fault-free ' + label)
            return faults

        try:
            r['cpu_protocol_selftest'] = selftest()
            timer_path = B / 'xe-source-llvm-profiling-tags-root-v1/record.json'
            timer = json.loads(timer_path.read_text())
            m.require(timer['passed'] and timer['complete'] and not timer['active'] and
                      timer['boot_id'] == boot and all(m.closed(c) for c in timer['commands']),
                      'closed same-boot raw profiling timestamp qualification before long profile')
            r['profiling_tag_qualification_sha256'] = m.sha(timer_path)
            small_path = B / 'xe-source-llvm-model8k-root-v5/record.json'
            small = json.loads(small_path.read_text())
            m.require(small['passed'] and small['complete'] and not small['active'] and small['boot_id'] == boot and all(m.closed(c) for c in small['commands']), 'closed same-boot real cache/MTP diagnostic before long profile')
            r['small_model_qualification_sha256'] = m.sha(small_path)
            cpu_path = B / 'xe-source-llvm-profile32k-cpu-root-v1/record.json'
            cpu = json.loads(cpu_path.read_text())
            m.require(cpu['passed'] and cpu['complete'] and not cpu['active'] and
                      all(m.closed(c) for c in cpu['commands']), 'closed CPU controller-helper qualification')
            m.require(all(m.sha(HERE / name) == digest for name, digest in cpu['source_pins'].items()),
                      'same CPU-qualified controller and helpers')
            r['cpu_helper_qualification_sha256'] = m.sha(cpu_path)
            m.require(os.geteuid() != 0 and not DUMP.exists(), 'unprivileged and no preexisting GPU dump')
            proofs = dict(build=B / 'xe-e32-free-full-build-root-v11/record.json',
                          asan=B / 'xe-e32-free-asan-full-build-root-v8/record.json',
                          lint=B / 'xe-e32-lints-baseline-compare-root-v6/record.json',
                          health=B / 'xe-source-llvm-small-health-root-v3/record.json',
                          cap=B / 'xe-free-host-capabilities-root-v3/record.json')
            for case in ('device-selftest', 'bf16-gemv', 'kv-q8', 'kv-q4', 'dense-free', 'dense-contrib'):
                proofs[case] = B / 'xe-source-llvm-gpu-parity-root-v2' / case / 'record.json'
            verified = {}
            for key, path in proofs.items():
                v = json.loads(path.read_text())
                m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']), 'closed prerequisite ' + key)
                if 'boot_id' in v:
                    m.require(v['boot_id'] == boot, 'same qualified boot ' + key)
                if key in ('device-selftest', 'bf16-gemv', 'kv-q8', 'kv-q4', 'dense-free', 'dense-contrib'):
                    m.require(m.sha(Path(v['binary']['path'])) == v['binary']['sha256'],
                              'same already GPU-qualified parity executable ' + key)
                verified[key] = v
            r['proof_sha256'] = {key: m.sha(path) for key, path in proofs.items()}
            r['gpu_parity_reuse_scope'] = 'Six closed same-boot v2 no-model cases; exact parity binaries unchanged after host-only cache selection/reader correction; all six parity binaries independently unchanged. No new model or full-quality proof inferred.'
            m.require(verified['build']['source_pins'] == verified['asan']['source_pins'], 'same release/host sanitizer source')
            r['source_pins'] = verified['build']['source_pins']
            m.require(all(m.sha(W / p) == digest for p, digest in r['source_pins'].items()), 'exact corrected source')
            head, _ = run('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
            status, _ = run('source-status', ['/usr/bin/git', 'status', '--porcelain'])
            m.require(head.strip() == r['candidate_revision'] and not status.strip(), 'committed clean candidate')
            r['binary'] = verified['build']['binary']
            binary = Path(r['binary']['path'])
            m.require(m.sha(binary) == r['binary']['sha256'], 'qualified engine binary')
            m.require(m.sha(HELPER) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'diagnostic helper')
            helper = types.ModuleType('read_only_diagnostics')
            exec(compile(HELPER.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(HELPER), 'exec'), helper.__dict__)
            m.require(m.sha(FIXTURE) == '5284d8fa53d1ef28f6ba916a629faf901c890bc0c669bf986ee04ed72aa2c3ff', 'canonical parent fixture')
            ids = [int(x) for x in re.split(r'[\s,]+', FIXTURE.read_text().strip())]
            m.require(len(ids) == 65536 and all(0 <= x < 248320 for x in ids), 'canonical parent count/domain')
            fixture_manifest_path = FIXTURE.with_name('manifest.json')
            m.require(m.sha(fixture_manifest_path) == '0fbf6a16a90b55a3767145d1217b2027e218c0537a487faa428ab6527e6771b1', 'immutable canonical task-tail proof')
            fixture_manifest = json.loads(fixture_manifest_path.read_text())
            m.require(fixture_manifest['passed'] and not fixture_manifest['active'] and
                      fixture_manifest['fixture']['sha256'] == m.sha(FIXTURE) and
                      fixture_manifest['suffix']['tokens'] == 49 and
                      fixture_manifest['suffix']['encode_exact'] and fixture_manifest['suffix']['decode_exact'],
                      'verified canonical 49-token closing task and assistant suffix')
            tokens = ids[:32768-49] + ids[-49:]
            m.require(len(tokens) == 32768 and tokens[-49:] == ids[-49:], 'complete task boundary preserved at exactly 32768 tokens')
            r['input_derivation'] = dict(parent_manifest_path=str(fixture_manifest_path),
                parent_manifest_sha256=m.sha(fixture_manifest_path), prefix_tokens=32719,
                suffix_tokens=49, rule='canonical first32719 body tokens plus canonical last49 verified task/chat suffix; middle omitted; no new tokenization',
                limitation='Historical body may end mid-code and this is not independent quality data; no guarantee of16 outputs or draft acceptance.')
            (out / 'tokens.txt').write_text(','.join(map(str, tokens)) + '\n')
            r['input'] = dict(parent_path=str(FIXTURE), parent_sha256=m.sha(FIXTURE),
                              count=32768, sha256=m.sha(out / 'tokens.txt'))
            m.require(m.sha(PROFILE) == '8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf', 'fixed expert profile')
            files = sorted(p for base in (PACK, MODEL, MTP) for p in base.rglob('*') if p.is_file()) + [PROFILE]
            prior_path = B / 'xe-source-llvm-model8k-root-v1/record.json'
            prior = json.loads(prior_path.read_text())
            m.require(not prior['active'] and not prior['model_launch_attempted'], 'closed CPU-only prior preflight')
            phase = next(c for c in prior['commands'] if c['label'] == 'model-input-hashes')
            m.require(m.closed(phase) and phase['exit_code'] == 0 and
                      prior['hash_helper_sha256'] == m.sha(HERE / 'hash_inputs.py'), 'actual successful full hash phase')
            r['model_inputs'] = prior['model_inputs']
            m.require([str(p.resolve()) for p in files] == [x['path'] for x in r['model_inputs']], 'same complete model file set')
            m.require(all(stat_identity(Path(x['path'])) == x['stat'] for x in r['model_inputs']), 'full-hashed model files unchanged')
            r['model_hash_phase_source'] = dict(path=str(prior_path), sha256=m.sha(prior_path),
                scope='Successful full-file CPU hash phase only; original overall preflight failure remains failed. Exact inode/device/size/mtime/ctime verified for every unchanged file.')
            memory = {x.split(':')[0]: int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:', 'MemAvailable:'))}
            r['initial_memory'] = memory
            m.require(memory['MemAvailable'] >= 80 << 30, 'RAM headroom for known model/native arena')
            cgroup = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
            r['cgroup_limits'] = []
            for p in (cgroup, *cgroup.parents):
                if not p.is_relative_to('/sys/fs/cgroup'):
                    continue
                if (p / 'memory.max').exists():
                    maximum, current = (p / 'memory.max').read_text().strip(), int((p / 'memory.current').read_text())
                    r['cgroup_limits'].append(dict(path=str(p), maximum=maximum, current=current))
                    m.require(maximum == 'max' or int(maximum) - current >= 80 << 30, 'cgroup memory headroom')
            state, _ = run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'])
            m.require(state.strip() == 'ActiveState=inactive', 'embedding remains inactive')
            adapter = verified['health']['adapter']
            m.require(m.sha(Path(adapter['path'])) == adapter['sha256'], 'qualified adapter')
            env = helper.diagnostic_environment(dict(BASE, ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
                STRATA_GPU_PCI='0000:05:00.0', STRATA_EVENT_MARK='barrier',
                SYCL_CACHE_PERSISTENT='0', UR_ADAPTERS_FORCE_LOAD=adapter['path'],
                UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
                EnableDirectSubmission='0', STRATA_PREFILL_RING='8', STRATA_PREFILL_FIRST='0',
                STRATA_ARENA_HOST_USM='1', STRATA_PREFILL_ISSUER='0', STRATA_MTP_PREFILL_SYNC='1', STRATA_PREFILL_TIMING='1'))
            health_env = dict(env)
            env.update(ZEL_LOADER_LOGGING_LEVEL='warn', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',
                       UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                       UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr', UR_ENABLE_LAYERS='')
            env.pop('UR_LOG_TRACING', None)
            r['environment'] = env
            argv = [str(binary), '--pack', str(PACK), '--native', str(MODEL / 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'),
                    '--serve', '--max-context', '262144', '--spec', '4', '--spec-min-p', '0',
                    '--mtp', str(MTP), '--suffix-draft', '0', '--prefill', '4096', '--no-prefill-borrow',
                    '--expert-cache', '128', '--expert-profile', str(PROFILE), '--expert-cache-per-layer',
                    '--pool-workers', '5', '--pcie-frac', '0', '--spec-split', '--adapt-swaps', '0',
                    '--conversation-cache-mib', '0', '--conversation-cache-slots', '4', '--greedy',
                    '--short-read', '0', '--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '1',
                    '--prompt-cache-every', '262139', '--prompt-cache-root', '0', '--turn-token', '-1']
            r['argv'] = argv
            kernel('kernel-before', ['-b'])
            text, _ = run('kernel-cursor', ['/usr/bin/journalctl', '-k', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'^-- cursor: (.+)$', text, re.M).group(1)
            r['kernel_cursor_before'] = cursor
            probe = verified['health']['binary']
            m.require(m.sha(Path(probe['path'])) == probe['sha256'], 'qualified integer probe')
            r['health_probe'] = probe

            def health(label):
                text, _ = run(label, [probe['path'], '0000:05:00.0'], env=health_env, wall=30, rss=4 << 30)
                expected = ['STATUS 0000:05:00.0 0', 'QUEUE 0000:05:00.0', 'H2D 65536 complete',
                            'KERNEL 16384 complete', 'D2H 65536 complete',
                            'PASS 0000:05:00.0: 1 round, 16384 exact words']
                m.require(text.splitlines()[:6] == expected, 'exact integer ' + label)
                r[label + '_protocol'] = text.splitlines()

            health('health-before')
            kernel('kernel-after-prehealth', ['--after-cursor', cursor])
            admitted_at = time.monotonic()
            stderr = out / 'commands/model.stderr'

            def interaction(io, entry):
                original_poll = io.poll

                def poll():
                    original_poll()
                    m.require(not DUMP.exists(), 'new GPU dump: stop owned model')
                io.poll = poll
                m.require(time.monotonic() - admitted_at < 30, 'fresh health admission')
                r['startup'] = []
                while True:
                    line = io.line(120)
                    r['startup'].append(line)
                    m.require(len(r['startup']) < 1000 and not line.startswith('ERR'), 'startup progress')
                    if line.startswith('READY '):
                        m.require(line.split()[1] == '262144', 'physical configured context')
                        break
                info = next(x for x in r['startup'] if x.startswith('INFO '))
                settings = dict(x.split('=', 1) for x in info.split()[1:] if '=' in x)
                r['effective_settings'] = settings
                m.require(all(settings.get(k) == v for k, v in dict(context='262144', kv='int8',
                    kv_resident='32768', spec='4', pool_workers='5', pcie_frac='0.00').items()), 'effective model settings')
                m.require(int(settings['vram_free_mib']) >= 512, 'VRAM startup reserve')
                exe = Path('/proc', str(entry['pid']), 'exe')
                m.require(exe.resolve() == binary.resolve() and m.sha(exe) == r['binary']['sha256'], 'actual process executable')
                actual = dict(x.decode().split('=', 1) for x in Path('/proc', str(entry['pid']), 'environ').read_bytes().split(b'\0') if b'=' in x)
                m.require(actual == env, 'actual explicit environment')
                m.require(not ERROR.search(stderr.read_text()), 'no severe startup runtime error')
                r['request_protocol'] = []
                io.send(('GEN 16 ckpt=1 logprobs=5 ' + ','.join(map(str, tokens)) + '\n').encode())
                while True:
                    line = io.line(120)
                    r['request_protocol'].append(line)
                    m.require(len(r['request_protocol']) < 1000 and not line.startswith('ERR'), 'request progress')
                    if line.startswith('PP '):
                        print(line, flush=True)
                        save()
                    if line.startswith('DONE '):
                        break
                # Finish an already completed request normally before judging its coverage.
                # Output/trace validation below cannot turn a valid early EOS into SIGTERM.
                save()
                io.send(b'QUIT\n')
                io.exit(30)

            r['model_launch_attempted'] = True
            save()
            so, se = run('model', argv, env=env, wall=500, interaction=interaction, rss=88 << 30)
            r['model_normal_close'] = True
            r['validation'] = validate(r['request_protocol'])
            fill_matches = re.findall(r'^strata generate: pre-filled (\d+) of (\d+) slots from the profile .*; slot (\d+) verified$', se, re.M)
            resident_matches = re.findall(r'^strata generate: expert residency table: (\d+) resident experts$', se, re.M)
            m.require(len(fill_matches) == 1 and fill_matches[0][:2] == ('144', '144') and resident_matches == ['144'], 'actual full per-layer profile occupancy')
            m.require(r['effective_settings']['expert_slots'] == '144', 'actual cache capacity matches fill and residency')
            r['profile_fill_validation'] = dict(filled=144, capacity=144, residents=144, verified_slot=int(fill_matches[0][2]), per_layer_quota=3, geometry_provenance='Same full48x512 unique ranking/model; per-layer slot ranges and total occupancy imply each of48 layers fills its3 slots. Only one slot readback is the existing byte check, not all144 byte checks.')
            r['trace_validation'] = validate_trace(se)
            timing = re.findall(r'^strata prefill timing: (\d+) tokens, GPU timeline (\d+) ms, wall (\d+) ms, host staging (\d+) ms:(.*)$', se, re.M)
            m.require(len(timing) == 1 and timing[0][0] == '32767' and int(timing[0][1]) > 0 and int(timing[0][2]) > 0, 'complete actual32K phase timeline')
            r['phase_profile'] = dict(batched_positions=32767, timeline_ms=int(timing[0][1]), wall_ms=int(timing[0][2]), host_staging_ms=int(timing[0][3]), phase_text=timing[0][4], host_lines=[x for x in se.splitlines() if x.startswith('strata prefill timing: host:')], limitation='Instrumented ordered-stream event intervals charge host/copy idle to the preceding phase; not exclusive kernel service times, quiet throughput, independent profiler verification, or model quality.')
            r['runtime_error_lines'] = [line for line in (so + se).splitlines() if ERROR.search(line)]
            m.require(not r['runtime_error_lines'], 'no severe model runtime error')
            for x in verified['health']['loaded_libraries']:
                m.require(r['observed_libraries'].get(x['path'], {}).get('sha256') == x['sha256'], 'actual qualified runtime mapping')
            m.require(all('/opt/intel/oneapi/' not in p for p in r['observed_libraries']), 'source free runtime only')
            kernel('kernel-after-model', ['--after-cursor', cursor])
            health('health-after')
            kernel('kernel-after-health', ['--after-cursor', cursor])
            m.require(all(stat_identity(Path(x['path'])) == x['stat'] for x in r['model_inputs']), 'model inputs unchanged')
            m.require(all(m.sha(W / p) == digest for p, digest in r['source_pins'].items()), 'source stable')
            m.require(m.sha(binary) == r['binary']['sha256'], 'binary stable')
            m.require(m.sha(HERE / 'protocol.py') == r['protocol_sha256'] and m.sha(HERE / 'hash_inputs.py') == r['hash_helper_sha256'], 'helpers stable')
            m.require(m.sha(__file__) == r['controller_sha256'] and
                      m.sha(HELPER) == '51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457', 'controller/diagnostic helper stable')
            m.require(all(m.sha(Path(x['path'])) == x['sha256'] for x in r['observed_libraries'].values()), 'mapped runtime identities stable')
            m.require(r['validation']['mtp_counts'][1] > 0, 'actual speculative draft activity')
            r['complete'] = True
        except BaseException as e:
            r['error'] = repr(e)
        finally:
            if cursor and helper and owner.active is None:
                try:
                    r['new_fault_messages'] = kernel('kernel-final', ['--after-cursor', cursor])
                    r['kernel_gate_passed'] = True
                except BaseException as e:
                    r['kernel_gate_error'] = repr(e)
            r['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            r['active'] = owner.active is not None
            r['passed'] = bool(r['complete'] and r.get('kernel_gate_passed') and r['boot_unchanged']
                               and not r['active'] and not r.get('error') and not r.get('kernel_gate_error'))
            r['finished_utc'] = m.utc()
            save()
        print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'kernel_gate_error', 'model_launch_attempted')}))
        return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
