#!/usr/bin/env python3
"""Main-owned, bounded VTune attention collection; no privilege changes."""
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
ROOT = B / 'qsa-hardware-profile-current'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-hardware-profile-current.json'
BUILD = B / 'main-native-gdn-engine/record.json'
QUALIFIED = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-attention-current.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'
VTUNE = Path('/opt/intel/oneapi/vtune/latest/bin64/vtune')
BENCH = 'sycl_qsa_bf16_prefix_qcache_bench'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def emitted_resources(text, trace):
    """Join complete compiler resource blocks to successful traced launches."""
    created = {}
    launches = {}
    for line in trace.splitlines():
        if '<--- urKernelCreate(' in line and 'UR_RESULT_SUCCESS' in line:
            match = re.search(r'\((_ZTS\S+)\).*?\.phKernel = .*?\((0x[0-9a-f]+)\)', line)
            if match:
                created[match[2]] = match[1]
        if '<--- urEnqueueKernelLaunchWithArgsExp(' not in line or 'UR_RESULT_SUCCESS' not in line:
            continue
        handle = re.search(r'\.hKernel = (0x[0-9a-f]+)', line)[1]
        local = [dict(index=int(i), bytes=int(n)) for i, n in re.findall(
            r'UR_EXP_KERNEL_ARG_TYPE_LOCAL, \.index = (\d+), \.size = (\d+)', line)]
        name = created[handle]
        value = launches.setdefault(name, dict(launches=0, local_arguments=local,
                                               dynamic_local_bytes=sum(x['bytes'] for x in local)))
        guards.require(value['local_arguments'] == local, 'different dynamic-local configuration')
        value['launches'] += 1
    resources = []
    for block in re.split(r'(?m)^  - name:\s+', text)[1:]:
        name = block.splitlines()[0].strip()
        execution = re.search(r'(?ms)^    execution_env:\n(.*?)(?=^ {0,4}\S|\Z)', block)
        if not execution:
            continue
        fields = dict(re.findall(r'^      (\w+):\s*(.*?)\s*$', execution[1], re.M))
        scratch = re.search(r'(?ms)^    per_thread_memory_buffers:\n(.*?)(?=^ {0,4}\S|\Z)', block)
        # Keep executed kernels and distinct spill evidence, not every compiled template.
        if name not in launches and not scratch:
            continue
        resources.append(dict(kernel=name, execution_env=fields,
            per_thread_memory_buffers=scratch[1].strip() if scratch else None,
            observed_launches=launches.get(name)))
    guards.require(resources and all(any(r['kernel'] == name for r in resources) for name in launches),
                   'complete resource metadata for every observed kernel required')
    return resources


def main():
    guards.require(len(sys.argv) == 2 and sys.argv[1] in ('smoke', 'resources', 'isa', 'overview', 'memory'),
                   'use profile_attention.py smoke|resources|isa|overview|memory')
    mode = sys.argv[1]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        driver = guards.driver_guard()
        build = json.loads(BUILD.read_text())
        qualified = json.loads(QUALIFIED.read_text())
        for item in (build, qualified):
            guards.require(item['passed'] and item['complete'] and item['ownership_closed']
                           and not item['active'], 'closed qualified build/fixture required')
        fixture = build['fixtures'][BENCH]
        guards.require(fixture == qualified['fixtures'][BENCH], 'exact qualified fixture required')
        guards.require(digest(fixture['path']) == fixture['sha256'], 'fixture changed')
        for group in [build['project'], *build['dependencies']]:
            for name, expected in group['source_sha256'].items():
                guards.require(digest(Path(group['root']) / name) == expected, 'build input changed: ' + name)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        inputs = {}
        for path in (Path(__file__).resolve(), P / 'tools/sycl/measurement_guards.py',
                     P / 'tools/sycl/measurement_owner.py', SETTINGS, QUALIFIED):
            name = str(path.relative_to(P))
            expected = hashlib.sha256(subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)).hexdigest()
            guards.require(digest(path) == expected, 'uncommitted profile input: ' + name)
            inputs[name] = expected
        settings = json.loads(SETTINGS.read_text())
        for path, item in settings['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime changed: ' + path)
        if mode == 'smoke' or (mode in ('resources', 'isa') and not ROOT.exists()):
            guards.require(not ROOT.exists(), 'inspect/close the existing profile; no automatic replacement')
            ROOT.mkdir(mode=0o700)
            record = dict(source_commit=revision, input_sha256=inputs, fixture=fixture,
                          build_source_commit=build['source_commit'], driver=driver, checks=[],
                          consumer='Current QSA hardware resource attribution; retire raw collector output after extraction.',
                          model_performance_measured=False, hardware_metrics_qualified=False)
        else:
            record = json.loads(RESULT.read_text())
            guards.require(record['fixture'] == fixture and record['ownership_closed'] and not record['active'],
                           'matching, fully closed prior collection required')
            if mode not in ('resources', 'isa'):
                guards.require(record['passed'] and record['hardware_metrics_qualified'],
                               'prior successful collection with explicitly reviewed hardware coverage required')
        guards.require(not (ROOT / mode).exists(), 'mode already launched; inspect its existing receipt')
        run = ROOT / mode
        run.mkdir()
        for name in ('tmp', 'cache', 'user'):
            (run / name).mkdir()
        env = dict(settings['environment'])
        if mode not in ('smoke', 'resources', 'isa'):
            for key in list(env):
                if key.startswith('UR_LOG_') or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE'):
                    del env[key]
        env.update(TMPDIR=str(run / 'tmp'), XDG_CACHE_HOME=str(run / 'cache'))
        command = [str(VTUNE), '-collect', 'gpu-hotspots', '-knob', 'gpu-profiling-mode=characterization',
                   '-knob', 'characterization-mode=' + ('global-memory-accesses' if mode == 'memory' else 'overview'),
                   '-knob', 'dump-compute-task-binaries=false', '-knob', 'collect-programming-api=true',
                   '-knob', 'gpu-sampling-interval=1', '-data-limit=16', '-user-data-dir', str(run / 'user'),
                   '-result-dir', str(run / 'vtune'), '--', fixture['path'],
                   '--smoke' if mode == 'smoke' else '--component']
        if mode in ('resources', 'isa'):
            # Dump metadata only, without changing optimization or dumping IR/ISA.
            # Intel IGC documentation/configuration_flags.md: release-available.
            env['IGC_DumpZEInfoToConsole'] = '1'
            command = [fixture['path'], '--smoke']
        if mode == 'isa':
            # ShaderDumpEnable emits IR/vISA/GenISA without the early exit of
            # DumpASMToConsole. No override or optimization control is set.
            # Owner monitors run (the parent of owned), including these files.
            (run / 'isa').mkdir()
            env.update(IGC_ShaderDumpEnable='1', IGC_DumpToCustomDir=str(run / 'isa'))
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        record.update(mode=mode, complete=False, passed=False, active=False, environment=env,
                      current_recipe_commit=revision)
        record['permissions'] = {p: Path('/proc/sys', p).read_text().strip() for p in
                                 ('dev/xe/observation_paranoid', 'kernel/perf_event_paranoid', 'kernel/kptr_restrict')}
        record['profiler'] = dict(path=str(VTUNE.resolve()), sha256=digest(VTUNE),
            version=subprocess.check_output([str(VTUNE), '--version'], text=True, timeout=10).strip())
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (ROOT / 'record.json').write_text(json.dumps(
                dict(record, owned_commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(run / 'owned', persist)
        try:
            entry, out, err = owner.run('vtune', command, env, P, wall=240, cpu=480,
                rss_cap=4 << 30, file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20)
            stdout = out.read_text(errors='replace')
            stderr = err.read_text(errors='replace')
            text = stdout + '\n' + stderr
            native_pass = 'fixture_status=PASS' in stdout
            collector_ok = supervision.closed(entry) and entry.get('exit_code') == 0
            record['checks'].append(dict(mode=mode, command=command, recipe_commit=revision,
                input_sha256=inputs, environment=env, collector_exit_code=entry.get('exit_code'),
                collector_ok=collector_ok, target_reported_pass=native_pass,
                ownership_closed=supervision.closed(entry),
                target_cases=[s for s in stdout.splitlines() if s.startswith(('case ', 'fixture_status='))],
                collector_diagnostics=[s for s in text.splitlines() if re.search(
                    r'(?i)vtune:|error|warning|permission|not supported|unsupported|no data|GPU Time|XVE|bandwidth', s)][-80:],
                owner_error=entry.get('error')))
            record.update(complete=True, passed=bool(native_pass and collector_ok),
                          hardware_metrics_qualified=False)
            if mode in ('resources', 'isa'):
                record['emitted_metadata_observed'] = 'grf_count:' in text and 'slm_size:' in text
                record['passed'] = record['passed'] and record['emitted_metadata_observed']
                record['emitted_resource_summary'] = dict(kernels=emitted_resources(stdout, stderr),
                    raw_sha256={p.name: digest(p) for p in (out, err)},
                    limits=['Compiler metadata and traced dynamic arguments, not hardware counters or measured occupancy.',
                            'Static slm_size excludes separately supplied dynamic-local arguments; hardware rounding is unmeasured.',
                            'Compiled variants without a successful traced launch are not executed-path evidence.'])
            if mode == 'isa':
                files = sorted(p for p in (run / 'isa').rglob('*') if p.is_file())
                guards.require(all(not p.is_symlink() for p in files), 'unexpected shader dump link')
                record['isa_files'] = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                            sha256=digest(p)) for p in files]
                record['isa_question'] = ('Match executed attention kernels to their K load/store lowering; '
                    'typed four-element source does not establish emitted transaction width. '
                    'Retire dumps after extracting the current candidate decision; not hardware traffic counters.')
                guards.require(any(p.suffix in ('.asm', '.visaasm') for p in files), 'no ISA dump observed')
        except BaseException as error:
            record['error'] = str(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(e) for e in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [s for s in after.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', s)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:
                record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record['driver_guard_error'] = str(error)
                record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed']
                                    and record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            RESULT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'passed', 'complete', 'ownership_closed', 'error')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
