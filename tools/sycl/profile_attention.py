#!/usr/bin/env python3
"""Main-owned, bounded VTune capability/attention collection; no privilege changes."""
import fcntl
import hashlib
import csv
import io
import json
import math
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


def capability_assets():
    """Pin the installed64-bit collector, not just its small CLI launcher."""
    root = VTUNE.resolve().parents[1]
    names = sorted(p for folder in ('bin64', 'lib64') for p in (root / folder).rglob('*') if p.is_file())
    guards.require(0 < len(names) < 16384, 'bounded installed collector asset set')
    rows = [(str(p.relative_to(root)), digest(p)) for p in names]
    value = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    return dict(root=str(root), count=len(rows), sha256=value,
                scope='Sorted relative-path/SHA256 pairs of all bin64/lib64 files, including resolved file symlinks; installed tool assets, not project build outputs.')


def capabilities(action):
    """Fresh version-qualified probes; no model, GPU target workload or privilege changes."""
    root = B / 'profiler-capability-vtune-2026-6-v1'
    result = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([]); driver = guards.driver_guard()
        guards.require(not root.exists(), 'inspect existing capability job; no retry')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        paths = ('tools/sycl/profile_attention.py', 'tools/sycl/measurement_guards.py',
                 'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_environment.json')
        if action == 'prepare':
            previous = result.read_bytes()
            guards.require(previous == subprocess.check_output(['git', 'show', revision + ':' + str(result.relative_to(P))], cwd=P),
                           'preserve committed previous capability outcome')
            guards.require(RESULT.read_bytes() == subprocess.check_output(['git', 'show', revision + ':' + str(RESULT.relative_to(P))], cwd=P),
                           'preserve previous GPU collector outcome')
            old = json.loads(previous)
            guards.require(old['complete'] and old['ownership_closed'] and not old['active'], 'previous capability job still open')
            pins = {}
            for name in paths:
                data = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)
                guards.require(data == (P/name).read_bytes(), 'commit exact capability input: ' + name)
                pins[name] = hashlib.sha256(data).hexdigest()
            profiler = str(VTUNE.resolve())
            guards.require(profiler == '/opt/intel/oneapi/vtune/2026.6/bin64/vtune', 'new profiler revision needs a new plan/root')
            environment = dict(json.loads(SETTINGS.read_text())['environment'], TMPDIR=str(root/'tmp'),
                XDG_CACHE_HOME=str(root/'cache'), UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
            base = [profiler, '-data-limit=16', '-user-data-dir', str(root/'user')]
            cpu = base + ['-collect', 'hotspots', '-knob', 'sampling-mode=sw', '-knob', 'enable-stack-collection=true',
                '-knob', 'stack-size=4096', '-knob', 'enable-characterization-insights=false',
                '-result-dir', str(root/'cpu'), '--', '/usr/bin/openssl', 'speed', '-seconds', '3', '-bytes', '16384', '-evp', 'sha256']
            report = [profiler, '-report', 'hotspots', '-result-dir', str(root/'cpu'), '-format', 'csv',
                '-csv-delimiter', 'tab', '-group-by', 'function', '-limit', '12']
            gpu = base + ['-collect', 'gpu-hotspots', '-result-dir', str(root/'gpu'), '--', '/usr/bin/true']
            record = dict(schema='strata-profiler-capability-v2', stage='prepared', source_commit=revision,
                previous=dict(commit=revision, path=str(result.relative_to(P)), sha256=hashlib.sha256(previous).hexdigest(),
                    outcome='VTune2026.4 CPU ptrace refusal; perf cycles/task-clock denied. Preserve original failures in Git.'),
                previous_gpu=dict(commit=revision, path=str(RESULT.relative_to(P)), sha256=digest(RESULT),
                    outcome='VTune2026.4 GPU-hotspots rejected this Ryzen host; compiler metadata did not qualify hardware metrics.'),
                input_sha256=pins, profiler=dict(path=profiler, sha256=digest(profiler), assets=capability_assets(),
                    version=subprocess.check_output([profiler, '-version'], text=True, timeout=15).strip()),
                targets={n:dict(path=n, sha256=digest(n)) for n in ('/usr/bin/openssl','/usr/bin/true')},
                environment=environment, commands=dict(cpu_collect=cpu, cpu_report=report, gpu_launch_probe=gpu),
                gates='Run CPU collect once; report only if collection exits0 and closes normally. Require positive finite CPU Time and actual OpenSSL/libcrypto samples for CPU qualification. Then independent GPU collector-launch probe on /usr/bin/true, which contains no GPU workload: exit0 never qualifies GPU timing/counters. Stop on source/asset/owner/budget/fault failure. No retry, permission/sysctl/driver reset or model execution.',
                budgets=dict(wall_s_per_command=120, cpu_s_per_command=120, rss_bytes=4<<30, text_bytes_per_command=16<<20,
                             file_bytes=128<<20, total_bytes=256<<20, raw_text_bytes=64<<20),
                retention='Keep tool/source/environment identities, each capability outcome and compact refusal/sample evidence. Retire normal collector output after the current production-profile decision; preserve old failure via previous Git reference.',
                model_executed=False, target_gpu_work=False, GPU_hardware_metrics_qualified=False,
                CPU_sampling_qualified=False, GPU_collector_launch_probe_accepted=False,
                passed_meaning='CPU sampling qualification plus GPU collector-launch acceptance; never GPU hardware-metric or model qualification.',
                active=False, complete=False, passed=False)
            result.write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(dict(stage='prepared', source_commit=revision, assets=record['profiler']['assets']['count'], model_executed=False)))
            return 0
        raw = result.read_bytes()
        guards.require(raw == subprocess.check_output(['git','show','HEAD:'+str(result.relative_to(P))],cwd=P), 'commit generated capability plan first')
        record = json.loads(raw)
        guards.require(record['schema']=='strata-profiler-capability-v2' and record['stage']=='prepared', 'closed/mismatched plan; no retry')
        def admission():
            for name, sha in record['input_sha256'].items():
                original = subprocess.check_output(['git','show',record['source_commit']+':'+name],cwd=P)
                guards.require(hashlib.sha256(original).hexdigest()==sha and digest(P/name)==sha, 'capability input differs: '+name)
            guards.require(str(VTUNE.resolve())==record['profiler']['path'] and digest(VTUNE)==record['profiler']['sha256'] and
                           capability_assets()==record['profiler']['assets'], 'installed profiler changed')
            for name, item in record['targets'].items(): guards.require(digest(name)==item['sha256'], 'target changed')
        admission();root.mkdir(mode=0o700)
        for name in ('tmp','cache','user'):(root/name).mkdir()
        record.update(driver_before=driver, permissions={name:Path('/proc/sys',name).read_text().strip() for name in
            ('kernel/yama/ptrace_scope','kernel/perf_event_paranoid','dev/xe/observation_paranoid')}, observations={})
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
        cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '));owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
        owner=supervision.Owner(root/'owned',persist);persist()
        try:
            for label in ('cpu_collect','cpu_report','gpu_launch_probe'):
                if label=='cpu_report' and record['observations']['cpu_collect']['exit_code']!=0: continue
                guards.driver_guard()
                entry,out,err=owner.run(label,record['commands'][label],record['environment'],P,wall=120,cpu=120,
                    rss_cap=4<<30,text_cap=16<<20,file_cap=128<<20,total_cap=256<<20)
                guards.require(supervision.closed(entry),'collector ownership/budget failed')
                guards.driver_guard()
                faults=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                guards.require(not re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',faults), 'fresh fault; stop before another collector')
                stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
                record['observations'][label]=dict(exit_code=entry.get('exit_code'),ownership_closed=True,
                    diagnostics=[line for line in (stdout+'\n'+stderr).splitlines() if re.search(r'(?i)vtune:|error|warning|permission|unsupported|not supported|no data|ptrace',line)][-24:])
                if label=='cpu_report' and entry.get('exit_code')==0:
                    table=csv.DictReader(io.StringIO(stdout),delimiter='\t');rows=list(table)
                    column=next((k for k in (table.fieldnames or []) if 'CPU Time' in k),None)
                    if column and rows:
                        times=[float(row[column]) for row in rows]
                        qualified=all(math.isfinite(v) and v>=0 for v in times) and sum(times)>0 and any('libcrypto' in str(row).lower() or 'openssl' in str(row).lower() for row in rows)
                        record['CPU_sampling_qualified']=bool(qualified)
                        if qualified:record['cpu_hotspots']=dict(columns=table.fieldnames,rows=rows,reported_CPU_time_sum_s=sum(times),scope='Top12 software capability samples, not model CPU attribution.')
                if label=='gpu_launch_probe':record['GPU_collector_launch_probe_accepted']=entry.get('exit_code')==0
                persist()
            admission();record.update(stage='closed',complete=True,execution_passed=True)
        except BaseException as error:record['error']=str(error)
        finally:
            record['ownership_closed']=owner.active is None and all(supervision.closed(e) for e in owner.commands)
            try:
                text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
                record['fresh_faults']=[s for s in text.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
                record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip();record['driver_after']=guards.driver_guard()
            except Exception as error:record['final_probe_error']=str(error)
            record['execution_passed']=bool(record.get('execution_passed') and record['ownership_closed'] and record.get('fresh_faults')==[] and record.get('boot_unchanged') and 'final_probe_error' not in record)
            record['passed']=bool(record['execution_passed'] and record['CPU_sampling_qualified'] and record['GPU_collector_launch_probe_accepted'])
            record['decision']='A successful collector-launch probe on true is not GPU qualification. Use actual positive CPU samples only if qualified; otherwise retain refusal and select PTI/native instrumentation without changing permissions.'
            persist();result.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:record.get(k) for k in ('stage','complete','execution_passed','passed','CPU_sampling_qualified','GPU_collector_launch_probe_accepted','GPU_hardware_metrics_qualified','ownership_closed','error')}))
        return 0 if record['execution_passed'] else 1


def main():
    if len(sys.argv)==2 and sys.argv[1] in ('capabilities-prepare','capabilities-run'):
        return capabilities(sys.argv[1].split('-')[1])
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
