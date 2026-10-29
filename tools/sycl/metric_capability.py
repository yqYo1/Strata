#!/usr/bin/env python3
"""Own a finite direct-C Level Zero metric inventory, never a model workload.

Repository LICENSE. Independently authored; installed official API declarations
and the existing Main-authored owner are the only implementation dependencies.
Prepare -> commit -> build -> commit -> run. No implicit retries or GPU reset.
"""
import copy
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
ROOT = B / 'metric-capability-c-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'metric_capability'
SOURCE = P / 'bench/sycl/metric_capability.c'
CC = Path('/usr/bin/gcc').resolve()
LOADER = Path('/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0')
NEO = Path('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758')
MD = Path('/usr/lib/x86_64-linux-gnu/libigdmd.so.1.16.196')
ML = Path('/usr/lib/x86_64-linux-gnu/libigdml.so.1.0.240')
ELF_LOADER = Path('/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2')
INPUTS = ('bench/sycl/metric_capability.c', 'tools/sycl/metric_capability.py',
          'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_guards.py',
          'tools/sycl/measurement_environment.json')
FLAGS = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIE']
BUILD_LIMITS = dict(wall=60, cpu=45, rss_cap=1 << 30, text_cap=2 << 20,
                    total_cap=32 << 20, file_cap=8 << 20)
RUN_LIMITS = dict(wall=60, cpu=45, rss_cap=1 << 30, text_cap=8 << 20,
                  total_cap=32 << 20, file_cap=8 << 20)
require, digest = guards.require, supervision.sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], timeout=15)


def pins(revision):
    result = {}
    for name in INPUTS:
        raw = git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def public():
    current = json.loads(RESULT.read_text())
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(current.get(KEY) == committed.get(KEY), 'commit exact capability plan/result')
    return current


def environments():
    build = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                 TMPDIR=str(ROOT / 'tmp'), PYTHONDONTWRITEBYTECODE='1')
    native = dict(build, HOME='/home/yayoi', XDG_CACHE_HOME=str(ROOT / 'cache'),
                  LD_LIBRARY_PATH='/usr/lib/x86_64-linux-gnu:/usr/local/lib',
                  LD_DEBUG='libs,files', ZE_ENABLE_ALT_DRIVERS=str(NEO),
                  STRATA_METRIC_CAPABILITY_GATES='1', ZET_ENABLE_METRICS='1',
                  ZES_ENABLE_SYSMAN='0', NEOReadDebugKeys='1', EnableDirectSubmission='0',
                  UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', STRATA_SYCL_PREFILL_SAME_QUEUE='1')
    baseline = json.loads((P / INPUTS[-1]).read_text())['environment']
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys',
                'EnableDirectSubmission', 'STRATA_SYCL_PREFILL_SAME_QUEUE', 'ZES_ENABLE_SYSMAN'):
        require(native[key] == baseline[key], 'shared safety setting differs: ' + key)
    return build, native


def commands():
    return [
        [str(CC), *FLAGS, '-MD', '-MF', str(ROOT / 'actual.d'), '-MT', 'probe',
         '-c', str(SOURCE), '-o', str(ROOT / 'probe.o')],
        [str(CC), str(ROOT / 'probe.o'), str(LOADER), '-pie',
         '-Wl,-z,relro,-z,now', '-Wl,-t', '-o', str(ROOT / 'metric_capability')]]


def dependencies(path):
    body = path.read_text().replace('\\\n', ' ').split(':', 1)[1]
    require('\\' not in body, 'unexpected escaped dependency path')
    return {str(Path(name).resolve()) for name in body.split()}


def mappings(pid):
    result = set()
    for line in Path('/proc', str(pid), 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and 'x' in fields[1] and fields[5].startswith('/'):
            require(not fields[5].endswith(' (deleted)'), 'removed mapped executable')
            result.add(str(Path(fields[5]).resolve()))
    return result


def verify(row):
    require(pins(row['source_commit']) == row['sources'], 'source inputs changed')
    require(commands() == row['commands'] and list(environments()) == row['environments'],
            'compiler/environment recipe changed')
    manifest = ROOT / 'dependencies.json'
    require(digest(manifest) == row['dependencies']['sha256'], 'dependency manifest changed')
    external = json.loads(manifest.read_text())
    for name, expected in external['files'].items():
        require(digest(name) == expected, 'changed external input: ' + name)
    require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
            'driver/boot changed')
    return external


def prepare(record):
    require(KEY not in record and not ROOT.exists(), 'fresh one-use capability plan only')
    comparison = record['moe_split_accumulator']['clean_prefill_comparison']
    require(comparison['complete'] and comparison['ownership_closed'] and not comparison['active'],
            'close frozen model comparison first')
    revision = git('rev-parse', 'HEAD').decode().strip()
    sources = pins(revision)
    build_env, native_env = environments()
    driver = guards.driver_guard()
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache', 'preparation'):
        (ROOT / part).mkdir(mode=0o700)
    write(ROOT / 'preparation.json', dict(source_commit=revision, sources=sources))
    owner = None
    owner = supervision.Owner(ROOT / 'preparation', lambda: write(
        ROOT / 'preparation-execution.json', owner.commands if owner else []))

    def utility(label, argv, env=build_env):
        entry, out, err = owner.run(label, argv, env, P, **BUILD_LIMITS)
        require(supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                'preparation utility failed: ' + label)
        return out.read_text()

    # Preprocess only. Native compilation requires the generated plan's commit.
    utility('headers', [str(CC), *FLAGS, '-M', '-MF', str(ROOT / 'headers.d'),
                        '-MT', 'probe', str(SOURCE)])
    files = dependencies(ROOT / 'headers.d')
    runtime_roots = [LOADER, NEO, MD, ML, Path('/usr/local/lib/libigc.so.2'),
                     Path('/usr/local/lib/libigdfcl.so.2'), Path('/usr/local/lib/libigc-clang.so')]
    # Independently found optional dlopen edges in the installed NEO ELF.
    # Their presence in this closure is not evidence that initialization uses them.
    runtime_roots += [Path('/usr/lib/x86_64-linux-gnu') / name for name in
                      ('libigsc.so.1', 'libmetee.so.6.2.5.0', 'libnl-genl-3.so.200',
                       'libnuma.so.1', 'libva.so.2')]
    tools = [CC, Path('/usr/bin/as'), Path('/usr/bin/ld'), Path(sys.executable),
             Path('/usr/bin/journalctl')]
    for name in ('cc1', 'collect2'):
        tools.append(Path(utility('program-' + name, [str(CC), '-print-prog-name=' + name]).strip()))
    for name in ('Scrt1.o', 'crti.o', 'crtbeginS.o', 'crtendS.o', 'crtn.o',
                 'libgcc.a', 'libgcc_s.so', 'libc.so', 'libc_nonshared.a'):
        files.add(str(Path(utility('link-' + name, [str(CC), '-print-file-name=' + name]).strip()).resolve()))
    runtime_files = set()
    for i, path in enumerate([*runtime_roots, *tools]):
        path = path.resolve()
        files.add(str(path))
        with path.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'expected ELF input: ' + str(path))
        env = dict(build_env, LD_LIBRARY_PATH=native_env['LD_LIBRARY_PATH']) if i < len(runtime_roots) else build_env
        output = utility('elf-' + str(i), [str(ELF_LOADER), '--list', str(path)], env)
        require('not found' not in output, 'unresolved ELF dependency: ' + str(path))
        closure = {str(path)}
        closure.update(str(Path(s).resolve()) for s in re.findall(r'(/[^\s()]+)', output))
        files.update(closure)
        if i < len(runtime_roots):
            runtime_files.update(closure)
    files.update(str(p.resolve()) for p in tools)
    files.add(str(ELF_LOADER))
    files.update(str(p) for p in (Path('/etc/ld.so.cache'), Path('/usr/include/level_zero/ze_api.h'),
                                 Path('/usr/include/level_zero/zet_api.h')))
    require(all(Path(name).is_file() for name in files), 'missing compiler/provider input')
    manifest = dict(files={name: digest(name) for name in sorted(files)},
                    headers=sorted(dependencies(ROOT / 'headers.d')),
                    runtime_elfs=sorted(runtime_files))
    write(ROOT / 'dependencies.json', manifest)
    version = utility('compiler-version', [str(CC), '--version']).splitlines()[0]
    row = dict(stage='prepared', source_commit=revision, sources=sources, commands=commands(),
               compiler=version, environments=[build_env, native_env],
               dependencies=dict(path=str(ROOT / 'dependencies.json'), sha256=digest(ROOT / 'dependencies.json'),
                                 files=len(files), runtime_elfs=len(runtime_files)),
               driver=driver, boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               build_limits=BUILD_LIMITS, run_limits=RUN_LIMITS,
               scope='Direct-C property enumeration only;no application contexts,queues,group activation,queries or streamers.',
               criteria='Exact unique B570 PCI0000:05:00.0;three owned identity/environment/provider gates;all API success and complete group/member counts;exit0,closed owners,unchanged inputs and no fresh GPU fault. Empty enumeration,API errors,permission errors and unknown providers remain distinct. Enumeration is not counter-collection qualification.',
               protocol=dict(gates=['before-runtime', 'runtime-ready', 'before-exit'],
                             line_deadline_s=15, exit_deadline_s=10, maximum_records=13000,
                             strings='Reconstruct API bytes via latin1;strict UTF8 or explicit raw_hex.'),
               retention='One small tool build and bounded raw inventory for Main to choose the next source-documented tiny-workload metric. Review/retire after that choice and collection plan/result;no new model generation.',
               active=False, ownership_closed=True, passed=False, collection_qualified=False,
               hardware_utilization_measured=False, physical_pcie_bytes_measured=False)
    write(ROOT / 'plan.json', row)
    record[KEY] = row
    write(RESULT, record)
    print('Prepared direct-C recipe and external closure;commit before build.', flush=True)


def build(record):
    row = record[KEY]
    require(row['stage'] == 'prepared' and row == json.loads((ROOT / 'plan.json').read_text()),
            'original committed preparation required')
    external = verify(row)
    require(not (ROOT / 'build').exists(), 'no implicit build retry')
    owner = None
    owner = supervision.Owner(ROOT / 'build', lambda: write(
        ROOT / 'build-execution.json', owner.commands if owner else []))
    try:
        for i, command in enumerate(row['commands']):
            entry, out, err = owner.run('build-' + str(i), command, row['environments'][0], P, **BUILD_LIMITS)
            require(supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                    'compile/link failed')
            if i == 0:
                require(dependencies(ROOT / 'actual.d') == set(external['headers']), 'actual header closure differs')
            else:
                linked = {str(Path(line).resolve()) for line in out.read_text().splitlines() if line.startswith('/')}
                require(linked and linked <= set(external['files']) | {str(ROOT / 'probe.o')},
                        'unfrozen link input: ' + repr(linked - set(external['files']) - {str(ROOT / 'probe.o')}))
        verify(row)
        binary = ROOT / 'metric_capability'
        row.update(stage='built', binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=digest(binary)),
                   build_passed=True)
    except BaseException as error:
        row.update(stage='build-failed', build_passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        row['active'] = owner.active is not None
        write(ROOT / 'build-execution.json', owner.commands)
        write(RESULT, record)
    print(json.dumps({k: row.get(k) for k in ('stage', 'build_passed', 'ownership_closed', 'error')}), flush=True)
    require(row.get('build_passed') and row['ownership_closed'], 'direct-C build failed')


def api_strings(item):
    result = copy.deepcopy(item)
    if item['kind'] in ('device', 'group', 'metric'):
        for key in ('name', 'description', 'component', 'units'):
            if key not in item:
                continue
            raw = item[key].encode('latin1')
            try:
                result[key] = raw.decode('utf-8', errors='strict')
            except UnicodeDecodeError:
                result[key] = dict(raw_hex=raw.hex(), utf8=None)
    return result


def run(record):
    row = record[KEY]
    require(row['stage'] == 'built' and row['build_passed'] and row['ownership_closed'], 'commit successful build first')
    external = verify(row)
    require(digest(row['binary']['path']) == row['binary']['sha256'], 'actual executable changed')
    require(not (ROOT / 'native').exists(), 'no implicit native retry')
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
                                          text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in cursor_text.splitlines() if line.startswith('-- cursor: '))
    items, gates = [], []
    owner = None
    owner = supervision.Owner(ROOT / 'native', lambda: write(
        ROOT / 'native-owner.json', owner.commands if owner else []))
    audit = []
    try:
        def observe(io, entry):
            while True:
                try:
                    line = io.line(seconds=row['protocol']['line_deadline_s'])
                except RuntimeError as error:
                    if io.eof and str(error) == 'protocol EOF before expected line':
                        io.exit(row['protocol']['exit_deadline_s'])
                        break
                    raise
                item = json.loads(line)
                require(len(items) < row['protocol']['maximum_records'], 'record count cap')
                require(item['kind'] in ('schema', 'api', 'device', 'group', 'metric', 'complete', 'refusal', 'gate'),
                        'unexpected record type')
                items.append(api_strings(item))
                if item['kind'] != 'gate':
                    continue
                stage, pid = item['stage'], item['pid']
                require(len(gates) < 3 and stage == row['protocol']['gates'][len(gates)] and pid == entry['pid'],
                        'gate order/owned PID mismatch')
                require(str(Path('/proc', str(pid), 'exe').resolve()) == row['binary']['path'] and
                        digest(Path('/proc', str(pid), 'exe')) == row['binary']['sha256'], 'native executable mismatch')
                environment = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if v)
                require(environment == {k.encode(): v.encode() for k, v in row['environments'][1].items()},
                        'immutable pre-exec environment differs')
                mapped = mappings(pid)
                allowed = set(external['runtime_elfs']) | {row['binary']['path']}
                require(mapped <= allowed and str(LOADER) in mapped, 'unknown runtime provider: ' + repr(mapped - allowed))
                require(stage == 'before-runtime' or str(NEO) in mapped, 'target NEO provider absent')
                for path in mapped:
                    expected = row['binary']['sha256'] if path == row['binary']['path'] else external['files'][path]
                    require(digest(path) == expected, 'mapped provider bytes differ: ' + path)
                audit.append(dict(stage=stage, pid=pid, code_maps=sorted(mapped)))
                gates.append(stage)
                io.send(b'continue\n')
                if stage == 'before-exit':
                    io.exit(row['protocol']['exit_deadline_s'])
                    break

        entry, out, err = owner.run('inventory', [row['binary']['path']], row['environments'][1], P,
                                    interaction=observe, **RUN_LIMITS)
        require(supervision.closed(entry), 'native supervision/ownership did not close')
        # Include dynamically initialized-and-unloaded providers, not just maps at gates.
        initialized = {str(Path(s).resolve()) for s in re.findall(r'calling (?:init|fini):\s+(/\S+)', err.read_text())}
        require(initialized <= set(external['runtime_elfs']) | {row['binary']['path']},
                'unfrozen transient provider: ' + repr(initialized - set(external['runtime_elfs']) - {row['binary']['path']}))
        errors = [v for v in items if v['kind'] == 'api' and v['result'] != 0]
        names = {int(value, 0): name for name, value in re.findall(
            r'\b(ZE_RESULT_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)',
            Path('/usr/include/level_zero/ze_api.h').read_text())}
        for error in errors:
            error['result_name'] = names.get(error['result'], 'UNKNOWN_RAW_RESULT')
        refusals = [v for v in items if v['kind'] == 'refusal']
        complete = [v for v in items if v['kind'] == 'complete']
        groups = [v for v in items if v['kind'] == 'group']
        metrics = [v for v in items if v['kind'] == 'metric']
        devices = [v for v in items if v['kind'] == 'device']
        row.update(native_exit_code=entry['exit_code'], api_errors=errors, refusals=refusals)
        if errors or refusals:
            require(entry['exit_code'] == (10 if errors else 11) and not (errors and refusals),
                    'API/refusal exit differs')
            outcome = 'api-error' if errors else 'native-refusal'
            if any(v['result_name'] == 'ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS' for v in errors):
                outcome = 'permission-denied'
            elif any(v['result_name'] == 'ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE' for v in errors):
                outcome = 'dependency-unavailable'
            elif any(v['result_name'] == 'ZE_RESULT_ERROR_UNSUPPORTED_FEATURE' for v in errors):
                outcome = 'unsupported'
            row.update(stage='closed', outcome=outcome, enumeration_complete=False)
        else:
            require(gates == row['protocol']['gates'] and entry['exit_code'] == 0 and len(complete) == len(devices) == 1,
                    'incomplete successful enumeration protocol')
            require(len(groups) == complete[0]['groups'] and len(metrics) == complete[0]['metrics'], 'inventory totals differ')
            require([g['group'] for g in groups] == list(range(len(groups))), 'missing or duplicate group')
            for group in groups:
                members = [v['index'] for v in metrics if v['group'] == group['group']]
                require(members == list(range(group['metric_count'])), 'missing/duplicate metric member')
            require(all(v['group'] in range(len(groups)) for v in metrics), 'unrecognized metric group')
            row.update(stage='closed', outcome='enumerated' if groups else 'enumerated-empty',
                       enumeration_complete=True, groups=groups, metrics=len(metrics), device=devices[0])
        inventory = ROOT / 'inventory.json'
        write(inventory, [v for v in items if v['kind'] in ('device', 'group', 'metric', 'complete')])
        row['inventory'] = dict(path=str(inventory), sha256=digest(inventory), bytes=inventory.stat().st_size,
                                consumer='Choose one source-documented tiny-workload counter;retire full inventory after plan/result closes.')
        row['provider_gates'] = [dict(stage=a['stage'], pid=a['pid'], code_maps=len(a['code_maps']),
                                     metrics_discovery=str(MD) in a['code_maps'], metrics_library=str(ML) in a['code_maps']) for a in audit]
        write(ROOT / 'provider-gates.json', audit)
        row['observed_runtime_elfs'] = sorted(set().union(*(set(a['code_maps']) for a in audit), initialized))
        require(digest(row['binary']['path']) == row['binary']['sha256'], 'binary changed during inventory')
        verify(row)
    except BaseException as error:
        row.update(stage='failed', outcome='inconclusive', error=repr(error), enumeration_complete=False)
    finally:
        faults = []
        try:
            journal = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor,
                                               '--no-pager', '-o', 'json'], timeout=15)
            require(len(journal) <= 2 << 20, 'kernel journal evidence budget')
            messages = [json.loads(line).get('MESSAGE', '') for line in journal.decode().splitlines() if line]
            faults = [s for s in messages if re.search(r'(?:\bxe\b|drm|GPU).*(?:fault|hang|wedg|reset|timed out)', s, re.I)]
            row['driver_after'] = guards.driver_guard()
        except BaseException as error:
            row.update(stage='failed', fault_evidence_error=repr(error), enumeration_complete=False)
        row.update(fresh_faults=faults, active=owner.active is not None,
                   ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands))
        row['passed'] = bool(row.get('enumeration_complete') and not faults and row['ownership_closed'])
        row['native_process'] = {k: owner.commands[0].get(k) for k in
                                ('pid', 'exit_code', 'elapsed_seconds', 'peak_session_rss_bytes',
                                 'observation_complete', 'direct_child_reaped', 'session_empty', 'error')} if owner.commands else {}
        write(ROOT / 'native-execution.json', dict(commands=owner.commands, items=items))
        write(RESULT, record)
    print(json.dumps({k: row.get(k) for k in ('stage', 'outcome', 'passed', 'metrics', 'ownership_closed', 'error')}), flush=True)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'run'), 'choose prepare,build,run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        record = public()
        fresh = not ROOT.exists()
        try:
            {'prepare': prepare, 'build': build, 'run': run}[sys.argv[1]](record)
        except BaseException as error:
            if sys.argv[1] == 'prepare' and fresh and (ROOT / 'preparation.json').exists():
                row = json.loads((ROOT / 'preparation.json').read_text())
                path = ROOT / 'preparation-execution.json'
                entries = json.loads(path.read_text()) if path.exists() else []
                row.update(stage='preparation-failed', error=repr(error), active=False,
                           ownership_closed=bool(entries) and all(supervision.closed(c) for c in entries),
                           passed=False, GPU_executed=False)
                record[KEY] = row
                write(RESULT, record)
            raise


if __name__ == '__main__':
    main()
