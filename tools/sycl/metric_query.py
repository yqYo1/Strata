#!/usr/bin/env python3
"""Qualify direct-C metric queries on bounded exact integer work.

Independently authored under repository LICENSE. Prepare, commit, build,
commit, run. Reuse the verified inventory's system provider closure, never
the model's SYCL allowlist. No retries, reset, clock or permission changes.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import metric_capability as capability
import measurement_guards as guards
import measurement_owner as owner_api

P, B, RESULT = capability.P, capability.B, capability.RESULT
ROOT = B / 'metric-query-c-v1'
KEY = 'metric_query'
SDK = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.tools/intel-llvm-v711/install')
CLANG = (SDK / 'bin/clang').resolve()
TRANSLATOR = (SDK / 'bin/llvm-spirv').resolve()
SOURCE = P / 'bench/sycl/metric_query.c'
KERNEL = P / 'bench/sycl/metric_integer.cl'
INPUTS = ('bench/sycl/metric_query.c', 'bench/sycl/metric_integer.cl',
          'tools/sycl/metric_query.py', *capability.INPUTS)
INPUTS = tuple(dict.fromkeys(INPUTS))
CL_FLAGS = ['--no-default-config', '-target', 'spir64', '-x', 'cl',
            '-cl-std=CL2.0', '-Xclang', '-finclude-default-header', '-O2']
LIMITS = dict(wall=90, cpu=60, rss_cap=2 << 30, text_cap=8 << 20,
              file_cap=8 << 20, total_cap=32 << 20)
MODES = ['work-off', 'work-query', 'empty-query']
GATES = ['before-runtime', 'runtime-ready', 'before-work', 'before-exit']
require, digest, write, git = guards.require, owner_api.sha, capability.write, capability.git


def public():
    record = json.loads(RESULT.read_text())
    committed = json.loads(git('show', 'HEAD:' + str(RESULT.relative_to(P))))
    for key in (KEY, capability.KEY):
        require(record.get(key) == committed.get(key), 'commit exact query/inventory evidence: ' + key)
    return record


def sources(revision):
    pins = {}
    for name in INPUTS:
        raw = git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit query input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def environments():
    build, native = capability.environments()
    build, native = dict(build), dict(native)
    build['TMPDIR'] = native['TMPDIR'] = str(ROOT / 'tmp')
    native['XDG_CACHE_HOME'] = str(ROOT / 'cache')
    del native['STRATA_METRIC_CAPABILITY_GATES']
    native['STRATA_METRIC_QUERY_GATES'] = '1'
    return [build, native]


def commands():
    return [
        [str(capability.CC), *capability.FLAGS, '-MD', '-MF', str(ROOT / 'actual-c.d'),
         '-MT', 'query', '-c', str(SOURCE), '-o', str(ROOT / 'query.o')],
        [str(capability.CC), str(ROOT / 'query.o'), str(capability.LOADER), '-pie',
         '-Wl,-z,relro,-z,now', '-Wl,-t', '-o', str(ROOT / 'metric_query')],
        [str(CLANG), *CL_FLAGS, '-emit-llvm', '-MD', '-MF', str(ROOT / 'actual-cl.d'),
         '-MT', 'module', '-c', str(KERNEL), '-o', str(ROOT / 'work.bc')],
        [str(TRANSLATOR), '--spirv-max-version=1.2', str(ROOT / 'work.bc'), '-o', str(ROOT / 'work.spv')]]


def verify(record):
    row = record[KEY]
    capability.verify(record[capability.KEY])
    require(row['sources'] == sources(row['source_commit']), 'query source changed')
    require(row['commands'] == commands() and row['environments'] == environments(), 'query recipe/env changed')
    require(guards.driver_guard() == row['driver'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'query boot/driver changed')
    for field in ('dependencies', 'selection'):
        require(digest(row[field]['path']) == row[field]['sha256'], field + ' changed')
    external = json.loads(Path(row['dependencies']['path']).read_text())
    for name, expected in external['files'].items():
        require(digest(name) == expected, 'query dependency changed: ' + name)
    return external


def prepare(record):
    require(not ROOT.exists() and KEY not in record, 'no implicit query preparation retry')
    reference = record[capability.KEY]
    require(reference['passed'] and reference['ownership_closed'] and not reference['active'], 'closed inventory required')
    base = capability.verify(reference)
    require(digest(reference['inventory']['path']) == reference['inventory']['sha256'], 'inventory bytes changed')
    inventory = json.loads(Path(reference['inventory']['path']).read_text())
    groups = [x for x in inventory if x['kind'] == 'group' and x['name'] == 'ComputeBasic' and
              x['sampling_flags'] == 1 and x['domain'] == 3 and x['metric_count'] == 83]
    require(len(groups) == 1, 'unique actual event group required')
    selected = [{k: x[k] for k in ('index', 'name', 'units', 'result_type')} for x in inventory
                if x['kind'] == 'metric' and x['group'] == groups[0]['group']]
    require([x['index'] for x in selected] == list(range(83)), 'complete ordered selection')
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache', 'preparation'):
        (ROOT / part).mkdir(mode=0o700)
    write(ROOT / 'selection.json', selected)
    supervisor = None
    supervisor = owner_api.Owner(ROOT / 'preparation', lambda: write(ROOT / 'preparation-execution.json', supervisor.commands))

    def utility(label, argv):
        entry, out, err = supervisor.run(label, argv, environments()[0], P, **LIMITS)
        require(owner_api.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                'query preparation utility failed: ' + label)
        return out.read_text()

    utility('headers-c', [str(capability.CC), *capability.FLAGS, '-M', '-MF', str(ROOT / 'headers-c.d'),
                          '-MT', 'query', str(SOURCE)])
    utility('headers-cl', [str(CLANG), *CL_FLAGS, '-M', '-MF', str(ROOT / 'headers-cl.d'), '-MT', 'module', str(KERNEL)])
    files = set(base['files'])
    headers = {kind: sorted(capability.dependencies(ROOT / ('headers-' + kind + '.d'))) for kind in ('c', 'cl')}
    for paths in headers.values():
        files.update(paths)
    for i, path in enumerate((CLANG, TRANSLATOR)):
        with path.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'compiler/translator ELF required')
        output = utility('elf-' + str(i), [str(capability.ELF_LOADER), '--list', str(path)])
        require('not found' not in output, 'compiler/translator dependency absent')
        files.add(str(path))
        files.update(str(Path(s).resolve()) for s in re.findall(r'(/[^\s()]+)', output))
    manifest = dict(files={name: digest(name) for name in sorted(files)}, headers=headers,
                    runtime_elfs=base['runtime_elfs'])
    write(ROOT / 'dependencies.json', manifest)
    row = dict(stage='prepared', source_commit=revision, sources=pins, commands=commands(),
               environments=environments(), limits=LIMITS, modes=MODES, gates=GATES,
               driver=guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               dependencies=dict(path=str(ROOT / 'dependencies.json'), sha256=digest(ROOT / 'dependencies.json'), files=len(files)),
               selection=dict(path=str(ROOT / 'selection.json'), sha256=digest(ROOT / 'selection.json'), count=83,
                              name='ComputeBasic', sampling_flag=1, domain=3),
               inventory_source_commit=reference['source_commit'],
               criteria='Fresh work-off must pass before work-query;then empty-query. Exact65536 integer payload and128 guards;four immutable environment/executable/provider gates;completed begin/work/end,sync,raw read,typed calculation;safe query/pool/group/context cleanup;exit0,all owned processes closed,no fresh fault. A refused query stops further counter arms. Null/nonfinite values remain unqualified.',
               limitations='Collection qualification only. One tiny kernel is not an observer-overhead estimate,hardware ceiling,process-attribution proof,physicalPCIe measurement or model speed/quality evidence. Empty query is not globally idle.',
               provenance='Independent Main C/OpenCL code;installed official declarations. SDK supplies independent Intel LLVM compiler/translator only;no XeStrata implementation used.',
               retention='Tiny fixture/tool for imminent collection and SYCL/PTI interoperability decision. Retire successful raw API/loader logs after extracting ordered counter evidence;keep original failures. No new model generation.',
               active=False, ownership_closed=True, passed=False, collection_qualified=False)
    write(ROOT / 'plan.json', row)
    record[KEY] = row; write(RESULT, record)
    print('Query prepared;commit the plan before building.', flush=True)


def build(record):
    row = record[KEY]
    require(row['stage'] == 'prepared' and row == json.loads((ROOT / 'plan.json').read_text()), 'commit frozen preparation')
    external = verify(record)
    require(not (ROOT / 'build').exists(), 'no implicit query build retry')
    supervisor = None
    supervisor = owner_api.Owner(ROOT / 'build', lambda: write(ROOT / 'build-execution.json', supervisor.commands))
    try:
        for i, command in enumerate(row['commands']):
            entry, out, err = supervisor.run('build-' + str(i), command, row['environments'][0], P, **LIMITS)
            require(owner_api.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size, 'query build failed: ' + str(i))
            if i in (0, 2):
                kind = 'c' if i == 0 else 'cl'
                require(capability.dependencies(ROOT / ('actual-' + kind + '.d')) == set(external['headers'][kind]),
                        'query actual headers changed: ' + kind)
            if i == 1:
                linked = {str(Path(s).resolve()) for s in out.read_text().splitlines() if s.startswith('/')}
                require(linked and linked <= set(external['files']) | {str(ROOT / 'query.o')}, 'unfrozen query link inputs')
        verify(record)
        row.update(stage='built', build_passed=True, artifacts={name: dict(path=str(ROOT / name),
                   sha256=digest(ROOT / name), bytes=(ROOT / name).stat().st_size) for name in ('metric_query', 'work.bc', 'work.spv')})
    except BaseException as error:
        row.update(stage='build-failed', build_passed=False, error=repr(error))
    finally:
        row.update(active=supervisor.active is not None,
                   ownership_closed=supervisor.active is None and all(owner_api.closed(c) for c in supervisor.commands))
        write(RESULT, record)
    print(json.dumps({k: row.get(k) for k in ('stage', 'build_passed', 'ownership_closed', 'error')}), flush=True)
    require(row.get('build_passed') and row['ownership_closed'], 'query build failed')


def run(record):
    row = record[KEY]
    require(row['stage'] == 'built' and row['build_passed'] and row['ownership_closed'], 'commit successful query build')
    external = verify(record)
    for item in row['artifacts'].values():
        require(digest(item['path']) == item['sha256'], 'query artifact changed')
    require(not (ROOT / 'native').exists(), 'no implicit query execution retry')
    selection = json.loads(Path(row['selection']['path']).read_text())
    cursor_output = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(s.split(': ', 1)[1] for s in cursor_output.splitlines() if s.startswith('-- cursor: '))
    supervisor = None
    supervisor = owner_api.Owner(ROOT / 'native', lambda: write(ROOT / 'native-owner.json', supervisor.commands))
    samples = row['samples'] = []
    audits = []
    try:
        for mode in row['modes']:
            items, gates = [], []

            def observe(io, entry):
                while True:
                    try:
                        item = json.loads(io.line(seconds=25))
                    except RuntimeError as error:
                        if io.eof and str(error) == 'protocol EOF before expected line':
                            io.exit(10); break
                        raise
                    require(len(items) < 4000 and item['kind'] in ('schema', 'api', 'refusal', 'gate', 'device',
                            'property', 'launch', 'collection', 'value', 'complete'), 'query protocol bound/type')
                    items.append(item)
                    if item['kind'] != 'gate':
                        continue
                    stage, pid = item['stage'], item['pid']
                    require(len(gates) < len(GATES) and stage == GATES[len(gates)] and pid == entry['pid'], 'query gate/PID')
                    binary = row['artifacts']['metric_query']
                    proc = Path('/proc', str(pid))
                    require(str((proc / 'exe').resolve()) == binary['path'] and digest(proc / 'exe') == binary['sha256'], 'query executable')
                    environment = dict(v.split(b'=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if v)
                    require(environment == {k.encode(): v.encode() for k, v in row['environments'][1].items()}, 'query pre-exec environment')
                    mapped = capability.mappings(pid)
                    allowed = set(external['runtime_elfs']) | {binary['path']}
                    require(mapped <= allowed and str(capability.LOADER) in mapped, 'unfrozen query runtime: ' + repr(mapped - allowed))
                    require(stage == 'before-runtime' or str(capability.NEO) in mapped, 'query target provider absent')
                    for name in mapped:
                        expected = binary['sha256'] if name == binary['path'] else external['files'][name]
                        require(digest(name) == expected, 'query mapped provider changed: ' + name)
                    if stage == 'before-work':
                        require([{k: x[k] for k in ('index', 'name', 'units', 'result_type')} for x in items if x['kind'] == 'property'] == selection,
                                'actual query property order/name/unit/type differs')
                    gates.append(stage); audits.append(dict(mode=mode, stage=stage, pid=pid, maps=sorted(mapped)))
                    io.send(b'continue\n')
                    if stage == 'before-exit':
                        io.exit(10); break

            entry, out, err = supervisor.run(mode, [row['artifacts']['metric_query']['path'], mode,
                row['artifacts']['work.spv']['path']], row['environments'][1], P, interaction=observe, **LIMITS)
            write(ROOT / (mode + '-items.json'), items)
            errors = [x for x in items if x['kind'] == 'api' and x['result'] != 0]
            refusals = [x for x in items if x['kind'] == 'refusal']
            completed = [x for x in items if x['kind'] == 'complete']
            values = [x for x in items if x['kind'] == 'value']
            initialized = {str(Path(s).resolve()) for s in re.findall(r'calling (?:init|fini):\s+(/\S+)', err.read_text())}
            require(initialized <= set(external['runtime_elfs']) | {row['artifacts']['metric_query']['path']},
                    'unfrozen transient query provider: ' + repr(initialized - set(external['runtime_elfs'])))
            passed = bool(owner_api.closed(entry) and entry['exit_code'] == 0 and not errors and not refusals and
                          gates == GATES and len(completed) == 1 and completed[0]['work_checked'] and completed[0]['cleanup_complete'])
            if mode != 'work-off':
                collection = [x for x in items if x['kind'] == 'collection']
                passed = passed and len(collection) == 1 and bool(values) and all(x['value'] is not None for x in values)
                if passed:
                    require(len(values) == collection[0]['values'] and
                            [(x['set'], x['index'], x['type']) for x in values] ==
                            [(s, p['index'], p['result_type']) for s in range(collection[0]['sets']) for p in selection],
                            'query calculated order/count/type')
            else:
                require(not values and not any(x['kind'] == 'collection' for x in items), 'observer-off unexpectedly collected')
            sample = dict(mode=mode, passed=passed, exit_code=entry['exit_code'], errors=errors, refusals=refusals,
                          gates=gates, process={k: entry.get(k) for k in ('pid', 'elapsed_seconds', 'peak_session_rss_bytes',
                          'observation_complete', 'direct_child_reaped', 'session_empty', 'error')},
                          values=values, collection=[x for x in items if x['kind'] == 'collection'], complete=completed)
            samples.append(sample); write(RESULT, record)
            require(owner_api.closed(entry), 'query ownership/observation not closed')
            guards.driver_guard()
            journal = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], timeout=15)
            require(len(journal) <= 2 << 20, 'query inter-arm kernel-journal cap')
            require(not any(re.search(r'(?:\bxe\b|drm|GPU).*(?:fault|hang|wedg|reset|timed out)',
                            json.loads(s).get('MESSAGE', ''), re.I) for s in journal.decode().splitlines() if s),
                    'fresh driver fault; stop further query arms')
            if not passed:
                break
        verify(record)
        for item in row['artifacts'].values():
            require(digest(item['path']) == item['sha256'], 'query artifact changed during execution')
        row.update(stage='closed', passed=len(samples) == len(MODES) and all(x['passed'] for x in samples))
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        try:
            journal = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'], timeout=15)
            require(len(journal) <= 2 << 20, 'query kernel-journal cap')
            messages = [json.loads(s).get('MESSAGE', '') for s in journal.decode().splitlines() if s]
            faults = [s for s in messages if re.search(r'(?:\bxe\b|drm|GPU).*(?:fault|hang|wedg|reset|timed out)', s, re.I)]
            row.update(fresh_faults=faults, driver_after=guards.driver_guard())
        except BaseException as error:
            row.update(passed=False, fault_evidence_error=repr(error))
        row.update(active=supervisor.active is not None,
                   ownership_closed=supervisor.active is None and all(c.get('direct_child_reaped') and c.get('session_empty') and
                   not c.get('survivors') and c.get('ownership_status') == 'observed' for c in supervisor.commands),
                   observation_complete=all(owner_api.closed(c) for c in supervisor.commands))
        row['collection_qualified'] = bool(row.get('passed') and not row.get('fresh_faults') and
                                          row['ownership_closed'] and row['observation_complete'])
        row['passed'] = row['collection_qualified']
        write(ROOT / 'provider-gates.json', audits)
        write(RESULT, record)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'collection_qualified', 'ownership_closed', 'error')}), flush=True)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'run'), 'choose prepare,build,run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        record = public()
        if sys.argv[1] == 'prepare':
            try:
                prepare(record)
            except BaseException as error:
                execution = ROOT / 'preparation-execution.json'
                entries = json.loads(execution.read_text()) if execution.exists() else []
                record[KEY] = dict(stage='preparation-failed', passed=False, error=repr(error), GPU_executed=False,
                                   source_commit=git('rev-parse', 'HEAD').decode().strip(),
                                   ownership_closed=all(owner_api.closed(c) for c in entries),
                                   active=any(not c.get('direct_child_reaped') for c in entries))
                write(RESULT, record)
                raise
        else:
            {'build': build, 'run': run}[sys.argv[1]](record)


if __name__ == '__main__':
    main()
