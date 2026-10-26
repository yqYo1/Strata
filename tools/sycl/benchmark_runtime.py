#!/usr/bin/env python3
"""Bounded Python-runtime admission for Main's model controls; repository LICENSE.

The probe imports entrypoint definitions and compiles the chat template only.
It never calls their main functions, starts a server, or initializes an engine.
"""
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

P = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
if str(SOURCE.parent) not in sys.path: sys.path.insert(0, str(SOURCE.parent))
import measurement_owner as ownership

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HOST = B / 'benchmark-python-admission-v2'
SCOPE = 'benchmark_python_runtime_v2'
require = ownership.require


def file_pin(path):
    path = Path(path); before = path.stat()
    value = dict(realpath=str(path.resolve()), dev=before.st_dev, ino=before.st_ino,
        bytes=before.st_size, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)
    value['sha256'] = ownership.sha(path)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(after, k) == getattr(before, k) for k in fields), 'runtime file changed during digest: ' + str(path))
    return value


def selection(env):
    preferred = P / '.venv/bin/python'
    selected = str(preferred) if os.access(preferred, os.X_OK) else shutil.which('python3', path=env['PATH'])
    require(selected is not None, 'benchmark interpreter absent')
    return dict(workspace_venv=os.access(preferred, os.X_OK), path=selected, file=file_pin(selected))


def code_maps(pid):
    paths = set()
    for line in Path('/proc', str(pid), 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or 'x' not in fields[1] or not fields[5].startswith('/'): continue
        require(not fields[5].endswith(' (deleted)'), 'deleted runtime mapping')
        paths.add(str(Path(fields[5]).resolve()))
    return sorted(paths)


def inventory(trees, explicit):
    """Include lazy imports and directory changes, with finite path/byte budgets."""
    paths = set(map(str, explicit)); states = {}; directories = set(); visited = 0
    for text in trees:
        root = Path(text); states[text] = dict(exists=root.exists(), realpath=str(root.resolve()))
        if not root.exists(): continue
        require(root.is_dir(), 'runtime search directory changed type')
        for path in root.rglob('*'):
            if '__pycache__' in path.parts: continue  # Private empty cache prefix is required.
            visited += 1; require(visited <= 32768, 'runtime traversal budget')
            require(not (path.is_symlink() and path.is_dir()), 'unqualified runtime directory symlink')
            if path.is_dir(): directories.add(str(path))
            if path.is_file(): paths.add(str(path))
            require(len(paths) <= 16384, 'runtime inventory path budget')
    files = {}; total = 0
    for text in sorted(paths):
        pin = file_pin(text); total += pin['bytes']
        require(total <= 512 << 20, 'runtime inventory byte budget')
        files[text] = pin
    return dict(trees=states, directories=sorted(directories), files=files, bytes=total)


def validate_manifest(manifest):
    require(inventory(list(manifest['inventory']['trees']), manifest['explicit']) == manifest['inventory'],
        'benchmark Python/runtime inventory changed')


def require_environment(env):
    require(env.get('PYTHONNOUSERSITE') == env.get('PYTHONDONTWRITEBYTECODE') == '1' and
        not env.get('PYTHONPATH') and not env.get('PYTHONHOME'), 'unqualified Python startup environment')
    cache = Path(env['PYTHONPYCACHEPREFIX'])
    require(cache.is_absolute() and cache.is_relative_to(B) and
        (not cache.exists() or not any(cache.rglob('*'))), 'private empty Python cache prefix required')


def check_running(pid, script, selected, env, allowed_maps):
    require(selection(env) == selected, 'live benchmark interpreter selection changed')
    before = ownership.identity(pid); require(before is not None and before['state'] != 'Z', 'Python child absent')
    require(str(Path('/proc', str(pid), 'exe').resolve()) == selected['file']['realpath'] and
        ownership.sha(Path('/proc', str(pid), 'exe')) == selected['file']['sha256'], 'actual Python interpreter differs')
    args = [v.decode() for v in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if v]
    require(len(args) >= 2, 'Python script argv absent')
    invoked = Path(args[1]); invoked = invoked if invoked.is_absolute() else P / invoked
    require(invoked.resolve() == Path(script).resolve(), 'actual Python script differs')
    executable = args[0]
    if '/' not in executable: executable = shutil.which(executable, path=env['PATH'])
    elif not Path(executable).is_absolute(): executable = str(P / executable)
    require(executable and str(Path(executable).absolute()) == str(Path(selected['path']).absolute()) and
        str(Path(executable).resolve()) == selected['file']['realpath'], 'Python argv/venv resolution differs')
    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
    require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
        set(actual) - {k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
        'actual Python environment differs;only shell PWD/OLDPWD/SHLVL/_ additions allowed')
    require(set(code_maps(pid)) <= set(allowed_maps), 'unqualified Python code mapping')
    after = ownership.identity(pid)
    require(after and all(after[k] == before[k] for k in ('pid', 'start_ticks', 'sid')), 'Python identity changed during check')
    return {k: before[k] for k in ('pid', 'start_ticks', 'sid')}


def probe(entry, template):
    import importlib.metadata
    import site
    entry = Path(entry)
    require(entry in (P / 'tools/strata_bench.py', P / 'serve/server.py'), 'probe entrypoint not declared')
    require_environment(os.environ)
    sys.path[0] = str(entry.parent)
    sys.argv = [str(entry)]
    namespace = runpy.run_path(str(entry), run_name='_strata_runtime_definition_probe')
    if entry.name == 'server.py': namespace['ChatTemplate'](Path(template))
    require(sys.flags.dont_write_bytecode == 1 and sys.flags.no_user_site == 1 and
        not sys.flags.no_site and site.ENABLE_USER_SITE is False, 'actual Python startup flags differ')
    versions = {name: importlib.metadata.version(name) for name in ('jinja2', 'MarkupSafe')}
    files = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file(): files.add(str(Path(filename).resolve()))
    print(json.dumps(dict(executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
        search_path=[str(Path(v or os.getcwd()).resolve()) for v in sys.path], modules=sorted(files),
        code_maps=code_maps(os.getpid()), versions=versions)), flush=True)


def host_action(phase):
    import fcntl
    import supervision_model as control
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); control.runtime.guards.no_model_owners([])
        control.qualification()
        previous = control.runtime.validation.committed(control.RESULT,
            ('historical_regression_audit', 'supervision_model_control_v3'))
        require(previous['stage'] == 'closed-unlaunched' and not previous['native_launched'], 'preserve original runtime refusal')
        seed = dict(previous, environment_template=control.gen.committed(control.TEMPLATE)['environment_template'])
        env = control.environment(seed, HOST); require_environment(env); chosen = selection(env)
        if phase == 'prepare':
            require(not HOST.exists(), 'single-use Python runtime discovery')
            source = control.gen.git('rev-parse', 'HEAD').decode().strip(); pins = control.gen.benchmark_pins(source)
            HOST.mkdir(mode=0o700); (HOST / 'cache').mkdir()
            state = dict(stage='discovering', source_commit=source, benchmark_inputs=pins, selected=chosen,
                complete=False, passed=False, commands=[], active=False, GPU_executed=False)
            manifest = None
        else:
            state = json.loads((HOST / 'record.json').read_text())
            committed = control.runtime.validation.committed(control.RESULT, ('historical_regression_audit', SCOPE))
            require(state['stage'] == 'prepared' and state['ownership_closed'] and not state['active'] and
                all(ownership.closed(v) for v in state['commands']) and committed == compact(state),
                'commit exact closed-discovery host runtime plan before qualification')
            source = state['source_commit']; pins = control.gen.benchmark_pins(source)
            require(pins == state['benchmark_inputs'] and chosen == state['selected'] and
                control.identity(env) == state['plan']['environment_identity'], 'host runtime inputs changed')
            manifest = read_manifest(state['plan']['manifest']); validate_manifest(manifest)
        require(chosen['file']['realpath'] == str(Path(sys.executable).resolve()), 'host-qualified Python core required')
        owner = None
        def persist():
            state.update(active=bool(owner and owner.active), commands=owner.commands if owner else [])
            control.write(HOST / 'record.json', state)
        descriptors = {}; error = None
        try:
            owner = ownership.Owner(HOST / ('discovery' if phase == 'prepare' else 'qualified'), persist)
            template = Path(previous['configuration']['tokenizer']) / 'chat_template.jinja'
            original_assets = json.loads(control.gen.git('show', '1e676d554:' + str(control.TEMPLATE.relative_to(P))))['model_assets']
            require(ownership.sha(template) == original_assets[str(template)]['sha256'], 'declared template changed')
            for role, entry in [('benchmark', P / 'tools/strata_bench.py'), ('server', P / 'serve/server.py')]:
                control.gen.benchmark_pins(source); require(selection(env) == chosen, 'interpreter selection changed')
                if manifest is not None: validate_manifest(manifest)
                command = [chosen['path'], str(SOURCE), 'probe', str(entry), str(template)]
                result, out, err = owner.run(role, command, env, P, wall=30, cpu=20, rss_cap=256 << 20,
                    text_cap=256 << 10, file_cap=1 << 20, total_cap=4 << 20)
                require(ownership.closed(result) and result['exit_code'] == 0 and not err.stat().st_size, 'Python definition probe failed')
                descriptors[role] = json.loads(out.read_text())
                if manifest is not None:
                    require(descriptors[role] == manifest['descriptors'][role], 'qualified runtime probe differs from frozen discovery')
                    validate_manifest(manifest)
            trees = set(); explicit = {chosen['path']}; project_files = set()
            for meta in descriptors.values():
                require(Path(meta['executable']).resolve() == Path(chosen['file']['realpath']), 'probe interpreter differs')
                config = Path(meta['prefix']) / 'pyvenv.cfg'; require(config.is_file(), 'declared benchmark venv required'); explicit.add(str(config))
                for text in meta['search_path']:
                    path = Path(text)
                    if path.is_relative_to(P): continue
                    if path.is_file(): explicit.add(text)
                    else: trees.add(text)
                for text in [*meta['modules'], *meta['code_maps']]:
                    if Path(text).is_relative_to(P): project_files.add(text)
                    else: explicit.add(text)
            for text in project_files:
                name = str(Path(text).relative_to(P)); require(name in pins and ownership.sha(text) == pins[name], 'uncommitted imported project input: ' + name)
            control.gen.benchmark_pins(source); require(selection(env) == chosen, 'final interpreter differs')
            if phase == 'prepare':
                manifest = dict(selected=chosen, descriptors=descriptors, explicit=sorted(explicit),
                    inventory=inventory(sorted(trees), explicit))
                control.write(HOST / 'runtime.json', manifest); validate_manifest(manifest)
                contract = dict(path=str(HOST / 'runtime.json'), sha256=ownership.sha(HOST / 'runtime.json'),
                    files=len(manifest['inventory']['files']), bytes=manifest['inventory']['bytes'])
                state.update(stage='prepared', discovery_command_count=len(owner.commands),
                    plan=dict(source_commit=source, selected=chosen, manifest=contract,
                        environment_identity=control.identity(env), template=dict(path=str(template), sha256=ownership.sha(template)),
                        work='Two bounded definition-only discovery processes,then commit this manifest/plan before two fresh qualified probes. Never call entrypoint main,serve requests or initialize an engine.',
                        gate='Exact source/interpreter/environment/template,all runtime directory inventories and file hashes checked before and after both qualified CPU probes;descriptor equality,normal owned closure;no GPU/model or speed claim.',
                        budgets=dict(wall=30, cpu=20, rss_cap=256 << 20, text_cap=256 << 10, file_cap=1 << 20, total_cap=4 << 20)),
                    versions=descriptors['server']['versions'])
            else: state.update(stage='closed', complete=True, passed=True)
        except BaseException as exception: error = repr(exception)
        state.update(error=error, ownership_closed=bool(owner and not owner.active and all(ownership.closed(c) for c in owner.commands)))
        state['passed'] = bool(state['passed'] and state['ownership_closed'] and error is None); persist()
        # Keep only this current scope in Git; the manifest is a live external input.
        if error is not None or not state['ownership_closed']: state['stage'] = 'failed'
        persist(); public = compact(state)
        raw = control.RESULT.read_text(); whole = json.loads(raw)
        if SCOPE in whole['historical_regression_audit']:
            require(phase == 'run', 'preserve original host discovery')
            literal = json.dumps(SCOPE) + ':'; start = raw.index(literal) + len(literal)
            while raw[start].isspace(): start += 1
            _, count = json.JSONDecoder().raw_decode(raw[start:])
            raw = raw[:start] + json.dumps(public, indent=2).replace('\n', '\n    ') + raw[start + count:]
        else:
            anchor = '"historical_regression_audit": {'
            raw = raw.replace(anchor, anchor + '\n    ' + json.dumps(SCOPE) + ': ' + json.dumps(public, indent=2).replace('\n', '\n    ') + ',', 1)
        whole['historical_regression_audit'][SCOPE] = public; require(json.loads(raw) == whole, 'unrelated host evidence changed')
        control.RESULT.write_text(raw); print(json.dumps(public), flush=True)
        return int(error is not None or not state['ownership_closed'] or (phase == 'run' and not state['passed']))


def compact(state):
    value = {k:v for k,v in state.items() if k not in ('benchmark_inputs', 'commands')}
    value['command_count'] = len(state['commands'])
    return value


def read_manifest(contract):
    require(ownership.sha(contract['path']) == contract['sha256'], 'committed runtime manifest changed')
    value = json.loads(Path(contract['path']).read_text())
    require(len(value['inventory']['files']) == contract['files'] and value['inventory']['bytes'] == contract['bytes'], 'runtime manifest inventory differs')
    return value


def admitted(control):
    public = control.runtime.validation.committed(control.RESULT, ('historical_regression_audit', SCOPE))
    state = json.loads((HOST / 'record.json').read_text())
    require(public == compact(state) and public['stage'] == 'closed' and public['complete'] and
        public['passed'] and public['ownership_closed'] and not public['active'] and
        public['command_count'] == 2 and public['discovery_command_count'] == 2 and
        all(ownership.closed(v) for v in state['commands']), 'closed qualified Python runtime required')
    require(control.gen.benchmark_pins(public['source_commit']) == state['benchmark_inputs'], 'qualified Python source changed')
    manifest = read_manifest(public['plan']['manifest']); validate_manifest(manifest)
    return public, manifest


if __name__ == '__main__':
    if sys.argv[1:] == ['--help']: print('host-prepare,host-run; probe ENTRY TEMPLATE')
    elif len(sys.argv) == 4 and sys.argv[1] == 'probe': probe(sys.argv[2], sys.argv[3])
    elif sys.argv[1:] == ['host-prepare']: sys.exit(host_action('prepare'))
    elif sys.argv[1:] == ['host-run']: sys.exit(host_action('run'))
    else: raise RuntimeError('use host-prepare,host-run or probe ENTRY TEMPLATE')
