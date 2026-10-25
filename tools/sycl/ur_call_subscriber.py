#!/usr/bin/env python3
"""Build and qualify a pre-exec UR-call subscriber on the CPU before GPU use.

This diagnostic has no production engine changes. Retained original6e40 and
its12 fixtures stay selected; one small observer ELF serves the next GPU probe.
Repository LICENSE; no external implementation copied.
"""
import collections
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import work_ledger_fixture as probe

r = probe.runtime
P, B, RESULT = probe.P, probe.B, probe.RESULT
ROOT = B / 'ur-call-subscriber-v1'
SOURCE = 'bench/sycl/ur_call_subscriber.cpp'
KEY = ('event_reconciliation', 'ur_subscriber')
SDK_KEY = ('UR_PTI_identity_bridge_20261025', 'exact_SDK_inspection')
require, digest = r.require, r.digest


def source_pins(revision):
    pins = probe.source_pins(revision)
    for name in (SOURCE, 'tools/sycl/ur_call_subscriber.py', 'tests/core/test_ur_call_subscriber.py'):
        raw = r.validation.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact subscriber input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def parse_calls(text, pid):
    """Validate complete begin/end instance pairs without inventing PTI links."""
    require(len(text.encode()) <= 16 << 20, 'subscriber text budget')
    rows = [json.loads(line) for line in text.splitlines()]
    require(rows and rows[0]['kind'] == 'load', 'missing subscriber load')
    require(all(type(e.get('pid')) is int and e['pid'] == pid and type(e.get('raw_ns')) is int and
                e['raw_ns'] >= rows[0]['raw_ns'] for e in rows), 'invalid subscriber owner/clock')
    require(sum(e['kind'] == 'load' for e in rows) == 1, 'duplicate subscriber load')
    subscribed = finished = False
    pending, seen, stacks, functions = {}, set(), collections.defaultdict(list), {}
    counts = collections.Counter()
    for event in rows[1:]:
        kind = event['kind']
        if kind in ('subscribe', 'finish'):
            require(event.get('stream') == 'ur.call' and not pending, 'invalid stream boundary')
            if kind == 'subscribe':
                require(not subscribed and not finished, 'duplicate UR subscription')
                subscribed = True
            else:
                require(subscribed and not finished, 'invalid UR finish')
                finished = True
            continue
        require(subscribed and not finished and kind in ('begin', 'end'), 'unclassified/out-of-stream record')
        ident, tid, function = event.get('instance'), event.get('tid'), event.get('function_id')
        require(type(ident) is int and 0 <= ident < (1 << 64) and type(tid) is int and tid > 0 and
                type(function) is int and 0 <= function < (1 << 32) and
                isinstance(event.get('name'), str) and re.fullmatch(r'ur[A-Za-z0-9_]+', event['name']),
                'invalid UR call identity')
        require(function not in functions or functions[function] == event['name'], 'unstable UR function identity')
        functions[function] = event['name']
        if kind == 'begin':
            require(ident not in seen and event.get('status') is None, 'duplicate/invalid UR begin')
            seen.add(ident)
            pending[ident] = event
            stacks[tid].append(ident)
        else:
            begin = pending.pop(ident, None)
            require(begin is not None and stacks[tid] and stacks[tid][-1] == ident,
                    'unmatched/cross-thread/nonnested UR end')
            require((begin['tid'], begin['function_id'], begin['name']) == (tid, function, event['name']) and
                    begin['raw_ns'] <= event['raw_ns'] and type(event.get('status')) is int and
                    -(1 << 31) <= event['status'] < (1 << 31),
                    'UR call end identity/clock/status mismatch')
            stacks[tid].pop()
            counts[(event['name'], event['status'])] += 1
    require(subscribed and seen and not pending and all(not stack for stack in stacks.values()),
            'missing UR subscription/call/end')
    return dict(call_pairs=len(seen), source_threads=len(stacks),
                outcomes=[dict(name=name, status=status, calls=count) for (name, status), count in sorted(counts.items())],
                ordinary_runtime_teardown_qualified=False, device_work_qualified=False)


def runtime_assets():
    row = r.validation.committed(r.RESULT)
    r.current(row)
    sdk = r.validation.committed(r.validation.INDEX, SDK_KEY)
    loader = row['preload']['loader']
    require(digest(loader) == sdk['runtime_assets']['libur_loader.so.0.12.0']['installed_sha256'],
            'inspected SDK loader differs from qualified runtime')
    dispatcher = Path(loader).parent / 'libxptifw.so.1.0.1'
    require(digest(dispatcher) == sdk['runtime_assets'][dispatcher.name]['installed_sha256'], 'dispatcher changed')
    for name, sha in sdk['source_inputs'].items():
        require(digest(Path(sdk['source_root']) / name) == sha, 'inspected source changed')
    for name in ('xpti_data_types.h', 'xpti_trace_framework.h'):
        require(digest(dispatcher.parent.parent / 'include/xpti' / name) ==
                sdk['source_inputs']['xpti/include/xpti/' + name], 'installed XPTI header/source mismatch')
    return dict(runtime_result_sha256=digest(r.RESULT), preload=row['preload'],
                dispatcher=dict(path=str(dispatcher), sha256=digest(dispatcher)), sdk=sdk)


def admit(row, binary=False):
    require(source_pins(row['source_commit']) == row['input_sha256'] and runtime_assets() == row['assets'],
            'subscriber source/runtime assets changed')
    for name, sha in row['compiler_tools'].items():
        require(digest(name) == sha, 'compiler/linker provider changed')
    manifest = Path(row['compiler_inputs']['path'])
    require(digest(manifest) == row['compiler_inputs']['sha256'], 'header census changed')
    inputs = json.loads(manifest.read_text())
    require(len(inputs) == row['compiler_inputs']['files'], 'header census count differs')
    for name, sha in inputs.items():
        require(digest(name) == sha, 'compiler source/header changed: ' + name)
    if binary:
        require(digest(row['library']['path']) == row['library']['sha256'], 'observer ELF changed')


def prepare():
    require(not ROOT.exists(), 'inspect existing subscriber owner; no retry')
    container = r.validation.committed(RESULT)
    require('ur_subscriber' not in container[KEY[0]], 'preserve existing subscriber result')
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    pins, assets = source_pins(revision), runtime_assets()
    sdk = Path(assets['dispatcher']['path']).parent.parent
    compiler = str(Path('/usr/bin/g++').resolve())
    options = [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-pthread',
               '-DXPTI_CALLBACK_API_EXPORTS', '-I' + str(sdk / 'include')]
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
    tools = {compiler: digest(compiler)}
    for name in ('cc1plus', 'collect2', 'as', 'ld'):
        value = subprocess.check_output([compiler, '-print-prog-name=' + name], env=env, text=True, timeout=10).strip()
        path = Path(value) if '/' in value else Path(shutil.which(value, path=env['PATH']))
        tools[str(path.resolve())] = digest(path)
    raw = subprocess.check_output([*options, '-M', str(P / SOURCE)], env=env, timeout=30)
    require(len(raw) <= 1 << 20 and b':' in raw, 'finite compiler header census required')
    dependencies = shlex.split(raw.decode().split(':', 1)[1].replace('\\\n', ' '))
    require(0 < len(dependencies) < 4096, 'header population cap')
    headers = {str(Path(name).resolve()): digest(name) for name in dependencies}
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    manifest = ROOT / 'compiler-inputs.json'
    r.write(manifest, headers)
    library = ROOT / 'libstrata_ur_calls.so'
    command = [*options, str(P / SOURCE), '-L' + str(sdk / 'lib'), '-Wl,-rpath,' + str(sdk / 'lib'),
               '-Wl,--no-undefined', '-lxptifw', '-o', str(library)]
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins, assets=assets,
        compiler_tools=tools, compiler_inputs=dict(path=str(manifest), sha256=digest(manifest), files=len(headers)),
        command=command, environment=dict(env, TMPDIR=str(ROOT / 'tmp')),
        budgets=dict(wall=120, cpu=120, rss_cap=4 << 30, text_cap=4 << 20, file_cap=32 << 20, total_cap=64 << 20),
        gates='Compile one Main-authored host-only observer ELF;commit binary hash before checks. Four CPU mutation controls must each reach their marker then exit87 without changing environment. CPU-only original UR canary must exit0 with pre-main provider/validation proof,exactly two rejected urPlatformGetApiVersion pairs,status46,unique instance IDs and complete ordered per-thread timestamps. Never enumerate a device,create a queue or enqueue work. Any failure stops. No GPU/model/performance or ordinary teardown qualification.',
        retention='Original6e40 plus12 fixtures retained unchanged. Keep this one small observer ELF,exact source/recipe and header manifest only for the imminent GPU coverage fixture after host checks;retire normal build/API/loader output after compact evidence.',
        active=False, ownership_closed=True, passed=False, gpu_work=False, model_qualified=False)
    container[KEY[0]][KEY[1]] = row
    r.write(RESULT, container)
    r.write(ROOT / 'record.json', row)
    print('Prepared host-only UR subscriber;commit recipe before build.', flush=True)


def mutate_command(kind):
    code = '''import ctypes,sys
lib=ctypes.CDLL(None)
name=sys.argv[1]
fn=getattr(lib,name)
fn.restype=ctypes.c_int
print('mutation-case-begin:'+name,flush=True)
if name=='setenv':fn(b'STRATA_MUTATION_TEST',b'1',1)
elif name=='unsetenv':fn(b'STRATA_MUTATION_TEST')
elif name=='putenv':fn(ctypes.create_string_buffer(b'STRATA_MUTATION_TEST=1'))
else:fn()
raise SystemExit(99)
'''
    return ['/usr/bin/python3', '-B', '-c', code, kind]


def execute(action):
    row = r.validation.committed(RESULT, KEY)
    require(row['stage'] == ('prepared' if action == 'build' else 'built'), 'committed single-use component stage required')
    admit(row, action == 'check')
    owner = None
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(x.split(': ', 1)[1] for x in cursor_text.splitlines() if x.startswith('-- cursor: '))
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def persist():
        row['active'] = bool(owner and owner.active is not None)
        r.write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else []))

    try:
        require(not (ROOT / (action + '-owned')).exists(), 'inspect original component process;no retry')
        owner = r.supervision.Owner(ROOT / (action + '-owned'), persist)
        row['stage'] = action + 'ing'
        if action == 'build':
            entry, _, _ = owner.run('compile', row['command'], row['environment'], P, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'subscriber compile failed')
            library = ROOT / 'libstrata_ur_calls.so'
            row.update(stage='built', library=dict(path=str(library), sha256=digest(library), bytes=library.stat().st_size))
        else:
            base = r.validation.committed(r.SETTINGS)['environment']
            row['outcomes'] = []
            for kind in ('setenv', 'unsetenv', 'putenv', 'clearenv', 'canary'):
                folder = ROOT / kind
                folder.mkdir()
                env = dict(base, STRATA_UR_CALL_LOG=str(folder / 'calls.jsonl'),
                    XPTI_FRAMEWORK_DISPATCHER=row['assets']['dispatcher']['path'],
                    XPTI_SUBSCRIBERS=row['library']['path'], XPTI_TRACE_ENABLE='1', SYCL_UR_TRACE='0',
                    LD_PRELOAD=row['library']['path'], TMPDIR=str(ROOT / 'tmp'), LD_DEBUG='libs',
                    UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'ur.log'))
                command = mutate_command(kind)
                if kind == 'canary':
                    env.update(LD_PRELOAD=row['assets']['preload']['library']['path'] + ':' + row['library']['path'],
                               STRATA_UR_PROBE_EXPECTED_LOADER=row['assets']['preload']['loader'])
                    command = [row['assets']['preload']['canary']['path']]
                entry, out, err = owner.run(kind, command, env, P, wall=45, cpu=45, rss_cap=2 << 30,
                                           text_cap=4 << 20, file_cap=32 << 20, total_cap=64 << 20)
                require(r.supervision.closed(entry), 'component owner closure failed')
                if kind != 'canary':
                    require(entry['exit_code'] == 87 and out.read_text() == 'mutation-case-begin:' + kind + '\n' and
                            'strata UR subscriber: environment mutation blocked' in err.read_text(), 'mutation control failed')
                    result = dict(kind=kind, exit_code=87, mutation_blocked=True)
                else:
                    require(entry['exit_code'] == 0, 'CPU UR subscriber canary failed')
                    validation = r.validation.preload_evidence(err.read_text(), entry['pid'])
                    require(json.loads(out.read_text()) == dict(pid=entry['pid'], initialized=0, checked=46,
                        expected=46, version=(1 << 32) - 1, closed=0), 'original UR canary result changed')
                    calls = parse_calls((folder / 'calls.jsonl').read_text(), entry['pid'])
                    require(calls['call_pairs'] == 2 and calls['source_threads'] == 1 and calls['outcomes'] ==
                        [dict(name='urPlatformGetApiVersion', status=46, calls=2)], 'CPU UR callback contract differs')
                    result = dict(kind=kind, exit_code=0, validation=validation, calls=calls)
                row['outcomes'].append(result)
                persist()
            row.update(stage='host-checked', passed=True)
        admit(row, True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner is None or (owner.active is None and all(r.supervision.closed(e) for e in owner.commands))
        try:
            check = dict(host_probe_accepted=True)
            r.validation.finish(check, owner, cursor, boot)
            row.update(fresh_faults=check['fresh_faults'], boot_unchanged=check['boot_unchanged'])
            require(check['passed'] and row['ownership_closed'], 'component fault/closure check failed')
        except BaseException as error:
            row.update(stage='failed', passed=False, fault_check_error=repr(error))
        persist()
        container = json.loads(RESULT.read_text())
        container[KEY[0]][KEY[1]] = row
        r.write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    return 1 if row['stage'] == 'failed' else 0


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check'),
            'use ur_call_subscriber.py prepare|build|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r.guards.no_model_owners([])
        raise SystemExit(prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1]))
