#!/usr/bin/env python3
"""One-use Main-owned host gate for the current native-image observer.

Repository LICENSE. This controller cannot enqueue GPU work.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_qsa_k_staging as c
b = c.b
P, RESULT = b.P, b.RESULT
ROOT = b.ROOT / 'native-image-v1'
SOURCE = 'bench/sycl/ur_kernel_binary_subscriber.cpp'
PACKETS = 'bench/sycl/ur_kernel_binary_packets.cpp'
KEY = 'native_image_diagnostic'
require = b.guards.require
FAILURES = {'bad-geometry': 'selected launch geometry',
    'multi-device': 'sole program device required for v2 native getter',
    'other-device': 'queue/program device differs', 'wrong-backend': 'Level Zero backend required',
    'wrong-gpu': 'qualified physical B570 required', 'extension-overflow': 'extension inventory budget',
    'no-extension': 'kernel binary extension unsupported', 'old-extension': 'kernel binary extension unsupported',
    'wrong-name': 'native name differs', 'wrong-arguments': 'selected native argument/subgroup contract',
    'unsupported': 'binary size', 'oversize': 'binary size budget', 'empty-binary': 'binary size budget',
    'size-change': 'binary size changed', 'native-change': 'native identity changed',
    'recreated': 'recreated selected kernel', 'wrong-api': 'create API identity', 'null-packet': 'callback packet'}


def pins(revision):
    result = {**b.inputs(revision), **c.inputs(revision)}
    for name in [SOURCE, PACKETS, str(Path(__file__).resolve().relative_to(P))]:
        raw = b.gen.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'commit exact observer/host recipe: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def prepare():
    record = b.gen.committed(RESULT)
    data = record[KEY]
    require(not ROOT.exists() and 'host_plan' not in data, 'fresh host observer gate only')
    require(not data['GPU_executed'] and not data['built'] and
        not record['decision_summary']['performance_gate_passed'], 'preserve rejected component decision')
    rt = c.run_admission(record, record['actual_id_plan'])
    b.qualified_runtime.current(rt)
    review = c.validation.committed(b.gen.A / 'research-index.json',
        ('qsa_UR_native_binary_contract_20261028',))
    require(review['worker_closed'] and review['review']['status'] == 'completed' and
        review['review']['reasoning_effort'] == 'max', 'completed independent contract review required')
    revision = b.gen.git('rev-parse', 'HEAD').decode().strip()
    source = pins(revision)
    sdk = Path(rt['preload']['loader']).parent.parent
    framework = (sdk / 'lib/libxptifw.so.1.0.1').resolve()
    inspected = c.validation.committed(b.gen.A / 'research-index.json',
        ('UR_PTI_identity_bridge_20261025', 'exact_SDK_inspection'))
    require(b.digest(framework) == inspected['runtime_assets'][framework.name]['installed_sha256'],
        'inspected XPTI framework changed')
    compiler = '/usr/bin/g++'
    options = ['-std=c++17', '-O2', '-fPIC', '-Wall', '-Wextra', '-Werror', '-pedantic',
        '-pthread', '-I', str(sdk / 'include')]
    host = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'LD_LIBRARY_PATH': str(sdk / 'lib')}
    ROOT.mkdir(mode=0o700)
    owner = b.supervision.Owner(ROOT / 'discover-owned')
    dependencies = {}
    for label, name in [('observer', SOURCE), ('packets', PACKETS)]:
        entry, out, err = owner.run(label, [compiler, *options, '-M', '-MT', 'fixture', str(P / name)],
            host, P, wall=30, cpu=30, rss_cap=1 << 30, text_cap=1 << 20,
            file_cap=1 << 20, total_cap=8 << 20)
        require(b.supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
            'bounded header discovery failed: ' + label)
        text = out.read_text().replace('\\\n', ' ')
        for value in shlex.split(text.split(':', 1)[1]):
            path = Path(value).resolve(); dependencies[str(path)] = b.digest(path)
    require(len(dependencies) > 20, 'actual observer header closure absent')
    tools = {'compiler': compiler}
    for name in ['cc1plus', 'collect2', 'as', 'ld']:
        path = subprocess.check_output([compiler, '-print-prog-name=' + name],
            text=True, env=host, timeout=10).strip()
        tools[name] = str(Path(path if '/' in path else shutil.which(path, path=host['PATH'])).resolve())
    libraries = ['-L', str(sdk / 'lib'), '-Wl,-rpath,' + str(sdk / 'lib'), '-lxptifw']
    commands = {
        'observer': [compiler, *options, '-MD', '-MF', str(ROOT / 'observer.d'), '-shared',
            str(P / SOURCE), *libraries, '-lur_loader', '-lze_loader', '-o', str(ROOT / 'libstrata_ur_kernel_binary.so')],
        'packets': [compiler, *options, '-MD', '-MF', str(ROOT / 'packets.d'), str(P / PACKETS),
            *libraries, '-o', str(ROOT / 'packets')]}
    plan = dict(stage='prepared', source_commit=revision, inputs=source, dependencies=dependencies,
        compiler_tools={path: b.digest(path) for path in tools.values()}, framework=dict(path=str(framework),
            sha256=b.digest(framework)), runtime_files=rt['runtime_files'], host_environment=host,
        build_commands=commands, kernel_names=c.kernel_ids(record), failures=FAILURES,
        budgets=dict(wall=60, cpu=60, rss_cap=1 << 30, text_cap=2 << 20,
            file_cap=8 << 20, total_cap=32 << 20),
        acceptance='Both host builds close0 with actual .d inputs matching predeclared closure. Packet binary does not link UR/LevelZero/SYCL. Success captures all3arms,2launch APIs,recursion and unlocked native queries;18adversarial cases stop124;ignored/failed launches query nothing. All raw outputs bounded. No GPU initialization/enqueue,subscriber activation,native image,performance or model qualification.',
        retention='Immediate actual subscriber activation and one frozen same-fixture extraction. Retire consumed success logs/objects after compact host evidence;retire observer generation after extraction comparison.')
    data['host_plan'] = plan
    b.write(ROOT / 'plan.json', plan); b.write(RESULT, record)
    print('Prepared host-only observer gate;commit exact plan before run.')


def run():
    record = b.gen.committed(RESULT);data = record[KEY];plan = data['host_plan']
    require(plan['stage'] == 'prepared' and plan == json.loads((ROOT / 'plan.json').read_text())
        and 'host_check' not in data, 'fresh committed host plan only')
    require(pins(plan['source_commit']) == plan['inputs'], 'host source/recipe changed')
    for path, sha in {**plan['dependencies'], **plan['compiler_tools'], **plan['runtime_files'],
            plan['framework']['path']: plan['framework']['sha256']}.items():
        require(b.digest(path) == sha, 'frozen observer build input changed: ' + path)
    outcome = dict(passed=False, complete=False, active=False, ownership_closed=False,
        source_commit=plan['source_commit'], GPU_executed=False, samples=[])
    owner = None
    def persist():
        outcome['active'] = bool(owner and owner.active is not None)
        b.write(ROOT / 'owned.json', dict(outcome, commands=owner.commands if owner else []))
    owner = b.supervision.Owner(ROOT / 'owned', persist)
    try:
        for label, command in plan['build_commands'].items():
            entry, out, err = owner.run('build-' + label, command, plan['host_environment'], P, **plan['budgets'])
            require(b.supervision.closed(entry) and entry['exit_code'] == 0, 'observer build failed: ' + label)
            actual = shlex.split((ROOT / (label + '.d')).read_text().replace('\\\n', ' ').split(':', 1)[1])
            require(actual and all(plan['dependencies'].get(str(Path(x).resolve())) == b.digest(x)
                for x in actual), 'unfrozen actual observer header')
        entry, out, err = owner.run('packet-dynamic', ['/usr/bin/readelf', '-d', str(ROOT / 'packets')],
            plan['host_environment'], P, **plan['budgets'])
        require(b.supervision.closed(entry) and entry['exit_code'] == 0 and
            not any(x in out.read_text() for x in ['libur_loader', 'libze_loader', 'libsycl']),
            'host packets must not link a native GPU runtime')
        for mode in ['success', 'ignored', 'failed-launch', *plan['failures']]:
            folder = ROOT / ('host-' + mode);folder.mkdir(mode=0o700)
            library = str(ROOT / 'libstrata_ur_kernel_binary.so')
            env = dict(plan['host_environment'], XPTI_TRACE_ENABLE='0', XPTI_SUBSCRIBERS=library,
                STRATA_NATIVE_CAPTURE_SUBSCRIBER=library, STRATA_NATIVE_CAPTURE_ROOT=str(folder),
                STRATA_NATIVE_PACKET_MODE=mode)
            env.update({'STRATA_NATIVE_KERNEL_' + str(i): name for i, name in enumerate(plan['kernel_names'])})
            entry, out, err = owner.run(mode, [str(ROOT / 'packets')], env, P, **plan['budgets'])
            expected = 124 if mode in plan['failures'] else 0
            require(b.supervision.closed(entry) and entry['exit_code'] == expected,
                'unexpected host packet result: ' + mode)
            if mode in plan['failures']:
                require('native kernel capture: ' + plan['failures'][mode] in err.read_text(),
                    'wrong host refusal mechanism: ' + mode)
            rows = [json.loads(x) for x in (folder / 'captures.jsonl').read_text().splitlines()]
            captured = [x for x in rows if x['kind'] == 'capture']
            launches = [x for x in rows if x['kind'] == 'launch']
            if mode == 'success':
                require(len(captured) == 3 and {x['arm'] for x in captured} == {0, 1, 2} and len(launches) == 6,
                    'host capture/launch census')
                for arm in range(3):
                    require((folder / ('kernel%d.gen' % arm)).read_bytes() == bytes(range(64)) and
                        (folder / ('kernel%d.zebin' % arm)).stat().st_size == 128, 'host copied bytes')
            if mode in ['ignored', 'failed-launch']:
                require(not captured and not launches, 'ignored/failed callback captured')
            outcome['samples'].append(dict(mode=mode, exit_code=entry['exit_code'], captures=len(captured),
                accepted_launches=len(launches)))
        for path, sha in plan['dependencies'].items():require(b.digest(path) == sha, 'header changed during gate')
        outcome.update(passed=True, complete=True, ownership_closed=True, artifacts={
            name: dict(path=str(ROOT / name), sha256=b.digest(ROOT / name), bytes=(ROOT / name).stat().st_size)
            for name in ['packets', 'libstrata_ur_kernel_binary.so']})
    except BaseException as error:
        outcome.update(error=str(error), ownership_closed=owner.active is None)
        raise
    finally:
        outcome['artifact_inventory'] = {name: dict(path=str(ROOT / name), sha256=b.digest(ROOT / name),
            bytes=(ROOT / name).stat().st_size) for name in ['packets', 'libstrata_ur_kernel_binary.so']
            if (ROOT / name).is_file()}
        persist();data['host_check'] = outcome;data['built'] = bool(outcome.get('artifacts'))
        b.write(RESULT, record)
    print('PASS host-only build/packets;GPU extraction and subscriber activation remain unqualified.')


if __name__ == '__main__':
    with (b.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
