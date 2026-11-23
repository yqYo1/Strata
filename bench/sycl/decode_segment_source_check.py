"""Project only the segment observer onto the measured SYCL source and check both production TUs."""

import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import decode_segment_clocks as reader
import decode_mtp_prefix_source as source

BASE = 'a41599ff3cace916dcdc6bb54a6db0e75d5b3855'
CHANGE = '82f4ab777063f599a8a98ad20e471459f39dcc93'
HEADER = 'include/strata/sycl/decode_segment_clocks.hpp'
UNITS = {'src/core/verify.cpp': 'strata_engine', 'src/program/generate.cpp': 'strata'}
PROJECTED = {HEADER: 'fd0b57f2d5dcc956a189a8018695004de3c736fbef764513aff7a46fe58b3056',
    'src/core/verify.cpp': '849737cd2b8975659d97e86c15486f40221ba879fa599155f56d27e0d0a0a8f2',
    'src/program/generate.cpp': '248e20855234aceb369a4436ef568fb27484be83a81e8dc08d78a9af578a320c'}
require = source.require


def check_reader():
    lines = []
    for i in range(196):
        window, segment = divmod(i, 49)
        lines.append(f'strata decode segment: request=5 window={9+window} pos={100+window} '
            f'qsa_interval=4 segment={segment} status=0 fields=7 submit={10*i+1} start={10*i+2} '
            f'end={10*i+6} resolution_ns=1 host_wait_ns=7 pool_callback_ns={0 if segment == 48 else 3}')
    result = reader.summarize(lines, 5)
    expected = {'input_to_gdn': 1, 'gdn_to_gdn': 24, 'gdn_to_qsa': 12, 'qsa_to_gdn': 11, 'qsa_to_head': 1}
    for window in result:
        bins = window['boundaries']
        require({k: v['segments'] for k, v in bins.items()} == expected and
                sum(v['graph_ns'] for v in bins.values()) == 196 and
                sum(v['pool_callback_ns'] for v in bins.values()) == 144, 'reader boundary totals differ')
    for bad in (lines[:-1], [lines[0], *lines], [lines[0].replace('request=5', 'request=4'), *lines[1:]],
                [lines[0].replace('end=6', 'end=1'), *lines[1:]], [*lines[:49], lines[49].replace('pos=101', 'pos=100'), *lines[50:]]):
        try:
            reader.summarize(bad, 5)
        except ValueError:
            continue
        raise ValueError('reader accepted invalid segment evidence')
    walls = [f'strata decode sample: request=5 window={9+i} pos={100+i} device=0 wall_ns=1000'
             for i in range(4)]
    control = reader.window_walls(walls, 5, False)
    device = reader.window_walls([s.replace('device=0', 'device=1') for s in walls], 5, True)
    require(reader.compare_windows(device, control)['within_ten_percent'], 'matched local clocks rejected')
    device[101] = 1500
    require(not reader.compare_windows(device, control)['within_ten_percent'], 'one distorted window went undetected')
    try:
        reader.window_walls([*walls, lines[0]], 5, False)
    except ValueError:
        return
    raise ValueError('wall-only control accepted a device row')


def check(directory):
    parent = source.PARENT
    require(source.digest(parent / 'record.json') == source.RECORD, 'parent build record changed')
    row = json.loads((parent / 'record.json').read_bytes())
    elf, ext = row['binaries']['strata'], row['external_inputs']
    require(row['complete'] and row['passed'] and row['ownership_closed'] and
            source.digest(elf['path']) == elf['sha256'] and source.digest(ext['path']) == ext['sha256'],
            'executable or external toolchain inventory changed')
    build = parent / 'build'
    table = build / 'compile_commands.json'
    require(source.digest(table) == row['configured_files'][str(table)], 'compile table changed')
    commands = json.loads(table.read_bytes())
    external = json.loads(Path(ext['path']).read_bytes())
    known = row['actual_inputs']['compiled']
    for unit in UNITS:
        original = parent / 'source' / unit
        require(known[str(original)] == source.digest(original), 'original translation unit changed')
        dest = directory / unit
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(original.read_bytes())
    patch = subprocess.check_output(['git', 'diff', BASE, CHANGE, '--', *PROJECTED], cwd=source.PROJECT)
    subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', '-'], cwd=directory, input=patch, check=True)
    require(all(source.digest(directory / p) == h for p, h in PROJECTED.items()), 'source projection differs')
    for unit, target in UNITS.items():
        original = parent / 'source' / unit
        entry, = [e for e in commands if e['file'] == str(original) and e['output'] == f'CMakeFiles/{target}.dir/{unit}.o']
        args = shlex.split(entry['command'])
        require(entry['directory'] == str(build) and args[-4:] == ['-o', entry['output'], '-c', str(original)]
                and '-Werror' in args and '-fsycl' in args and external[args[0]] == source.digest(args[0]),
                'production compiler recipe differs')
        depfile = directory / (target + '.d')
        candidate = directory / unit
        cmd = [args[0], '-I'+str(directory/'include'), *args[1:-4], '-iquote', str(original.parent),
               '-fsyntax-only', '-MD', '-MF', str(depfile), '-MT', 'checked', str(candidate)]
        subprocess.run(cmd, cwd=build, env=dict(row['recipe']['environment'], TMPDIR=str(directory),
                       XDG_CACHE_HOME=str(directory)), check=True, timeout=240)
        deps = {(build / p).resolve() for p in shlex.split(depfile.read_text().replace('\\\n', ' ').split(':', 1)[1])}
        added = {candidate, directory / HEADER}
        require(added <= deps and original not in deps and all(known.get(str(p)) == source.digest(p) for p in deps-added),
                'compiled source or a transitive include differs')
        print(f'{target} syntax passed, {len(deps-added)} original dependencies matched', flush=True)
    check_reader()
    print('reader boundary sums, five damaged rows and local overhead controls passed', flush=True)


if __name__ == '__main__':
    for path in (Path(__file__), Path(reader.__file__), Path(source.__file__), source.PROJECT / 'tools/sycl/measurement_guards.py'):
        require(path.read_bytes() == source.blob('HEAD', str(path.resolve().relative_to(source.PROJECT))), 'uncommitted check input')
    require(source.blob(CHANGE, 'bench/sycl/decode_segment_clocks.py') == Path(reader.__file__).read_bytes(), 'reader changed')
    with (source.STATE / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source.no_model_owners([])
        with tempfile.TemporaryDirectory(prefix='decode-segment-check-', dir=source.STATE) as temp:
            check(Path(temp))
