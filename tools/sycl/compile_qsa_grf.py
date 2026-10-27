#!/usr/bin/env python3
"""Main-authored compile-only GRF discriminator; repository LICENSE.

Uses this fork's own immutable QSA proposal and existing bounded process owner.
Never links or executes a device program. Emitted metadata is not B570 support,
machine-code GRF allocation, kernel admission, correctness or speed evidence.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_guards as guards
import measurement_owner as supervision
import qsa_grf_sources as proposal

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'qsa-grf-lowering-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-grf-current.json'
ARCHIVE = '208c591b0'
OLD_RESULT = 'bench/results/2026-10-15-pr6-recovery-audit/qsa-barrier-current.json'
require = guards.require


def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], timeout=30)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def committed(path):
    raw = git('show', 'HEAD:' + str(path.relative_to(P)))
    require(raw == path.read_bytes(), 'commit exact input: ' + str(path))
    return json.loads(raw)


def inputs(revision):
    names = {'tools/sycl/' + n for n in ('compile_qsa_grf.py',
        'qsa_grf_sources.py', 'measurement_guards.py', 'measurement_owner.py')}
    names.add(proposal.PREFIX)
    names.add('tests/core/test_qsa_grf.py')
    names.update(git('ls-files', 'include').decode().splitlines())
    result = {}
    for name in sorted(names):
        raw = git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'changed compilation input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    require(git('show', proposal.BASE + ':' + proposal.PREFIX) ==
        (P / proposal.PREFIX).read_bytes(), 'selected original prefix changed')
    return result


def baseline(row):
    require(not row['active'] and row['ownership_closed'] and row['complete'] and
        row['passed'] and row['source_commit'] == proposal.BASE,
        'keep the selected original engine')
    assets = [row['binary'], *row['fixtures'].values()]
    require(len(assets) == 13, 'selected engine and twelve fixtures required')
    for asset in assets:
        require(digest(asset['path']) == asset['sha256'], 'selected artifact changed')
    return dict(source_commit=proposal.BASE, engine=row['binary'],
        assets_verified=len(assets), result='native-gdn-engine-build.json')


def admission(row):
    require(inputs(row['source_commit']) == row['input_sha256'], 'source changed')
    for name, body in proposal.generated(P).items():
        path = ROOT / name
        require(path.read_bytes() == body and digest(path) == row['generated'][str(path)],
            'generated proposal changed')
    manifest = row['compiler_inputs']
    require(digest(manifest['path']) == manifest['sha256'], 'compiler manifest changed')
    external = json.loads(Path(manifest['path']).read_text())
    require(len(external) == manifest['files'], 'compiler inventory count changed')
    for item in external:
        require(digest(item['path']) == item['sha256'], 'compiler dependency changed: ' + item['path'])
    require(digest(row['compiler']['path']) == row['compiler']['sha256'], 'compiler changed')
    require(digest(row['host_python']['path']) == row['host_python']['sha256'], 'host Python changed')
    require(baseline(committed(RESULT.with_name('native-gdn-engine-build.json'))) ==
        row['retained_engine'], 'selected engine qualification changed')


def prepare():
    require(not ROOT.exists() and not RESULT.exists(), 'fresh plan only; inspect existing handle')
    prior = committed(RESULT.with_name('qsa-barrier-current.json'))
    require(prior['retirement']['complete'] and not (B / 'qsa-barrier-v1').exists() and
        not prior['comparison_decision']['advance_to_model'], 'retire the rejected generation first')
    revision = git('rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    old = json.loads(git('show', ARCHIVE + ':' + OLD_RESULT))
    origin = old['retirement']['compiler_manifest_reconstruction']['origin']
    require(digest(origin['path']) == origin['sha256'], 'original toolchain inventory changed')
    rows = json.loads(Path(origin['path']).read_text())
    require(len(rows) == origin['files'], 'original inventory count changed')
    changed = {}
    for item in rows:
        sha = digest(item['path'])
        if sha != item['sha256']:
            expected = old['declared_compiler_header_delta'].get(item['path'])
            require(expected == dict(original_sha256=item['sha256'], new_build_sha256=sha),
                'undeclared toolchain change: ' + item['path'])
            changed[item['path']] = expected
        item['sha256'] = sha
    require(changed == old['declared_compiler_header_delta'], 'declared header delta changed')
    compiler = old['compiler']
    require(digest(compiler['path']) == compiler['sha256'], 'original compiler changed')
    retained = baseline(committed(RESULT.with_name('native-gdn-engine-build.json')))
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    generated = proposal.generated(P)
    for name, body in generated.items():
        (ROOT / name).write_bytes(body)
    manifest = ROOT / 'compiler-inputs.json'
    write(manifest, rows)
    install = Path(compiler['path']).parents[1]
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-g0', '-fsycl',
        '-fsycl-id-queries-range=size_t', '-fno-fast-math', '-ffp-contract=off',
        '-DSTRATA_ENABLE_SYCL=1', '-DSTRATA_QSA_PACKED_GUARD=17',
        '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat')]
    sources = dict(original=str(P / proposal.PREFIX),
        grf_128=str(ROOT / 'grf_128.cpp'), grf_256=str(ROOT / 'grf_256.cpp'))
    commands = {name: [compiler['path'], *flags, '-fsycl-device-only', '-S',
        '-emit-llvm', '-MD', '-MF', str(ROOT / (name + '.d')), '-MT',
        str(ROOT / (name + '.ll')), source, '-o', str(ROOT / (name + '.ll'))]
        for name, source in sources.items()}
    row = dict(stage='prepared', source_commit=revision, input_sha256=pins,
        generated={str(ROOT / n): digest(ROOT / n) for n in generated},
        compiler=compiler, compiler_inputs=dict(path=str(manifest), files=len(rows),
            sha256=digest(manifest), origin=origin, declared_header_delta=changed),
        retained_engine=retained, commands=commands,
        host_python=dict(path='/usr/bin/python3.12', sha256=digest('/usr/bin/python3.12')),
        environment=dict(PATH=str(install / 'bin') + ':/usr/bin:/bin',
            LD_LIBRARY_PATH=str(install / 'lib'), TMPDIR=str(ROOT / 'tmp'),
            LANG='C', LC_ALL='C', PYTHONDONTWRITEBYTECODE='1'),
        criteria='Three compile-only commands; no link/native program/device submission. Verify complete actual dependencies. Original LLVM has no sycl-grf-size request; each private arm must attach exactly one matching128/256 request to a SIMD16 spir_kernel,with no request on pack or helper kernels. Keep compilation completion separate from metadata proof. Stop on compile/dependency failure; no retry. LLVM request is not SPIR-V/native lowering,legal B570 mode,kernel workgroup admission,correctness or performance.',
        budgets=dict(per_command_wall_s=180, total_wall_s=600, cpu_s=240,
            rss_bytes=8 << 30, text_bytes=4 << 20, file_bytes=128 << 20, total_bytes=256 << 20),
        retention='Only LLVM/dependency outputs for the imminent lowering discriminator;no second executable. Keep compact attributes/dependency conservation and failures. Retire raw IR/compiler output once the source-contract decision closes;selected engine and12fixtures remain unchanged. Separate legacy provenance holds remain in native-gdn-engine-build.json.',
        active=False, ownership_closed=True, complete=False, compilation_passed=False,
        llvm_grf_request_confirmed=False, GPU_executed=False, inference_adopted=False)
    admission(row)
    write(ROOT / 'plan.json', row)
    write(RESULT, row)
    print('Prepared compile-only GRF discriminator; commit this exact plan before compile.', flush=True)


def compiled_dependencies(row):
    allowed = {str((P / n).resolve()): sha for n, sha in row['input_sha256'].items()}
    allowed.update(row['generated'])
    allowed.update({str(Path(i['path']).resolve()): i['sha256'] for i in
        json.loads(Path(row['compiler_inputs']['path']).read_text())})
    actual = {}
    for name in row['commands']:
        text = (ROOT / (name + '.d')).read_text().replace('\\\n', ' ')
        target, tail = text.split(':', 1)
        require(Path(target).resolve() == ROOT / (name + '.ll'), 'dependency target differs')
        for value in shlex.split(tail):
            path = str(Path(value).resolve()); sha = digest(path)
            require(allowed.get(path) == sha, 'unfrozen compilation dependency: ' + path)
            actual[path] = sha
    require(len(actual) > 100, 'incomplete compilation dependency coverage')
    return dict(files=len(actual), sha256=hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def metadata(path, expected):
    text = path.read_text()
    attrs = dict(re.findall(r'^attributes #(\d+) = (.*)$', text, re.M))
    kernels = []
    for line in text.splitlines():
        if not line.startswith('define ') or 'spir_kernel ' not in line:
            continue
        match = re.search(r' #([0-9]+)(?: |$)', line)
        require(match is not None, 'kernel attribute group absent')
        group = attrs[match[1]]
        requests = re.findall(r'"sycl-grf-size"="([0-9]+)"', group)
        if requests:
            subgroup = re.search(r'!intel_reqd_sub_group_size !(\d+)', line)
            size = re.search(r'^!' + subgroup[1] + r' = !\{i32 (\d+)\}$', text, re.M) if subgroup else None
            kernels.append(dict(symbol=line.split('@', 1)[1].split('(', 1)[0],
                grf_requests=requests, subgroup_size=int(size[1]) if size else None))
    all_requests = re.findall(r'"sycl-grf-size"="([0-9]+)"', text)
    confirmed = (not all_requests and not kernels) if expected is None else (
        len(kernels) == 1 and kernels[0]['grf_requests'] == [str(expected)] and
        kernels[0]['subgroup_size'] == 16 and all_requests == [str(expected)])
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size,
        requested_grf=expected, request_attributes=all_requests,
        kernels_with_request=kernels, metadata_pattern_confirmed=confirmed)


def compile_only():
    row = committed(RESULT)
    require(row['stage'] == 'prepared' and not (ROOT / 'record.json').exists() and
        row == json.loads((ROOT / 'plan.json').read_text()), 'inspect existing compile;no retry')
    admission(row)
    owner = None
    def persist():
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(row, owned_commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / 'owned', persist)
    deadline = time.monotonic() + row['budgets']['total_wall_s']
    row['stage'] = 'compiling'; persist()
    try:
        b = row['budgets']
        host_command = ['/usr/bin/python3.12', '-I', '-S', '-B', 'tests/core/test_qsa_grf.py']
        entry, _, _ = owner.run('host-control', host_command, row['environment'], P,
            wall=30, cpu=30, rss_cap=1 << 30, text_cap=b['text_bytes'],
            file_cap=b['file_bytes'], total_cap=b['total_bytes'])
        require(supervision.closed(entry) and entry['exit_code'] == 0, 'metadata host controls failed')
        row['host_control'] = dict(command=host_command, passed=True)
        for name, command in row['commands'].items():
            remaining = deadline - time.monotonic(); require(remaining > 1, 'whole compile deadline')
            print('Compile-only ' + name, flush=True)
            entry, _, _ = owner.run(name, command, row['environment'], P,
                wall=min(remaining, b['per_command_wall_s']), cpu=b['cpu_s'],
                rss_cap=b['rss_bytes'], text_cap=b['text_bytes'],
                file_cap=b['file_bytes'], total_cap=b['total_bytes'])
            require(supervision.closed(entry) and entry['exit_code'] == 0, 'compile failed: ' + name)
        admission(row)
        row['actual_dependencies'] = compiled_dependencies(row)
        row['metadata'] = {name: metadata(ROOT / (name + '.ll'), expected)
            for name, expected in (('original', None), ('grf_128', 128), ('grf_256', 256))}
        row.update(stage='compiled', compilation_passed=True, complete=True,
            llvm_grf_request_confirmed=all(x['metadata_pattern_confirmed'] for x in row['metadata'].values()))
    except BaseException as error:
        row.update(stage='failed', compilation_passed=False, complete=False, error=repr(error))
    finally:
        row['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
        if not row['ownership_closed']:
            row.update(stage='failed', compilation_passed=False, complete=False, error='compile ownership not closed')
        persist(); write(RESULT, row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'compilation_passed',
        'llvm_grf_request_confirmed', 'complete', 'ownership_closed', 'GPU_executed', 'error')}), flush=True)
    return 0 if row['compilation_passed'] else 1


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'compile'), 'use prepare|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        sys.exit(prepare() if sys.argv[1] == 'prepare' else compile_only())
