"""Run the existing CPU parity fixture on the packed IQ3_S sign-word change."""

import fcntl
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import iq3_pair as recipe

P, B = recipe.P, recipe.B
ROOT = B / 'cpu-iq3s-sign-parity-v1'
BUILD = ROOT / 'build'
KEY = 'cpu_iq3s_sign_parity_v1'
TARGET = 'cpu_iq_single_exact_test'
FIX = '200ce45c3'
PRIOR = '4cb47ff93933d509788afca106291b1caec12e54'
SOURCE = 'src/kernels/cpu/iq_single_avx2.cpp'
UNITS = (SOURCE, 'src/kernels/cpu/native_expert.cpp', 'tests/core/iq_single_exact_test.cpp')
COMPILED = ('bench/sycl/cpu_roofline/CMakeLists.txt', 'src/kernels/cpu', 'include', 'third_party/ggml', UNITS[-1])
CASES = dict(single=384, IQ3_XXS_pair=80, IQ3_S_pair=80)
EXPECTED = ('IQ single exact PASS: 384 single and 80 paired cases, bitwise GGML parity and row canaries\n'
            'IQ3_S pair exact PASS: 80 cases,512 grids,256 high/sign bytes,16 scales\n')
need, sha, git = recipe.need, recipe.digest, recipe.git


def committed_rows():
    raw = recipe.RESULT.read_bytes()
    need(raw == git(P, 'show', 'HEAD:' + recipe.RESULT.relative_to(P).as_posix()),
         'commit the result file before advancing')
    return json.loads(raw)


def write(path, row):
    temporary = path.with_suffix('.tmp')
    recipe.write(temporary, row)
    temporary.replace(path)


def save(row, checked_rows):
    checked_rows[KEY] = recipe.compact(row)
    write(ROOT / 'record.json', row)
    write(recipe.RESULT, checked_rows)


def baseline(ggml):
    key = 'cpu_iq3s_pair_component_v1'
    prior = json.loads(git(P, 'show', PRIOR + ':' + recipe.RESULT.relative_to(P).as_posix()))[key]
    check = prior['correctness']
    configure = prior['configure'].copy()
    configure[configure.index('-B') + 1] = str(BUILD)
    need(prior['stage'] == 'failed' and prior['ownership_closed'] and check['passed']
         and (check['default_single_cases'], check['default_IQ3_XXS_pair_cases'], check['IQ3_S_pair_cases'])
             == tuple(CASES.values())
         and prior['dependency']['revision'] == recipe.GGML_REV
         and configure[configure.index('-S') + 1] == str(P / 'bench/sycl/cpu_roofline')
         and prior['ggml_inputs']['aggregate_sha256'] == recipe.fingerprint(ggml)
         and all(sha(path) == prior['tools'][path] for path in (recipe.CC, recipe.CXX)),
         'recover the original passing parity stage and leave its later timing failure intact')
    relevant = (*COMPILED[:2], 'include/strata/kernels/cpu', *COMPILED[3:])
    need(not git(P, 'diff', '--name-only', prior['source_commit'], FIX + '^', '--', *relevant),
         'original CPU parity inputs differ from the parent sign implementation')
    return configure, dict(evidence=PRIOR, key=key, source=prior['source_commit'], cases=CASES)


def prepare(checked_rows):
    need(not ROOT.exists() and KEY not in checked_rows, 'retain the existing parity attempt')
    revision = git(P, 'rev-parse', 'HEAD').decode().strip()
    need(not git(P, 'diff', '--name-only', FIX, revision, '--', *COMPILED)
         and git(P, 'diff', '--name-only', FIX + '^', revision, '--', *COMPILED).decode().splitlines() == [SOURCE],
         'this parity run must contain only the committed IQ3_S sign-word change')
    need(git(recipe.GGML, 'rev-parse', 'HEAD').decode().strip() == recipe.GGML_REV
         and sha(recipe.SDK) == recipe.SDK_SHA, 'known GGML source or SDK inventory changed')
    project = recipe.pins(P, revision, recipe.PROJECT)
    ggml = recipe.pins(recipe.GGML, recipe.GGML_REV, ('ggml',))
    allowed = json.loads(recipe.SDK.read_bytes())
    need(len(allowed) == 10945 and all(sha(path) == digest for path, digest in allowed.items()), 'SDK drift')
    environment = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8', LC_ALL='C.UTF-8',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib', OMP_NUM_THREADS='1',
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    allowed.update(recipe.host_inventory(environment)['files'])
    tools = {str(Path(path).resolve()): sha(path) for path in
             (recipe.CC, recipe.CXX, '/usr/bin/cmake', '/usr/bin/ninja', sys.executable)}
    allowed.update(tools)
    for directory in Path('/usr/share').glob('cmake-*/Modules'):
        allowed.update({str(path.resolve()): sha(path) for path in directory.rglob('*') if path.is_file()})
    header = '#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "3cf03257f"\n'
    generated = {str(BUILD / 'ggml/src/ggml-version.h'): sha256(header.encode()).hexdigest()}
    configure, reference = baseline(ggml)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    inventory = ROOT / 'external-inputs.json'
    write(inventory, allowed)
    row = dict(stage='prepared', source_commit=revision, fix=FIX, ggml_revision=recipe.GGML_REV,
        project_inputs=project, ggml_inputs=ggml, tools=tools, machine=recipe.machine(), environment=environment,
        external_inputs=dict(path=str(inventory), sha256=sha(inventory), files=len(allowed)),
        generated_expected=generated, configure=configure, reference=reference,
        build=['/usr/bin/cmake', '--build', str(BUILD), '--target', TARGET, '--parallel', '4'],
        test=[str(BUILD / TARGET), '--iq3s-pair'], test_expected=EXPECTED,
        budgets=dict(build_wall=300, test_wall=45, build_rss=8 << 30, test_rss=1 << 30,
                     text=2 << 20, file=64 << 20, total=384 << 20),
        active=False, ownership_closed=True, complete=False, passed=False, GPU_executed=False)
    save(row, checked_rows)


def run(stage, checked_rows):
    row = json.loads((ROOT / 'record.json').read_bytes())
    need(checked_rows[KEY] == recipe.compact(row) and row['ownership_closed'] and not row['active']
         and row['stage'] == ('prepared' if stage == 'build' else 'built'), 'commit the preceding stage')
    need(recipe.pins(P, row['source_commit'], recipe.PROJECT) == row['project_inputs']
         and recipe.pins(recipe.GGML, row['ggml_revision'], ('ggml',)) == row['ggml_inputs']
         and all(sha(path) == digest for path, digest in row['tools'].items())
         and recipe.machine() == row['machine'], 'frozen source, toolchain, or machine changed')
    recipe.input_map(row)
    scope = ROOT / ('owned-' + stage)
    need(not scope.exists(), 'inspect the original stage owner before running again')

    def persist():
        write(ROOT / 'execution.json', dict(stage=stage, active=bool(owner.active), commands=owner.commands))

    owner = recipe.owned.Owner(scope, persist=persist)
    budgets = row['budgets']

    def command(label, argv, *, build=False):
        entry, output, errors = owner.run(label, argv, row['environment'], ROOT,
            wall=budgets['build_wall' if build else 'test_wall'], cpu=1200 if build else 120,
            rss_cap=budgets['build_rss' if build else 'test_rss'],
            text_cap=budgets['text'], file_cap=budgets['file'], total_cap=budgets['total'])
        need(recipe.owned.closed(entry) and entry['exit_code'] == 0,
             label + ' failed: ' + errors.read_text(errors='replace')[-1500:])
        return entry, output, errors

    binary = BUILD / TARGET
    try:
        if stage == 'build':
            command('configure', row['configure'], build=True)
            command('compile', row['build'], build=True)
            command('deps', ['/usr/bin/ninja', '-C', str(BUILD), '-t', 'deps'])
            row['build_evidence'] = recipe.build_check(row, scope, build_root=BUILD, required_units=UNITS)
            row['binary'] = dict(path=str(binary), bytes=binary.stat().st_size, sha256=sha(binary))
            row['stage'] = 'built'
        else:
            identity = row['binary']
            census = row['build_evidence']['compiler_inputs']
            need(identity['path'] == str(binary) and identity['bytes'] == binary.stat().st_size
                 and identity['sha256'] == sha(binary) and census['sha256'] == sha(census['path'])
                 and all(sha(path) == digest for path, digest in row['build_evidence']['external_link_inputs'].items()),
                 'the committed test executable or build evidence changed')
            entry, output, errors = command('parity', row['test'])
            need(output.read_text() == row['test_expected'] and errors.stat().st_size == 0
                 and sha(binary) == identity['sha256'], 'the exact executable or GGML parity contract differed')
            row.update(stage='closed', complete=True, passed=True, test_exit=entry['exit_code'],
                       cases=CASES)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
        raise
    finally:
        row['ownership_closed'] = owner.active is None and all(recipe.owned.closed(e) for e in owner.commands)
        row['active'] = not row['ownership_closed']
        row.setdefault('owners', {})[stage] = [
            {k: entry.get(k) for k in ('label', 'exit_code', 'direct_child_reaped', 'session_empty')}
            for entry in owner.commands]
        save(row, checked_rows)


if __name__ == '__main__':
    action, = sys.argv[1:]
    need(action in ('prepare', 'build', 'check'), 'select prepare, build or check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe.guards.no_model_owners([])
        checked = committed_rows()
        if action == 'prepare':
            prepare(checked)
        else:
            run(action, checked)
