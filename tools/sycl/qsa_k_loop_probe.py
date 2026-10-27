#!/usr/bin/env python3
"""Finite CPU-only device-IR probe; Main-authored under repository LICENSE."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsa_k_loop_sources as source
import measurement_owner as supervision
from measurement_guards import require

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'qsa-direct-k-invariant-v2'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/qsa-codegen-current.json'
OLD = RESULT.with_name('qsa-k-staging-current.json')
KEY = 'direct_K_loop_invariant'
SELF = str(Path(__file__).resolve().relative_to(P))
SOURCES = [SELF, 'tools/sycl/qsa_k_loop_sources.py', 'tools/sycl/measurement_owner.py',
           'tools/sycl/measurement_guards.py', source.PREFIX]
DEVICE_ONLY_HEADERS = {
    'image_ocl_types.hpp': 'e98f86d24c09c0385f4c37563f9055dda13db74913bf8222b37c637d3fec2cab',
    'group_sort_impl.hpp': '439f7ade46f4b439bede26d8ca5a749fe6de111f228d994df0ffc247034a5757',
    'key_value_iterator.hpp': 'f00494103db0e8e7a1cf6159296869ba218372711c52bb738c9d11a853d7a5e8',
}


def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], timeout=30)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def committed(path):
    raw = git('show', 'HEAD:' + str(path.relative_to(P)))
    require(raw == path.read_bytes(), 'uncommitted measurement input: ' + str(path))
    return json.loads(raw)


def pins(revision):
    result = {}
    for name in SOURCES:
        raw = git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'source changed: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def prepare():
    record = committed(RESULT)
    require(KEY not in record and not ROOT.exists(), 'fresh probe only; inspect previous ownership before retry')
    require(all(x['original_outcome']['ownership_closed'] and not x['original_outcome']['active']
                for x in record.get('direct_K_loop_attempts', [])), 'previous compiler ownership must be closed')
    revision = git('rev-parse', 'HEAD').decode().strip()
    inputs = pins(revision)
    prior = committed(OLD)
    require(prior['generation_retirement']['complete'] and
            not prior['decision_summary']['advance_to_model'], 'prior generation/rejection must stay closed')
    compiler = prior['compiler']
    require(digest(compiler['path']) == compiler['sha256'], 'compiler changed')
    sdk = Path(compiler['path']).parents[1]
    dependencies = {}
    for path, sha in prior['actual_compiled_inputs']['source_sha256'].items():
        if not Path(path).is_file():
            require(path.startswith(str(P / 'bench/sycl/qsa_k_staging')) or
                    path.startswith(str(ROOT.parent / 'qsa-k-staging-packed32-v1')),
                    'unexpected missing original dependency: ' + path)
            continue
        require(digest(path) == sha, 'changed existing compiler input: ' + path)
        dependencies[path] = sha
    # The previous full-build closure omits three device-only headers. Admit
    # them prospectively against the exact official SDK source, not by changing
    # the closed failed plan or refreshing its hashes.
    reference = Path('/home/yayoi/ghq/github.com/intel/llvm/.worktree/toolchain-sycl-v711-b570-20261011')
    extra = {}
    for name in DEVICE_ONLY_HEADERS:
        raw = subprocess.check_output(['git', '-C', str(reference), 'show',
            '504366f4b82ff00bbac7b0c956635eed8bc599d8:sycl/include/sycl/detail/' + name], timeout=30)
        header = sdk / 'include/sycl/detail' / name
        require(raw == header.read_bytes() and b'SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception' in raw,
                'official device-only SDK header differs: ' + name)
        sha = hashlib.sha256(raw).hexdigest()
        require(sha == DEVICE_ONLY_HEADERS[name], 'device-only header qualification changed')
        dependencies[str(header)] = sha
        extra[name] = sha
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    generated = source.generated(P)
    for name, body in generated.items():
        (ROOT / name).write_bytes(body)
    environment = dict(PATH=str(sdk / 'bin') + ':/usr/bin:/bin',
                       LD_LIBRARY_PATH=str(sdk / 'lib'), LANG='C.UTF-8', LC_ALL='C.UTF-8',
                       TMPDIR=str(ROOT / 'tmp'), PYTHONDONTWRITEBYTECODE='1')
    owner = supervision.Owner(ROOT / 'dependency-discovery')
    entry, out, err = owner.run('compiler', ['/usr/bin/ldd', compiler['path']], environment, P,
        wall=20, cpu=20, rss_cap=1 << 30, text_cap=1 << 20, file_cap=1 << 20, total_cap=4 << 20)
    require(supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
            'compiler dependency discovery failed')
    for path in re.findall(r'(?:=>\s+)?(/\S+)\s+\(0x[0-9a-f]+\)', out.read_text()):
        path = str(Path(path).resolve())
        dependencies[path] = digest(path)
    commands = []
    for name in generated:
        stem = Path(name).stem
        commands.append([compiler['path'], '-std=c++20', '-O3', '-DNDEBUG', '-g0',
            '-fsycl', '-fsycl-device-only', '-fsycl-id-queries-range=size_t',
            '-fno-fast-math', '-ffp-contract=off', '-DSTRATA_ENABLE_SYCL=1',
            '-I' + str(P / 'include'), '-I' + str(P / 'include/strata/sycl/compat'),
            '-MD', '-MF', str(ROOT / (stem + '.d')), '-S', '-emit-llvm', str(ROOT / name),
            '-o', str(ROOT / (stem + '.ll'))])
    plan = dict(stage='prepared', source_commit=revision, inputs=inputs, compiler=compiler,
        generated={name: hashlib.sha256(body).hexdigest() for name, body in generated.items()},
        dependencies=dependencies, device_only_headers=extra, commands=commands, environment=environment,
        mapping_proof=source.mapping_proof(),
        budgets=dict(wall=180, cpu=180, rss_cap=4 << 30, text_cap=2 << 20,
                     file_cap=8 << 20, total_cap=32 << 20),
        question='Does row-invariant direct K staging remove the repeated SLM row read/predicate/address dependency from the inner16-step loop,without adding subgroup exchanges or changing any store/read/owner/math/barrier?',
        native_evidence='Source-bound original smoke ASM lines1152..1245,backedge L17232->L16480:load.slm.d64 and row validity/address formation are inside the vector loop. Signed IR division is already lowered to shifts;unsigned spelling alone is not a predicted speed gain.',
        prior_nonduplication='QSA full-unroll experiment changed cached-query/QK matrix loops,not direct-K staging;Page4 changed logical paging. Both failed and remain failed. Both subgroup staging variants also remain rejected.',
        criteria='Compile only2isolated translation units;all actual headers within predeclared pins,exactly1attention and1pack kernel each. Inspect changed IR control/dataflow before selecting a native build. No GPU,linking,model/timing,numerical or production qualification.',
        retention='Keep2source/IR pairs only through immediate lowering/source review;no executable or model generation. Retire ordinary build/dependency logs after compact outcome. Original6e40engine/12fixtures unchanged.')
    record[KEY] = plan
    write(ROOT / 'plan.json', plan)
    write(RESULT, record)
    print('Prepared2CPU-only device-IR commands;commit before run.')


def run():
    record = committed(RESULT)
    plan = record[KEY]
    require(plan['stage'] == 'prepared' and plan == json.loads((ROOT / 'plan.json').read_text()),
            'fresh committed probe only')
    require(pins(plan['source_commit']) == plan['inputs'], 'probe source changed')
    require(digest(plan['compiler']['path']) == plan['compiler']['sha256'], 'compiler changed')
    for path, sha in plan['dependencies'].items():
        require(digest(path) == sha, 'predeclared compiler/header dependency changed: ' + path)
    for name, sha in plan['generated'].items():
        require(digest(ROOT / name) == sha, 'generated source changed')
    outcome = dict(passed=False, complete=False, active=False, ownership_closed=False,
                   GPU_executed=False, source_commit=plan['source_commit'], outputs=[])
    owner = None
    def persist():
        outcome['active'] = owner is not None and owner.active is not None
        write(ROOT / 'owned.json', dict(outcome, commands=owner.commands if owner else []))
    owner = supervision.Owner(ROOT / 'owned', persist)
    try:
        for command in plan['commands']:
            stem = Path(command[-1]).stem
            entry, out, err = owner.run(stem, command, plan['environment'], P, **plan['budgets'])
            require(supervision.closed(entry) and entry['exit_code'] == 0, 'IR compile failed')
            dep = (ROOT / (stem + '.d')).read_text().replace('\\\n', ' ')
            paths = [str(Path(x).resolve()) for x in shlex.split(dep.split(':', 1)[1])]
            actual = {}
            for path in paths:
                sha = digest(path)
                expected = plan['dependencies'].get(path)
                if path == str(ROOT / (stem + '.cpp')):
                    expected = plan['generated'][stem + '.cpp']
                require(sha == expected, 'unqualified actual compiler input: ' + path)
                actual[path] = sha
            target = ROOT / (stem + '.ll')
            text = target.read_text()
            kernels = re.findall(r'^define .*spir_kernel void @([^ (]+)', text, re.M)
            attention = [x for x in kernels if x.startswith('_ZTSZZN') and 'prefix_batch_impl' in x]
            require(len(kernels) == 2 and len(attention) == 1, 'exact pack/attention definitions required')
            outcome['outputs'].append(dict(arm=stem, path=str(target), sha256=digest(target),
                bytes=target.stat().st_size, actual_dependencies=len(actual),
                actual_dependency_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
                attention_kernel=attention[0]))
        outcome.update(passed=True, complete=True)
    except BaseException as error:
        outcome['error'] = repr(error)
        raise
    finally:
        outcome['ownership_closed'] = owner.active is None and all(x.get('direct_child_reaped') and
            x.get('session_empty') and not x.get('survivors') for x in owner.commands)
        persist()
        plan['stage'] = 'closed'
        plan['outcome'] = outcome
        write(RESULT, record)
    print('PASS2CPU-only source-closed IR compilations;native execution remains unqualified.')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
