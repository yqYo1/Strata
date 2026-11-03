#!/usr/bin/env python3
"""Build the independent QSA row-publication comparison; repository LICENSE.

CPU compilation only. Keep the selected model engine intact; no GPU, numerical,
performance, model-quality or full-context qualification follows from this build.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

sys.dont_write_bytecode = True
import cache_map_model_entry as entry

base, P, B, RESULT = entry.base, entry.P, entry.B, entry.RESULT
ROOT, KEY = B/'qsa-subgroup-rows-component-v1', 'qsa_subgroup_rows_component_v1'
SELF = Path(__file__).resolve()
SCOPE = 'qsa_subgroup_rows_contract_20261103'
KERNEL = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
HEADER = 'include/strata/kernels/qsa_bf16_prefix.hpp'
TEST = 'tests/sycl/qsa_bf16_prefix.cpp'
BENCH = 'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
FILES = (KERNEL, HEADER, TEST, BENCH, 'cmake/sycl_backend.cmake')
RUNTIME = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
           'src/kernels/sycl/prompt_attention.cpp')
require, digest, write = base.require, base.digest, base.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('sources', 'commands_executed')}
    out['source_inputs'] = dict(files=len(row['sources']), aggregate_sha256=fingerprint(row['sources']))
    return out


def save(row):
    write(ROOT/'record.json', row)
    data = json.loads(RESULT.read_text());data[KEY] = compact(row);write(RESULT, data)


def sources(revision):
    pins = base.controls(revision, controller=SELF)
    for name in FILES:
        raw = base.basis.build.git(P, 'show', revision + ':' + name)
        reviewed = base.basis.build.git(P, 'show', entry.HISTORY_COMMIT + ':' + name)
        require(raw == reviewed == (P/name).read_bytes(), 'reviewed QSA source changed: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def qualification():
    retired = base.committed(RESULT)['adaptive_fixture_basis_retirement_v1']
    require(all(retired[k] is True for k in ('passed', 'complete', 'deleted_verified',
                'current_admission_after_deletion')) and not entry.model.prior.ROOT.exists(),
            'retire the old generation and verify its actual reader before another build')
    native, gate, providers = entry.proof()
    require(native['binaries']['strata']['sha256'] == retired['selected_engine_sha256'],
            'preserve the selected model fallback')
    review = entry.template.shared.runtime.validation.committed(entry.INDEX, (SCOPE,))
    require(review['status'] == 'reviewed_source_only' and review['ownership_closed'] is True
            and review['worker_done_accepted'] is True
            and all(review['source_sha256'][name] == digest(P/name) for name in (KERNEL, HEADER))
            and review['fixture_followup']['file'] == TEST
            and review['fixture_followup']['sha256'] == digest(P/TEST)
            and review['timing_gate_source_followup']['file'] == BENCH
            and review['timing_gate_source_followup']['sha256'] == digest(P/BENCH),
            'exact independent QSA source review required')
    return native, gate, providers


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row), 'commit the exact component plan/result')
    require(sources(row['source_commit']) == row['sources'], 'frozen component source changed')
    native, gate, providers = qualification()
    require(row['native_source_commit'] == native['source_commit']
            and row['selected_engine'] == native['binaries']['strata']
            and row['fixture_basis'] == gate['mathematical_fixture_basis'], 'selected fallback changed')
    require(digest(row['compiler']['path']) == row['compiler']['sha256']
            and row['compiler'] == native['compiler'], 'pinned compiler changed')
    allowed = entry.device.allowed_inputs(native)
    for name, sha in row['sources'].items(): allowed[str((P/name).resolve())] = sha
    path = ROOT/HEADER
    require(path.read_bytes() == (P/HEADER).read_bytes() and digest(path) == row['overlay_sha256'],
            'typed-header overlay changed')
    allowed[str(path)] = row['overlay_sha256']
    require(base.host_link_inventory.inventory(base.COMPILER, row['environment']) == row['host_link_inputs'],
            'compiler/host-link inputs changed')
    return allowed


def prepare():
    require(KEY not in base.committed(RESULT) and not ROOT.exists(), 'one fresh component generation')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = sources(revision);native, gate, providers = qualification()
    ROOT.mkdir(mode=0o700)
    for name in ('build', 'tmp', 'cache'): (ROOT/name).mkdir()
    overlay = ROOT/HEADER;overlay.parent.mkdir(parents=True);overlay.write_bytes((P/HEADER).read_bytes())
    environment = dict(native['environment'], TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache'))
    flags = ['-std=c++20', '-O3', '-DNDEBUG', '-g0', '-fsycl', '-fsycl-id-queries-range=size_t',
        '-fno-fast-math', '-ffp-contract=off', '-DSTRATA_ENABLE_SYCL=1', '-DSTRATA_VERSION="0.1.38"',
        '-I'+str(ROOT/'include'), '-I'+str(entry.model.ROOT/'source/include'),
        '-I'+str(entry.model.ROOT/'source/include/strata/sycl/compat')]
    native_files = [entry.model.ROOT/'source'/n for n in RUNTIME] + [P/KERNEL, P/TEST, P/BENCH]
    objects = [ROOT/'build'/f'input-{i}.o' for i in range(len(native_files))]
    commands = []
    for i,(path,obj) in enumerate(zip(native_files, objects)):
        defines = ['-DSTRATA_QSA_TEST_SUBGROUP_ROWS=1', '-DSTRATA_QSA_COMPARE_PREFIX33=1',
                   '-DSTRATA_QSA_PACKED_GUARD=17'] if path == P/TEST else \
                  ['-DSTRATA_QSA_BENCH_SUBGROUP_ROWS=1'] if path == P/BENCH else []
        commands.append(dict(label=f'object{i}', argv=[native['compiler']['path'], *flags, *defines,
            '-MD', '-MF', str(obj.with_suffix('.d')), '-c', str(path), '-o', str(obj)]))
    for i,(name,selected) in enumerate((('numerical', objects[:5]+[objects[5]]),
                                       ('paired', objects[:5]+[objects[6]]))):
        commands.append(dict(label=f'link{i}', argv=[native['compiler']['path'], *flags,
            *map(str,selected), '-pthread', '-ldl', '-Wl,--trace', '-o', str(ROOT/'build'/name)]))
    row = dict(stage='prepared', source_commit=revision, sources=pins,
        native_source_commit=native['source_commit'], selected_engine=native['binaries']['strata'],
        fixture_basis=gate['mathematical_fixture_basis'], reviewed_source_commit=entry.HISTORY_COMMIT,
        reviewed_scope=SCOPE, compiler=native['compiler'], environment=environment,
        host_link_inputs=base.host_link_inventory.inventory(base.COMPILER, environment),
        overlay_sha256=digest(overlay), commands=commands,
        source_origin='Current independently authored QSA/fixtures unchanged from reviewed c23a snapshot;four runtime/prompt files and remaining headers come from the selected479a native projection. No production selector or engine replacement.',
        budgets=dict(wall=420, cpu=1200, rss_cap=12<<30, text_cap=16<<20,
                     file_cap=512<<20, total_cap=2<<30), whole_wall_seconds=1800,
        plan='Compile seven units and link numerical/paired fixtures only;strict C++20/SYCL math. Freeze exact source,generated overlay,compiler,all actual dependencies and external host-link inputs. No GPU execution follows automatically.',
        device_next='Commit a bounded traced smoke and full numerical plan before device use. Only then three fresh --current-prefill processes0..2,each six alternating pairs per six shapes;10ms interval floor and clock cost<1percent. Preserve original failed/ineligible outcomes;no model gain implied.',
        retention='Selected552b nine-ELF engine remains the known-good fallback. This is the sole additional active component build;legacy PR6 provenance hold is separate. Retire closed raw build/JIT/trace output and this comparison generation after a recorded selection and source/recipe recovery.',
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False,
        model_executed=False, inference_adopted=False)
    save(row)
    print('Prepared nine CPU compile/link commands;commit before build. GPU UNRUN.', flush=True)


def build():
    row = json.loads((ROOT/'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT/'owned-build').exists(), 'fresh committed build only')
    allowed = current(row);deadline = time.monotonic()+row['whole_wall_seconds'];owner = None
    def persist():
        write(ROOT/'execution.json', dict(active=bool(owner and owner.active),
              commands=owner.commands if owner else [], GPU_executed=False))
    try:
        owner = base.supervision.Owner(ROOT/'owned-build', persist)
        links = []
        for command in row['commands']:
            remaining = deadline-time.monotonic();require(remaining>1, 'whole build deadline')
            require(base.supervision.artifact_bytes(ROOT) <= row['budgets']['total_cap'], 'whole build bytes')
            print('CPU compile/link:', command['label'], flush=True)
            result,out,err = owner.run(command['label'], command['argv'], row['environment'], P,
                **dict(row['budgets'], wall=min(remaining,row['budgets']['wall'])))
            require(base.supervision.closed(result) and result['exit_code'] == 0,
                    command['label']+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-5000:])
            if command['label'].startswith('link'): links.append(out.read_bytes())
        current(row)
        actual = {}
        for dep in (ROOT/'build').glob('*.d'):
            for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
                path = str(Path(name).resolve());sha = digest(path)
                require(allowed.get(path) == sha, 'unfrozen actual compiler input: '+path)
                actual[path] = sha
        require(len(actual)>600, 'incomplete compiler dependency closure')
        write(ROOT/'compiler-inputs.json', actual)
        # Retain the two original linker traces; this combined private view is
        # consumed by the existing observed-link-input checker only.
        (ROOT/'owned-build/compile.stdout').write_bytes(b'\n'.join(links))
        link_inputs = base.traced_link_inputs(allowed,ROOT)
        row.update(stage='built-not-executed', passed=True, complete=True,
            compiler_inputs=dict(path=str(ROOT/'compiler-inputs.json'), sha256=digest(ROOT/'compiler-inputs.json'), files=len(actual)),
            actual_link_inputs=link_inputs,
            binaries={name:dict(path=str(ROOT/'build'/name), sha256=digest(ROOT/'build'/name),
                bytes=(ROOT/'build'/name).stat().st_size) for name in ('numerical','paired')})
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None
            and all(base.supervision.closed(v) for v in owner.commands)),
            commands_executed=owner.commands if owner else [])
        save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    require(row['passed'] and row['complete'] and row['ownership_closed'], 'component build did not qualify')


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'],['build']), 'use prepare|build;GPU execution is separate')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        {'prepare':prepare,'build':build}[sys.argv[1]]()
