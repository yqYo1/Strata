#!/usr/bin/env python3
"""Isolated MIT selector build, never a production selection or GPU launch.

Compile the exact selected runtime/selector and the independently authored
batched selector with the selected selector's flags. Reuse the admitted SDK and
provider inventories, not application code from another engine/worktree.
"""
import ast
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shlex
import sys
import time

sys.dont_write_bytecode = True
import build_qsa_k_model as base
import relink_qsa_k_model as link

P, B, RESULT = base.P, base.B, base.RESULT
ROOT = B / 'qsa-batched-radix-v1'
KEY = 'batched_radix_selector'
HEADER = 'include/strata/kernels/qsa_radix_batched.hpp'
NATIVE = 'src/kernels/sycl/qsa_radix_batched.cpp'
FIXTURE = 'tests/sycl/qsa_batched_radix.cpp'
ORACLE = 'tests/sycl/qsa_select_radix.cpp'
CAPTURE = 'tests/sycl/qsa_event_capture.hpp'
SELECTOR = 'src/kernels/sycl/qsa_select.cpp'
SELECTED = ['src/sycl/runtime.cpp', 'src/sycl/compat.cpp',
            'src/sycl/command_profile.cpp', SELECTOR]
require, digest, write = base.require, base.digest, base.write


def inputs(revision):
    names = {HEADER, NATIVE, FIXTURE, ORACLE, CAPTURE}
    pending, seen = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = str(path.relative_to(P))
        raw = base.component.gen.git('show', revision + ':' + name)
        require(path.read_bytes() == raw, 'commit exact controller first: ' + name)
        names.add(name)
        for node in ast.walk(ast.parse(raw)):
            modules = [v.name for v in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                local = path.parent / ((module or '').split('.')[0] + '.py')
                if local.is_file():
                    pending.append(local)
    pins = {}
    for name in sorted(names):
        raw = base.component.gen.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'source differs: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def foundation():
    # This is a source/SDK dependency, not adoption of the direct-K engine.
    row = link.compiled()
    allowed = base.admit(row)
    for name in SELECTED:
        path = base.ROOT / 'source' / name
        require(path.read_bytes() == base.component.gen.git('show', base.model.BASE + ':' + name),
                'selected native source differs: ' + name)
    return row, allowed


def compact(plan):
    return {k: v for k, v in plan.items() if k not in ('input_sha256',)} | {
        'controller_inputs': dict(files=len(plan['input_sha256']), sha256=hashlib.sha256(
            json.dumps(plan['input_sha256'], sort_keys=True).encode()).hexdigest())}


def generated():
    result = {}
    for name, source, count in [('original-observed.cpp',base.ROOT/'source'/SELECTOR,4),
                                ('proposal-observed.cpp',P/NATIVE,1)]:
        raw = source.read_bytes()
        before = b'finish(stream, event);'
        after = b'::strata::qsa_fixture::observed_finish(stream, event);'
        require(raw.count(before) == count and after not in raw, 'nonunique finish observation seam')
        body = raw.replace(before,after)
        require(body.replace(after,before) == raw, 'observation changed another source region')
        result[name] = ('#include "' + str(P / CAPTURE) + '"\n').encode() + body
    return result


def admit(plan):
    origin, allowed = foundation()
    require(origin['compile_database_sha256'] == plan['selected_compile_database_sha256'],
            'selected compile recipes differ')
    require(inputs(plan['source_commit']) == plan['input_sha256'], 'build inputs changed')
    require(link.provider_inventory() == json.loads(Path(plan['providers']['path']).read_text()) and
            digest(plan['providers']['path']) == plan['providers']['sha256'],
            'prospectively frozen link providers differ')
    providers = json.loads(Path(plan['providers']['path']).read_text())
    allowed.update({v['resolved']: v['sha256'] for v in providers.values()})
    allowed.update({str((P / n).resolve()): sha for n, sha in plan['input_sha256'].items()})
    header = ROOT / HEADER
    require(header.read_bytes() == (P / HEADER).read_bytes() and
            digest(header) == plan['generated_header_sha256'], 'private header differs')
    allowed[str(header.resolve())] = plan['generated_header_sha256']
    for name, raw in generated().items():
        path = ROOT / name
        require(path.read_bytes() == raw and digest(path) == plan['generated_input_sha256'][name],
                'generated observed source differs')
        allowed[str(path.resolve())] = plan['generated_input_sha256'][name]
    return allowed


def prepare():
    record = base.component.gen.committed(RESULT)
    cohort = record[base.KEY]['official_model_qualification']
    require(cohort['complete'] and not cohort['active'] and cohort['ownership_closed'],
            'close the frozen cohort before another build; no simultaneous work')
    require(KEY not in record and not ROOT.exists(), 'fresh isolated generation only')
    review = base.runtime.validation.committed(base.runtime.validation.INDEX,
        ('qsa_batched_radix_histogram_contract_20261028',))
    require(review['worker_closed'] and review['native_review']['decision'].startswith('Source contract'),
            'closed independent selector source review required')
    revision = base.component.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    require(all(pins[n] == sha for n, sha in review['source_files'].items()),
            'selector changed since the independent source review')
    origin, _ = foundation()
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    (ROOT / HEADER).parent.mkdir(parents=True)
    (ROOT / HEADER).write_bytes((P / HEADER).read_bytes())
    generated_inputs = generated()
    for name, raw in generated_inputs.items(): (ROOT / name).write_bytes(raw)
    providers = link.provider_inventory()
    write(ROOT / 'providers.json', providers)
    database = json.loads((base.ROOT / 'build/compile_commands.json').read_text())
    commands, objects = [], []
    template = None
    for index, name in enumerate(SELECTED):
        path = base.ROOT / 'source' / name
        rows = [r for r in database if Path(r['file']) == path]
        require(len(rows) == 1, 'one selected compile recipe per source')
        args = shlex.split(rows[0]['command'])
        require(args[0] == origin['compiler']['path'] and args.count('-c') == args.count('-o') == 1,
                'unexpected selected compile recipe')
        if name == SELECTOR:
            template = list(args)
        obj = ROOT / ('input-' + str(index) + '.o')
        args[args.index('-o') + 1] = str(obj)
        args += ['-MD', '-MF', str(obj.with_suffix('.d'))]
        commands.append(('compile-' + str(index), args))
        objects.append(obj)
    require(template is not None and '-ffp-contract=off' in template and '-ffast-math' not in template,
            'selected selector math flags differ')
    for name in (NATIVE, FIXTURE):
        obj = ROOT / ('input-' + str(len(objects)) + '.o')
        args = list(template)
        args[args.index('-c') + 1] = str(P / name)
        args[args.index('-o') + 1] = str(obj)
        args.insert(1, '-I' + str(ROOT / 'include'))
        args += ['-MD', '-MF', str(obj.with_suffix('.d'))]
        commands.append(('compile-' + str(len(objects)), args))
        objects.append(obj)
    for path in [ROOT/'original-observed.cpp',ROOT/'proposal-observed.cpp',P/FIXTURE]:
        obj = ROOT / ('input-' + str(len(objects)) + '.o')
        args = list(template)
        args[args.index('-c') + 1] = str(path)
        args[args.index('-o') + 1] = str(obj)
        args.insert(1, '-I' + str(ROOT / 'include'))
        if path == P / FIXTURE: args.append('-DSTRATA_QSA_EVENT_CAPTURE=1')
        args += ['-MD','-MF',str(obj.with_suffix('.d'))]
        commands.append(('compile-' + str(len(objects)),args)); objects.append(obj)
    compiler = origin['compiler']['path']
    for name, chosen in [('fixture',objects[:6]),('fixture-observed',objects[:3]+objects[6:])]:
        commands.append((name, [compiler, '-std=c++20', '-O3', '-DNDEBUG', '-fsycl',
            '-fsycl-id-queries-range=size_t', '-ffp-contract=off', *map(str, chosen),
            '-pthread', '-ldl', '-save-temps=obj', '-v', '-Wl,--trace', '-o', str(ROOT / name)]))
    for name, path in [('original', base.ROOT / 'source' / SELECTOR), ('proposal', P / NATIVE),
                       ('observed-original',ROOT/'original-observed.cpp'),
                       ('observed-proposal',ROOT/'proposal-observed.cpp')]:
        args = list(template)
        del args[args.index('-o'):args.index('-o') + 2]
        del args[args.index('-c'):args.index('-c') + 2]
        args.insert(1, '-I' + str(ROOT / 'include'))
        args += ['-fsycl-device-only', '-MD', '-MF', str(ROOT / (name + '.d')),
                 '-S', '-emit-llvm', str(path), '-o', str(ROOT / (name + '.ll'))]
        commands.append((name + '-IR', args))
    environment = dict(origin['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    plan = dict(stage='prepared', source_commit=revision, input_sha256=pins,
        selected_native_source=base.model.BASE, selected_compile_database_sha256=origin['compile_database_sha256'],
        compiler=origin['compiler'], sdk=origin['sdk'],
        providers=dict(path=str(ROOT / 'providers.json'), sha256=digest(ROOT / 'providers.json'), files=len(providers)),
        generated_header_sha256=pins[HEADER], environment=environment, commands=commands,
        generated_input_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in generated_inputs.items()},
        budgets=dict(wall=420, cpu=1200, rss_cap=12 << 30, text_cap=16 << 20,
                     file_cap=512 << 20, total_cap=2 << 30), total_wall_seconds=2400,
        active=False, ownership_closed=True, GPU_executed=False, model_selected=False,
        provenance='Own MIT Strata radix rank/emission contract and reviewed batched histogram;independent weighted-cell oracle. Selected6e40 native/runtime sources and exact selected flags. Apache LLVM SDK only;no XeStrata application/helpers/tests.',
        criteria='Build only. Complete actual compiler/link closure;record every LLVM kernel definition without assuming whole-TU counts from a native image. Native resources,small traced correctness,matching output IDs and independent-process clean/service/CPU speed are subsequent gates. Decode and selected production unchanged.',
        retention='One component generation with two necessary fixtures:unmodified-source clean calls and host-only captured-event observation. No third full engine. Require identical device IR after removing only module/source-file labels. Retire normal output/IR/intermediates and rejected source/exports after compact decision and exact source/recipe recovery;do not extend closed direct-K generation retention.')
    admit(plan)
    write(ROOT / 'plan.json', plan)
    record[KEY] = dict(build=compact(plan), passed=False, model_selected=False)
    write(RESULT, record)
    print('Prepared selector recipe;commit this frozen plan before building.', flush=True)


def build():
    record = base.component.gen.committed(RESULT)
    plan = json.loads((ROOT / 'plan.json').read_text())
    require(plan['stage'] == 'prepared' and record[KEY]['build'] == compact(plan) and
            not (ROOT / 'owned').exists(), 'fresh committed build plan required')
    allowed = admit(plan)
    outcome = dict(passed=False, complete=False, active=False, ownership_closed=False, GPU_executed=False)
    owner = None
    def persist():
        write(ROOT / 'record.json', dict(outcome, active=bool(owner and owner.active),
                                        commands=owner.commands if owner else []))
    owner = base.runtime.supervision.Owner(ROOT / 'owned', persist)
    deadline = time.monotonic() + plan['total_wall_seconds']
    def monitor(io, entry):
        io.proc.stdin.close()
        next_size = 0
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(base.runtime.supervision.artifact_bytes(ROOT) <= plan['budgets']['total_cap'],
                        'whole selector build artifact budget')
                next_size = time.monotonic() + .5
            if select.select([io.proc.stdout],[],[],.02)[0]:
                io.read(); io.pending.clear()
        io.pending.clear(); io.exit(1)
    try:
        for label, args in plan['commands']:
            require(time.monotonic() < deadline, 'whole build deadline')
            print('CPU-only selector build:', label, flush=True)
            entry, out, err = owner.run(label, args, plan['environment'], ROOT,interaction=monitor,
                **(plan['budgets'] | {'wall': min(plan['budgets']['wall'], deadline - time.monotonic())}))
            require(base.runtime.supervision.closed(entry) and entry['exit_code'] == 0,
                    'compile/link failed: ' + err.read_text(errors='replace')[-2000:])
            if label.startswith('compile-'):
                obj = Path(args[args.index('-o') + 1]); allowed[str(obj)] = digest(obj)
            if label in ('fixture','fixture-observed'):
                outcome.setdefault('link_inputs',{})[label] = link.traced_inputs(out.read_text(), err.read_text(), allowed, ROOT)
        admit(plan)
        actual = {}
        for label, args in plan['commands']:
            if '-MF' not in args:
                continue
            dep = Path(args[args.index('-MF') + 1])
            require(dep.is_file(), 'missing actual dependencies: ' + label)
            names = shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
            require(names, 'empty actual dependencies')
            for name in names:
                path = str(Path(name).resolve()); sha = digest(path)
                require(allowed.get(path) == sha, 'unfrozen actual compiler input: ' + path)
                actual[path] = sha
        write(ROOT / 'compiler-inputs.json', actual)
        def asset(path):
            return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
        inventories = {}
        for name in ('original', 'proposal','observed-original','observed-proposal'):
            text = (ROOT / (name + '.ll')).read_text()
            headers = re.findall(r'^define[^\n]*spir_kernel[^\n]*', text, re.M)
            require(headers, 'missing compiled kernel definitions')
            inventories[name] = [dict(name=re.search(r'@([^ (]+)\(', h)[1],
                arguments=re.search(r'@[^ (]+\((.*)\)[^{]*\{', h)[1],
                subgroup_metadata=re.search(r'!intel_reqd_sub_group_size !(\d+)', h)[1]
                    if '!intel_reqd_sub_group_size ' in h else None) for h in headers]
        write(ROOT / 'kernel-inventory.json', inventories)
        for name in ('original','proposal'):
            canonical = lambda path: re.sub(r'^(?:; ModuleID = .*|source_filename = .*)\n','',path.read_text(),flags=re.M)
            require(canonical(ROOT/(name+'.ll')) == canonical(ROOT/('observed-'+name+'.ll')),
                    'host observation seam changed device IR: ' + name)
        outcome.update(passed=True, complete=True, binary=asset(ROOT / 'fixture'),
            observed_binary=asset(ROOT/'fixture-observed'), device_IR_identical_across_observation=True,
            actual_compiler_inputs=asset(ROOT / 'compiler-inputs.json') | {'files': len(actual), 'uncovered': 0},
            LLVM={n:asset(ROOT / (n + '.ll')) for n in inventories},
            kernel_inventory=asset(ROOT / 'kernel-inventory.json') | {
                'definitions': {n:len(v) for n,v in inventories.items()}})
    except BaseException as error:
        outcome['error'] = repr(error)
        raise
    finally:
        outcome['ownership_closed'] = owner.active is None and all(
            base.runtime.supervision.closed(c) for c in owner.commands)
        outcome['passed'] = bool(outcome['passed'] and outcome['ownership_closed'] and 'error' not in outcome)
        persist()
        record[KEY]['build']['stage'] = 'closed'
        record[KEY]['build']['outcome'] = outcome
        write(RESULT, record)
    print('Build closed;native/property/correctness/service/model gates remain unrun.', flush=True)


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'build': build}[sys.argv[1]]()
