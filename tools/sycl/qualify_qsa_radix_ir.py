#!/usr/bin/env python3
"""Prospective source-identity-only IR gate on an unchanged compiled generation.

The original build refusal remains failed. No compilation or GPU work occurs.
Only the exact expected input path in three source-identity fields is ignored;
every instruction, function, other attribute and metadata byte must match.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import build_qsa_radix as b

KEY = 'source_identity_IR_qualification'
require, digest, write = b.require, b.digest, b.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inputs(revision):
    pins = b.inputs(revision)
    name = str(Path(__file__).resolve().relative_to(b.P))
    raw = b.base.component.gen.git('show', revision + ':' + name)
    require(raw == (b.P / name).read_bytes(), 'commit exact IR qualifier first')
    pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def asset(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def compiled(scope):
    old = scope['build']
    require(old['stage'] == 'closed' and not old['outcome']['passed'] and
            old['outcome']['ownership_closed'] and not old['outcome']['GPU_executed'] and
            old['outcome'].get('error') ==
                "RuntimeError('host observation seam changed device IR: original')",
            'only the exact closed source-identity refusal is eligible')
    plan = json.loads((b.ROOT / 'plan.json').read_text())
    b.admit(plan)
    private = json.loads((b.ROOT / 'record.json').read_text())
    commands = private.pop('commands')
    require(private == old['outcome'] and len(commands) == len(plan['commands']) == 15 and
            [c['label'] for c in commands] == [c[0] for c in plan['commands']] and
            all(b.base.runtime.supervision.closed(c) and c['exit_code'] == 0 for c in commands),
            'all original compile/link/IR commands must have closed successfully')
    actual = json.loads((b.ROOT / 'compiler-inputs.json').read_text())
    require(actual and all(digest(p) == sha for p, sha in actual.items()),
            'original actual compiled inputs changed')
    assets = {k: asset(b.ROOT / n) for k, n in (
        ('binary', 'fixture'), ('observed_binary', 'fixture-observed'),
        ('actual_compiler_inputs', 'compiler-inputs.json'),
        ('kernel_inventory', 'kernel-inventory.json'))}
    assets['actual_compiler_inputs'].update(files=len(actual), uncovered=0)
    sources = {}
    for label, command in plan['commands']:
        if label.endswith('-IR'):
            require(command[-2] == '-o' and command[-1] == str(b.ROOT / (label[:-3] + '.ll')),
                    'unexpected original IR recipe')
            sources[label[:-3]] = command[-3]
    require(set(sources) == {'original', 'proposal', 'observed-original', 'observed-proposal'},
            'four exact original IR recipes required')
    assets['LLVM'] = {name: asset(b.ROOT / (name + '.ll')) for name in sources}
    return assets, sources


def canonical(path, expected_source):
    body = path.read_text()
    require(re.findall(r"^; ModuleID = '([^']+)'$", body, re.M) == [expected_source] and
            re.findall(r'^source_filename = "([^"\n]+)"$', body, re.M) == [expected_source],
            'IR module/source_filename is not the expected original recipe input')
    paths = re.findall(r'"sycl-module-id"="([^"\n]+)"', body)
    require(paths and set(paths) == {expected_source}, 'unexpected SYCL source-identity attribute')
    normalized = body.replace('"sycl-module-id"="' + expected_source + '"',
                              '"sycl-module-id"="<verified-source-identity>"')
    normalized = re.sub(r'^(?:; ModuleID = .*|source_filename = .*)\n', '', normalized, flags=re.M)
    return normalized, len(paths)


def verify(scope, plan):
    require(inputs(plan['source_commit']) == plan['input_sha256'] and
            fingerprint(scope['build']) == plan['original_failed_build_sha256'] and
            digest(b.ROOT / 'plan.json') == plan['original_plan_sha256'] and
            digest(b.ROOT / 'record.json') == plan['original_record_sha256'],
            'frozen method/original build evidence changed')
    assets, sources = compiled(scope)
    require(assets == plan['compiled_assets'] and sources == plan['IR_source_paths'],
            'original compiled artifacts or effective IR recipes changed')
    pairs = {}
    for name in ('original', 'proposal'):
        clean, clean_count = canonical(b.ROOT / (name + '.ll'), sources[name])
        observed, observed_count = canonical(b.ROOT / ('observed-' + name + '.ll'),
                                              sources['observed-' + name])
        require(clean == observed and clean_count == observed_count,
                'host observation changed nonidentity device IR: ' + name)
        pairs[name] = dict(canonical_sha256=hashlib.sha256(clean.encode()).hexdigest(),
                           verified_source_attributes_per_unit=clean_count)
    return assets, pairs


def prepare():
    record = b.base.component.gen.committed(b.RESULT); scope = record[b.KEY]
    require(KEY not in scope and not scope.get('checks'), 'fresh source-identity gate before native work')
    assets, sources = compiled(scope)
    revision = b.base.component.gen.git('rev-parse', 'HEAD').decode().strip()
    plan = dict(stage='prepared', source_commit=revision, input_sha256=inputs(revision),
        original_failed_build_sha256=fingerprint(scope['build']),
        original_plan_sha256=digest(b.ROOT / 'plan.json'),
        original_record_sha256=digest(b.ROOT / 'record.json'), compiled_assets=assets,
        IR_source_paths=sources, GPU_executed=False, binary_rebuilt=False,
        criteria='New prospective method gate,not a retry or relabeling of the original build failure. Keep exact original compiled artifact bytes,15successful closed command receipts,all actual compiler inputs and recipe. Require each IR module/source_filename and every sycl-module-id attribute to name its exact recipe input. Normalize only those verified source identities;all remaining IR text must be identical for both original/proposal observation pairs. Native resources/correctness/event/service/model gates remain unrun.')
    scope[KEY] = dict(plan=plan, passed=False)
    write(b.RESULT, record)
    print('Prepared exact source-identity method gate;commit before execution.', flush=True)


def run():
    record = b.base.component.gen.committed(b.RESULT); scope = record[b.KEY]
    entry = scope[KEY]; plan = entry['plan']
    require(plan['stage'] == 'prepared' and 'outcome' not in entry, 'fresh committed method plan required')
    outcome = dict(passed=False, complete=False, ownership_closed=True,
                   GPU_executed=False, binary_rebuilt=False)
    try:
        _, pairs = verify(scope, plan)
        outcome.update(passed=True, complete=True, pairs=pairs,
                       device_IR_identical_except_verified_source_identity=True)
    except BaseException as error:
        outcome['error'] = repr(error)
        raise
    finally:
        entry.update(passed=outcome['passed'], outcome=outcome)
        write(b.RESULT, record)
    print('Source-identity-only IR gate passed;no GPU execution or model qualification.', flush=True)


def qualified(scope):
    entry = scope[KEY]; outcome = entry['outcome']
    require(entry['passed'] and outcome['passed'] and outcome['complete'] and
            outcome['ownership_closed'] and not outcome.get('error') and
            outcome['device_IR_identical_except_verified_source_identity'],
            'new prospective source-identity gate must pass')
    assets, pairs = verify(scope, entry['plan'])
    require(pairs == outcome['pairs'], 'qualified IR pair identities changed')
    return dict(assets, passed=True, complete=True, ownership_closed=True,
                device_IR_identical_across_observation=True)


if __name__ == '__main__':
    with (b.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
