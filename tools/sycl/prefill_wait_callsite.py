#!/usr/bin/env python3
"""Resolve host queue waits by caller on the unchanged native ELF. Repository LICENSE."""

import collections
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import compat_wait_audit as audit
import engine_once_wait_audit as launch
import prefill_prefix_expert_profile as cpu

B, P, RESULT = audit.B, audit.P, audit.RESULT
basis = audit.basis
require, sha, signature = audit.require, audit.sha, audit.signature
ROOT = B / 'prefill-compat-wait-callsite-v1'
KEY = 'prefill_compat_wait_callsite_v1'
WRAPPER = 'tools/sycl/engine_once_wait_audit.py'
THIS = 'tools/sycl/prefill_wait_callsite.py'
START = 'strata MoE accumulator admission: '
STOP = 'strata prefill cpu wait: '
ROLES = ('unloaded-before', 'loader-audit', 'unloaded-after')
READ, RUNS = 32769, 2
MIN_COVERAGE = .98


def read(path):
    return json.loads(path.read_bytes())


def caller_table(elf):
    text = audit.clean_command(['/usr/bin/objdump', '-d', elf])
    heading = re.compile(r'^[0-9a-f]+ <(.+)>:$')
    native_call = re.compile(r'^\s*([0-9a-f]+):\s*((?:[0-9a-f]{2}\s+)+)call\s+[0-9a-f]+ <'
                             + re.escape(audit.WRAPPER) + r'>$')
    current = None
    callers = {}
    for line in text.splitlines():
        header = heading.fullmatch(line)
        if header:
            current = header[1]
        match = native_call.fullmatch(line)
        if match:
            require(current, 'compatibility call has no enclosing symbol')
            address = int(match[1], 16) + len(match[2].split())
            require(format(address, 'x') not in callers, 'ambiguous native direct caller')
            callers[format(address, 'x')] = current
    require(len(callers) >= 20 and any('run_layer_major' in s for s in callers.values()),
            'unexpected compatibility caller population')
    return callers


def qualified_audit():
    row = read(audit.ROOT / 'record.json')
    audit.current(row)
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and row['ownership_closed']
            and not row['active'] and row['host_control']['clean']['exit'] == row['host_control']['audit']['exit'] == 0,
            'closed CPU caller, exception and result control required')
    native = row['native']
    return dict(library=row['artifacts'][audit.A.name], return_pc=native['native_return'][2:],
                parent_offset=native['stack_offset']), signature(audit.compact(row))


def summarize(log, callers):
    lines = log.splitlines()
    begin, = [i for i, line in enumerate(lines) if line.startswith(START)]
    end, = [i for i, line in enumerate(lines) if line.startswith(STOP)]
    require(begin < end, 'long-prefill observation window differs')
    expected = cpu.profile.d.gen.common.extract_cpu_observers(log, 4)['wait']
    expected_count = int(expected['compat_count'])
    expected_wall = float(expected['compat_wall_s'])
    expected_cpu = float(expected['compat_cpu_s'])
    counts = collections.defaultdict(lambda: dict(calls=0, cpu_ns=0, wall_ns=0))
    matched = 0
    previous = 0
    for line in lines[begin:end]:
        if not line.startswith('strata wait site:'): continue
        match = re.fullmatch(r'strata wait site: ordinal=(\d+) caller=([0-9a-f]+) wall_ns=(\d+) cpu_ns=(\d+)', line)
        require(match, 'malformed loader wait sample')
        ordinal, address, wall, cpu_ns = match.groups()
        ordinal, wall, cpu_ns = int(ordinal), int(wall), int(cpu_ns)
        require(ordinal > previous and 0 <= cpu_ns <= wall, 'invalid sample order or clock')
        previous = ordinal
        matched += int(address in callers)
        row = counts[address]
        row['calls'] += 1
        row['cpu_ns'] += cpu_ns
        row['wall_ns'] += wall
    n = sum(row['calls'] for row in counts.values())
    sampled_cpu = sum(row['cpu_ns'] for row in counts.values()) / 1e9
    sampled_wall = sum(row['wall_ns'] for row in counts.values()) / 1e9
    require(expected_count > 0 and expected_cpu > 0 and expected_wall > 0
            and MIN_COVERAGE * expected_count <= n <= expected_count
            and matched >= MIN_COVERAGE * n
            and MIN_COVERAGE * expected_cpu <= sampled_cpu <= expected_cpu * 1.02
            and MIN_COVERAGE * expected_wall <= sampled_wall <= expected_wall * 1.02,
            'sampled call and clock coverage insufficient for attribution')
    return dict(compatibility_calls=expected_count, observed_calls=n, resolved_calls=matched,
        scoped_cpu_s=expected_cpu, scoped_wall_s=expected_wall, observed_cpu_s=sampled_cpu,
        observed_wall_s=sampled_wall,
        by_callsite=[dict(caller_pc=address, symbol=callers.get(address), **value)
            for address,value in sorted(counts.items(), key=lambda a: a[1]['wall_ns'], reverse=True)])


def campaign():
    ref = cpu.stage.model_context('timing')
    proof, audit_sha = qualified_audit()
    scope = format(int(proof['return_pc'], 16), 'x')
    audit_record = read(audit.ROOT / 'record.json')['native']
    source = cpu.read(cpu.build.STAGE_ROOT / 'record.json')
    path = cpu.DECISIONS
    doc_digest = sha(P / path)
    doc_delta = {path: dict(original=source['controller_sources'][path], current=doc_digest)}
    spec = importlib.util.spec_from_file_location('prefill_wait_callsite', basis.core.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (READ,) * 3, [(READ, 1)] * 3, RUNS, True
    orig_current = ctx.current
    table = None

    def callsites(row):
        nonlocal table
        manifest = row['caller_table']
        if table is None:
            path = Path(manifest['path'])
            require(sha(path) == manifest['sha256'], 'frozen direct caller table differs')
            table = read(path)
        return table

    def environment(row, plan):
        env = ref.environment(row,plan)
        env.update({cpu.stage.SELECTOR: 'all', cpu.SELECTOR: '1'})
        require(not env.get('LD_AUDIT') and not env.get('STRATA_TRACE')
                and env['STRATA_SYCL_PREFILL_MOE_CENSUS'] == '0',
                'preserve the matched CPU wait scopes and private native audit')
        return env

    def current(row):
        providers = orig_current(row)
        require(row['loader_audit_host_proof'] == audit_sha
                and qualified_audit() == (proof, audit_sha)
                and row['binary'] == audit_record['binary']
                and len(callsites(row)) == row['caller_table']['entries']
                and sha(row['caller_table']['path']) == row['caller_table']['sha256']
                and all(p.get('queue_wait_audit') == (proof if i == 1 else None) and p['role'] == ROLES[i]
                        and not p['trace_progress'] and p['read'] == READ for i,p in enumerate(row['plans'])),
                'frozen native loader audit comparison changed')
        return providers

    def request_route(log, plan):
        route = ref.request_route(log, plan)
        route['cpu_partition'] = cpu.cpu_partition(log)
        if plan.get('queue_wait_audit'):
            route['wait_callsites'] = summarize(log, table)
        else:
            require('strata wait site:' not in log and 'strata wait audit:' not in log,
                    'unloaded control has audit records')
        return route

    def validate_model_text(log, plan):
        require(sum(line.startswith(STOP) for line in log.splitlines()) == RUNS, 'expected CPU scope population differs')
        if not plan.get('queue_wait_audit'):
            require('strata wait audit:' not in log, 'unloaded native child has audit output')
            return
        headers = [line for line in log.splitlines() if line.startswith('strata wait audit:')]
        require(len(headers)==2 and headers[0] == f'strata wait audit: ready=1 return_pc={scope} parent_offset={proof["parent_offset"]}',
                'native loader audit activation missing')
        done = re.fullmatch(r'strata wait audit: entered=(\d+) exited=(\d+) rejected=0', headers[1])
        require(done and int(done[1]) == int(done[2]) > 0
                and int(done[1]) == sum(s.startswith('strata wait site:') for s in log.splitlines()),
                'missing or incomplete loader exit population')

    def decision(processes):
        record = ctx.profile_decision(processes)
        record.update(limits='Compares only the additional loader observer with already qualified CPU wait scopes. '
                     'Source-bound wrapper intervals show wait service. A caller interval can contain queued GPU work.')
        return record

    ctx.basis = lambda: cpu.admitted_stage_basis(documentation_delta=doc_delta)
    ctx.current, ctx.environment, ctx.check_sample = current, environment, ref.check_sample
    ctx.request_route, ctx.validate_model_text, ctx.decision = request_route, validate_model_text, decision
    return ctx, proof, audit_sha


def prepare(ctx, proof, audit_sha):
    ctx.prepare()
    row = read(ROOT / 'record.json')
    envs = []
    row['configuration']['exe'] = str(P / WRAPPER)
    for i, plan in enumerate(row['plans']):
        plan.update(role=ROLES[i], variant='all', bank=512, window=10, relocate=0, trace_progress=False,
                    upload='jit', cpu_wait_scopes=True)
        directory = Path(plan['root'])
        f = directory / 'config.json'
        ctx.write(f, dict(read(f), exe=str(P / WRAPPER)))
        if i == 1:
            plan['queue_wait_audit'] = proof
            f2 = directory / 'plan.json'
            ctx.write(f2, dict(read(f2), queue_wait_audit=proof))
        require(ctx.comparison.fingerprint(ctx.environment(row,plan)) == plan['environment_sha256'],
                'prepared parent environment differs')
        envs.append({k:v.replace(plan['root'],'<case>') for k,v in ctx.environment(row,plan).items()})
        plan['generated'] = {s:sha(directory / s) for s in plan['generated']}
    require(all(x == envs[0] for x in envs), 'another parent environment setting changed')
    calls = caller_table(row['binary']['path'])
    out = ROOT / 'caller-table.json'
    ctx.write(out, calls)
    row.update(loader_audit_host_proof=audit_sha, caller_table=dict(path=str(out), sha256=sha(out), entries=len(calls)),
        plan=dict(question='Which real long-prefill compatibility-queue calls consume the measured wait time?',
            fixed='Same retained ELF and providers, early prefix issue, native1, deferred first reader, all-stage, '
                  '512 bank slots, 8192 chunks, BF16, 32769 inputs, 24 thinking-enabled outputs, immutable decode.',
            comparison='Three fresh owned processes, two requests per process. All three use CPU wait scopes. '
                       'Only the middle native executable loads the validated x86-64 rtld audit library.',
            gate='Charge all observed overhead. Require prompt/TTFT no more than 1.20x either corresponding '
                 'control, exact dispatch and logical work, resolved and conserved queue wait accounting, and '
                 'normal closed ownership, unchanged driver and quiet validation.',
            stopping='Stop first invalid gate, ownership, fault or resource-budget problem. Preserve every earlier outcome.',
            limits='Direct wrapper callsites at the exact known ABI boundary. No dynamic library calls or GPU '
                   'execution metrics elsewhere. Do not interpret wait intervals as removable critical-path cost.'))
    ctx.save(row)


def execute(ctx, index):
    row = read(ROOT / 'record.json')
    target = row['plans'][index]

    def check_native_env(observed, expected, plan):
        if plan.get('queue_wait_audit'):
            audit_plan = dict(native_validation_probe=row['native_validation_probe'], queue_wait_audit=plan['queue_wait_audit'])
            ref = ctx.environment(row, plan)
            expected = launch.native_environment(audit_plan, ref)
        require(all(observed.get(k.encode()) == v.encode() for k,v in expected.items())
                and set(observed) - {k.encode() for k in expected} <= {b'PWD',b'OLDPWD',b'SHLVL',b'_'},
                'actual immutable native launch environment differs')

    def after_execution(outcome, original, plan):
        cpu.compare_work(outcome, original, plan)
        outcome['audit_loaded'] = bool(plan.get('queue_wait_audit'))

    ctx.after_execution, ctx.check_native_environment = after_execution, check_native_env
    ctx.native_extra_maps = lambda _: {target['queue_wait_audit']['library']['path']} if target.get('queue_wait_audit') else set()
    basis.comparison.run(index, ctx)


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    action, = sys.argv[1:]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit.guards.no_model_owners([])
        cpu.build.ROOT, cpu.build.KEY = cpu.build.STAGE_ROOT, cpu.build.STAGE_BUILD
        ctx, proof, audit_sha = campaign()
        if action == 'prepare':
            prepare(ctx, proof, audit_sha)
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            execute(ctx, int(action))
