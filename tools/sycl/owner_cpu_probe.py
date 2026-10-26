#!/usr/bin/env python3
"""Main-authored CPU-only attribution of the current process supervisor.

Repository LICENSE. No GPU API, native build or inference speed qualification.
"""
import cProfile
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time
sys.dont_write_bytecode = True
import prefill_clock_diagnostic as diagnostic

P, B, RESULT = diagnostic.P, diagnostic.B, diagnostic.RESULT
ROOT = B / 'owner-cpu-probe-v1'
KEY = 'owner_cpu_probe_v1'
require, digest, write = diagnostic.require, diagnostic.digest, diagnostic.write


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    literal = json.dumps(KEY) + ':'; value = json.dumps(row, indent=2).replace('\n', '\n    ')
    if KEY in desired['historical_regression_audit']:
        require(raw.count(literal) == 1, 'ambiguous owner probe scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"historical_regression_audit": {'
        require(raw.count(anchor) == 1, 'ambiguous audit scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + value + ',', 1)
    desired['historical_regression_audit'][KEY] = row
    require(json.loads(raw) == desired, 'owner probe changed unrelated results')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def prepare():
    require(not ROOT.exists() and KEY not in diagnostic.gen.committed(RESULT)['historical_regression_audit'], 'single-use owner probe')
    failed = diagnostic.runtime.validation.committed(RESULT, ('historical_regression_audit', diagnostic.KEY))
    require(failed['stage'] == 'failed' and not failed['passed'] and
        failed['closure_reconciliation']['all_owned_processes_terminated'] and
        failed['outcome']['fresh_faults'] == [], 'close rejected model diagnostic first')
    native = diagnostic.observer_inputs()
    revision = diagnostic.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = diagnostic.gen.benchmark_pins(revision)
    ROOT.mkdir(mode=0o700)
    row = dict(stage='prepared', source_commit=revision, controller_fingerprint=diagnostic.host.identity(pins),
        binary=native['binary'], boot=native['boot'], processes=[], complete=False, passed=False,
        active=False, ownership_closed=True, GPU_executed=False,
        plan=dict(question='Which supervision calls consume host CPU while waiting for an otherwise idle child?',
            order=['plain', 'CPU-profile', 'plain'], commands='Same libc-only qualified clock-access --observe;stop by private stdin EOF after tick10. '
                'No model or GPU. Existing Owner and failed-model artifact parent unchanged. Exact source/build/provider checks per launch.',
            timing='Record whole Owner.run wall and controller process CPU for each command. Plain arms report CPU fraction,not '
                'model overhead. CPU-profile uses cProfile(timer=process_time);its cost is charged and values are attribution only.',
            gate='Normal EOF/exit/maps/session closure,11ticks and no stderr. Record whether BOTH unprofiled host fractions<=0.05. '
                'Do not change the rejected model gate or advance to a new model trial from this control alone.',
            budgets='Per command20s awake wall,2s childCPU,32MiBRSS,1MiBtext/file,128MiBmodel-root artifacts. '
                'Reject BOOTTIME/MONOTONIC divergence>50ms;finite owner cleanup. Whole control240BOOTTIME seconds.',
            retention='Keep3CPU/wall samples and a compact CPU profile attribution,source/recipe/identity and limits. '
                'Retire normal process/tick logs after verified replay;do not keep a full profile transaction.'),
        environment=dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8'))
    save(row); print('Prepared CPU-only owner control;commit before run.', flush=True)


def run(row):
    require(diagnostic.runtime.validation.committed(RESULT, ('historical_regression_audit', KEY)) == row and
        row['stage'] == 'prepared', 'commit exact single-use owner plan')
    require(diagnostic.host.identity(diagnostic.gen.benchmark_pins(row['source_commit'])) == row['controller_fingerprint'], 'owner control source changed')
    initial = diagnostic.clock_pair(); owner = None
    def persist(): write(ROOT / 'execution.json', dict(active=bool(owner and owner.active), commands=owner.commands if owner else []))
    try:
        owner = diagnostic.owners.Owner(diagnostic.ROOT / 'supervision-cpu-probe', persist)
        for index, mode in enumerate(row['plan']['order']):
            native = diagnostic.observer_inputs()
            require(native['binary'] == row['binary'] and native['boot'] == row['boot'], 'native observer changed')
            def interact(io, entry):
                poll = io.poll
                def bounded():
                    require(diagnostic.awake(initial) < 240000000000, 'whole CPU probe BOOTTIME deadline'); poll()
                io.poll = bounded
                while True:
                    line = io.line(15)
                    if line.startswith('clock_tick=10 '): break
                io.proc.stdin.close()
                require(io.line(3) == f'observer_done pid={io.proc.pid} ticks=10 failed=0', 'native EOF framing')
                io.exit(3)
            profile = cProfile.Profile(timer=time.process_time) if mode == 'CPU-profile' else None
            wall = diagnostic.clock_pair(); cpu = time.process_time_ns()
            if profile: profile.enable()
            entry, out, err = owner.run(str(index), [row['binary']['path'], '--observe'], row['environment'], P,
                interaction=interact, wall=20, cpu=2, rss_cap=32 << 20, text_cap=1 << 20, file_cap=1 << 20, total_cap=128 << 20)
            if profile: profile.disable()
            elapsed = diagnostic.awake(wall); consumed = time.process_time_ns() - cpu
            require(diagnostic.owners.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size, 'CPU owner control failed')
            text = out.read_text()
            require(diagnostic.host.verify_maps(text, native, entry['pid']) == {entry['pid']} and
                sum(v.startswith('clock_tick=') for v in text.splitlines()) == 11, 'CPU owner actual provider/extent differs')
            sample = dict(index=index, mode=mode, elapsed_ns=elapsed, controller_cpu_ns=consumed,
                controller_cpu_fraction=consumed / elapsed, native_pid=entry['pid'], ownership_closed=True)
            if profile:
                profile.create_stats()
                rows = [dict(file=f, line=n, function=name, calls=values[1], self_CPU_s=values[2], cumulative_CPU_s=values[3])
                    for (f, n, name), values in profile.stats.items()]
                sample['top_self_CPU'] = sorted(rows, key=lambda v: v['self_CPU_s'], reverse=True)[:12]
                sample['supervisor_functions'] = [v for v in rows if v['file'].endswith('measurement_owner.py')]
            row['processes'].append(sample)
            require(diagnostic.awake(initial) < 240000000000, 'whole CPU probe deadline')
        row.update(stage='closed', complete=True, passed=True)
        plain = [p['controller_cpu_fraction'] for p in row['processes'] if p['mode'] == 'plain']
        row['analysis'] = dict(unprofiled_controller_fractions=plain, both_within_rejected_model_budget=all(v <= .05 for v in plain),
            limits='Host-only supervisor workload;profile changes overhead. No causal inference loss,clock/model qualification or speedup. Main must address measured overhead before a separate model plan.')
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row['ownership_closed'] = bool(owner and owner.active is None and all(diagnostic.owners.closed(c) for c in owner.commands))
        row['passed'] = bool(row['passed'] and row['ownership_closed']); persist(); save(row)
    print(json.dumps(row['analysis'] if row.get('analysis') else row), flush=True)
    return int(not row['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); diagnostic.host.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['run']: return run(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or run')


if __name__ == '__main__': sys.exit(main())
