#!/usr/bin/env python3
"""Main-authored, request-scoped CPU0 counter diagnostic; repository LICENSE.

Uses the unchanged selected engine and the qualified libc-only observer. Ratios
describe CPU0 intervals, not model-thread MHz. No clean-speed/adoption claim.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import resource
import select
import signal
import socket
import statistics
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import amd_clock_observer as host
import cpu_capacity as models
import engine_once
import generation_engine as gen
import measurement_owner as owners
import runtime_probe as runtime

P, B, RESULT = host.P, host.B, host.RESULT
ROOT = B / 'prefill-cpu-clock-v1'
TEMPLATE = gen.A / 'phase-prefill-speed-current.json'
KEY = 'model_cpu_clock_diagnostic_v1'
require, digest, write = host.require, host.digest, host.write


def compact(row):
    return {k: v for k, v in row.items() if k not in ('controller_inputs', 'models', 'environment_template')}


def save(row):
    raw = RESULT.read_text(); before = json.loads(raw)
    literal = json.dumps(KEY) + ':'; value = json.dumps(compact(row), indent=2).replace('\n', '\n    ')
    if KEY in before['historical_regression_audit']:
        require(raw.count(literal) == 1, 'ambiguous clock diagnostic scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"historical_regression_audit": {'
        require(raw.count(anchor) == 1, 'ambiguous audit scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + value + ',', 1)
    before['historical_regression_audit'][KEY] = compact(row)
    require(json.loads(raw) == before, 'clock update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def observer_inputs():
    row = json.loads((host.ROOT / 'record.json').read_text()); host.current(row)
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and row['analysis']['overhead_screen_passed'] and
        not row['analysis']['model_overhead_qualified'], 'closed exact host gate required')
    for key in ('binary', 'object', 'actual_link_manifest'):
        require(digest(row[key]['path']) == row[key]['sha256'], 'observer build artifact changed')
    linked = json.loads(Path(row['actual_link_manifest']['path']).read_text())
    require(len(linked) == row['actual_link_manifest']['files'] and
        all(digest(p) == sha for p, sha in linked.items()), 'actual observer link inputs changed')
    return row


def clock_pair():
    # Both MONOTONIC clocks exclude suspend. BOOTTIME supplies a separate guard;
    # a suspended run is rejected, never silently treated as an awake interval.
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()


def awake(initial):
    now = clock_pair(); elapsed = [a - b for a, b in zip(now, initial)]
    require(all(v >= 0 for v in elapsed) and abs(elapsed[0] - elapsed[1]) <= 50000000,
        'suspend or clock discontinuity: diagnostic ineligible')
    return elapsed[0]


class Observer:
    """Own only a fresh, non-forking native child through its pinned pidfd.

    Its session leader stays unreaped until the session scan. EOF is the normal
    stop; signals are reserved for a failed owned child, never GPU recovery.
    """
    def __init__(self, root, row):
        self.root, self.row = root, row
        self.proc = None; self.pidfd = None; self.identity = None
        self.out = self.err = None; self.closed = False; self.killed = False
        self.start = clock_pair(); self.next_check = 0
        self.cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)

    def start_child(self):
        require(not self.root.exists(), 'one observer launch per request')
        self.root.mkdir(mode=0o700)
        observer_inputs()  # Exact source, headers, executable and providers at EACH exec.
        self.out = (self.root / 'stdout').open('xb'); self.err = (self.root / 'stderr').open('xb')
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
            resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 10, 256 << 10))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        self.proc = subprocess.Popen([self.row['binary']['path'], '--observe'], cwd=P,
            env=dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8'),
            stdin=subprocess.PIPE, stdout=self.out, stderr=self.err,
            start_new_session=True, preexec_fn=limits)
        self.identity = owners.identity(self.proc.pid)
        require(self.identity and self.identity['sid'] == self.identity['pgid'] == self.proc.pid,
            'observer session identity unavailable')
        self.pidfd = os.pidfd_open(self.proc.pid)
        require(owners.identity(self.proc.pid)['start_ticks'] == self.identity['start_ticks'], 'observer identity raced')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.check(force=True)
            text = (self.root / 'stdout').read_text()
            if text.endswith(f'observer_ready pid={self.proc.pid} period_ns=1000000000 max_ticks=900\n'):
                require(host.verify_maps(text, self.row, self.proc.pid) == {self.proc.pid}, 'observer mappings differ')
                caps, = (host.fields(v) for v in text.splitlines() if v.startswith('cpu_signature='))
                require(caps == self.row['processes'][0]['cpu_identity'], 'observer CPU contract changed')
                return
            time.sleep(.01)
        raise RuntimeError('observer ready deadline')

    def check(self, force=False):
        require(awake(self.start) < 850000000000, 'observer absolute deadline')
        now = time.monotonic()
        if not force and now < self.next_check: return
        self.next_check = now + .25
        require(self.pidfd is not None and not select.select([self.pidfd], [], [], 0)[0], 'observer exited early')
        current = owners.identity(self.proc.pid)
        require(current and current['start_ticks'] == self.identity['start_ticks'] and
            current['sid'] == self.identity['sid'] and current['rss_bytes'] < 32 << 20,
            'observer identity/RSS changed')
        require((self.root / 'stdout').stat().st_size <= 256 << 10 and
            not (self.root / 'stderr').stat().st_size, 'observer output/stderr budget')

    def stop(self):
        if self.proc is None: return
        if self.proc.stdin and not self.proc.stdin.closed: self.proc.stdin.close()
        if self.pidfd is None: self.pidfd = os.pidfd_open(self.proc.pid)
        if not select.select([self.pidfd], [], [], 5)[0]:
            signal.pidfd_send_signal(self.pidfd, signal.SIGTERM); self.killed = True
            if not select.select([self.pidfd], [], [], 2)[0]:
                signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
        require(select.select([self.pidfd], [], [], 2)[0], 'owned observer unreaped after finite stop')
        remaining = []
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal(): continue
            try: v = owners.identity(int(path.name))
            except PermissionError: continue
            if v and v['sid'] == self.proc.pid and v['state'] != 'Z': remaining.append(v['pid'])
        require(not remaining, 'observer session still has a live member')
        code = self.proc.wait(timeout=1)
        self.out.close(); self.err.close(); os.close(self.pidfd); self.pidfd = None; self.closed = True
        self.elapsed_ns = awake(self.start)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        # During a request only this direct child is reaped; the benchmark owner
        # is still live. Thus this delta includes the complete observer process.
        self.cpu_ns = round(((after.ru_utime - self.cpu_before.ru_utime) +
            (after.ru_stime - self.cpu_before.ru_stime)) * 1e9)
        require(code == 0 and not self.killed and not (self.root / 'stderr').stat().st_size,
            'observer abnormal exit or stderr')

    def summary(self, minimum_ticks):
        require(self.closed, 'observer ownership still open')
        text = (self.root / 'stdout').read_text(); lines = text.splitlines()
        allowed = ('native_map_pid=', 'cpu_signature=', 'phase=before-RDPRU ', 'clock_tick=',
            'observer_ready ', 'observer_done ')
        require(all(v.startswith(allowed) for v in lines), 'unknown observer framing')
        ready, = (host.fields(v) for v in lines if v.startswith('observer_ready '))
        done, = (host.fields(v) for v in lines if v.startswith('observer_done '))
        ticks = [host.fields(v) for v in lines if v.startswith('clock_tick=')]
        require(ready == dict(pid=str(self.proc.pid), period_ns='1000000000', max_ticks='900') and
            done == dict(pid=str(self.proc.pid), ticks=str(len(ticks) - 1), failed='0') and
            len(ticks) > minimum_ticks and host.verify_maps(text, self.row, self.proc.pid) == {self.proc.pid},
            'observer framing, extent or provider differs')
        bounds, spans, gaps = [], [], []
        for i, t in enumerate(ticks):
            t = {k: int(v) for k, v in t.items()}
            require(set(t) == {'clock_tick', 'cpu', 'raw_before_ns', 'raw_after_ns', 'mperf_before',
                'aperf', 'mperf_after'} and t['clock_tick'] == i and t['cpu'] == 0 and
                0 < t['raw_before_ns'] <= t['raw_after_ns'] and
                t['raw_after_ns'] - t['raw_before_ns'] <= 100000 and
                0 < t['mperf_before'] <= t['mperf_after'] < 1 << 64, 'snapshot/skew bound')
            spans.append(t['raw_after_ns'] - t['raw_before_ns'])
            if i:
                previous = {k: int(v) for k, v in ticks[i - 1].items()}
                lo = t['mperf_before'] - previous['mperf_after']; hi = t['mperf_after'] - previous['mperf_before']
                delta = t['aperf'] - previous['aperf']; gap = t['raw_before_ns'] - previous['raw_before_ns']
                require(0 < lo <= hi < 1 << 63 and 0 < delta < 1 << 63 and
                    (hi - lo) / lo <= .001 and 800000000 <= gap <= 1200000000, 'counter envelope/gap bound')
                bounds.append((delta / hi, delta / lo)); gaps.append(gap)
        require(0 <= self.cpu_ns / self.elapsed_ns <= .002, 'actual observer CPU fraction exceeded')
        return dict(identity={k: self.identity[k] for k in ('pid', 'start_ticks', 'sid')},
            elapsed_ns=self.elapsed_ns, cpu_ns=self.cpu_ns, cpu_fraction=self.cpu_ns / self.elapsed_ns,
            intervals=len(bounds), ratio_range=[min(b[0] for b in bounds), max(b[1] for b in bounds)],
            ratio_median=statistics.median((a + b) / 2 for a, b in bounds),
            max_snapshot_ns=max(spans), gap_ns_range=[min(gaps), max(gaps)],
            ownership_closed=True, speed_eligible=False, MHz_qualified=False,
            limits='CPU0 interval aggregate,not model-thread clock. No historical paired counters or causal frequency attribution.')


def qualification():
    rt = runtime.validation.committed(runtime.RESULT); full = runtime.current(rt)
    observer = observer_inputs()
    audit = runtime.validation.committed(runtime.validation.INDEX, ('amd_clock_observer_source_contract_20261027',))
    require(audit['status'] == 'reviewed_host_scope_only_local_model_gates_pending' and
        audit['native_review']['state'] == 'accepted_host_scope_only' and
        all(digest(P / p) == sha for p, sha in audit['sources'].items()), 'reviewed exact observer source required')
    return full, rt, observer, host.identity(audit)


def current(row):
    require(runtime.validation.committed(RESULT, ('historical_regression_audit', KEY)) == compact(row), 'commit exact diagnostic plan')
    full, rt, observer, review = qualification()
    require(gen.benchmark_pins(row['source_commit']) == row['controller_inputs'] and
        digest(TEMPLATE) == row['template_sha256'] and full['binary'] == row['binary'] and
        digest(runtime.RESULT) == row['runtime_sha256'] and row['observer_binary'] == observer['binary'] and
        rt['boot'] == row['boot'] and review == row['review_identity'], 'diagnostic inputs/platform changed')
    for name, sha in row['generated_inputs'].items(): require(digest(ROOT / name) == sha, 'generated input changed')
    models.model_pins(row['configuration'], row['models'])
    return full, rt, observer


def prepare():
    require(not ROOT.exists() and KEY not in gen.committed(RESULT)['historical_regression_audit'], 'single-use diagnostic')
    full, rt, observer, review = qualification()
    reference = gen.committed(RESULT)['historical_regression_audit']['near_dense_original_reference_v1']
    require(reference['passed'] and reference['ownership_closed'] and reference['binary'] == full['binary'], 'closed same-engine reference')
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    cfg = copy.deepcopy(reference['configuration']); cfg['log'] = str(ROOT / 'engine.stderr')
    template = gen.committed(TEMPLATE)['environment_template']; old = template['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in template.items()}
    require(not any(k in env for k in ('LD_PRELOAD', 'STRATA_TRACE', 'STRATA_STATE_HASH', 'STRATA_SNAPSHOT_VERIFY',
        'STRATA_DECODE_TIMING', 'STRATA_VERIFY_PROFILE', 'STRATA_CPU_WORK_CENSUS', 'ZE_ENABLE_ALT_DRIVERS')),
        'unexpected base observer/provider')
    neo = next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p)
    env.update(UR_ENABLE_LAYERS='', ZE_ENABLE_ALT_DRIVERS=neo,
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    loader = rt['preload']['loader']
    probe = dict(library=rt['preload']['library'], loader=dict(path=loader, sha256=rt['runtime_files'][loader]))
    expected = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    pinned_models = models.model_pins(cfg, expected, full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    write(ROOT / 'config.json', cfg); write(ROOT / 'plan.json', dict(binary=full['binary'], native_validation_probe=probe))
    with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    row = dict(stage='prepared', source_commit=revision, controller_inputs=gen.benchmark_pins(revision),
        binary=full['binary'], observer_binary=observer['binary'], runtime_sha256=digest(runtime.RESULT), review_identity=review,
        template_sha256=digest(TEMPLATE), environment_template=template, environment=env,
        configuration=cfg, models=pinned_models, probe=probe, NEO=neo, boot=rt['boot'],
        request_hashes=reference['request_hashes'], generated_inputs={n: digest(ROOT / n) for n in ('config.json', 'plan.json')},
        command=['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
            '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '4', '--profile-request-gates',
            '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(ROOT / 'bench')],
        budgets=dict(wall=3000, cpu=72000, rss_cap=104 << 30, text_cap=32 << 20, file_cap=64 << 20, total_cap=128 << 20),
        plan=dict(question='Does CPU0 APERF/MPERF vary during the currently slower original-engine long requests?',
            work='One process,four fresh117489-token/zero-reuse/thinking-on requests,24outputs,8192chunks,1256decode slots. '
                'No engine,math,placement,driver or safety change. Original private validation preload only in native child.',
            host_gate='Fresh standalone observer with stdin PIPE;>=2periodic ticks,ready/EOF/exit/maps/CPU/snapshot/envelope checks. '
                'Commit its closure before model. Repeat exact source/build/provider checks before EACH observer exec.',
            model_gate='Request begin/end gates start/stop an independent sanitized libc-only process on CPU0. '
                'Require>=30complete intervals/request,snapshot<=100us,relative counter envelope<=0.001,gaps0.8..1.2s, '
                'observer complete CPU/elapsed<=0.002,request controller CPU/gate elapsed<=0.05. '
                'BOOTTIME vs MONOTONIC divergence>50ms rejects suspend. '
                'Observer cap850BOOTTIME seconds,32MiBRSS,256KiBstdout,empty stderr. EOF stop5s,TERM2s,KILL2s; '
                'signals make result fail. Pinned pidfd/session closure before next request. Main CPU accounting retained.',
            interpretation='Diagnostic only; model observer overhead is unqualified. Record PP clocks descriptively,never clean '
                'speed or adoption. APERF/MPERF describes CPU0 interval aggregate,not thread MHz,CPU critical path or history. '
                'Past equivalent counter records absent; current steady frequency alone cannot explain historical decline.',
            stop='First source/provider/request/clock/contract/fault/ownership/budget error;no retries,reset,service or policy change.',
            retention='Keep4 individual prompt/ratio/observer CPU/coverage samples,source/recipe/runtime identities and limits. '
                'Retire normal tick timelines,build and API logs after independent replay;tiny helper consumer ends at closure.'),
        host=None, outcome=None, active=False, ownership_closed=True, complete=False, passed=False,
        speed_eligible=False, inference_adoption=False)
    save(row); print('Prepared host gate plus one sampled model process;commit before execution.', flush=True)


def run_host(row):
    _, _, native = current(row); require(row['stage'] == 'prepared' and row['host'] is None, 'single-use host gate')
    obs = Observer(ROOT / 'host-observer', native); result = dict(passed=False, ownership_closed=True)
    try:
        obs.start_child()
        end = time.monotonic() + 2.2
        while time.monotonic() < end: obs.check(); time.sleep(.02)
        obs.stop(); result = dict(obs.summary(2), passed=True)
    except BaseException as error:
        result['error'] = repr(error)
        if obs.proc is not None and not obs.closed:
            try: obs.stop()
            except BaseException as cleanup: result['cleanup_error'] = repr(cleanup)
    result['ownership_closed'] = obs.proc is None or obs.closed
    row.update(host=result, stage='host-passed' if result['passed'] else 'failed', ownership_closed=result['ownership_closed'])
    save(row); print(json.dumps(result), flush=True); return int(not result['passed'])


def run_model(row):
    _, rt, native = current(row)
    require(row['stage'] == 'host-passed' and row['host']['passed'] and row['host']['ownership_closed'] and
        not (ROOT / 'execution.json').exists(), 'commit successful standalone gate first;one model launch')
    start = next(v.split(': ', 1)[1] for v in subprocess.check_output(['journalctl', '-k', '-b', '-n', '0',
        '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    owner = None; obs = None; captures = []; initial = clock_pair(); cpu_begin = time.process_time_ns()
    request_cpu_begin = request_wall_begin = None
    outcome = dict(complete=False, host_probe_accepted=False)
    def persist(): write(ROOT / 'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, leader):
        nonlocal obs, request_cpu_begin, request_wall_begin
        previous_poll = io.poll
        def supervised_poll():
            require(awake(initial) < 3000000000000, 'whole model BOOTTIME deadline')
            if obs is not None: obs.check()
            previous_poll()
        io.poll = supervised_poll
        for request in range(4):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(1500)
                    if line.startswith('STRATA_PROFILE_GATE '): break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=stage, phase='prefill-only', run=request), 'request gate differs')
                require(not (ROOT / 'validation.log').exists() or not (ROOT / 'validation.log').stat().st_size, 'parameter warning')
                if stage == 'begin':
                    claim = json.loads((ROOT / 'engine-start.json').read_text()); pid = claim['native']['pid']
                    ident = owners.identity(pid)
                    require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == leader['pid'] and
                        str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native identity differs')
                    expected = engine_once.native_environment(dict(native_validation_probe=row['probe']), row['environment'])
                    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
                        set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                    mapped = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    require(mapped <= set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve())} and
                        {row['NEO'], row['probe']['loader']['path'], row['probe']['library']['path']} <= mapped,
                        'native provider mappings differ')
                    obs = Observer(ROOT / f'clock-{request}', native); obs.start_child()
                    request_cpu_begin = time.process_time_ns(); request_wall_begin = clock_pair()
                else:
                    controller_cpu = time.process_time_ns() - request_cpu_begin
                    request_elapsed = awake(request_wall_begin)
                    require(0 <= controller_cpu / request_elapsed <= .05, 'request controller CPU fraction exceeded')
                    obs.stop(); capture = obs.summary(30)
                    capture.update(controller_cpu_ns=controller_cpu, request_gate_elapsed_ns=request_elapsed,
                        controller_cpu_fraction=controller_cpu / request_elapsed)
                    captures.append(capture); obs = None
                    print(f'Sampled long request {request + 1}/4 completed', flush=True)
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'model teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = owners.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('bench', row['command'], row['environment'], P, interaction=interact, **row['budgets'])
        require(owners.closed(entry) and entry['exit_code'] == 0, 'model benchmark/owner failed')
        result_file, = (ROOT / 'bench').glob('*/results.json'); result = json.loads(result_file.read_text())
        samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == len(captures) == 4, 'incomplete official benchmark')
        for s, expected in zip(samples, row['request_hashes']):
            require(s['ok'] and s['eligible'] and s['stream_complete'] and s['thinking_requested'] and s['reasoning_observed'] and
                s['request_sha256'] == expected and s['actual_prompt_tokens'] == s['actual_read_tokens'] == 117489 and
                s['actual_reused_tokens'] == 0 and s['generated_tokens'] == 24 and s['prompt_tps_source'] == 'engine' and
                10000 <= s['prompt_ms'] < 1500000, 'workload/timing differs')
        sys.path.insert(0, str(P / 'tools')); from strata_bench import verify_request_journal
        journal, = (ROOT / 'bench').glob('*/requests.jsonl'); verified = verify_request_journal(journal.parent)
        require(verified['valid'] and verified['complete'], 'journal invalid/incomplete')
        events = [json.loads(v) for v in journal.read_text().splitlines()]
        require([dict(v['sample'], run=v['run']) for v in events if v['kind'] == 'result' and v['phase'] == 'prefill-only'] == samples,
            'journal/request binding differs')
        end = gen.native_end(ROOT, dict(binary=row['binary'], native_validation_probe=row['probe']),
            dict(config=row['configuration']), row['boot'], entry['pid'])
        text = (ROOT / 'engine.stderr').read_text()
        runtime.validation.preload_evidence(text, end['native']['pid'])
        require(text.count('strata prefill: layer-major candidate, tokens=117488 chunk=8192 residual_dtype=bf16') == 4 and
            text.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 4 and
            not any(s in text for s in ('strata trace:', 'strata decode timing:', 'STATE_HASH', 'SNAPSHOT_VERIFY',
                'UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'route/runtime differs')
        current(row); models.model_pins(row['configuration'], row['models'], full=True)
        fields = ('run', 'request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens', 'generated_tokens', 'prompt_ms')
        outcome.update(complete=True, host_probe_accepted=True, native_identity=end['native'],
            requests=[dict(**{k: s[k] for k in fields}, diagnostic_tok_s=117489000 / s['prompt_ms'], cpu0=c)
                for s, c in zip(samples, captures)], controller_cpu_ns=time.process_time_ns() - cpu_begin,
            elapsed_ns=awake(initial), speed_eligible=False, model_observer_overhead_qualified=False,
            historical_frequency_cause_established=False)
    except BaseException as error:
        outcome['error'] = repr(error)
        if (ROOT / 'engine.stderr').exists(): outcome['error_excerpt'] = (ROOT / 'engine.stderr').read_text()[-4096:]
    finally:
        if obs is not None and not obs.closed:
            try: obs.stop()
            except BaseException as cleanup: outcome['observer_cleanup_error'] = repr(cleanup)
        if owner: runtime.validation.finish(outcome, owner, start, row['boot'])
        outcome['ownership_closed'] = bool(outcome.get('ownership_closed') and (obs is None or obs.closed))
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'] and outcome['ownership_closed'])
        persist(); row.update(outcome=outcome, stage='closed' if outcome['passed'] else 'failed',
            complete=outcome['complete'], passed=outcome['passed'], ownership_closed=outcome['ownership_closed'], active=False)
        save(row)
    print(json.dumps(outcome), flush=True); return int(not outcome['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); host.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        row = json.loads((ROOT / 'record.json').read_text())
        if sys.argv[1:] == ['host']: return run_host(row)
        if sys.argv[1:] == ['model']: return run_model(row)
        raise RuntimeError('use prepare,host,model;commit each frozen plan/closed gate before next')


if __name__ == '__main__': sys.exit(main())
