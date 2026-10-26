#!/usr/bin/env python3
"""CPU-only complete RDPRU observer control; Main authored, repository LICENSE.

Do not execute during model timing. No GPU API, frequency change or MHz claim.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import shlex
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import generation_engine as gen
import measurement_owner as owners
import measurement_guards as guards
import ur_validation as validation

P, B = gen.P, gen.B
ROOT = B / 'amd-clock-observer-v2'
SOURCE = P / 'bench/sycl/amd_clock_access.c'
RESULT = gen.A / 'prefill-runtime-comparison-current.json'
KEY = 'amd_clock_observer_v2'
ORDER = ('on', 'off', 'off', 'on', 'on', 'off')
require, digest = guards.require, validation.digest


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def compact(row):
    return {k: v for k, v in row.items() if k not in ('controller_inputs', 'native_inputs', 'headers', 'link_paths')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw); desired['historical_regression_audit'][KEY] = compact(row)
    literal = json.dumps(KEY) + ':'; value = json.dumps(compact(row), indent=2).replace('\n', '\n    ')
    if KEY in json.loads(raw)['historical_regression_audit']:
        require(raw.count(literal) == 1, 'ambiguous clock scope'); start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"historical_regression_audit": {'; require(raw.count(anchor) == 1, 'ambiguous audit scope')
        raw = raw.replace(anchor, anchor + '\n    ' + literal + ' ' + value + ',', 1)
    require(json.loads(raw) == desired, 'clock update changed unrelated results')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def dependencies(text): return {str(Path(v).resolve()) for v in shlex.split(text.replace('\\\n', ' '))[1:]}


def prepare():
    require(not ROOT.exists() and KEY not in gen.committed(RESULT)['historical_regression_audit'], 'single-use complete observer control')
    prior = gen.committed(RESULT)['historical_regression_audit']['amd_clock_observer_v1']
    require(prior['stage'] == 'failed' and not prior['passed'] and prior['ownership_closed'] and
        not prior['active'] and not (B / 'amd-clock-observer-v1').exists(),
        'preserve original failure and retire its closed build before new cohort')
    cohort = gen.committed(RESULT)['historical_regression_audit']['validation_cost_control_v1']
    require(cohort['stage'] == 'closed' and cohort['complete'] and cohort['passed'] and
        cohort['ownership_closed'] and not cohort['active'], 'close frozen model cohort first')
    access = gen.committed(RESULT)['historical_regression_audit']['amd_clock_access_v1']
    require(access['passed'] and access['counter_access_observed'] and access['ownership_closed'] and
        access['boot'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'same-boot access pilot required')
    previous = gen.git('show', access['source_commit'] + ':bench/sycl/amd_clock_access.c')
    require(hashlib.sha256(previous).hexdigest() == access['native_source']['sha256'], 'original access source unrecoverable')
    revision = gen.git('rev-parse', 'HEAD').decode().strip(); pins = gen.benchmark_pins(revision)
    require(SOURCE.read_bytes() == gen.git('show', revision + ':' + str(SOURCE.relative_to(P))), 'commit exact C source')
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
    compiler = str(Path('/usr/bin/cc').resolve()); tools = dict(cc=compiler, readelf=str(Path('/usr/bin/readelf').resolve()))
    for name in ('cc1', 'as', 'ld', 'collect2'):
        v = subprocess.check_output([compiler, '-print-prog-name=' + name], env=env, text=True, timeout=10).strip()
        tools[name] = str(Path(shutil.which(v, path=env['PATH']) or v).resolve()); require(Path(tools[name]).is_file(), 'missing host compiler tool')
    flags = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIE']
    headers = dependencies(subprocess.check_output([compiler, *flags, '-M', str(SOURCE)], env=env, text=True, timeout=15))
    linked = set()
    for name in ('Scrt1.o', 'crti.o', 'crtn.o', 'crtbeginS.o', 'crtendS.o', 'libgcc.a', 'libgcc_s.so',
                 'libgcc_s.so.1', 'libc.so', 'libc.so.6', 'libc_nonshared.a', 'ld-linux-x86-64.so.2'):
        v = subprocess.check_output([compiler, '-print-file-name=' + name], env=env, text=True, timeout=10).strip()
        require(Path(v).is_file(), 'missing prospective link input'); linked.add(str(Path(v).resolve()))
    plugin = subprocess.check_output([compiler, '-print-file-name=liblto_plugin.so'], env=env, text=True, timeout=10).strip()
    require(Path(plugin).is_file(), 'missing linker plugin'); tools['linker_plugin'] = str(Path(plugin).resolve())
    files = set(tools.values()) | headers | linked
    for path in tools.values():
        out = subprocess.check_output(['/usr/bin/ldd', path], env=env, text=True, timeout=10)
        require('not found' not in out, 'unresolved host tool dependency')
        for line in out.splitlines():
            name = next((v for v in line.split() if v.startswith('/')), None)
            if name: files.add(str(Path(name).resolve()))
    require(all('/MistVVK/' not in p and '/oneapi/' not in p for p in files), 'CPU pilot must use system compiler inputs')
    inputs = {p: digest(p) for p in sorted(files)}
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); env['TMPDIR'] = str(ROOT / 'tmp')
    write(ROOT / 'native-inputs.json', inputs)
    row = dict(stage='prepared', source_commit=revision, controller_inputs=pins, native_inputs=inputs,
        native_source=dict(path=str(SOURCE), sha256=digest(SOURCE)), compiler=dict(path=compiler, sha256=digest(compiler),
            version=subprocess.check_output([compiler, '--version'], env=env, text=True, timeout=10).splitlines()[0]),
        headers=sorted(headers), link_paths=sorted(linked), environment=env,
        manifest=dict(path=str(ROOT / 'native-inputs.json'), sha256=digest(ROOT / 'native-inputs.json'), files=len(inputs)),
        compile_argv=[compiler, *flags, '-MD', '-MF', str(ROOT / 'actual-deps.d'), '-c', str(SOURCE), '-o', str(ROOT / 'clock.o')],
        link_argv=[compiler, str(ROOT / 'clock.o'), '-Wl,-t', '-o', str(ROOT / 'clock-access')],
        input_identity=identity(dict(headers=sorted(headers), link_paths=sorted(linked), environment=env)),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        budgets=dict(wall=90, cpu=90, rss_cap=1 << 30, text_cap=4 << 20, file_cap=8 << 20, total_cap=32 << 20),
        order=list(ORDER), processes=[], whole_wall=900, whole_bytes=96 << 20,
        active=False, ownership_closed=True, passed=False, complete=False, GPU_executed=False,
        counter_access_observed=False, model_clock_qualified=False, MHz_qualified=False,
        criteria='System-C compile/link/ELF and fresh access pilot,then6fixed-work CPU-only processes '
            'on/off/off/on/on/off. CPU0 actor:1006632960 warmup iterations outside clock,then10066329600 '
            'identical serial unsigned LCG iterations. Timed interval includes observer fork/exec,initialization, '
            '1Hz polling/counter/output,EOF shutdown and wait. Child uses same CPU0 and private EOF pipe. '
            'Require normal closure/checksums,wall>=5s,>=5periodic samples/on process,positive ordered '
            'MPERF/APERF deltas,gaps0.8..1.2s,no missed timer expiration and childCPU/wall<=0.002. '
            'Overhead screen:off-wall/on-wall each>=0.98 and geometric mean>=0.99. No speed-based '
            'early stop/retries;report all pairs. Stop first source/contract/fault/ownership/budget failure.',
        limits='CPU fixture overhead and observer protocol only. No GPU,model clock,MHz,P0/SMT '
            'interpretation,resource saturation,historical cause or throughput/adoption claim. Passing '
            'permits a separately pinned sampled model diagnostic,with clean speed collected separately; '
            'it does not establish identical perturbation for GPU/syscall/polling workloads.',
        retention='Keep individual6work/CPU/overhead values,source/build/contract identities and decision. '
            'Retire normal logs/tick sequences/private receipts after verified replay. Retain this tiny '
            'host observer ELF only for imminent model diagnostic after audit and this host gate; '
            'no new engine generation. Original engine and officialv4fallback are untouched.')
    save(row); print('Prepared complete1Hz observer host plan;commit before build/run.', flush=True)


def current(row):
    require(validation.committed(RESULT, ('historical_regression_audit', KEY)) == compact(row), 'commit exact host plan')
    require(gen.benchmark_pins(row['source_commit']) == row['controller_inputs'] and
        all(digest(p) == sha for p, sha in row['native_inputs'].items()) and
        digest(row['manifest']['path']) == row['manifest']['sha256'] and
        json.loads(Path(row['manifest']['path']).read_text()) == row['native_inputs'] and
        identity(dict(headers=row['headers'], link_paths=row['link_paths'], environment=row['environment'])) == row['input_identity'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'host inputs/platform changed')


def fields(line):
    parts = line.split()
    if parts and parts[0] in ('observer_ready', 'observer_done'): parts = parts[1:]
    return dict(v.split('=', 1) for v in parts)


def verify_maps(text, row, main_pid):
    groups = {}
    for line in text.splitlines():
        if line.startswith('native_map_pid='):
            v = fields(line); groups.setdefault(int(v['native_map_pid']), set()).add(str(Path(v['path']).resolve()))
    expected = {row['binary']['path']} | {p for p in row['link_paths']
        if Path(p).name in ('libc.so.6', 'ld-linux-x86-64.so.2')}
    require(len(expected) == 3 and main_pid in groups and all(v == expected for v in groups.values()),
        'actual host executable/provider mapping differs')
    return set(groups)


def lcg(blocks):
    mask = (1 << 64) - 1; count = blocks * 4096; mul, add, out_mul, out_add = 6364136223846793005, 1, 1, 0
    while count:
        if count & 1: out_mul, out_add = out_mul * mul & mask, (mul * out_add + add) & mask
        mul, add = mul * mul & mask, (mul * add + add) & mask; count >>= 1
    return (out_mul + out_add) & mask


def host_result(text, index, mode):
    caps = [fields(v) for v in text.splitlines() if v.startswith('cpu_signature=')]
    require(len(caps) == (2 if mode == 'on' else 1) and all(v == caps[0] for v in caps) and
        caps[0]['hypervisor'] == '0' and caps[0]['rdpru'] == '1' and int(caps[0]['max_index']) >= 1,
        'host/observer capability differs')
    summary, = (fields(v) for v in text.splitlines() if v.startswith('control='))
    require(summary['control'] == mode and int(summary['iterations']) == 10066329600 and
        int(summary['warm_iterations']) == 1006632960 and int(summary['checksum']) == lcg(2457600) and
        int(summary['warm_checksum']) == lcg(245760), 'fixed work/checksum differs')
    wall, actor, observer = (int(summary[k]) for k in ('wall_ns', 'actor_cpu_ns', 'observer_cpu_ns'))
    require(wall >= 5000000000 and 0 < actor <= wall * 1.01 and observer >= 0, 'host duration/CPU clock ineligible')
    row = dict(index=index, mode=mode, wall_ns=wall, actor_cpu_ns=actor, observer_cpu_ns=observer,
        checksum=int(summary['checksum']), warm_checksum=int(summary['warm_checksum']), cpu_identity=caps[0])
    ticks = [fields(v) for v in text.splitlines() if v.startswith('clock_tick=')]
    ready = [fields(v) for v in text.splitlines() if v.startswith('observer_ready ')]
    done = [fields(v) for v in text.splitlines() if v.startswith('observer_done ')]
    if mode == 'off':
        require(not ticks and not ready and not done and observer == 0 and summary['observer_pid'] == '-1', 'off arm observed')
        return row
    require(len(ready) == len(done) == 1 and len(ticks) >= 6 and done[0]['failed'] == '0' and
        ready[0]['pid'] == done[0]['pid'] == summary['observer_pid'] and int(summary['observer_pid']) > 0 and
        int(done[0]['ticks']) == len(ticks) - 1 and ready[0]['period_ns'] == '1000000000' and
        ready[0]['max_ticks'] == '900', 'observer framing/closure differs')
    require(observer / wall <= .002, 'complete observer CPU budget exceeded')
    bounds, gaps, skew = [], [], []
    for i, sample in enumerate(ticks):
        require(int(sample['clock_tick']) == i and sample['cpu'] == '0' and
            0 < int(sample['raw_before_ns']) <= int(sample['raw_after_ns']) and
            int(sample['mperf_after']) >= int(sample['mperf_before']), 'snapshot/cpu/skew invalid')
        skew.append(int(sample['raw_after_ns']) - int(sample['raw_before_ns']))
        if i:
            previous = ticks[i - 1]
            lo = int(sample['mperf_before']) - int(previous['mperf_after'])
            hi = int(sample['mperf_after']) - int(previous['mperf_before'])
            delta = int(sample['aperf']) - int(previous['aperf'])
            gap = int(sample['raw_before_ns']) - int(previous['raw_before_ns'])
            require(0 < lo <= hi and delta > 0 and 800000000 <= gap <= 1200000000, 'interval progression/skew/gap invalid')
            bounds.append((delta / hi, delta / lo)); gaps.append(gap)
    row.update(periodic_samples=len(ticks) - 1, observer_cpu_fraction=observer / wall,
        ratio_bounds=[min(v[0] for v in bounds), max(v[1] for v in bounds)],
        raw_gap_ns_range=[min(gaps), max(gaps)], max_snapshot_wall_ns=max(skew),
        observer_pid=int(summary['observer_pid']), bounded_protocol_passed=True)
    return row


def run(row):
    current(row); require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use host execution')
    initial = copy.deepcopy(row); owner = None; outcome = dict(host_probe_accepted=False)
    start_wall = time.monotonic()
    start = next(v.split(': ', 1)[1] for v in subprocess.check_output(['journalctl', '-k', '-b', '-n', '0',
        '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist(): write(ROOT / 'execution.json', dict(commands=owner.commands if owner else [], active=bool(owner and owner.active)))
    def checked(name, argv):
        entry, out, err = owner.run(name, argv, row['environment'], P, **row['budgets'])
        require(owners.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size, name + ' owner/exit/stderr failure')
        require(time.monotonic() - start_wall < row['whole_wall'] and owners.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole host budget')
        text = out.read_text()
        if name == 'access' or name[0].isdigit():
            mapped = verify_maps(text, row, entry['pid'])
            control = [fields(v) for v in text.splitlines() if v.startswith('control=')]
            expected = {entry['pid']}
            if control and control[0]['control'] == 'on': expected.add(int(control[0]['observer_pid']))
            require(mapped == expected, 'native actor/observer process identities differ')
        return text
    try:
        owner = owners.Owner(ROOT / 'owned', persist)
        checked('compile', row['compile_argv'])
        require(dependencies((ROOT / 'actual-deps.d').read_text()) == set(row['headers']), 'actual header closure differs')
        row['object'] = dict(path=str(ROOT / 'clock.o'), sha256=digest(ROOT / 'clock.o'))
        link = checked('link', row['link_argv'])
        require(digest(ROOT / 'clock.o') == row['object']['sha256'], 'object changed')
        linked = {str(Path(v).resolve()) for v in link.splitlines() if v.startswith('/')}
        require(linked <= set(row['link_paths']) | {row['object']['path']} and row['object']['path'] in linked, 'link closure uncovered')
        write(ROOT / 'actual-link-inputs.json', {p: digest(p) for p in sorted(linked)})
        row['actual_link_manifest'] = dict(path=str(ROOT / 'actual-link-inputs.json'),
            sha256=digest(ROOT / 'actual-link-inputs.json'), files=len(linked), uncovered=0)
        row['binary'] = dict(path=str(ROOT / 'clock-access'), sha256=digest(ROOT / 'clock-access'))
        elf = checked('elf', [str(Path('/usr/bin/readelf').resolve()), '-d', row['binary']['path']])
        require(re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', elf) == ['libc.so.6'], 'unexpected native dependency')
        access = checked('access', [row['binary']['path']])
        require('status=ACCESS_OBSERVED MHz=UNQUALIFIED SMT_scope=UNQUALIFIED model_clock=UNMEASURED' in access,
            'fresh access pilot incomplete')
        row['counter_access_observed'] = True
        row['fresh_access'] = dict(passed=True, snapshot_batch_wall_ns=int(re.search(
            r'snapshot_count=1000 snapshot_batch_wall_ns=(\d+)', access)[1]))
        for index, mode in enumerate(ORDER):
            text = checked(str(index) + '-' + mode, [row['binary']['path'], '--host-control', mode])
            row['processes'].append(host_result(text, index, mode))
            print('Closed host process', index, mode, json.dumps(row['processes'][-1]), flush=True)
        pairs = []
        for i in (0, 2, 4):
            group = {p['mode']: p['wall_ns'] for p in row['processes'][i:i + 2]}
            pairs.append(group['off'] / group['on'])
        mean = statistics.geometric_mean(pairs)
        row['analysis'] = dict(off_wall_over_on_wall_pairs=pairs, geometric_mean=mean,
            overhead_screen_passed=all(v >= .98 for v in pairs) and mean >= .99,
            source_audit_and_separate_model_plan_required=True,
            model_overhead_qualified=False, MHz_qualified=False, inference_adoption=False)
        current(initial); require(digest(row['binary']['path']) == row['binary']['sha256'], 'host binary changed')
        outcome['host_probe_accepted'] = True; row['complete'] = True
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner: validation.finish(outcome, owner, start, row['boot'])
        row.update(stage='closed' if outcome.get('passed') else 'failed', passed=bool(outcome.get('passed')),
            active=False, ownership_closed=outcome.get('ownership_closed', True), outcome=outcome)
        persist(); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'analysis', 'outcome')}), flush=True)
    return int(not row['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['run']: return run(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or run')


if __name__ == '__main__': sys.exit(main())
