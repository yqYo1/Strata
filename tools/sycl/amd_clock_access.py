#!/usr/bin/env python3
"""Bounded CPU-only RDPRU capability pilot; Main authored, repository LICENSE.

Do not execute during model timing. No GPU API, frequency change or MHz claim.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
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
ROOT = B / 'amd-clock-access-v1'
SOURCE = P / 'bench/sycl/amd_clock_access.c'
RESULT = gen.A / 'prefill-runtime-comparison-current.json'
KEY = 'amd_clock_access_v1'
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
    require(not ROOT.exists() and KEY not in gen.committed(RESULT)['historical_regression_audit'], 'single-use host pilot')
    cohort = gen.committed(RESULT)['historical_regression_audit']['validation_cost_control_v1']
    require(cohort['stage'] == 'closed' and cohort['ownership_closed'] and not cohort['active'], 'close frozen model cohort first')
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
        budgets=dict(wall=30, cpu=30, rss_cap=1 << 30, text_cap=4 << 20, file_cap=8 << 20, total_cap=32 << 20),
        active=False, ownership_closed=True, passed=False, complete=False, GPU_executed=False,
        counter_access_observed=False, model_clock_qualified=False, MHz_qualified=False,
        criteria='Fresh system-C compile/link plus one contained CPU0 process after model cohort closure. Exact '
            'source/compiler/header/link/ELF identities;normal owner closure. CPU/vendor/CPUID guards,1000 '
            'serialized snapshots and three32ms busy intervals with positive ordered counter deltas and finite '
            'read-skew bounds. Exit77 or SIGILL records unavailable access,not a usable clock. Other failures '
            'remain failures. No retries,signals masked,root,GPU/runtime calls or system frequency changes.',
        limits='Primitive access/progression/skew/timer evidence only. CPU0 counters are not per-process clocks. '
            'No verified P0/MHz/SMT semantics,model attribution,sampling overhead,physical bus or speed claim. '
            'A later model sampler needs a separate matched complete-observer overhead gate.',
        retention='Keep small counter samples and source/recipe/capability decision. Retire host ELF/object,normal '
            'build output and private execution records after source recovery and verified replay. No new engine generation.')
    save(row); print('Prepared CPU-only capability plan;commit before build/run.', flush=True)


def current(row):
    require(validation.committed(RESULT, ('historical_regression_audit', KEY)) == compact(row), 'commit exact host plan')
    require(gen.benchmark_pins(row['source_commit']) == row['controller_inputs'] and
        all(digest(p) == sha for p, sha in row['native_inputs'].items()) and
        digest(row['manifest']['path']) == row['manifest']['sha256'] and
        json.loads(Path(row['manifest']['path']).read_text()) == row['native_inputs'] and
        identity(dict(headers=row['headers'], link_paths=row['link_paths'], environment=row['environment'])) == row['input_identity'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'host inputs/platform changed')


def run(row):
    current(row); require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use host execution')
    initial = copy.deepcopy(row); owner = None; outcome = dict(host_probe_accepted=False)
    start = next(v.split(': ', 1)[1] for v in subprocess.check_output(['journalctl', '-k', '-b', '-n', '0',
        '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist(): write(ROOT / 'execution.json', dict(commands=owner.commands if owner else [], active=bool(owner and owner.active)))
    try:
        owner = owners.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('compile', row['compile_argv'], row['environment'], P, **row['budgets'])
        require(owners.closed(entry) and entry['exit_code'] == 0 and
            dependencies((ROOT / 'actual-deps.d').read_text()) == set(row['headers']), 'compile/header closure failed')
        row['object'] = dict(path=str(ROOT / 'clock.o'), sha256=digest(ROOT / 'clock.o'))
        entry, out, _ = owner.run('link', row['link_argv'], row['environment'], P, **row['budgets'])
        require(owners.closed(entry) and entry['exit_code'] == 0 and digest(ROOT / 'clock.o') == row['object']['sha256'], 'link failed')
        linked = {str(Path(v).resolve()) for v in out.read_text().splitlines() if v.startswith('/')}
        require(linked <= set(row['link_paths']) | {row['object']['path']} and row['object']['path'] in linked,
            'actual link closure uncovered')
        write(ROOT / 'actual-link-inputs.json', {p: digest(p) for p in sorted(linked)})
        row['actual_link_manifest'] = dict(path=str(ROOT / 'actual-link-inputs.json'),
            sha256=digest(ROOT / 'actual-link-inputs.json'), files=len(linked), uncovered=0)
        row['binary'] = dict(path=str(ROOT / 'clock-access'), sha256=digest(ROOT / 'clock-access'))
        entry, out, _ = owner.run('elf', [str(Path('/usr/bin/readelf').resolve()), '-d', row['binary']['path']],
            row['environment'], P, **row['budgets'])
        require(owners.closed(entry) and entry['exit_code'] == 0 and
            re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', out.read_text()) == ['libc.so.6'], 'unexpected native dependency')
        entry, out, err = owner.run('access', [row['binary']['path']], row['environment'], P,
            wall=10, cpu=10, rss_cap=1 << 30, text_cap=1 << 20, file_cap=8 << 20, total_cap=32 << 20)
        require(owners.closed(entry) and not err.stat().st_size, 'host access owner/stderr failure')
        text = out.read_text(); row['native_exit_code'] = entry['exit_code']
        if entry['exit_code'] == 0:
            require('status=ACCESS_OBSERVED MHz=UNQUALIFIED SMT_scope=UNQUALIFIED model_clock=UNMEASURED' in text and
                'phase=before-RDPRU cpu=0 selected=0' in text, 'access proof missing')
            samples = [dict(v.split('=', 1) for v in line.split()) for line in text.splitlines() if line.startswith('sample=')]
            cpu_line, = (line for line in text.splitlines() if line.startswith('cpu_signature='))
            cpu = dict(v.split('=', 1) for v in cpu_line.split())
            require(cpu['hypervisor'] == '0' and cpu['rdpru'] == '1' and int(cpu['max_index']) >= 1,
                'CPU capability contract differs')
            require(len(samples) == 3, 'missing busy intervals')
            for i, sample in enumerate(samples):
                lo, hi = (float(sample[k]) for k in ('ratio_lower', 'ratio_upper'))
                require(int(sample['sample']) == i and int(sample['delta_aperf']) > 0 and
                    0 < int(sample['delta_mperf_lower']) <= int(sample['delta_mperf_upper']) and
                    math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi, 'invalid ordered counter/skew result')
            row.update(counter_access_observed=True, cpu_identity=cpu, samples=samples,
                snapshot_batch_wall_ns=int(re.search(r'snapshot_count=1000 snapshot_batch_wall_ns=(\d+)', text)[1]))
        else:
            require(entry['exit_code'] == 77 or (entry['exit_code'] == -4 and 'phase=before-RDPRU cpu=0 selected=0' in text),
                'unexpected native failure')
            row['capability_unavailable_excerpt'] = text[-2048:]
        current(initial); outcome['host_probe_accepted'] = True; row['complete'] = True
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner: validation.finish(outcome, owner, start, row['boot'])
        row.update(stage=('access-observed' if row['counter_access_observed'] else 'access-unavailable') if outcome.get('passed') else 'failed',
            passed=bool(outcome.get('passed')), active=False, ownership_closed=outcome.get('ownership_closed', True), outcome=outcome)
        persist(); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'counter_access_observed', 'outcome')}), flush=True)
    return int(not row['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['run']: return run(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or run')


if __name__ == '__main__': sys.exit(main())
