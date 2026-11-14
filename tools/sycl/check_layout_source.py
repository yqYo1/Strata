#!/usr/bin/env python3
"""Main-owned host diagnostic hash-order gate. Repository LICENSE; no GPU."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import check_full_context_source as source

P, B, RESULT = source.P, source.B, source.RESULT
ROOT = B / 'prefill-layout-source-v3'
KEY = 'prefill_layout_source_v3'
TEST = 'tests/core/full_context_diagnostic_test.cpp'
INPUTS = [*source.INPUTS, 'include/strata/program/diagnostic_kv_ranges.hpp',
          'tools/sycl/check_layout_source.py']
ENV = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8')
require = source.guards.require
digest = source.build.digest
write = source.write


def pins(revision):
    result = {}
    for name, raw in source.build.blobs(P, revision, sorted(set(INPUTS))):
        require(raw == (P / name).read_bytes(), 'commit exact source input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def prepare(container):
    require(KEY not in container and not ROOT.exists(), 'fresh source gate only')
    revision = source.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    inputs = pins(revision)
    _, baseline = source.baseline()
    compiler = '/usr/bin/g++'
    options = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-I', str(P / 'include')]
    dep_command = [compiler, *options, '-M', '-MT', 'fixture', str(P / TEST)]
    dep = subprocess.run(dep_command, cwd=P, env=ENV, capture_output=True, text=True, timeout=15)
    require(dep.returncode == 0 and not dep.stderr and len(dep.stdout) <= 1 << 20,
            'bounded compiler dependency discovery failed')
    names = shlex.split(dep.stdout.replace('\\\n', ' ').split(':', 1)[1])
    dependencies = {str(Path(n).resolve()): digest(n) for n in names}
    require(str(P / TEST) in dependencies and
            str(P / 'include/strata/program/diagnostic_kv_ranges.hpp') in dependencies,
            'actual host include closure incomplete')
    ROOT.mkdir(mode=0o700)
    record = dict(stage='prepared', source_commit=revision, input_sha256=inputs,
        compiler=dict(path=compiler, sha256=digest(compiler)), dependencies=dependencies,
        dependency_command=dep_command, selected_original=baseline,
        command=[compiler, *options, str(P / TEST), '-o', str(ROOT / 'host_bounds')],
        environment=ENV, active=False, passed=False, complete=False, ownership_closed=True,
        GPU_executed=False, model_adoption_qualified=False, decode_performance_qualified=False,
        criteria='Exactly134 existing admission/terminal bounds plus9459 hash-span checks:independent original page/head/cell byte-order oracle over all intervals of19cells,3page sizes,3head counts,5row widths;full262144 spans coalesce to1;consumer failure stops;negative/overflow domains reject before visiting. Owned compile/test exit0;unchanged source/compiler/actual standard-header closure and selected13 assets. Source-only,no native/model/throughput claim.',
        retention='Keep source,recipe,compact counts and ownership outcome;retire the host executable and successful logs after this gate closes, before rebuilding the single unselected engine generation.')
    write(ROOT / 'record.json', record); container[KEY] = record; write(RESULT, container)
    print('Prepared host source gate;commit before execution.', flush=True)


def admission(record, container):
    require(RESULT.read_bytes() == source.build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))) and
            container[KEY] == {k:v for k,v in record.items() if k != 'commands'} and
            pins(record['source_commit']) == record['input_sha256'], 'commit exact host source plan first')
    require(digest(record['compiler']['path']) == record['compiler']['sha256'] and
            all(digest(p) == h for p,h in record['dependencies'].items()), 'compiler/header closure changed')
    require(source.baseline()[1] == record['selected_original'], 'selected fallback changed')


def run(container):
    record = json.loads((ROOT / 'record.json').read_text()); admission(record, container)
    require(record['stage'] == 'prepared', 'no retry')
    original = dict(record); owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(record, commands=owner.commands if owner else []))
    owner = source.supervision.Owner(ROOT / 'owned', persist)
    record['stage'] = 'running'; persist()
    try:
        limits = dict(wall=90, cpu=90, rss_cap=2 << 30, text_cap=1 << 20,
                      file_cap=16 << 20, total_cap=16 << 20)
        for label, command in [('compile', record['command']), ('fixture', [str(ROOT / 'host_bounds')])]:
            entry, out, err = owner.run(label, command, ENV, P, **limits)
            require(source.supervision.closed(entry) and entry['exit_code'] == 0,
                    label + ' failed: ' + err.read_text(errors='replace')[-4096:])
            if label == 'fixture':
                require(out.read_text() == 'cases=134 full_context_request_and_terminal_bounds=PASS\n'
                    'bounds_cases=134 kv_hash_cases=9459 byte_order_and_bounds=PASS full_extent_spans=1 GPU=UNRUN\n',
                    'hash-order/bounds fixture output differs')
        admission(original, container)
        record.update(passed=True, complete=True, bounds_cases=134, kv_hash_cases=9459)
    except BaseException as error:
        record['error'] = repr(error)
    finally:
        record['ownership_closed'] = owner.active is None and all(source.supervision.closed(c) for c in owner.commands)
        record['passed'] = record['complete'] = bool(record['passed'] and record['ownership_closed'])
        record['stage'] = 'closed' if record['passed'] else 'failed'
        persist(); container[KEY] = record; write(RESULT, container)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','error')}), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['check']), 'prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); source.guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        if sys.argv[1] == 'prepare': prepare(container)
        else: sys.exit(run(container))
