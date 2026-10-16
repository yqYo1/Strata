#!/usr/bin/env python3
"""One bounded syntax check; does not initialize or execute a GPU runtime."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = A / 'moe-window-fixture-syntax-v2.json'
    root = B / 'main-moe-window-fixture-syntax-v2'
    assert not out.exists() and not root.exists()
    helper = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(helper) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('owned', helper)
    owned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owned)
    database = B / 'main-bf16-prefix-tilepack-engine-build-v1/build/compile_commands.json'
    old_test = 'tests/sycl/native16_prefill.cpp'
    new_test = 'tests/sycl/moe_window.cpp'
    entry = next(e for e in json.loads(database.read_text()) if e['file'].endswith('/' + old_test))
    argv = shlex.split(entry['command'])
    command = [str(Path(shutil.which('nice'))), '-n', '19', argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        i += 1
        if arg == '-o':
            i += 1
            continue
        if arg == '-c' or arg.startswith('-O'):
            continue
        if arg.startswith('-DSTRATA_VERSION='):
            arg = '-DSTRATA_VERSION="0.1.38"'
        arg = arg.replace('/home/yayoi/orca/workspaces/Strata/sycl', str(P))
        if arg.endswith('/' + old_test):
            arg = str(P / new_test)
        command.append(arg)
    command += ['-O0', '-fsyntax-only']
    names = [new_test, 'cmake/sycl_backend.cmake', 'include/strata/prefill/moe_window_routes.hpp',
             'include/strata/prefill/native16.hpp', 'include/strata/prefill/draft_uploads.hpp',
             'include/strata/sycl/launch.hpp', 'include/strata/sycl/runtime.hpp',
             'src/prefill/sycl/native16.cpp', 'src/prefill/sycl/postops.cpp',
             str(Path(__file__).relative_to(P))]
    pins = {name: sha(P / name) for name in names}
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir()
        (root / 'tmp').mkdir()
        env = {key: os.environ[key] for key in ('PATH', 'LD_LIBRARY_PATH') if key in os.environ}
        env.update(LANG='C', LC_ALL='C', TMPDIR=str(root / 'tmp'))
        record = dict(schema='strata-moe-window-fixture-syntax-v2', passed=False,
                      complete=False, GPU_executed=False, code_generation=False, sources=pins,
                      owner_sha256=sha(helper), database_sha256=sha(database),
                      compiler=dict(path=argv[0], sha256=sha(argv[0])),
                      limits=dict(priority=19, wall_seconds=45, CPU_seconds_per_process=20,
                                  session_RSS_bytes=2 << 30, artifact_bytes=8 << 20),
                      scope='SYCL syntax only. CMake configure/link and every GPU fixture case remain unexecuted.',
                      retention=dict(root=str(root), budget_bytes=8 << 20,
                                     consumer='Main MoE window fixture syntax diagnostics',
                                     review_point='After actual fixture build or correction of distinct syntax failure.'))

        def save():
            out.write_text(json.dumps(record, indent=2) + '\n')

        owner = owned.Owner(root / 'owned', save)
        record['commands'] = owner.commands
        save()
        try:
            assert all(sha(P / n) == h for n, h in pins.items())
            result, _, _ = owner.run('moe_window_fixture', command, env, P, wall=45, cpu=20,
                                     rss_cap=2 << 30, text_cap=64 << 10,
                                     total_cap=8 << 20, file_cap=1 << 20)
            assert owned.closed(result) and result['exit_code'] == 0
            assert all(sha(P / n) == h for n, h in pins.items())
            record.update(passed=True, complete=True, ownership_closed=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
        print(json.dumps(dict(passed=record['passed'], complete=record['complete'], receipt=str(out))), flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
