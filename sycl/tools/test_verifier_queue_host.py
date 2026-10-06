"""Reproduce borrowed-default-queue UAF using production verifier code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def extract_destructor(source):
    start = source.index('Verifier::~Verifier() try {')
    return source[start:source.index('bool Verifier::init(', start)].strip()


def extract_queues(source):
    return '\n'.join(re.search(r'dpct::queue_ptr ' + name + r'\s*=[^;]+;', source).group()
                     for name in ('cs_', 'copy_'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    control = '900025461a0fd70906e5b54a3380c4b1341f4aea'
    paths = ['sycl/src/core/verify.cpp', 'sycl/include/strata/core/verify.hpp']
    sources = {
        'control': [subprocess.check_output(['git', 'show', control + ':' + p], cwd=root, text=True) for p in paths],
        'candidate': [(root / p).read_text() for p in paths],
    }
    record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  control_commit=control, scope='Production destructor and initializers; CPU queue/USM stand-ins, no GPU',
                  sources={}, runs=[], passed=False)
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:abort_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    for variant, (cpp, hpp) in sources.items():
        out = args.output / variant; out.mkdir()
        (out / 'destructor.inc').write_text(extract_destructor(cpp) + '\n')
        (out / 'queues.inc').write_text(extract_queues(hpp) + '\n')
        record['sources'][variant] = dict(zip(paths, [hashlib.sha256(s.encode()).hexdigest() for s in (cpp, hpp)]))
        command = ['g++', '-std=c++20', '-O1', '-g', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie',
                   '-fsanitize=address,undefined', '-I' + str(out),
                   str(Path(__file__).with_name('verifier_queue_host_test.cpp')), '-o', str(out / 'probe')]
        build = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (out / 'build.stderr').write_text(build.stderr)
        assert build.returncode == 0, build.stderr
        modes = ['unused', 'copy-only'] if variant == 'control' else ['unused', 'copy-only', 'compute-only', 'both', 'both-remote', 'failed-drain']
        for mode in modes:
            run = subprocess.run([str(out / 'probe'), mode], env=env, capture_output=True, text=True, timeout=5)
            (out / (mode + '.stdout')).write_text(run.stdout)
            (out / (mode + '.stderr')).write_text(run.stderr)
            expected_code = 86 if mode == 'failed-drain' else 0
            passed = ('AddressSanitizer: heap-use-after-free' in run.stderr and run.returncode != 0
                      if variant == 'control' else run.returncode == expected_code and 'PASS ' + mode in run.stdout)
            record['runs'].append(dict(variant=variant, mode=mode, exit_code=run.returncode, expected_result=passed))
            (args.output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
            assert passed, (variant, mode, run.stdout, run.stderr)
    record.update(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
