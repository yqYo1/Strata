"""Test the production DPCT queue ownership guard without a GPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def extract(source):
    start = source.index('  void destroy_queue(sycl::queue *&queue) {')
    return source[start:source.index('  [[deprecated(', start)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    path = 'sycl/include/dpct/device.hpp'
    control = '900025461a0fd70906e5b54a3380c4b1341f4aea'
    sources = dict(control=subprocess.check_output(['git', 'show', control + ':' + path], cwd=root, text=True),
                   candidate=(root / path).read_text())
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:abort_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  control_commit=control, scope='Actual destroy_queue method; CPU queue objects, no SYCL/Level Zero/GPU',
                  sources={}, runs=[], passed=False)
    for variant, source in sources.items():
        out = args.output / variant; out.mkdir()
        (out / 'destroy.inc').write_text(extract(source))
        record['sources'][variant] = hashlib.sha256(source.encode()).hexdigest()
        build = subprocess.run(['g++', '-std=c++20', '-O1', '-g', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie',
                                '-fsanitize=address,undefined', '-I' + str(out),
                                str(Path(__file__).with_name('queue_ownership_host_test.cpp')), '-o', str(out / 'probe')],
                               capture_output=True, text=True, timeout=30)
        (out / 'build.stderr').write_text(build.stderr)
        assert build.returncode == 0, build.stderr
        modes = ['default-in', 'default-out'] if variant == 'control' else ['default-in', 'default-out', 'owned', 'foreign', 'null', 'repeat']
        for mode in modes:
            run = subprocess.run([str(out / 'probe'), mode], capture_output=True, text=True, env=env, timeout=5)
            (out / (mode + '.stdout')).write_text(run.stdout)
            (out / (mode + '.stderr')).write_text(run.stderr)
            passed = ('AddressSanitizer: heap-use-after-free' in run.stderr and run.returncode != 0
                      if variant == 'control' else run.returncode == 0 and 'PASS ' + mode in run.stdout)
            record['runs'].append(dict(variant=variant, mode=mode, exit_code=run.returncode, expected_result=passed))
            (args.output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
            assert passed, (variant, mode, run.stdout, run.stderr)
    record.update(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
