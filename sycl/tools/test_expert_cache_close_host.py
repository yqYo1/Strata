"""Check production cache-close ordering with deferred CPU memory consumers."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def extract(source):
    start = source.index('void ExpertCache::close() {')
    return source[start:source.index('\n/// R4.2g.', start)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source_path = root / 'sycl/src/core/expert_cache.cpp'
    control = 'fde2eed5524034a94b30c5fb8e870ccebddbea90'
    sources = dict(control=subprocess.check_output(['git', 'show', control + ':sycl/src/core/expert_cache.cpp'], cwd=root, text=True),
                   candidate=source_path.read_text())
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1:abort_on_error=1',
               UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  control_commit=control, scope='Actual production close method; CPU queue/USM stubs, no SYCL/Level Zero/GPU',
                  sources={}, runs=[], passed=False)
    for variant, source in sources.items():
        output = args.output / variant; output.mkdir()
        (output / 'close.inc').write_text(extract(source))
        record['sources'][variant] = hashlib.sha256(source.encode()).hexdigest()
        command = ['g++', '-std=c++20', '-O1', '-g', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie',
                   '-fsanitize=address,undefined', '-I' + str(output),
                   str(Path(__file__).with_name('expert_cache_close_host_test.cpp')), '-o', str(output / 'probe')]
        build = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (output / 'build.stderr').write_text(build.stderr)
        assert build.returncode == 0, build.stderr
        modes = ['read', 'write'] if variant == 'control' else ['read', 'write', 'two-queues', 'failed-drain', 'repeat-close']
        for mode in modes:
            result = subprocess.run([str(output / 'probe'), mode], env=env, capture_output=True, text=True, timeout=5)
            (output / (mode + '.stdout')).write_text(result.stdout)
            (output / (mode + '.stderr')).write_text(result.stderr)
            passed = ('AddressSanitizer: heap-use-after-free' in result.stderr and result.returncode != 0
                      if variant == 'control' else result.returncode == 0 and 'PASS ' + mode in result.stdout)
            record['runs'].append(dict(variant=variant, mode=mode, exit_code=result.returncode, expected_result=passed))
            (args.output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
            assert passed, (variant, mode, result.stdout, result.stderr)
    record.update(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
