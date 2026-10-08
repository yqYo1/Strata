"""Exercise the production cache-copy methods without a GPU, under ASan/UBSan."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    root = Path(__file__).resolve().parents[2]
    source = (root / 'sycl/src/core/expert_cache.cpp').read_text()
    begin = source.index('namespace {\nbool slot_copy_fits(')
    end = source.index('\n}  // namespace strata::core', begin)
    # Include the actual methods, including their validation and copy calls.
    (args.output / 'copy.inc').write_text(source[begin:end])
    command = ['g++', '-std=c++20', '-O1', '-g', '-fno-pie', '-no-pie',
               '-fsanitize=address,undefined', '-I' + str(args.output),
               str(Path(__file__).with_name('cache_copy_bounds_host_test.cpp')),
               '-o', str(args.output / 'probe')]
    build = subprocess.run(command, capture_output=True, text=True, timeout=30)
    (args.output / 'build.stderr').write_text(build.stderr)
    record = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  scope='Production slot-copy methods with CPU queue stand-ins; no GPU proof',
                  build_exit_code=build.returncode, passed=False)
    if build.returncode == 0:
        result = subprocess.run([str(args.output / 'probe')], capture_output=True, text=True, timeout=10)
        (args.output / 'run.stdout').write_text(result.stdout)
        (args.output / 'run.stderr').write_text(result.stderr)
        record.update(exit_code=result.returncode,
                      passed=result.returncode == 0 and 'PASS cache-copy bounds' in result.stdout)
    (args.output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record['passed'] else 1)


if __name__ == '__main__':
    main()
