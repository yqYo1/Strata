#!/usr/bin/env python3
"""CPU ASan/UBSan regression test of extracted verifier watchdog code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
source = root / 'sycl/src/core/verify.cpp'
test = Path(__file__).with_name('verifier_registry_host_test.cpp')
control_revision = '4a9b24724da2a683186d1ce4199c39c6090b51ae'
env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:abort_on_error=1',
           UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
record = dict(scope='Actual production registry/callbacks and registration/removal snippets; CPU verifier substitutes payload for mapped flags/queues; no SYCL/Level Zero/GPU',
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              tests=[], controls=[], passed=False)

def save():
    (output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

def fragment(text, control):
    callback = text.index('void diag_active_verifier(')
    begin = text.rfind('namespace {', 0, callback) + len('namespace {')
    end = text.index('// #649: the host side', callback)
    globals_code = text[begin:end]
    init = text.index('bool Verifier::init(')
    prefix_begin = text.index('try {', init) + len('try {')
    prefix_end = text.index('    device_ =', prefix_begin)
    init_prefix = text[prefix_begin:prefix_end]
    destructor = text.index('Verifier::~Verifier() try {')
    removal_begin = destructor + len('Verifier::~Verifier() try {')
    removal_end = text.index('    if (cs_)', removal_begin)
    removal = text[removal_begin:removal_end]
    init_suffix = ''
    if not control:
        suffix_begin = text.index('    // Publish only fully initialized', init)
        suffix_end = text.index('    std::fprintf(stderr, "strata verify: window', suffix_begin)
        init_suffix = text[suffix_begin:suffix_end]
    return ('namespace {\n' + globals_code + '\n}\n'
            'bool Verifier::init(bool valid) {\n' + init_prefix +
            '    if (!valid) return false;\n'
            '    delete payload_; payload_ = new int(4242);\n'
            '    std::string err;\n' + init_suffix + '    return true;\n}\n'
            'Verifier::~Verifier() {\n' + removal +
            '    gate_->retired.store(true);\n    delete payload_;\n}\n')

def build(directory, text, control):
    directory.mkdir(exist_ok=True)
    header = directory / 'registry-under-test.hpp'
    header.write_text(fragment(text, control))
    binary = directory / 'registry-test'
    command = ['g++', '-std=c++20', '-O1', '-g', '-pthread', '-fsanitize=address,undefined',
               '-fno-omit-frame-pointer', '-DSTRATA_REGISTRY_CONTROL=' + str(int(control)),
               '-I' + str(directory), str(test), '-o', str(binary)]
    with (directory / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
    return binary, dict(compile=command, fragment_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())

save()
binary, record['candidate'] = build(output, source.read_text(), False)
for mode in ['diag-race', 'release-race', 'failed-init', 'capacity', 'stress']:
    started = time.monotonic()
    result = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=30, env=env)
    (output / (mode + '.stdout')).write_text(result.stdout)
    (output / (mode + '.stderr')).write_text(result.stderr)
    assert result.returncode == 0 and 'PASS: ' + mode in result.stdout, (mode, result.returncode, result.stderr)
    record['tests'].append(dict(mode=mode, exit_code=result.returncode, elapsed_seconds=time.monotonic() - started, passed=True))
    save()
control_source = subprocess.check_output(['git', 'show', control_revision + ':sycl/src/core/verify.cpp'], cwd=root, text=True)
control, record['control_build'] = build(output / 'control', control_source, True)
record['control_build'].update(revision=control_revision, source_sha256=hashlib.sha256(control_source.encode()).hexdigest())
for mode in ['diag-race', 'release-race']:
    result = subprocess.run([str(control), mode], capture_output=True, text=True, timeout=15, env=env)
    (output / 'control' / (mode + '.stdout')).write_text(result.stdout)
    (output / 'control' / (mode + '.stderr')).write_text(result.stderr)
    assert result.returncode != 0 and 'heap-use-after-free' in result.stderr and 'CONTROL: object deleted with callback in flight' in result.stdout, (mode, result.returncode, result.stderr)
    record['controls'].append(dict(mode=mode, exit_code=result.returncode, reproduced='ASan heap-use-after-free'))
    save()
dependencies = subprocess.check_output(['ldd', str(binary)], text=True)
(output / 'ldd.txt').write_text(dependencies)
assert all(name not in dependencies for name in ['libsycl', 'libze_', 'libur_'])
record['helper_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [test, Path(__file__)]}
record['passed'] = True
save()
print(json.dumps(record, indent=2))
