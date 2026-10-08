"""Exercise the actual shell wrapper with a recording Docker stub; no GPU."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile

base = Path(__file__).parent
out = base / 'upstream-v0141-server-env-forwarding-fix-v1'
candidate = out / 'strata-sycl.sh'
original = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-v0.1.41-20261009/sycl/serve/strata-sycl.sh')
preparation = json.loads((out / 'source-preparation.json').read_text())
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == preparation['candidate_sha256']
assert hashlib.sha256(original.read_bytes()).hexdigest() == preparation['base_sha256']
assert not (out / 'cpu-test-record.json').exists()
cases = []

with tempfile.TemporaryDirectory(prefix='wrapper-fixture-', dir=out) as temp:
    root = Path(temp)
    bin_dir = root / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'docker'
    stub.write_text('#!/usr/bin/python3\nimport json,os,sys\nwith open(os.environ["DOCKER_STUB_CAPTURE"],"a") as f:\n f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    stub.chmod(0o700)
    copy = root / 'repo/sycl/serve/strata-sycl.sh'
    copy.parent.mkdir(parents=True)
    capture = root / 'docker.jsonl'

    def run(source, env, args=()):
        shutil.copyfile(source, copy)
        capture.unlink(missing_ok=True)
        actual_env = {'PATH': str(bin_dir) + ':/usr/bin:/bin', 'DOCKER_STUB_CAPTURE': str(capture)}
        actual_env.update(env)
        result = subprocess.run(['/bin/bash', str(copy), *args], env=actual_env, capture_output=True, text=True, timeout=10)
        calls = [json.loads(x) for x in capture.read_text().splitlines()] if capture.exists() else []
        values = {}
        if result.returncode == 0:
            assert len(calls) == 2 and calls[0][0] == 'rm' and calls[1][0] == 'run'
            argv = calls[1]
            for i, key in enumerate(argv[:-1]):
                if key == '-e':
                    name, value = argv[i + 1].split('=', 1)
                    assert name not in values
                    values[name] = value
        return result, calls, values

    result, calls, values = run(candidate, {})
    assert result.returncode == 0 and values == {'STRATA_VERIFY_DEVICE_PLAN': '1', 'STRATA_STAGER_THREADS': '12'}
    cases.append({'name': 'existing_host_completion_defaults', 'passed': True})

    zeros = {'NEOReadDebugKeys': '1', 'EnableDirectSubmission': '0', 'SYCL_CACHE_PERSISTENT': '0',
             'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD': '0', 'IGC_TestSetting': '0', 'ZES_TestSetting': '0',
             'OverrideDefaultFP64Settings': '0'}
    result, calls, values = run(original, zeros)
    assert result.returncode == 0 and 'EnableDirectSubmission' not in values and 'SYCL_CACHE_PERSISTENT' not in values
    cases.append({'name': 'actual_original_reproduces_missing_runtime_zeros', 'passed': True,
                  'original_forwarded_settings': values})
    result, calls, values = run(candidate, zeros)
    assert result.returncode == 0 and all(values.get(k) == v for k, v in zeros.items())
    cases.append({'name': 'runtime_and_driver_zeros_retained', 'passed': True, 'forwarded': zeros})

    numeric = {'STRATA_PREFILL_FIRST': '0', 'STRATA_STAGER_THREADS': '0', 'STRATA_TRACE': '0',
               'STRATA_PREFILL_TIMING': '0', 'STRATA_VERIFY_DEVICE_PLAN': '0'}
    result, calls, values = run(candidate, numeric)
    assert result.returncode == 0 and values['STRATA_PREFILL_FIRST'] == values['STRATA_STAGER_THREADS'] == '0'
    assert all(k not in values for k in ['STRATA_TRACE', 'STRATA_PREFILL_TIMING', 'STRATA_VERIFY_DEVICE_PLAN'])
    cases.append({'name': 'numeric_zeros_retained_presence_switches_disabled', 'passed': True})

    env = {'ONEAPI_DEVICE_SELECTOR': 'level_zero:gpu', 'UR_LOG_TRACING': 'level:info;flush:info;output:stderr',
           'STRATA_PREFILL_COMPACT': '2', 'STRATA_SYCL_NAME': 'wrapper-fixture',
           'STRATA_SYCL_HOST_BOUNDARY': '1', 'PRIVATE_API_TOKEN': 'fixture-not-forwarded'}
    args = ['--pack', '/work/models/model with spaces', '--test-value', 'literal; value $chars']
    result, calls, values = run(candidate, env, args)
    assert result.returncode == 0
    assert all(values[k] == env[k] for k in ['ONEAPI_DEVICE_SELECTOR', 'UR_LOG_TRACING', 'STRATA_PREFILL_COMPACT'])
    assert all(k not in values for k in ['PRIVATE_API_TOKEN', 'STRATA_SYCL_NAME', 'STRATA_SYCL_HOST_BOUNDARY'])
    cmd = shlex.split(calls[1][-1])
    assert cmd[:1] == ['cd'] and cmd[2:4] == ['&&', 'exec']
    assert cmd[5:] == args, cmd
    cases.append({'name': 'upstream_environment_support_and_literal_arguments_preserved', 'passed': True})

    result, calls, values = run(candidate, {'SYCL_TEST_EMPTY_SETTING': ''})
    assert result.returncode == 0 and values['SYCL_TEST_EMPTY_SETTING'] == ''
    cases.append({'name': 'explicit_empty_runtime_value_retained', 'passed': True})

    for value in ['0', '1', '']:
        result, calls, values = run(candidate, {'STRATA_VERIFY_NO_HOST': value})
        assert result.returncode == 2 and not calls and 'device-spin' in result.stderr
        cases.append({'name': 'retired_no_host_' + repr(value) + '_rejected_before_docker', 'passed': True})
    result, calls, values = run(candidate, {'STRATA_SYCL_HOST_BOUNDARY': '0'})
    assert result.returncode == 2 and not calls
    cases.append({'name': 'disabled_host_boundaries_rejected_before_docker', 'passed': True})

result = subprocess.run(['/bin/bash', '-n', str(candidate)], capture_output=True, text=True)
assert result.returncode == 0, result.stderr
record = {'active': False, 'passed': True, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'CPU-only execution of the actual wrapper with a recording Docker stub. Original failure reproduced, corrected behavior and retired-path admission checked. No real Docker/container/GPU run.',
          'source_sha256': preparation['base_sha256'], 'candidate_sha256': preparation['candidate_sha256'],
          'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'cases': cases, 'shell_syntax_exit_code': result.returncode, 'tracked_source_modified': False}
(out / 'cpu-test-record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, ensure_ascii=False))
