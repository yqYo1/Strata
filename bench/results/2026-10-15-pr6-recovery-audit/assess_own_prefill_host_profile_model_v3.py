#!/usr/bin/env python3
"""Offline v3 analysis; preserve the original stale-version-check FAIL unchanged."""
import fcntl
import json
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import require, sha, load_owner


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = A / 'own-prefill-host-profile-model-v3.json'
        r = json.loads(source.read_text())
        root = B / 'main-own-prefill-host-profile-model-root-v3'
        require(not r['passed'] and not r['complete'] and not r['active'] and
                r['error'] == "RuntimeError('actual extended observer version')", 'preserved original observer-version FAIL')
        require(sha(A / 'run_own_prefill_host_profile_model_v3.py') == r['recipe_sha256'] and
                sha(root / 'record.json') == sha(source), 'original recipe and receipt unchanged')
        require(r['GPU_executed'] and not r['fresh_faults'] and r['boot_unchanged'] and
                not r['driver_dump_exists'], 'closed healthy GPU evidence')
        for c in r['commands']:
            require(module.closed(c) and c['exit_code'] == 0, 'normal owned children')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'old owned process alive')
        require(r['result']['passed'] and r['result']['server_exit_code'] ==
                r['result']['real_engine_exit']['exit_code'] == 0, 'native/server succeeded')
        require(json.loads((root / 'engine-exit.json').read_text()) == r['result']['real_engine_exit'], 'native exit identity')
        require(all(sha(P / n) == h for n, h in r['source_pins'].items()), 'same frozen native sources')
        native = root / 'owned/engine.stderr'
        text = native.read_text()
        require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', text), 'no native errors')
        require(list(map(int, re.findall(r'layer-major layer (\d+) of 48: completed 10 chunks', text))) == list(range(48)), 'all48 layers10chunks')
        def fields(prefix):
            lines = [s for s in text.splitlines() if s.startswith(prefix)]
            require(len(lines) == 1, 'one long scope per line type')
            value = dict(re.findall(r'(\w+)=([^ ]+)', lines[0]))
            require(value['version'] == '3' and value['available'] == '1', 'actual v3 available summary')
            seconds = {k: float(v) for k, v in value.items() if k.endswith('_s')}
            require(all(math.isfinite(v) and v >= 0 for v in seconds.values()), 'finite nonnegative clocks')
            return dict(line=lines[0], fields=value, seconds=seconds)
        wait = fields('strata prefill cpu wait: ')
        host = fields('strata prefill cpu host: ')
        w, h = wait['fields'], host['fields']
        c, t = wait['seconds'], host['seconds']
        require(wait['line'] == r['CPU_wait_diagnostic']['line'], 'same actual line parsed before original failure')
        require(w['complete'] == '1' and all(w[k] == '0' for k in ('clock_or_accounting_error', 'open_waits', 'wait_throws')), 'valid completed wait accounting')
        require(h['open_regions'] == '0', 'no open host regions')
        counts = dict(source_copy=48, host_group=480, gemm_submit=239458, route_exchange=480,
                      upload_submit=48, residual_input_host=282, residual_input_device=188,
                      residual_store_host=282, residual_store_device=188)
        for k, expected in counts.items():
            require(h[k + '_count'] == str(expected) and h[k + '_throws'] == '0', 'exact new and original phase counts')
            require(t[k + '_cpu_s'] >= t[k + '_outside_wait_cpu_s'] and
                    t[k + '_wall_s'] >= t[k + '_outside_wait_wall_s'], 'inclusive/wait overlap')
        require(w['wait_count'] == '2053', 'all explicit waits')
        for unit, total, waited in [('cpu_s', 'main_cpu_s', 'wait_cpu_s'), ('wall_s', 'wall_s', 'wait_wall_s')]:
            residual = c[total] - c[waited] - sum(t[k + '_outside_wait_' + unit] for k in counts)
            require(abs(residual - t['unclassified_' + unit]) < 1e-7, 'full disjoint ' + unit + ' partition')
        require(abs(c['main_cpu_s'] - c['wait_cpu_s'] - c['outside_wait_cpu_s']) < 3e-8, 'wait union arithmetic')
        require(r['benchmark_result']['params']['think'] and r['request_journal_verification']['valid'] and
                r['request_journal_verification']['complete'], 'original benchmark evidence passed')
        requests = r['benchmark_result']['results']['prefill_only']['requests']
        require(len(requests) == 1 and requests[0]['actual_prompt_tokens'] == requests[0]['actual_read_tokens'] == 78453 and
                requests[0]['actual_reused_tokens'] == 0 and requests[0]['eligible'], 'actual request scope')
        for item in r['request_journals']:
            require(sha(root / item['path']) == item['sha256'], 'verified original journal unchanged')
        result = dict(passed=True, complete=True, active=False, GPU_executed=False,
            original_receipt=source.name, original_receipt_sha256=sha(source),
            original_result_status_unchanged=True, original_passed=False,
            original_failure='Main runner retained two stale version=2 assertions after adding actual version=3 observers.',
            assessment='Offline analysis of original normally closed native; no rerun or modification of original FAIL.',
            recipe_sha256=sha(Path(__file__)), native_log_sha256=sha(native),
            native_binary_sha256=r['binary_sha256'], original_native_execution_succeeded=True,
            CPU_wait_diagnostic=wait, CPU_host_diagnostic=host,
            route_wait_overlap_cpu_s=t['route_exchange_cpu_s'] - t['route_exchange_outside_wait_cpu_s'],
            route_wait_overlap_wall_s=t['route_exchange_wall_s'] - t['route_exchange_outside_wait_wall_s'],
            residual_host_copy_submit_cpu_s=t['residual_input_host_cpu_s'] + t['residual_store_host_cpu_s'],
            pure_route_transfer_time_measured=False, performance_qualified=False, quality_qualified=False,
            whole_model_CPU_fallback_excluded=False, physical_262144_qualified=False,
            scope='Route wait includes prior GPU computation; transfer-call CPU may include driver work/implicit blocking. CPU seconds are not removable latency.')
        target = A / 'own-prefill-host-profile-model-assessment-v3.json'
        with target.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=True, original_passed=False, wait=c, host=t, residual_host_copy_submit_cpu_s=result['residual_host_copy_submit_cpu_s'])), flush=True)


if __name__ == '__main__': main()
