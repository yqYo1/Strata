#!/usr/bin/env python3
"""GPU-free regression contract for PfTimer extracted verbatim from engine source.

Run with Python 3 and a host C++17 compiler; no SYCL runtime/device is required.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

SOURCE = Path(__file__).resolve().parents[1] / 'src/prefill/prefill.cpp'
STUBS = r'''
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
static int calls = 0, waits = 0, submits = 0, destroyed = 0;
static int fail_query = -1;
static bool unknown_completion = false, incomplete_completion = false;
namespace sycl {
namespace property { namespace queue { struct enable_profiling {}; } }
enum class aspect { queue_profiling };
namespace info {
namespace event { struct command_execution_status {}; }
namespace event_profiling { struct command_end {}; }
enum class event_command_status { complete, running };
}
struct device {
 bool profiling;
 bool has(aspect) const { ++calls; return profiling; }
};
struct event {
 int id = 0;
 uint64_t endpoint = 0;
 ~event() { ++destroyed; }
 template<class T> info::event_command_status get_info() const {
  ++calls;
  if (unknown_completion) throw std::runtime_error("completion unknown");
  return incomplete_completion ? info::event_command_status::running : info::event_command_status::complete;
 }
 template<class T> uint64_t get_profiling_info() const {
  ++calls;
  if (id == fail_query) throw std::runtime_error("profiling unavailable");
  return endpoint;
 }
};
struct queue {
 bool property = true, aspect_ok = true, async_fault = false;
 std::vector<uint64_t> ends;
 template<class T> bool has_property() const { ++calls; return property; }
 device get_device() const { ++calls; return {aspect_ok}; }
 void wait() { ++calls; ++waits; }
 void throw_asynchronous() { ++calls; if (async_fault) throw std::runtime_error("async fault"); }
};
}
namespace dpct {
using event_ptr = sycl::event*;
using queue_ptr = sycl::queue*;
void sync_barrier(event_ptr event, queue_ptr q) {
 ++calls; ++submits;
 event->id = submits;
 event->endpoint = q->ends.empty() ? uint64_t(submits) * 1000000 : q->ends.at(submits - 1);
}
int destroy_event(event_ptr event) { ++calls; delete event; return 0; }
}
#define DPCT_CHECK_ERROR(x) (x)
'''
MAIN = r'''
struct Canary { ~Canary() { std::fprintf(stdout, "UNWOUND\n"); std::fflush(stdout); } };
int main(int argc, char** argv) {
 const std::string mode = argv[1];
 sycl::queue q, other;
 Canary canary;
 {
  PfTimer timer;
  if (mode == "missing_property") q.property = false;
  if (mode == "missing_aspect") q.aspect_ok = false;
  if (mode == "reversed") q.ends = {9000000, 1000000};
  if (mode == "equal") q.ends = {7000000, 7000000};
  if (mode == "query_begin") fail_query = 1;
  if (mode == "query_end") fail_query = 2;
  if (mode == "zero") { timer.report(5, 10, 0, 0); }
  else {
   timer.mark(kPfStart, &q);
   if (mode != "one") {
    timer.mark(mode == "phase_invalid" ? kPfCount : kPfHc,
               mode == "queue_change" ? &other : &q);
   }
   if (mode == "incomplete" || mode == "destructor_incomplete") incomplete_completion = true;
   if (mode == "unknown") unknown_completion = true;
   if (mode == "async") q.async_fault = true;
   if (mode != "destructor_incomplete") {
    timer.fold();
    if (mode == "normal") {
     timer.mark(kPfGdn, &q); timer.mark(kPfQsa, &q); timer.fold();
     timer.mark(kPfStart, &q); timer.fold();
    }
    timer.report(5, 10, 0, 0);
   }
  }
 }
 std::printf("COUNTERS calls=%d waits=%d submits=%d destroyed=%d\n", calls, waits, submits, destroyed);
}
'''


def extract():
    source = SOURCE.read_text()
    start = source.index('enum PfPhase {')
    end = source.index('\n// Profile the memcpy commands themselves', start)
    actual = source[start:end]
    assert 'struct PfTimer {' in actual and 'struct ExpertTransferTimer' not in actual
    return actual


def main():
    with tempfile.TemporaryDirectory(prefix='strata-pftimer-contract-') as tmp:
        cpp, exe = Path(tmp) / 'contract.cpp', Path(tmp) / 'contract'
        cpp.write_text(STUBS + extract() + MAIN)
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O0', '-Wall', '-Wextra',
                        str(cpp), '-o', str(exe)], check=True)
        base_env = os.environ.copy()
        for name in ('STRATA_PREFILL_TIMING', 'STRATA_PREFILL_SYNC'):
            base_env.pop(name, None)

        def run(mode, enabled=True, sync=False):
            env = base_env.copy()
            if enabled:
                env['STRATA_PREFILL_TIMING'] = '0'  # Presence, including zero, enables it.
            if sync:
                env['STRATA_PREFILL_SYNC'] = '0'
            result = subprocess.run([str(exe), mode], env=env, text=True, capture_output=True)
            receipts = [json.loads(line.split(': ', 1)[1]) for line in result.stderr.splitlines()
                        if line.startswith('strata prefill phase validity: ')]
            return result, receipts

        fatal = {'incomplete': 'marker_incomplete', 'unknown': 'completion_query_failure',
                 'async': 'queue_async_failure', 'destructor_incomplete': 'destruction_marker_incomplete'}
        invalid = {'missing_property': 'queue_profiling_unavailable',
                   'missing_aspect': 'queue_profiling_unavailable',
                   'query_begin': 'profiling_query_failure', 'query_end': 'profiling_query_failure',
                   'reversed': 'nonmonotonic_endpoints', 'queue_change': 'queue_changed',
                   'phase_invalid': 'phase_index', 'zero': 'final_reconciliation',
                   'one': 'final_reconciliation'}
        for mode, reason in {**invalid, **fatal}.items():
            result, receipts = run(mode)
            assert result.returncode == (1 if mode in fatal else 0), (mode, result)
            assert len(receipts) == 1 and receipts[0]['status'] == 'invalid', (mode, receipts)
            assert receipts[0]['reason'] == reason, (mode, receipts)
            assert not re.search(r'strata prefill phases:|GPU timeline|\([0-9.]+%\)', result.stderr), (mode, result.stderr)
            if mode in fatal:
                assert 'UNWOUND' not in result.stdout and 'COUNTERS' not in result.stdout, (mode, result.stdout)
                assert receipts[0]['incomplete'] == 1
            else:
                assert 'UNWOUND' in result.stdout
            if mode.startswith('query_'):
                assert receipts[0]['query_attempts'] == 2 and receipts[0]['query_failures'] == 1
                assert receipts[0]['query_successes'] == 1 and receipts[0]['intervals_valid'] == 0
                assert 'phase query failure: interval 1' in result.stderr
            if mode == 'reversed':
                assert receipts[0]['raw_begin_ns'] == 9000000 and receipts[0]['raw_end_ns'] == 1000000
                assert receipts[0]['nonmonotonic'] == 1 and receipts[0]['intervals_valid'] == 0
                assert 'begin 9000000 end 1000000' in result.stderr
            if mode.startswith('missing_'):
                assert receipts[0]['markers_submitted'] == 0 and receipts[0]['query_attempts'] == 0

        for mode, marks, intervals in [('normal', 5, 4), ('equal', 2, 1)]:
            result, receipts = run(mode)
            assert result.returncode == 0 and len(receipts) == 1, (mode, result)
            receipt = receipts[0]
            assert receipt['status'] == 'valid' and receipt['queue_admitted']
            assert receipt['marker_attempts'] == receipt['markers_submitted'] == marks
            assert receipt['retained_markers'] == 1
            assert receipt['intervals_attempted'] == receipt['intervals_valid'] == intervals
            assert receipt['query_attempts'] == receipt['query_successes'] == 2 * intervals
            assert receipt['query_failures'] == receipt['nonmonotonic'] == receipt['incomplete'] == 0
            phase_lines = [line for line in result.stderr.splitlines() if line.startswith('strata prefill phases: ')]
            assert len(phase_lines) == 1
            phases = json.loads(phase_lines[0].split(': ', 1)[1])
            assert phases['gpu_timeline_ms'] == (4 if mode == 'normal' else 0)
            assert sum(phases['phase_ms'].values()) == phases['gpu_timeline_ms']
            assert re.search(r'waits=0 submits=' + str(marks), result.stdout)

        result, receipts = run('normal', enabled=False)
        assert result.returncode == 0 and not receipts and not result.stderr
        assert 'calls=0 waits=0 submits=0 destroyed=0' in result.stdout
        result, receipts = run('normal', enabled=False, sync=True)
        assert result.returncode == 0 and not receipts
        assert 'calls=5 waits=5 submits=0 destroyed=0' in result.stdout
        assert result.stderr.count('strata prefill sync: mark ') == 5
        print('PfTimer contract: all GPU-free cases passed (actual source extraction).')


if __name__ == '__main__':
    main()
