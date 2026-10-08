"""Real owned CPU/GDB tests of pointer-offset reads and failure-only guards."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(root/'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity
from capture_owned_native_counter_v01402_v1 import capture, event_frames
out = base/'owned-native-counter-v01402-cpu-v1'
out.mkdir(mode=0o700)
source = r'''#include <cstdint>
#include <csignal>
#include <cstdlib>
namespace NEO {
struct InOrderExecEventData { uint64_t counterValue; uint32_t counterOffset; uint32_t hostPartitions; uint32_t devicePartitions; };
struct SharableEventDataHelper { InOrderExecEventData *eventDataPtr; };
struct InOrderExecEventHelper { SharableEventDataHelper sharableEventDataHelper; bool dataAssigned; uint64_t *baseHostCpuAddress; };
}
namespace L0 {
template <typename T> struct EventImp {
 NEO::InOrderExecEventHelper inOrderExecHelper;
 bool heapfullCbEventWithProfiling;
 __attribute__((noinline)) int queryCounterBasedEventStatus(long timeSinceWait) { raise(SIGTRAP); return timeSinceWait; }
};
}
int main(int argc,char **argv) {
 uint64_t counters[2]={999, static_cast<uint64_t>(std::strtoull(argv[1],nullptr,10))};
 NEO::InOrderExecEventData data{17,8,1,1};
 L0::EventImp<unsigned long> event{{{&data},true,counters},false};
 return event.queryCounterBasedEventStatus(0);
}
'''
(out/'fixture.cpp').write_text(source)
subprocess.run(['/usr/bin/g++', '-g', '-O0', '-std=c++20', str(out/'fixture.cpp'), '-o', str(out/'fixture')], check=True)
receipt = {'active': True, 'passed': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'scope': 'CPU synthetic layout only. Actual owned GDB reads pointers, nonzero byte offset and uint64 memory; not a GPU/native-runtime compatibility test.', 'cases': []}
for value in [11, 17, 29]:
    case_dir = out/str(value)
    case_dir.mkdir()
    g = OwnedGdb([str(out/'fixture'), str(value)], case_dir, dict(os.environ))
    try:
        g.command('-gdb-set may-call-functions off')
        g.run()
        end = time.monotonic()+15
        while not g.stops and time.monotonic() < end:
            g.poll(.05)
        assert g.stops and 'signal-name="SIGTRAP"' in g.stops[-1]
        snapshot = g.snapshot('cpu', resume=False)
        r = capture(g, snapshot, case_dir/'counter.json')
        assert r['all_matching_fields_captured'] and len(r['events']) == 1
        e = r['events'][0]
        assert e['expected_counter'] == 17 and e['first_host_counter'] == value
        assert e['counter_offset'] == 8 and e['host_partitions'] == 1 and e['all_host_partitions_read']
        assert e['first_counter_ge_expected'] == (value >= 17) and e['plain_counter_branch']
        prior = g.stops[-1]
        g.stops.append('resumed')
        try:
            capture(g, snapshot, case_dir/'must-not-read-running.json')
        except AssertionError:
            assert not (case_dir/'must-not-read-running.json').exists()
        else:
            raise AssertionError('running guard failed')
        finally:
            g.stops.pop()
        assert g.stops[-1] == prior
        receipt['cases'].append({'counter': value, 'capture_passed': True, 'running_guard_passed': True,
                                 'inferior': g.inferior, 'debugger': g.debugger_identity})
    finally:
        cleanup = g.close()
        assert not cleanup['inferior_survived'] and not cleanup['gdb_survived']
    for identity in [g.inferior, g.debugger_identity]:
        now = process_identity(identity['pid'])
        assert not now or now['start_ticks'] != identity['start_ticks']
assert event_frames('Thread 1 (Thread CPU)\n#0 main ()\n') == []
receipt.update(active=False, passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               capture_controller_sha256=hashlib.sha256((base/'capture_owned_native_counter_v01402_v1.py').read_bytes()).hexdigest(),
               controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               owned_helper_sha256=hashlib.sha256((root/'sycl/tools/owned_gdb.py').read_bytes()).hexdigest(),
               fixture_source_sha256=hashlib.sha256((out/'fixture.cpp').read_bytes()).hexdigest())
(out/'record.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'passed': receipt['passed'], 'cases': len(receipt['cases']), 'gpu_used': False}, indent=2))
