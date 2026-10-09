#!/usr/bin/python3
"""Exercise the actual bounded IQ hint helpers on CPU; never initialize a GPU."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-iq-prefetch-bounds-v0141-20261009')
OUT = B / 'iq-prefetch-bounds-host-v1'
SOURCE = W / 'src/kernels/cpu/iq_avx2.cpp'
SHA = '6b12af8b57c12698de6a240c85706fdcfadccb67fb4aa72b2baa715cc3f00d4e'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def function(text, start):
    begin = text.index(start)
    brace = text.index('{', begin)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0: return text[begin:i + 1]
    raise RuntimeError('unterminated actual helper')

assert digest(SOURCE) == SHA
prior = json.loads((B / 'owned-pool-tasks-v0141-code32k-tasks6-diagnostic-r1/record.json').read_text())
assert not prior['active'] and prior['completed'] and prior['healthy'] and prior['math_gate_passed'] and prior['exit_code'] == 0
assert digest(B / 'owned-pool-tasks-v0141-code32k-tasks6-diagnostic-r1/record.json') == '116604dfa1206464b668c9e391d6a66f09350fe35abec7071402a1621641323f'
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists()
OUT.mkdir()
record = dict(active=True, passed=False, gpu_executed=False, measurement_executed=False,
              created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source=str(SOURCE), source_sha256=SHA, controller_sha256=digest(Path(__file__)),
              scope='Actual extracted parser/hint helpers; hint calls captured with bounds oracle and volatile byte reads. CPU ASan/UBSan only, no math-kernel or runtime equivalence claim.',
              steps=[])
rp = OUT / 'record.json'
def save(): rp.write_text(json.dumps(record, indent=2) + '\n')
save()
try:
    text = SOURCE.read_text()
    parser = function(text, 'int prefetch_distance() {')
    helper = function(text, 'inline void rows_ahead(')
    assert helper.count('_mm_prefetch(') == 2
    record['actual_parser_sha256'] = hashlib.sha256(parser.encode()).hexdigest()
    record['actual_helper_sha256'] = hashlib.sha256(helper.encode()).hexdigest()
    body = r'''
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <algorithm>
#define _MM_HINT_T0 3
static uintptr_t base_address, end_address;
static std::vector<uintptr_t> captured;
static uint64_t reads = 0;
static void capture_prefetch(const char* p, int hint) {
    const auto a = reinterpret_cast<uintptr_t>(p);
    if (hint != _MM_HINT_T0 || a < base_address || a >= end_address) {
        std::fprintf(stderr, "invalid hint address %llu outside [%llu,%llu)\n",
                     (unsigned long long)a, (unsigned long long)base_address,
                     (unsigned long long)end_address);
        std::exit(3);
    }
    captured.push_back(a);
    volatile uint8_t observed = *reinterpret_cast<const uint8_t*>(p);
    (void)observed;
    ++reads;
}
'''
    body += parser + '\n' + helper.replace('_mm_prefetch(', 'capture_prefetch(') + '\n'
    body += r'''
int main(int argc, char** argv) {
    if (argc != 2) return 9;
    const int expected = std::atoi(argv[1]);
    if (prefetch_distance() != expected || prefetch_distance() != expected) return 10;
    const size_t lengths[] = {0,1,2,63,64,65,66,127,128,129,360,720,2048,2112,4096,65536};
    const int distances[] = {INT_MIN,-2048,-1,0,1,2,63,64,65,512,1024,2048,INT_MAX,expected};
    uint64_t cases = 0;
    for (size_t length : lengths) {
        // Prefix/suffix sentinels ensure the oracle checks the caller's logical
        // matrix/assigned range, not just the larger allocation's address range.
        std::vector<uint8_t> storage(length + 96, 0xa5);
        const uint8_t* begin = storage.data() + 32;
        const uint8_t* end = begin + length;
        base_address = reinterpret_cast<uintptr_t>(begin);
        end_address = reinterpret_cast<uintptr_t>(end);
        std::vector<size_t> offsets;
        for (size_t i=0; i<=std::min<size_t>(length,256); ++i) offsets.push_back(i);
        offsets.push_back(length / 2); offsets.push_back(length);
        if (length) offsets.push_back(length-1);
        if (length > 1) offsets.push_back(length-2);
        std::sort(offsets.begin(), offsets.end());
        offsets.erase(std::unique(offsets.begin(),offsets.end()),offsets.end());
        for (size_t offset : offsets) for (int pf : distances) {
            captured.clear();
            rows_ahead(begin+offset,end,pf);
            std::vector<uintptr_t> wanted;
            // Independent integer-address oracle: enumerate the original two
            // hint targets and retain exactly the targets in the logical range.
            if (pf > 0) for (uint64_t extra : {uint64_t(0), uint64_t(64)}) {
                const uint64_t target = uint64_t(offset) + uint64_t(pf) + extra;
                if (target < length) wanted.push_back(base_address + target);
            }
            if (captured != wanted) return 11;
            ++cases;
        }
        for (uint8_t v : storage) if (v != 0xa5) return 12;
    }
    std::printf("{\"cases\":%llu,\"bounded_reads\":%llu,\"parsed_distance\":%d}\n",
                (unsigned long long)cases,(unsigned long long)reads,expected);
}
'''
    cpp = OUT / 'actual-hint-host-test.cpp'
    cpp.write_text(body)
    exe = OUT / 'actual-hint-host-test'
    env = dict(os.environ)
    env['PATH'] = '/usr/bin:/bin'
    env.pop('LD_LIBRARY_PATH', None)
    env.pop('LIBRARY_PATH', None)
    env['ASAN_OPTIONS'] = 'detect_leaks=1:halt_on_error=1'
    env['UBSAN_OPTIONS'] = 'halt_on_error=1:print_stacktrace=1'
    def run(label, argv, process_env=env):
        r = subprocess.run(argv, env=process_env, capture_output=True, text=True, timeout=120)
        (OUT / (label+'.stdout')).write_text(r.stdout)
        (OUT / (label+'.stderr')).write_text(r.stderr)
        record['steps'].append(dict(label=label, argv=argv, exit_code=r.returncode,
                                    stdout_sha256=digest(OUT/(label+'.stdout')),
                                    stderr_sha256=digest(OUT/(label+'.stderr'))))
        save()
        assert r.returncode == 0, (label,r.returncode,r.stderr)
        return r.stdout
    run('compile', ['/usr/bin/g++','-std=c++20','-O1','-g','-fno-omit-frame-pointer','-fno-pie','-no-pie','-fsanitize=address,undefined',str(cpp),'-o',str(exe)])
    cases = [(None,2048),('',0),('0',0),('-1',0),('-2147483648',0),('1',1),('2',2),('63',63),('64',64),('65',65),('512',512),('1024',1024),('2048',2048),('2147483647',2147483647),('2147483648',0),('9999999999999999999999999999999',0),('-9999999999999999999999999999999',0),('+2048',0),(' 2048',0),('2048 ',0),('2048junk',0),('junk',0),('00002048',2048),('0000',0)]
    results = []
    for i,(value,expected) in enumerate(cases):
        child = dict(env)
        child.pop('STRATA_IQ_PREFETCH',None)
        if value is not None: child['STRATA_IQ_PREFETCH'] = value
        answer = json.loads(run(f'case{i}',[str(exe),str(expected)],child))
        results.append(dict(environment_value=value,expected=expected,**answer))
    assert digest(SOURCE) == SHA
    record.update(active=False, passed=True, cases=results, parser_processes=len(cases),
                  helper_cases=sum(x['cases'] for x in results), bounded_reads=sum(x['bounded_reads'] for x in results),
                  harness_sha256=digest(cpp), binary_sha256=digest(exe))
except BaseException as exc:
    record.update(active=False,error=repr(exc))
    raise
finally:
    save()
    fcntl.flock(lock,fcntl.LOCK_UN)
    lock.close()
print(json.dumps({k:record[k] for k in ['passed','gpu_executed','parser_processes','helper_cases','bounded_reads']}))
