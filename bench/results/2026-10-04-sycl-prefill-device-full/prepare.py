"""Prepare diagnostic sources only, after committing the accepted kernel change."""
from pathlib import Path
import subprocess
import sys
ref = sys.argv[1] if len(sys.argv) > 1 else 'HEAD'
subprocess.run([sys.executable, str(Path(__file__).with_name('base-prepare.py')), ref], check=True)
header = Path(__file__).with_name('trace.hpp').read_text()
anchor = 'struct Scope {'
assert header.count(anchor) == 1
header = header.replace(anchor, '''// Only device duration is measured here; submit_us=0 means not collected.
inline void put_event(const char* file, sycl::event e, int line) {
    if (!state().active.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(state().mutex);
    state().items.push_back({file,e,0,line,0,0,0,0.0});
}
''' + anchor)
Path('/tmp/strata-sycl-goal-prefill-device-full.hpp').write_text(header)
path='include/strata/sycl/launch.hpp'
s=subprocess.check_output(['git','show',f'{ref}:{path}']).decode()
s=s.replace('#include <initializer_list>', '#include <initializer_list>\n#include <source_location>\n#include "/tmp/strata-sycl-goal-prefill-device-full.hpp"')
s=s.replace('inline void finish(void *stream, sycl::event event) {', '''inline void finish(void *stream, sycl::event event,
                   std::source_location location = std::source_location::current()) {
  strata_prefill_trace::put_event(location.file_name(), event, int(location.line()));''')
s=s.replace('void for_each(int64_t count, void *stream, Function function) {', '''void for_each(int64_t count, void *stream, Function function,
              std::source_location location = std::source_location::current()) {''')
s=s.replace('  finish(stream, event);', '  finish(stream, event, location);')
Path('/tmp/strata-sycl-goal-prefill-device-full-launch.hpp').write_text(s)
for name in ('prefill.cpp','compat.cpp','xmx.cpp'):
 p=Path('/tmp/strata-sycl-goal-prefill-device-trace-'+name)
 s=p.read_text().replace('/tmp/strata-sycl-goal-prefill-device-trace.hpp','/tmp/strata-sycl-goal-prefill-device-full.hpp')
 Path('/tmp/strata-sycl-goal-prefill-device-full-'+name).write_text(s)
print('Prepared full event capture without modifying production sources.')
