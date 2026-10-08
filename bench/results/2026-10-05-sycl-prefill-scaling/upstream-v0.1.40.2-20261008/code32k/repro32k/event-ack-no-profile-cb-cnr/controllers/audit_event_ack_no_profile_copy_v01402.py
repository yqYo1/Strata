"""Record source boundaries of the private, unregistered copy-queue experiment."""
from pathlib import Path
import datetime
import hashlib
import json

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source = root/'build-sycl-event-ack-no-profile-copy-v2-20261008/source/prefill.cpp'
header = root/'sycl/include/dpct/device.hpp'
out = base/'event-ack-no-profile-copy-v01402-source-audit'
out.mkdir(mode=0o700)
lines = source.read_text().splitlines()
checks = {}
for name, text in {
    'same_context_device': 'm.copy_owner = std::make_unique<sycl::queue>(m.cs->get_context(), m.cs->get_device(),',
    'release_wait': 'if (impl_->copy) impl_->copy->wait_and_throw();',
    'release_stager': 'impl_->stager.reset();',
    'release_copy_owner': 'impl_->copy_owner.reset(); impl_->copy = nullptr;',
    'relayout_wait': 'DPCT_CHECK_ERROR(m.copy->wait()) != 0 ||',
    'successful_normal_run_wait': 'if (const dpct::err0 cst = DPCT_CHECK_ERROR(m.copy->wait()); cst != 0) {',
    'cache_lease_registry_wait': 'dpct::get_current_device().queues_wait_and_throw();',
}.items():
    matches = [i+1 for i, line in enumerate(lines) if text in line]
    assert len(matches) == 1, (name, matches)
    checks[name] = {'source_line': matches[0], 'source': text}
assert checks['release_wait']['source_line'] < checks['release_stager']['source_line'] < checks['release_copy_owner']['source_line']
assert 'std::vector<std::shared_ptr<sycl::queue>> current_queues(\n        _queues);' in header.read_text()
record = {
    'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'candidate_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'dpct_header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
    'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'checked_boundaries': checks,
    'normal_path_explicit_drains_present': True,
    'registered_with_device_ext': False,
    'adoption_allowed': False,
    'limitation': 'The private owned copy queue is absent from device_ext::_queues. Explicit successful run, relayout and release drains are present, but device-wide waits and sync_barrier on the default queue no longer include this queue. In particular, experimental cache-lease shrink uses the registry wait. This candidate tests the profiling/status hypothesis only; it cannot be generally adopted without preserving the global-wait contract and validating all source lifetimes/graph retirement. drain_pipeline itself only joins successor runs; it is not a copy-queue wait.',
    'scope': 'Source audit, not a runtime, full-context, reproduction or hang-prevention proof.',
}
(out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
