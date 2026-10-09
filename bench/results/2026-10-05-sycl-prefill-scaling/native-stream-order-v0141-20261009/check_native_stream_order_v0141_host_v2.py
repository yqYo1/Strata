"""Check the stream-all skip publication protocol with host vector clocks.

This models the source's plain used_of index, not a SYCL runtime or GPU.
Every case is a legal pause after the issuer acquires one old release and
reads a wrapped slot, before it publishes that entry's copy. The consumer
then skips ahead. An unordered read/write is a host ownership counterexample.
"""
from pathlib import Path
import datetime
import hashlib
import json

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree')
OLD = W / 'perf-sycl-prefill-native-copy-v0141-20261009/sycl/src/prefill/prefill.cpp'
NEW = W / 'perf-sycl-native-stream-order-v0141-20261009/sycl/src/prefill/prefill.cpp'
OUT = B / 'native-stream-order-v0141-host-ownership-proof-v2.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD) == '6399d65feea884f932a5a1079a91c80228293bfc3be371732b48413977229f84'
assert sha(NEW) == '2b37d84a973dbb11b78cd76bdba74e7632cd1bfb8695da4b772067b9dcc32109'
assert not OUT.exists()
before = """                                    if (gg_nslots > 0) flush();   // the open group's slots get their event first
                                    const int sl = (int) (k % (size_t) m.ring);
                                    dpct::sync_barrier(m.used[sl], m.cs);
"""
after = """                                    if (gg_nslots > 0) flush();   // the open group's slots get their event first
                                    // Even an unrouted entry must finish publishing its copy before
                                    // the consumer replaces this slot's event and used_of mapping.
                                    // Otherwise skip-ahead can race the issuer reading a reused slot.
                                    wait_issued(k);
                                    const int sl = (int) (k % (size_t) m.ring);
                                    dpct::sync_barrier(m.used[sl], m.cs);
"""
old, new = OLD.read_text(), NEW.read_text()
assert old.count(before) == new.count(after) == 1
assert new.replace(after, before) == old
# Bind the modeled objects/publications to the actual source.
for text in (old, new):
    assert 'int used_of[RING_MAX] = {};' in text
    assert '*m.used[m.used_of[sl]]' in text
    assert 'a_consumed.load(std::memory_order_acquire) + (size_t) m.ring' in text
    assert 'a_issued.store(idx + 1, std::memory_order_release);' in text
    assert 'a_consumed.store(upto, std::memory_order_release);' in text
    assert 'a_issued.load(std::memory_order_acquire) > k' in text
    assert 'm.used_of[sl] = sl;\n                                    consumed = ++k;\n                                    give_back(consumed);' in text


class Clocks:
    def __init__(self):
        self.clock = [[0, 0], [0, 0]]  # consumer, issuer
        self.publication = {}
        self.trace = []

    def tick(self, thread, description):
        self.clock[thread][thread] += 1
        stamp = tuple(self.clock[thread])
        self.trace.append({'thread': ('consumer', 'issuer')[thread],
                           'operation': description, 'clock': list(stamp)})
        return stamp

    def release(self, thread, name, value):
        self.publication[name] = value, self.tick(thread, f'{name}.release({value})')

    def acquire(self, thread, name):
        value, stamp = self.publication[name]
        self.clock[thread] = [max(a, b) for a, b in zip(self.clock[thread], stamp)]
        self.tick(thread, f'{name}.acquire() -> {value}')
        return value


def precedes(a, b):
    return all(x <= y for x, y in zip(a, b))


def scenario(ring, slot, require_publication):
    c = Clocks()
    k = ring + slot
    assert 0 <= slot < ring and k < 512
    # First fill [0,ring) needs no prior used-event read: stage_live is false.
    # Every wrapped copy then acquires actual consumer credit before issuing.
    c.release(0, 'consumed', 0)
    for i in range(ring):
        assert i < c.acquire(1, 'consumed') + ring
        c.tick(1, f'initial copy entry{i}; copied[{i % ring}]; stage_live = true')
        c.release(1, 'issued', i + 1)
    for j in range(slot + 1):
        if require_publication:
            assert c.acquire(0, 'issued') > j
        c.tick(0, f'used_of[{j % ring}] = {j % ring} (skipped entry{j})')
        c.release(0, 'consumed', j + 1)
    for i in range(ring, k):
        assert i < c.acquire(1, 'consumed') + ring
        c.tick(1, f'read used_of[{i % ring}]; wrapped copy entry{i}; copied[{i % ring}]')
        c.release(1, 'issued', i + 1)
    credit = c.acquire(1, 'consumed')
    assert k < credit + ring
    read = c.tick(1, f'read used_of[{slot}] for wrapped copy entry{k}')
    # The issuer may be descheduled here, before copy/barrier/publication.
    for j in range(slot + 1, k):
        if require_publication:
            assert c.acquire(0, 'issued') > j
        c.tick(0, f'used_of[{j % ring}] = {j % ring} (skipped entry{j})')
        c.release(0, 'consumed', j + 1)
    blocked = False
    if require_publication:
        blocked = c.acquire(0, 'issued') <= k
        assert blocked
        c.tick(1, f'copy entry{k}; copied[{slot}] barrier; slot-live publication')
        c.release(1, 'issued', k + 1)
        assert c.acquire(0, 'issued') > k
    write = c.tick(0, f'used_of[{slot}] = {slot} (skipped entry{k})')
    unordered = not precedes(read, write) and not precedes(write, read)
    c.release(0, 'consumed', k + 1)
    if not require_publication:
        c.tick(1, f'copy entry{k}; copied[{slot}] barrier; slot-live publication')
        c.release(1, 'issued', k + 1)
    c.acquire(1, 'consumed')
    subsequent = c.tick(1, f'read released used_of[{slot}] on next wrap')
    assert precedes(write, subsequent)
    assert unordered == (not require_publication)
    return {'ring': ring, 'slot': slot, 'paused_copy_entry': k,
            'variant': 'with-publication-acquire' if require_publication else 'original-skip-release',
            'read_clock': list(read), 'replacement_clock': list(write),
            'unordered_plain_index_access': unordered,
            'consumer_waited_for_paused_copy_publication': blocked,
            'next_wrap_acquires_replacement': True, 'all_initial_and_wrapped_copies_obey_credit': True, 'trace': c.trace}


cases = [scenario(r, slot, fixed) for r in (16, 32, 64, 128)
         for slot in (0, r // 2, r - 1) for fixed in (False, True)]
assert len(cases) == 24
assert sum(x['unordered_plain_index_access'] for x in cases) == 12
report = {'active': False, 'passed': True, 'gpu_executed': False,
          'compiled_engine': False, 'runtime_race_detector_executed': False,
          'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'checker_sha256': sha(Path(__file__)), 'old_source_sha256': sha(OLD),
          'new_source_sha256': sha(NEW), 'exact_delta_only_skip_publication_acquire': True,
          'source_bound_plain_index_counterexamples': 12,
          'replacement_source_closes_all_checked_counterexamples': True,
          'all_cases_have_legal_copy_credit_admission': all(x['all_initial_and_wrapped_copies_obey_credit'] for x in cases),
          'supersedes_rejected_model_v1': 'The earlier v1 warm start pretended entries beyond ring capacity were issued before receiving credit for nonzero target slots. This v2 explicitly admits every initial/wrapped copy and keeps v1 as rejected historical evidence.',
          'cases': cases,
          'limits': ['This deterministic C++ happens-before model checks twelve valid paused-issuer/skipped-prefix schedules. It is not exhaustive and does not execute SYCL, UR or the GPU.',
                     'The conflicting object is the project plain int used_of field; no claim about SYCL API thread-safety is required.',
                     'The native copy-only routedSTAGE8 full256K process does not enable this stream-all path and remains unchanged.',
                     'This does not prove the cause of any observed hang or numerical correctness/performance of the new source. Full engine build and bounded logged GPU checks remain pending.']}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': True, 'gpu_executed': False,
                  'cases': len(cases), 'old_counterexamples': 12, 'all_copy_credit_admission_checked': True,
                  'new_unordered_checked_accesses': 0,
                  'receipt_sha256': sha(OUT)}, indent=2))
