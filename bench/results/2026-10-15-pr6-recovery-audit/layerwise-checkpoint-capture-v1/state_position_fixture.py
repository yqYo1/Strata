#!/usr/bin/env python3
"""Standalone CPU contract oracle. PROPOSAL; NOT RUN by author.

No Strata imports, environment changes, device calls, files or process control.
Toy state is an independent prefix function, not Strata numerical validation.
Main may run `python3 state_position_fixture.py`; a failing invariant raises.
"""
from dataclasses import dataclass, field
from itertools import product
I64 = (1 << 63) - 1
LAYERS = 48


def ids_at(p):
    return (17 * p + 13) % 100003


def state_at(layer, prefix):
    # Closed-form cumulative recurrent state, intentionally unrelated to the
    # proposal's storage/masks/offsets. Different absolute prefix => different
    # state even when a QSA block position happens to coincide.
    if layer % 4 == 3:
        return ('QSA', prefix, prefix % 4,
                tuple(ids_at(p) for p in range(max(0, prefix - 3), prefix)),
                prefix * (prefix + 1) // 2)
    return ('GDN', prefix, (layer + 1) * prefix * (prefix + 1) // 2)


def ple_at(prefix):
    return tuple(ids_at(p) for p in range(max(0, prefix - 3), prefix))


def prev_at(prefix):
    return tuple(ids_at(p) if p >= 0 else -1 for p in (prefix - 2, prefix - 1))


def reference_events(begin, pieces, interval, initial_threshold=None):
    """Token-time interpreter, preserving windows and non-global-grid semantics.
    pieces: (end, actual chunk, 'batch'/'windows', root_or_turn_event).
    The oracle counts elapsed tokens and inspects actual chunk ends; it does
    not use the proposal's planner or a ceil/division boundary formula.
    """
    assert 0 <= begin <= I64 and 0 <= interval <= I64
    elapsed = 0 if initial_threshold is None else interval - (initial_threshold - begin)
    events, pos = [], begin
    for end, chunk, mode, boundary in pieces:
        assert pos < end <= I64 and chunk > 0
        chunk_tokens = 0
        while pos < end:
            pos += 1
            elapsed += 1
            chunk_tokens += 1
            if chunk_tokens == chunk or pos == end:
                if mode == 'batch' and interval > 0 and elapsed >= interval:
                    events.append(pos)
                    elapsed = 0
                chunk_tokens = 0
        if boundary:
            events.append(pos)  # does not reset the periodic threshold
    return events


def retain_by_spec(events, cap, old=(), initial_clock=0):
    """Reference policy on tiny records: pin first chain node at cap>=2;
    rotate other nodes by last use; duplicate lengths are uses, not new nodes.
    """
    chain = [dict(pos=p, use=u) for p, u in old]
    clock = initial_clock
    for pos in events:
        clock += 1
        if clock > (1 << 64) - 1:
            raise OverflowError('stamp exhausted')
        match = next((node for node in chain if node['pos'] == pos), None)
        if match is None:
            chain.append(dict(pos=pos, use=clock))
        else:
            match['use'] = clock
        if len(chain) > cap:
            eligible = chain[1:] if cap >= 2 else chain[:1]
            victim = min(eligible, key=lambda node: node['use'])
            chain.remove(victim)
    return [(node['pos'], node['use']) for node in chain], clock


@dataclass
class Auditor:
    """Checks position provenance and transaction ordering, not storage code."""
    positions: tuple
    published: tuple = ()
    run: int = 77
    pieces: dict = field(default_factory=dict)
    ple: dict = field(default_factory=dict)
    mtp: set = field(default_factory=set)
    lifecycle: list = field(default_factory=list)
    failed: bool = False

    def piece(self, run, layer, prefix, value):
        assert run == self.run and 0 <= layer < LAYERS
        assert prefix in self.positions and value == state_at(layer, prefix)
        key = (prefix, layer)
        assert key not in self.pieces
        self.pieces[key] = value

    def history(self, run, layer, prefix, value):
        assert run == self.run and layer == 1 and prefix in self.positions
        assert value == ple_at(prefix) and prefix not in self.ple
        self.ple[prefix] = value

    def prompt(self, run, prefix):
        assert run == self.run and prefix in self.positions
        assert all((prefix, layer) in self.pieces for layer in range(LAYERS))
        assert prefix not in self.mtp
        self.mtp.add(prefix)

    def end(self, kind, success=True):
        assert kind in ('run', 'expert_restore', 'refill', 'cancel', 'copy_error', 'mtp_error')
        if not success or kind in ('cancel', 'copy_error', 'mtp_error'):
            self.failed = True
        self.lifecycle.append(kind)

    def publish(self):
        before = self.published
        if self.failed:
            assert self.published == before
            return False
        assert self.lifecycle == ['run', 'expert_restore', 'refill']
        for prefix in self.positions:
            assert prefix in self.ple and prefix in self.mtp
            assert all((prefix, layer) in self.pieces for layer in range(LAYERS))
            # Restore derives previous IDs from EXACT prefix, not final session.
            assert prev_at(prefix) == tuple(ids_at(p) if p >= 0 else -1 for p in (prefix-2, prefix-1))
        self.published = self.positions
        return True


def expect_rejected(fn):
    try:
        fn()
    except (AssertionError, OverflowError):
        return
    raise AssertionError('invalid event history was accepted')


def assemble(positions, end, poison=False):
    """Synthetic producer visits a full segment layer-major; independent state
    oracle knows the expected prefix without observing that producer's state.
    Poison reproduces final-callback whole-session save's mixed positions.
    """
    audit = Auditor(tuple(positions), published=(2048,))
    for layer in range(LAYERS):
        for prefix in positions:
            actual = end if poison and layer != LAYERS - 1 else prefix
            audit.piece(audit.run, layer, prefix, state_at(layer, actual))
            if layer == 1:
                audit.history(audit.run, layer, prefix, ple_at(actual))
            if layer == LAYERS - 1:
                audit.prompt(audit.run, prefix)
    return audit


def run():
    # Explicit sizes, tails, shifted starts, intervals not aligned to chunks.
    for length, chunk, interval, cap in product((32768, 65536, 262144), (4096, 8192),
                                              (1, 16384, 16385), (1, 2, 6)):
        events = reference_events(0, [(length, chunk, 'batch', False)], interval)
        retained, clock = retain_by_spec(events, cap)
        assert len(retained) <= cap and clock == len(events)
        positions = [p for p, _ in retained]
        assert len(set(positions)) == len(positions)
        if cap >= 2:
            assert positions[0] == events[0]
        audit = assemble(positions, length)
        for event in ('run', 'expert_restore', 'refill'):
            audit.end(event)
        assert audit.publish()
        assert audit.published == tuple(positions)
    for length, expected in ((32768, [16384, 32768]),
                             (65536, [16384, 32768, 49152, 65536]),
                             (262144, [16384, 196608, 212992, 229376, 245760, 262144])):
        events = reference_events(0, [(length, 8192, 'batch', False)], 16384)
        assert [p for p, _ in retain_by_spec(events, 6)[0]] == expected
    assert reference_events(0, [(34602, 8192, 'batch', False)], 16384) == [16384,32768]
    assert reference_events(0, [(65536, 8192, 'batch', False)], 16385) == [24576,49152]
    assert reference_events(4000, [(10000, 4096, 'batch', True),
                                  (26000, 8192, 'batch', True)], 16384) == [10000,26000,26000]
    assert reference_events(0, [(17000, 1, 'windows', False),
                               (25192, 8192, 'batch', False)], 16384) == [25192]
    assert reference_events(0, [(32768, 8192, 'batch', True)], 0) == [32768]
    assert reference_events(I64-9, [(I64, 4, 'batch', False)], I64) == []
    assert reference_events(I64-9, [(I64, 4, 'batch', False)], 1) == [I64-5,I64-1,I64]
    expect_rejected(lambda: reference_events(I64-1, [(I64+1,1,'batch',False)],1))
    expect_rejected(lambda: reference_events(-1, [(4,1,'batch',False)],1))
    expect_rejected(lambda: reference_events(0, [(4,1,'batch',False)],I64+1))
    # Boundary periodic+turn duplicate: exactly two stamps at same length.
    chain, clock = retain_by_spec([16384,32768,32768],6)
    assert chain == [(16384,1),(32768,3)] and clock == 3
    # Existing root is preserved; de-dup touches change the next LRU victim.
    chain, clock = retain_by_spec([100,400],3,((100,1),(200,2),(300,3)),3)
    assert chain == [(100,4),(300,3),(400,5)] and clock == 5
    expect_rejected(lambda: retain_by_spec([1],1,initial_clock=(1<<64)-1))
    expect_rejected(lambda: assemble((16384,32768),65536,poison=True))
    # Cancel with genuinely incomplete state, then admit a separate next run.
    for stopped_layer in (0, 1, 23, 46, 47):
        a = Auditor((16384,32768), published=(2048,))
        for layer in range(stopped_layer):
            for prefix in a.positions:
                a.piece(a.run,layer,prefix,state_at(layer,prefix))
                if layer == 1: a.history(a.run,layer,prefix,ple_at(prefix))
        a.end('cancel')
        assert not a.publish() and a.published == (2048,)
        b = assemble((16384,),32768)
        for event in ('run','expert_restore','refill'): b.end(event)
        assert b.publish()
    for failure in ('cancel', 'copy_error', 'mtp_error'):
        a = assemble((16384,),32768)
        old = a.published
        a.end(failure)
        assert not a.publish() and a.published == old
    for failure_at in ('run','expert_restore','refill'):
        a = assemble((16384,),32768)
        old = a.published
        for stage in ('run','expert_restore','refill'):
            a.end(stage, success=stage != failure_at)
        assert not a.publish() and a.published == old
    # Missing layer, PLE, MTP, final lifecycle or wrong run cannot publish.
    for missing in ('layer','ple','mtp','refill'):
        a = assemble((16384,),32768)
        if missing == 'layer': del a.pieces[(16384,0)]
        if missing == 'ple': a.ple.clear()
        if missing == 'mtp': a.mtp.clear()
        for stage in ('run','expert_restore','refill'):
            if stage != missing: a.end(stage)
        expect_rejected(a.publish)
        assert a.published == (2048,)
    a = Auditor((16384,))
    expect_rejected(lambda: a.piece(76,0,16384,state_at(0,16384)))
    expect_rejected(lambda: a.history(77,47,16384,ple_at(16384)))
    # Admission arithmetic is logical, no claim of physical capacity measured.
    gdn = 117669888
    old = 6*gdn
    new = 6*gdn
    reserve = 8*(1<<30)
    free_after_old = 10*(1<<30)
    assert old + new + reserve <= old + free_after_old
    assert not (new + reserve <= reserve + new - 1)
    # Modelled output state has correct previous IDs at prefixes 0/1/2.
    assert prev_at(0) == (-1,-1) and prev_at(1) == (-1,ids_at(0))
    print('PASS: independent schedule/retention/position/transaction contract oracle; no engine numerical qualification')

if __name__ == '__main__':
    run()
