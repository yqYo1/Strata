"""Reconcile independently planned work, API results and device observations.

The route owner must construct the plan before submission. API adapters expand
their own captured graph templates; a graph parent is not a device operation.
Collector adapters supply instance IDs and observed host-source links, including
handle generations. Never manufacture either ledger from the other one.

This is the host reconciliation gate, not a PTI/UR adapter or a qualification of
their coverage, clock conversion or overhead. Those require a native fixture on
the exact source/runtime before a model observation can use this gate.
Repository LICENSE applies; independently authored, no external code copied.
"""
from dataclasses import dataclass


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def identity(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class Work:
    kind: str
    name: str = ''
    shape: tuple = ()
    bytes: int = 0  # Requested extent, never a physical bus-traffic measurement.

    def validate(self):
        require(self.kind in ('kernel', 'copy', 'fill', 'barrier'), 'unknown device work')
        require(isinstance(self.name, str) and type(self.shape) is tuple and
                all(integer(v, 1) for v in self.shape) and integer(self.bytes), 'invalid work geometry')
        if self.kind == 'kernel':
            require(identity(self.name) and bool(self.shape) and not self.bytes, 'invalid kernel geometry')
        elif self.kind in ('copy', 'fill'):
            require(identity(self.name) and self.bytes > 0 and not self.shape, 'invalid transfer geometry')
        else:
            require(not self.shape and not self.bytes, 'invalid barrier geometry')


@dataclass(frozen=True)
class Operation:
    id: str  # Stable route-owner identity, independent of collector allocation.
    queue: str  # Queue identity AND lifetime generation, supplied by the adapter.
    phase: str
    work: tuple = ()  # Graph nodes expanded by the independent route plan.
    host_task: bool = False
    result: str = 'accepted'


@dataclass(frozen=True)
class Submission:
    id: str
    queue: str
    phase: str
    result: str
    work: tuple = ()  # Actual API submissions, expanded by the API adapter.
    event_ids: tuple = ()
    host_completed: bool = False


@dataclass(frozen=True)
class DeviceEvent:
    id: str  # Execution INSTANCE, not a reused event/command-list handle.
    origin: str  # Actual observed host-source link, not copied from the plan.
    queue: str
    work: Work
    begin_ns: int
    end_ns: int


class CommandBuffers:
    """API-side lifetime tracking. The route-plan producer must not use this.

    Call only after the corresponding API succeeded. Failed append/finalize/
    replay calls must be retained by the adapter as failures, not expanded work.
    Explicit generations prevent a recycled native handle aliasing an old graph.
    """
    def __init__(self):
        self.live = {}
        self.generations = {}

    def create(self, handle):
        require(identity(handle) and handle not in self.live, 'live command-buffer handle reused')
        generation = self.generations.get(handle, 0) + 1
        self.generations[handle] = generation
        key = (handle, generation)
        self.live[handle] = (key, [], False)
        return key

    def _get(self, key):
        require(type(key) is tuple and len(key) == 2 and identity(key[0]) and
                integer(key[1], 1), 'invalid command-buffer identity')
        current = self.live.get(key[0])
        require(current is not None and current[0] == key, 'stale command-buffer generation')
        return current

    def append(self, key, work):
        _, nodes, finalized = self._get(key)
        require(not finalized, 'append after finalize')
        require(isinstance(work, Work), 'invalid captured work')
        work.validate()
        nodes.append(work)

    def finalize(self, key):
        _, nodes, finalized = self._get(key)
        require(nodes and not finalized, 'empty or repeated finalize')
        self.live[key[0]] = (key, nodes, True)

    def replay(self, key):
        _, nodes, finalized = self._get(key)
        require(finalized, 'unfinalized replay')
        return tuple(nodes)

    def destroy(self, key):
        self._get(key)
        del self.live[key[0]]


def indexed(rows, expected_type, label):
    result = {}
    for row in rows:
        require(isinstance(row, expected_type) and identity(row.id), 'invalid ' + label)
        require(row.id not in result, 'duplicate ' + label + ' identity')
        result[row.id] = row
    return result


def reconcile(plan, submissions, events, phase_bounds, *, drained, async_errors):
    """Require exact work/identity conservation; raise on unknown or missing work.

    phase_bounds are independently observed, drained, half-open native windows
    in the qualified common clock domain. Device intervals may overlap; their
    sum is not a critical path. Host tasks need explicit completion receipts and
    contribute no synthetic device event. Rejected calls produce no device work.
    """
    require(drained is True and type(async_errors) is tuple and not async_errors,
            'uncertain drain or asynchronous error')
    require(isinstance(phase_bounds, dict) and bool(phase_bounds), 'missing phase bounds')
    for phase, bounds in phase_bounds.items():
        require(identity(phase) and type(bounds) is tuple and len(bounds) == 2 and
                all(integer(v) for v in bounds) and bounds[1] > bounds[0], 'invalid phase clock')
    ordered = sorted(phase_bounds.values())
    require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), 'overlapping phase windows')
    expected = indexed(plan, Operation, 'planned operation')
    accepted = indexed(submissions, Submission, 'API submission')
    observed = indexed(events, DeviceEvent, 'device event')
    require(bool(expected) and expected.keys() == accepted.keys(), 'planned/API operation mismatch')
    linked = set()
    counts = dict(kernel=0, copy=0, fill=0, barrier=0)
    requested_bytes = dict(copy=0, fill=0)
    hosts = rejected = 0
    for ident, op in expected.items():
        api = accepted[ident]
        require(identity(op.queue) and op.phase in phase_bounds and
                (op.queue, op.phase, op.result) == (api.queue, api.phase, api.result),
                'API route/result mismatch')
        require(op.result in ('accepted', 'rejected') and type(op.host_task) is bool and
                type(api.host_completed) is bool and type(op.work) is tuple and
                type(api.work) is tuple and type(api.event_ids) is tuple, 'invalid operation schema')
        require(all(isinstance(w, Work) for w in op.work + api.work), 'invalid operation work')
        for work in op.work + api.work:
            work.validate()
        require(not op.host_task or not op.work, 'host task cannot own device work')
        if op.result == 'rejected':
            require(not api.work and not api.event_ids and not api.host_completed,
                    'rejected submit produced work')
            rejected += 1
            continue
        require(op.work or op.host_task, 'unclassified accepted operation')
        require(op.work == api.work and len(api.event_ids) == len(api.work), 'API work expansion mismatch')
        require(api.host_completed == op.host_task, 'host task completion mismatch')
        hosts += int(op.host_task)
        begin, end = phase_bounds[op.phase]
        for work, event_id in zip(api.work, api.event_ids, strict=True):
            require(identity(event_id) and event_id not in linked, 'duplicate device link')
            require(event_id in observed, 'missing device event')
            event = observed[event_id]
            require((event.origin, event.queue, event.work) == (ident, op.queue, work),
                    'device origin/queue/work mismatch')
            require(integer(event.begin_ns) and integer(event.end_ns) and
                    begin <= event.begin_ns < end and event.begin_ns <= event.end_ns <= end and
                    (work.kind == 'barrier' or event.end_ns > event.begin_ns), 'invalid device clock')
            linked.add(event_id)
            counts[work.kind] += 1
            if work.kind in requested_bytes:
                requested_bytes[work.kind] += work.bytes
    require(linked == observed.keys(), 'unmapped device event')
    return dict(planned_operations=len(expected), accepted_operations=len(expected) - rejected,
                rejected_operations=rejected, host_tasks=hosts, device_counts=counts,
                requested_bytes=requested_bytes)
