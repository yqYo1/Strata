"""Deterministic maintenance-window assignment workload for DFlash2."""

from dataclasses import dataclass
import hashlib
import json
from random import Random

from dflash2_reconciliation_fixture import JSON_BLOCK


VERSION = 'maintenance-window-assignment-v1'
SEED = 20261126
JOBS = 36
WINDOWS = 56


@dataclass(frozen=True)
class Option:
    allowed: bool
    risk: int
    disruption: int


def minimum_assignment(costs):
    """Rectangular Hungarian algorithm with integer reduced costs."""
    if not costs or len(costs) > len(costs[0]) or any(len(row) != len(costs[0]) for row in costs):
        raise ValueError('expected a nonempty assignment matrix with sufficient windows')
    jobs, windows = len(costs), len(costs[0])
    job_price = [0] * (jobs + 1)
    window_price = [0] * (windows + 1)
    owner = [0] * (windows + 1)
    previous = [0] * (windows + 1)
    infinity = 1 + sum(max(row) for row in costs)
    for job in range(1, jobs + 1):
        owner[0] = job
        window = 0
        distance = [infinity] * (windows + 1)
        visited = [False] * (windows + 1)
        while True:
            visited[window] = True
            current_job = owner[window]
            step, next_window = infinity, 0
            for candidate in range(1, windows + 1):
                if visited[candidate]:
                    continue
                reduced = (costs[current_job - 1][candidate - 1]
                           - job_price[current_job] - window_price[candidate])
                if reduced < distance[candidate]:
                    distance[candidate] = reduced
                    previous[candidate] = window
                if distance[candidate] < step:
                    step, next_window = distance[candidate], candidate
            for candidate in range(windows + 1):
                if visited[candidate]:
                    job_price[owner[candidate]] += step
                    window_price[candidate] -= step
                else:
                    distance[candidate] -= step
            window = next_window
            if not owner[window]:
                break
        while window:
            predecessor = previous[window]
            owner[window] = owner[predecessor]
            window = predecessor
    assigned = [0] * jobs
    for window, job in enumerate(owner[1:]):
        if job:
            assigned[job - 1] = window
    return assigned


def solve(options):
    """Minimize risk, disruption and the ordered assignment lexicographically."""
    jobs, windows = len(options), len(options[0])
    radix = windows + 1
    tie_scale = radix ** jobs
    disruption_scale = 1 + sum(max(item.disruption for item in row) for row in options)
    # Forbidden choices must cost more than any complete feasible assignment.
    risk_bound = sum(max(item.risk for item in row) for row in options)
    forbidden = (risk_bound + 1) * disruption_scale * tie_scale
    costs = []
    for job, row in enumerate(options):
        position = radix ** (jobs - job - 1)
        costs.append([(item.risk * disruption_scale + item.disruption) * tie_scale + slot * position
                      if item.allowed else forbidden for slot, item in enumerate(row)])
    assigned = minimum_assignment(costs)
    chosen = [options[job][slot] for job, slot in enumerate(assigned)]
    if not all(item.allowed for item in chosen):
        raise ValueError('the packet has no complete feasible assignment')
    return dict(risk=sum(item.risk for item in chosen), disruption=sum(item.disruption for item in chosen),
                assignments=[dict(job=f'J{job:02}', window=f'W{slot:02}',
                                  risk=item.risk, disruption=item.disruption)
                             for job, (slot, item) in enumerate(zip(assigned, chosen, strict=True))])


def build():
    random = Random(SEED)
    feasible = random.sample(range(WINDOWS), JOBS)
    options = [[Option(random.randrange(7) >= 2 or slot == feasible[job],
                       random.randrange(25, 1000), random.randrange(5, 400))
                for slot in range(WINDOWS)] for job in range(JOBS)]
    expected = solve(options)
    table = '\n'.join(
        f'change J{job:02}, window W{slot:02}, approved {int(item.allowed)}, '
        f'risk {item.risk}, disruption {item.disruption}'
        for job, row in enumerate(options) for slot, item in enumerate(row))
    text = f"""Prepare the final overnight maintenance schedule for the change desk.

Every change J00 through J{JOBS - 1:02} is mandatory and requires one whole window.
The windows W00 through W{WINDOWS - 1:02} are consecutive, and one crew can run at
most one change in each window. A row with approved 0 is forbidden because
the change has unmet staffing, dependency or approval requirements there.
The packet is complete. Every legal choice and its two scores are listed.

Find the feasible schedule with the smallest total risk. If two schedules
have equal risk, choose the one with smaller total disruption. If both
totals tie, compare their window labels at the first job where they differ,
in ascending job ID order, and choose the earlier window.

Give a concise handover explaining a few scheduling conflicts using the
packet. Finish with one fenced JSON block and no text after it. The object
must contain exactly risk and disruption integer totals and an assignments
array ordered by job ID. Every assignment contains exactly job, window,
risk, and disruption, using the exact labels and scores in the packet.

## Approval and impact packet

{table}
"""
    return dict(version=VERSION, seed=SEED, records=JOBS * WINDOWS,
                prompt_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                messages=[dict(role='user', content=text)], expected=expected)


def validate_answer(text, expected):
    block = JSON_BLOCK.search(text)
    if block is None:
        raise ValueError('expected a final JSON maintenance schedule')

    def unique_object(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError('duplicate JSON fields in maintenance schedule')
        return obj

    actual = json.loads(block[1], object_pairs_hook=unique_object)
    # JSON serialization keeps booleans, floats and exact integer scores distinct.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('the schedule or objective differs from the verified optimum')
