"""Deterministic event reconciliation workload for DFlash2 generation."""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re


VERSION = 'rollout-reconciliation-v1'
SEED = 20261126
BUCKETS = 24
JSON_BLOCK = re.compile(r'(?ms)^\x60\x60\x60json\s*\n(.*?)\n\x60\x60\x60\s*\Z')


@dataclass(frozen=True)
class Reading:
    event: str
    service: str
    bucket: int
    requests: int
    errors: int
    ready: int
    delivered: int

    def line(self):
        return ','.join(map(str, asdict(self).values()))


def stable_number(*parts):
    key = ':'.join(map(str, (VERSION, SEED, *parts)))
    return int.from_bytes(hashlib.sha256(key.encode('ascii')).digest()[:8], 'big')


def observed_violation(delivered, rollout):
    """Resolve duplicate and delayed readings before evaluating the invariant."""
    readings = {}
    for reading in delivered:
        key = (reading.service, reading.bucket)
        first = readings.setdefault(key, reading)
        if replace(first, delivered=reading.delivered) != reading:
            raise ValueError('fixture has conflicting readings for one service/bucket')

    violations = []
    for service, entry in rollout.items():
        start, target, canary, _, _ = entry
        if canary < 25:
            continue
        pending = None
        for bucket in range(start, BUCKETS):
            reading = readings[service, bucket]
            if reading.errors * 100 < reading.requests * 2 or reading.ready >= target:
                pending = None
            elif pending is None:
                pending = reading
            else:
                violations.append((pending, reading))
                break
    if len(violations) != 1:
        raise ValueError('fixture must have one rollout violation')
    return violations[0]


def build(service_count):
    if not 32 <= service_count <= 96:
        raise ValueError('service_count is outside 32..96')
    services = [f'svc-{n:02d}' for n in range(service_count)]
    culprit_index = stable_number('incident') % service_count
    culprit = services[culprit_index]
    shadow = services[(culprit_index + 1) % service_count]
    incident_bucket = 12 + stable_number('bucket') % 6
    rollout, delivery = {}, []
    for service_index, service in enumerate(services):
        facts = stable_number(service, 'release')
        start, target = 4 + facts % 5, 5 + (facts >> 8) % 6
        canary = 10 if service == shadow else 25 + (facts >> 16) % 3 * 25
        change, stable = f'ch-{service_index:02d}', f'r{100 + (facts >> 24) % 200}'
        rollout[service] = (start, target, canary, change, stable)
        amber = 10 + (facts >> 32) % 8
        for bucket in range(BUCKETS):
            sample = stable_number(service, bucket, 'reading')
            requests = 900 + sample % 1201
            errors, ready = (sample >> 14) % 12, target + (sample >> 20) % 2
            delayed = sample % 97 == 0
            if service in (culprit, shadow):
                if incident_bucket <= bucket <= incident_bucket + 1:
                    errors, ready = requests // 24, target - 1
                if service == culprit and bucket == incident_bucket:
                    delayed = True
            elif bucket == amber:
                errors = (requests + 49) // 50
                ready = target - 1
            reading = Reading(f'm{service_index:02d}-{bucket:02d}', service, bucket, requests,
                              errors, ready, bucket + 3 if delayed else bucket)
            delivery.append(reading)
            if sample % 211 == 0 or (service == culprit and bucket == incident_bucket):
                delivery.append(replace(reading, delivered=reading.delivered + 1))

    delivery.sort(key=lambda row: (row.delivered, stable_number('order', row.event, row.delivered), row.event))
    first, second = observed_violation(delivery, rollout)
    if (first.service, first.bucket, second.bucket) != (culprit, incident_bucket, incident_bucket + 1):
        raise ValueError('derived violation differs from the planted scenario')
    _, _, _, change, stable = rollout[culprit]
    answer = dict(service=culprit, change=change, first_bad_bucket=first.bucket, violation_bucket=second.bucket,
                  rollback_revision=stable, evidence_ids=[first.event, second.event])

    releases = '\n'.join(','.join(map(str, (service, *rollout[service]))) for service in services)
    readings = '\n'.join(reading.line() for reading in delivery)
    text = f"""Produce the on-call handoff for an application rollout incident. All records below are synthetic.

## Rules

Rollouts stay open until the end of this interval at their recorded canary percentage.
A rollout violates its guard at the second consecutive observed bucket after or including its start
with errors at least 2% of requests and ready replicas below target, when canary percentage is at least
25. Evaluate the thresholds together in each bucket. Deduplicate monitor readings by event ID.
Order readings by observed bucket. Roll back only the violating rollout to its listed stable revision.
Ignore isolated threshold crossings and canaries below 25%.

Write a substantive handoff with these sections: incident decision, timeline and arithmetic for both
thresholds, one isolated alert and one sub-threshold canary ruled out by evidence, delayed and
duplicate telemetry effects, and a staged rollback and post-rollback verification plan.
Use exact values from the records when making factual claims. Finish with a single fenced json
block containing exactly the keys service, change, first_bad_bucket, violation_bucket,
rollback_revision, and evidence_ids. evidence_ids is the two relevant monitor IDs in observed order.

## Rollout ledger

service,start_bucket,target_replicas,canary_percent,change,stable_revision
{releases}

## Monitor delivery log

event,service,observed_bucket,requests,errors,ready_replicas,delivered_bucket
{readings}
"""
    return dict(version=VERSION, seed=SEED, service_count=service_count, records=len(delivery),
                prompt_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                messages=[dict(role='user', content=text)], expected=answer)


def validate_answer(text, expected):
    """Keep the structured decision separate from engine/token eligibility."""
    record = JSON_BLOCK.search(text)
    if record is None or len(text[:record.start()].strip()) < 512:
        raise ValueError('expected a visible handoff before one final JSON block')

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError('duplicate JSON fields in the handoff')
        return result

    actual = json.loads(record[1], object_pairs_hook=unique_object)
    if actual != expected:
        raise ValueError('the reported decision or evidence differs from the verified ledger')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--services', type=int, default=80)
    parser.add_argument('--check-answer', type=Path)
    args = parser.parse_args()
    fixture = build(args.services)
    if args.check_answer:
        validate_answer(args.check_answer.read_text(encoding='utf-8'), fixture['expected'])
        fixture.pop('messages')
    print(json.dumps(fixture, ensure_ascii=False, indent=2))
