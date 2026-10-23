#!/usr/bin/env python3
"""Mandatory actual-order timing qualification; repository LICENSE.

Add an independent gate without changing the completed capture's frozen
producer/controller/native inputs. Native eligible/PASS is execution evidence;
only this entrypoint may qualify a subsequent GPU-local implementation test.
"""
import fcntl
import json
import sys

sys.dont_write_bytecode = True
import measure_qsa_order_replay as cohort

GATE = 'all-540-clean-timer-pairs-strictly-positive-v1'


def positive_resolution(record):
    require = cohort.guards.require
    require(record['stage'] == 'closed' and record['passed'] and record['complete'] and
            record['ownership_closed'] and not record['active'] and
            record['postchecks_complete'] and record['boot_unchanged'] and
            record['fresh_faults'] == [], 'closed successful cohort required')
    require(record['sample_columns'] ==
            ['layer', 'round', 'order', 'arm', 'elapsed_ns', 'timer_pair_ns'] and
            record['sample_calls'] == 32, 'exact timing sample schema required')
    groups = record['samples']
    require(len(groups) == 4 and groups[0] == [] and
            all(len(rows) == 180 for rows in groups[1:]), 'all 540 clean samples required')
    clocks = []
    for process, rows in enumerate(groups[1:]):
        keys = set()
        for row in rows:
            require(len(row) == 6 and all(type(x) is int for x in row), 'integer timing row required')
            layer, round_, order, arm, elapsed, clock = row
            keys.add((layer, round_, order))
            require(arm == (process + round_ + order) % 3 and clock > 0 and
                    elapsed >= 100000000 and elapsed >= 1000 * clock,
                    'positive timer median and original timing floors required')
            clocks.append(clock)
        require(keys == {(l, r, o) for l in range(12) for r in range(5) for o in range(3)},
                'clean sample conservation failed')
    return dict(gate=GATE, passed=True, samples=len(clocks),
                minimum_timer_pair_ns=min(clocks), maximum_timer_pair_ns=max(clocks))


def prepare():
    cohort.prepare()
    record = json.loads(cohort.RESULT.read_text())
    record['resolution_qualification'] = dict(gate=GATE, passed=False)
    record['gpu_local_experiment_eligible'] = False
    record['prospective_gates'] += (
        ' Mandatory entrypoint:tools/sycl/qualify_qsa_order_resolution.py. '
        'The original native/controller PASS and eligible fields establish execution only. '
        'Require strictly positive timer_pair_ns on all540clean samples before any '
        'performance advancement. Preserve the original execution outcome if this gate fails.')
    cohort.write(cohort.ROOT / 'plan.json', record)
    cohort.write(cohort.RESULT, record)


def run():
    plan = cohort.gen.committed(cohort.RESULT)
    cohort.guards.require(plan['resolution_qualification'] == dict(gate=GATE, passed=False) and
                          plan['gpu_local_experiment_eligible'] is False, 'committed resolution gate required')
    code = cohort.run()
    record = json.loads(cohort.RESULT.read_text())
    record['execution_decision'] = record.get('decision')
    try:
        record['resolution_qualification'] = positive_resolution(record)
        record['gpu_local_experiment_eligible'] = bool(record['passing_methods'])
        record['decision'] = ('GPU-local implementation experiment justified;model adoption unqualified'
                              if record['gpu_local_experiment_eligible'] else
                              'Neither order meets prospective mechanism gate;no model trial')
    except (AssertionError, KeyError, TypeError, ValueError, RuntimeError) as error:
        record['resolution_qualification'] = dict(gate=GATE, passed=False, error=repr(error))
        record['gpu_local_experiment_eligible'] = False
        record['decision'] = 'No advancement:mandatory timing-resolution qualification failed'
        code = 1
    owned = json.loads((cohort.ROOT / 'record.json').read_text())['owned_commands']
    cohort.write(cohort.ROOT / 'record.json', dict(record, owned_commands=owned))
    cohort.write(cohort.RESULT, record)
    print(json.dumps({key: record[key] for key in
                      ('resolution_qualification', 'gpu_local_experiment_eligible', 'decision')}), flush=True)
    return code


if __name__ == '__main__':
    cohort.guards.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'prepare|run')
    with (cohort.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cohort.guards.no_model_owners([])
        cohort.guards.driver_guard()
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run())
