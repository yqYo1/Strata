#!/usr/bin/env python3
"""Six serial, fresh-process 8192-query QSA32/16 component measurements."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert len(sys.argv) == 2 and int(sys.argv[1]) in (32768, 131072, 262144)
    cells = int(sys.argv[1])
    path = A / f'qsa-prefix16-cohort-cells{cells}-v1.json'
    assert not path.exists()
    recipe = A / 'run_qsa_prefix16_bench_v1.py'
    recipe_hash = sha(recipe)
    order = [('base', 0), ('tile16', 0), ('tile16', 1), ('base', 1), ('base', 2), ('tile16', 2)]
    record = dict(schema='strata-qsa-prefix16-cohort-v1', passed=False, complete=False,
                  active_sample=None, completed=[], cells=cells, order=order,
                  recipe=dict(path=str(recipe), sha256=recipe_hash),
                  scope='8192 queries,2051 selectedcells/query, bothKVheads and12real/16paddedQheads perKVhead. Pack included; same binary/inputs and2 explicit warmups plus an initial oracle call. Three fresh processes per arm with10timings each. Different32/16-cell probability rounding has contract-specific oracle. No model PP, physicaltraffic, hardwarebusy or quality/RESTORE qualification.')

    def save():
        path.write_text(json.dumps(record, indent=2) + '\n')

    save()
    try:
        for arm, repetition in order:
            mode = f'{arm}-{cells}-{repetition}'
            receipt = A / f'qsa-prefix16-bench-{mode}-v1.json'
            assert not receipt.exists() and sha(recipe) == recipe_hash
            record['active_sample'] = dict(mode=mode, receipt=str(receipt))
            save()
            command = [sys.executable, str(recipe), mode]
            print('begin ' + mode, flush=True)
            result = subprocess.run(command)
            value = json.loads(receipt.read_text())
            assert result.returncode == 0 and value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
            assert value['fresh_faults'] == [] and value['boot_unchanged'] and not value['driver_dump_entries']
            samples = [float(v['whole_wall_ms']) for v in value['samples']]
            assert len(samples) == 10
            row = dict(arm=arm, repetition=repetition, receipt=str(receipt), receipt_sha256=sha(receipt),
                       command=command, sample_wall_ms=samples, median_wall_ms=statistics.median(samples),
                       binary=value['binary'], frame=value['frame'], cases=value['cases'])
            record['completed'].append(row)
            record['active_sample'] = None
            save()
            print(json.dumps(dict(mode=mode, median_wall_ms=row['median_wall_ms'])), flush=True)
        for arm in ('base', 'tile16'):
            rows = [x for x in record['completed'] if x['arm'] == arm]
            medians = [x['median_wall_ms'] for x in rows]
            record[arm] = dict(process_medians_ms=medians, median_of_process_medians_ms=statistics.median(medians),
                               mean=statistics.mean(medians), stdev=statistics.stdev(medians))
        assert len({r['binary']['sha256'] for r in record['completed']}) == 1
        for key in ('q', 'k', 'v', 'ks', 'vs', 'table', 'ids', 'steps', 'device_bytes', 'packed_scratch_payload_bytes'):
            assert len({x['frame'][key] for x in record['completed']}) == 1, key
        record['latency_reduction_percent'] = (1 - record['tile16']['median_of_process_medians_ms'] / record['base']['median_of_process_medians_ms']) * 100
        record.update(passed=True, complete=True)
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print(json.dumps(dict(receipt=str(path), passed=True, base=record['base'], tile16=record['tile16'],
                          latency_reduction_percent=record['latency_reduction_percent'])), flush=True)


if __name__ == '__main__':
    main()
