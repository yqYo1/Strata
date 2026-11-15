#!/usr/bin/env python3
"""Build a host-only MoE census successor without overwriting validated engines.

Main-authored; repository LICENSE applies. No GPU run or model qualification.
"""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_qsa_cached_model as base

P, B, RESULT = base.P, base.B, base.RESULT
ROOT, KEY = B/'moe-host-census-native-v1', 'moe_host_census_native_v1'
TARGETS = [*base.TARGETS, 'sycl_moe_upload_census_host_test']
require, sha = base.require, base.sha


def context():
    # Reuse the complete qualified compiler/dependency/source/link census,
    # including the explicitly declared dual qsa_native16 production/test TU.
    base.ROOT, base.KEY, base.TARGETS = ROOT, KEY, TARGETS
    return base.context()


def prepare():
    previous = json.loads((B/'qsa-cached-subgroup-model-native-v3/record.json').read_text())
    public = base.original.validation.committed(RESULT, (base.KEY,))
    require(public['binaries'] == previous['binaries'] and previous['passed']
        and previous['complete'] and previous['ownership_closed'],
        'preserve the exact eight-ELF current build before a successor')
    for binary in previous['binaries'].values():
        require(Path(binary['path']).stat().st_size == binary['bytes']
            and sha(binary['path']) == binary['sha256'], 'current fallback changed')
    ctx = context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text())
    row['preserved_binaries'].extend(previous['binaries'].values())
    decode = json.loads((B/'q6-dense-head-backing-native-v3/record.json').read_text())
    require(decode['passed'] and decode['complete'] and decode['ownership_closed'],
        'preserve independent decode')
    row['preserved_binaries'].append(decode['binaries']['strata'])
    row['duplicate_source_admission'] = base.DUPLICATE_TARGETS
    row['census_contract'] = dict(
        source='include/strata/prefill/moe_upload_census.hpp',
        environment='STRATA_SYCL_PREFILL_MOE_CENSUS=1',
        semantics='Host metadata only. Existing router drain provides planned jobs;count accepted private-bank uploads/hits once per nonempty expert. Emit format/blob/row bins after the already checked HC-write drain. No added GPU event,transfer,wait,arithmetic or execution-order change.',
        disabled_observer='Keep STRATA_PREFILL_LAYER_MAJOR_PROFILE=0 for this route. Its coupled timing-enabled V1census stalled and remains failed;no cause or device-health conclusion is inherited.',
        default=False,decode_changed=False)
    row['previous_census_failure'] = dict(key='moe_job_geometry_v1',
        evidence_commit='237f6390fc64081d8a02bc4c6ba9f00f24807eee',
        native_exit_code=-6,watchdog_seconds=480,completed_requests=0,
        original_ownership_closed=False,physical_session_empty_observed=True,
        no_fresh_driver_fault_observed=True,device_health_verified=False)
    row['plan'] = ('Commit every stage before execution. Build9targets using the unchanged '
        'qualified compiler/SDK/strict-FP recipe and full generated/compiler/link census. '
        'Run the new CPU-only literal upload/hit/byte/format and row/expert-bound controls '
        'first. Then commit a bounded traced known-safe submission control and fresh native '
        'numerical/ownership gates before the successor small/long model plan. The new '
        'census selector is independent of GPU timing;do not enable the failed timing '
        'observer or retry V1. This is diagnostic preparation,not faster inference, '
        'GPU health,model quality,full262144/RESTORE or prefill/decode adoption.')
    row['retention'] = ('Temporary replacement for current9af,never an overwrite. Preserve '
        '9af and its8fixtures until this successor closes fresh applicable native/model '
        'admission;then select the validated build and retire the replaced9af inputs '
        'after exact source/recipe recovery. The separate56cad9/552b prefill references '
        'and a347 independent decode have existing named phase consumers. This finite '
        'transition is not a larger default generation quota;review before another '
        'build and at this successor admission boundary. Resolve the old two-QSA-fixture '
        'reader dependency separately;do not remove those bytes implicitly.')
    ctx.current(row, committed=False); ctx.save(row)
    print('Prepared9target host-census successor;commit before configure. No GPU run.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare','configure','compile'),
        'prepare|configure|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: context().build(sys.argv[1])
