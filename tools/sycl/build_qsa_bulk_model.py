#!/usr/bin/env python3
"""Build the full bulk-mirror candidate without replacing qualified references.

Main-authored from this repository's build/census controls; repository LICENSE.
The completed producer component supplies numerical evidence, not model speed.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_qsa16_stream as original
import qsa_bulk_mirror as component

P, B, RESULT = original.P, original.B, original.RESULT
ROOT, KEY = B / 'qsa16-bulk-mirror-model-native-v1', 'qsa16_bulk_mirror_model_native_v1'
THIS = 'tools/sycl/build_qsa_bulk_model.py'
TARGETS = ['strata', 'sycl_q6_head_owner_test', 'sycl_submission_test',
           'sycl_verifier_retirement_test', 'sycl_verify_window_policy_test']
require, sha = original.require, original.sha


def producer_component():
    ctx = component.context()
    path = component.ROOT / 'record.json'
    row = json.loads(path.read_text())
    public = json.loads(ctx.common.source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(public[component.KEY] == ctx.compact(row) and row['stage'] == 'closed'
        and all(row[k] for k in ('passed', 'complete', 'ownership_closed'))
        and not row['active'], 'closed committed off/on producer required')
    require(set(row['outcomes']) == {v[0] for v in component.CASES},
        'all four immutable component processes required')
    for label, mode, flag in component.CASES:
        outcome = row['outcomes'][label]
        require(all(outcome[k] for k in ('passed', 'complete', 'ownership_closed',
            'host_probe_accepted', 'boot_unchanged')) and outcome['fresh_faults'] == []
            and outcome['native_exit_code'] == 0 and outcome['mode'] == mode
            and outcome['flag'] == flag, 'original component outcome changed')
    full = row['outcomes']['full-on']
    require(full['physical_cells'] == 262144 and full['actual_bulk_calls'] == 33
        and full['accepted_copy_commands'] == 136
        and full['requested_D2H_bytes'] == 262145 * 1056,
        'full-capacity plus last-cell publication proof required')
    before = row['project']['source_sha256']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
        ctx.common.source.blobs(P, row['project']['revision'], before)}
    after = ctx.common.source.pin(P, ctx.common.PREFIXES)['source_sha256']
    require(recovered == before and set(after) - set(before) == {THIS}
        and not set(before) - set(after) and all(after[n] == h for n, h in before.items()),
        'only this declared build controller may extend original producer inputs')
    binary, actual = row['binary'], row['actual_inputs']
    require(sha(binary['path']) == binary['sha256']
        and Path(binary['path']).stat().st_size == binary['bytes']
        and sha(actual['path']) == actual['sha256']
        and actual['translation_units'] == 5 and actual['uncovered'] == 0,
        'original producer executable and complete census required')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for family in ('compiled', 'generated', 'links')
        for p, h in census[family].items()), 'producer compiler/link input changed')
    require(ctx.probe_qualification() == (row['native_validation_probe'],
        row['validation_qualification']), 'producer runtime qualification changed')
    return dict(key=component.KEY, path=str(path), sha256=sha(path),
        native_source_commit=row['native_source_commit'], binary=binary,
        original_source_files=len(before), all_original_inputs_recovered=True,
        only_added_controller={THIS: after[THIS]}, component_only=True,
        model_attention_RESTORE_performance_inherited=False)


def context():
    spec = importlib.util.spec_from_file_location('qsa_bulk_native_build', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.TARGETS = ROOT, KEY, TARGETS
    ctx.MODEL_BUILD = True
    ctx.component_reference = producer_component
    return ctx


def prepare():
    ctx = context()
    reference = json.loads(component.NATIVE.read_text())
    public = json.loads(ctx.common.source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    require(public['qsa16_stream_model_native_v2'] == ctx.compact(reference)
        and reference['stage'] == 'built' and reference['passed']
        and reference['complete'] and reference['ownership_closed']
        and not reference['active'], 'preserve exact original b9aa reference')
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['preserved_binaries'] += list(reference['binaries'].values())
    row['comparison_reference'] = dict(path=str(component.NATIVE),
        sha256=sha(component.NATIVE), native_source_commit=reference['native_source_commit'],
        binaries=reference['binaries'])
    row['plan'] = ('Commit each stage. Build the full engine and four production-linked '
        'ownership/submission/retirement/phase fixtures with the exact qualified compiler/SDK '
        'recipe. Require24census host controls,at least120actual TUs,complete compiler/generated/'
        'link coverage and no uncovered input. Reuse only the separately proved unchanged '
        'producer component bytes and off/on numerical evidence;no model result inherited. '
        'Before GPU/model execution commit native checks and a separate immutable same-binary '
        'off/on32K/64K comparison using bench.sh,thinking on,first-use and subsequent requests '
        'separated. Preserve current decode and checked handoff;no default adoption from build '
        'or short output. Full262144model/RESTORE and sustained TG remain open.')
    row['retention'] = ('One bounded temporary fourth engine generation for the imminent bulk '
        'mirror comparison:original b9aa is the measured clean32K/length-curve reference;'
        '552b is the faster117Kprefill fallback;a347 is the sustained decode reference. '
        'None is overwritten. This is not a new default retention quota. At the first '
        'closed native and matched32K/64Koff/on plus decode-preservation boundary,select '
        'candidate or original b9aa and retire the superseded one after exact source/recipe '
        'recovery. Retire the temporary producer fixture/raw output when its proof consumer '
        'migrates to the candidate same-build checks;review remaining phase references then. '
        'Existing configured-deployment provenance holds remain separate.')
    ctx.current(row, committed=False)
    ctx.save(row)
    print('Prepared full candidate and bounded comparison-generation plan;commit first.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'configure', 'compile'),
        'build_qsa_bulk_model.py prepare|configure|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            context().build(sys.argv[1])
