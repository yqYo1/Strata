#!/usr/bin/env python3
"""Qualify the existing collector on the current Q6-half compatibility fixture.

Main-authored from this fork's owned collector/measurement contracts; repository
LICENSE. No new engine/tool build. This small active-from-exec control is gated
on closure of the frozen MTP-width comparison; it cannot qualify model coverage,
physical traffic, hardware saturation, clean speed or ordinary provider teardown.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_mtp_width_model as width
import profile_iq3_decode as previous

P, B, RESULT, base = width.P, width.B, width.RESULT, width.base
SELF = Path(__file__).resolve()
ROOT = width.short.build.ROOT / 'collector-active-v1'
KEY = 'q6_native_half_collector_active_v1'
require, digest = width.require, width.digest


def selected_basis():
    """Requalify the current ELFs; never inherit a retired engine's control."""
    data = base.committed(RESULT)
    generation = width.generation_context()
    cohort = json.loads((width.SPEED_ROOT / 'record.json').read_text())
    require(data[width.SPEED_KEY] == generation.compact(cohort) and
            cohort['stage'] == 'closed' and cohort['passed'] and cohort['complete'] and
            cohort['ownership_closed'] and not cohort['active'] and
            len(cohort['processes']) == 6 and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and
                v['native_exit_code'] == 0 and v['fresh_faults'] == []
                for v in cohort['processes']),
            'close the frozen six-process width comparison before collector work')
    entry = width.context(4)
    prior = json.loads((entry.ROOT / 'record.json').read_text())
    files, _ = entry.current(prior)
    outcome = prior.get('outcome', {})
    require(data[entry.KEY] == entry.compact(prior) and prior['stage'] == 'closed' and
            prior['passed'] and prior['complete'] and prior['host_passed'] and
            prior['ownership_closed'] and not prior['active'] and prior['model_executed'] and
            outcome['native_exit_code'] == 0 and outcome['fresh_faults'] == [] and
            outcome['boot_unchanged'] and outcome['width_lifetime_checks_passed'] and
            all(cohort[k] == prior[k] for k in
                ('binary', 'native_source_commit', 'model_assets', 'sources')) and
            cohort['selected_half'] == prior['qualification_basis']['half_arm'],
            'closed source-matched fixed4 entry and generation required')
    native, _, providers = width.short.model_proof()
    require(files == providers and native['binaries']['strata'] == prior['binary'] and
            native['source_commit'] == prior['native_source_commit'],
            'current native/fixture/provider basis differs')
    fixture = native['binaries']['sycl_submission_test']
    require(digest(fixture['path']) == fixture['sha256'] and
            Path(fixture['path']).stat().st_size == fixture['bytes'],
            'actual same-build compatibility fixture changed')
    assets = previous.profile.collector_assets(data)
    return native, prior, providers, assets


def context():
    # Only this private module's globals are replaced. Imported helpers and the
    # process environment remain untouched; the old engine's basis is unused.
    spec = importlib.util.spec_from_file_location(
        'strata_q6_half_active_collector', previous.compat.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT = ctx.CONTROL_ROOT = ROOT
    ctx.KEY = ctx.CONTROL_KEY = KEY
    ctx.SOURCE = SELF
    ctx.ACTIVE_FROM_EXEC = True
    ctx.basis = selected_basis

    def current(row):
        require(base.committed(RESULT)[KEY] == ctx.compact(row) and
                base.controls(row['source_commit'], controller=SELF) == row['sources'],
                'commit the exact current collector controller and plan')
        native, prior, providers, assets = selected_basis()
        require(row['native_source_commit'] == native['source_commit'] and
                row['engine'] == native['binaries']['strata'] and
                row['binary'] == native['binaries']['sycl_submission_test'] and
                row['native_validation_probe'] == prior['native_validation_probe'] and
                row['environment_template'] == prior['environment'] and
                row['boot'] == prior['boot'] and row['driver'] == prior['driver'] and
                row['assets'] == assets and row['budgets'] == ctx.BUDGET and
                all(row['profile'][k] == v for k, v in assets.items()) and
                row['profile']['environment']['UNITRACE_TraceOutputDir'] == str(ROOT / 'trace') and
                row['whole_bytes'] == 128 << 20 and row['profile']['mode'] == 'aggregate',
                'current native/provider/collector/control envelope differs')
        return providers

    ctx.current = current
    return ctx


def prepare():
    ctx = context()
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['plan'].update(
        native='Exact ec4/cb7 engine and its six checked ELFs,including the current compatibility submission fixture. No build and no inheritance of552b/b310 collector qualification.',
        fixture='No-argument tests/sycl/compat.cpp,not --native or --submission. Source inventory:1direct kernel+12first replays+24host handoff replays+24cross-queue replays+1late replay=62. No in-process environment changes.',
        order='After all six frozen MTP-width processes close,one fresh active-from-exec compatibility-graph fixture. Keep native capture0,Sysman0 and all qualified runtime safety. Existing aggregate launcher,ChromeKernelLogging1,event buffer128;no collection toggle or process-environment mutation.',
        acceptance='Existing source/ELF/provider/pre-main-validation/owner/journal/fault gates,numerical/event/copy/graph/lifetime pass,normal exit and all62 named kernels with unique native host-flow joins. Both native gates must observe active collection;keep it active through normal exit and require flushed trace and removed named session.',
        next='Only this current-build collector admission can support a separately committed representative model-service plan. Reuse the existing guarded reader and clock/loss/overhead contracts;do not treat the old short/default-spin model capture as current long-generation attribution.',
        limits='Small collector/route qualification only;no model coverage,observer-cost,physical RAM/PCIe/VRAM traffic,resource saturation,critical path,clean speed,full context or inference adoption claim.')
    ctx.save(row)
    print('Prepared current Q6-half active collector;commit before the device check.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['check']), 'use prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            context().check()
