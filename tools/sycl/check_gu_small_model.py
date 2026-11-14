#!/usr/bin/env python3
"""Fresh native/phase qualification for the opt-in compact-GU model relink.

Main-authored from this fork's owned qualification controller; repository
LICENSE applies. Component results permit this trial,not model adoption.
"""
import fcntl
import json
import sys
sys.dont_write_bytecode = True
import build_gu_small_model as build
import check_moe_census_model as previous

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B/'gu-small-model-device-v1', 'gu_small_model_device_v1'
THIS = 'tools/sycl/check_gu_small_model.py'
STAGES = previous.STAGES
require = build.require


def context():
    driver = previous.context()
    driver.ROOT, driver.KEY, driver.THIS = ROOT, KEY, THIS
    driver.build, driver.native = build, build.native
    return driver


def prepare():
    driver = context(); ctx = driver.context(); ctx.prepare()
    row = json.loads((ROOT/'record.json').read_text())
    row['environment'].update(STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
        STRATA_SYCL_PREFILL_MOE_CENSUS='0', STRATA_SYCL_PREFILL_NATIVE16='bf16',
        STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm', STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',
        STRATA_SYCL_PREFILL_GU_SWIGLU='auto')
    row['plan'].update(order=list(STAGES),
        host='Fresh literal MoE census and54reader/census controls,11owner/switch refusals,phase policy and selected decode module equality against a347. The eight reused fixtures retain their original exact link inputs;none links the changed prefill archive.',
        submission='Fresh bounded traced submission control before head/attention/producer work. Immutable pre-exec safety settings and parameter validation. Stop on first fault,error,budget or ownership failure.',
        attention='Fresh small/full QSA numerical/guard gates,including262144live cells.',
        producer='Fresh off/on producer small/full gates,including final-cell overwrite and accepted bulk receipts.',
        next='Only after all native gates pass,prepare bounded actual-model long/short/long handoff,then separate repeated32K/64Kprefill comparisons and sustained independent decode. No component or native success alone qualifies model throughput or physical full-model context.')
    row['limits'] = ('Native prerequisite only. Eight fixtures are reused from the original build because '
        'they do not link the changed archive;the new model engine is freshly relinked after a byte-identical '
        'baseline control. Actual production compact-GU math has separate fresh component evidence. '
        'Full-model262144/RESTORE,task/agent quality,PP/TG/TTFT,phase preservation and adoption remain open.')
    row['retention'] = 'Keep compact gate outcomes for imminent model comparison;retire successful traces after closure. No extra fixture generation;97fda fallback and56cad9/a347 named comparisons remain separate.'
    ctx.context(row['native_observer_admission']).save(row)
    print('Prepared fresh compact-GU native prerequisites;commit before host/device.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', *STAGES), 'prepare|'+'|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.original.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] in build.original.STAGES: context().context().execute(sys.argv[1])
        else: context().extra(sys.argv[1])
