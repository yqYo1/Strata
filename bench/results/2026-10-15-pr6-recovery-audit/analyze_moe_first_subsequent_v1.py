#!/usr/bin/env python3
"""Independently qualify the closed clean 4-request mode8/10 comparison."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True

A = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = A / 'moe-window10-first-subsequent-analysis-v1.json'
    report = A / 'moe-window10-first-subsequent-results-v1.ja.md'
    assert not out.exists() and not report.exists()
    path = A / 'direct_owner_transient_artifacts_v1.py'
    assert digest(path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec = importlib.util.spec_from_file_location('moe_warm_main_owner', path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    arms = {}
    for mode in (8, 10):
        path = A / f'moe-accum-model-profile-window{mode}-tokens65536-warm4-v5.json'
        value = json.loads(path.read_text())
        assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
        assert value['fresh_faults'] == [] and not value['driver_dump_exists'] and value['boot_unchanged']
        for command in value['commands']:
            assert owner.closed(command) and command['exit_code'] == 0
            for old in command['owners'] + [command['direct_identity']]:
                now = owner.identity(old['pid'])
                assert now is None or now['start_ticks'] != old['start_ticks']
        env = value['environment']
        assert 'STRATA_TRACE' not in env and 'UR_ENABLE_LAYERS' not in env
        for name in ('STRATA_PREFILL_LAYER_MAJOR_PROFILE', 'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
                     'STRATA_PREFILL_CPU_WAIT_PROFILE', 'STRATA_SYCL_PREFILL_COMMAND_PROFILE'):
            assert env[name] == '0'
        requests = value['benchmark_result']['results']['prefill_only']['requests']
        assert len(requests) == 4
        for request in requests:
            assert request['ok'] and request['eligible'] and request['stream_complete']
            assert request['thinking_requested'] and request['reasoning_observed'] and request['generated_tokens'] == 24
            assert request['actual_reused_tokens'] == 0 and request['actual_read_tokens'] == request['actual_prompt_tokens']
        assert len(set(q['request_sha256'] for q in requests)) == 4
        proof = value['single_engine_proof']
        assert proof['four_successful_fresh_requests'] and proof['no_successful_engine_restart_possible']
        assert digest(proof['exclusive_pre_Popen_claim']['path']) == proof['exclusive_pre_Popen_claim']['sha256']
        samples = [q['prompt_tps'] for q in requests]
        later = samples[1:]
        arms[mode] = dict(receipt=str(path), receipt_sha256=digest(path), binary=value['binary'],
                         requests=requests, samples=samples, first=samples[0], later=later,
                         later_summary=dict(n=3, median=statistics.median(later), mean=statistics.mean(later),
                                            stdev=statistics.stdev(later), minimum=min(later), maximum=max(later)),
                         placements=value['placement_lines'], windows=value['window_lines'],
                         admissions=value['capacity_admission_lines'], single_engine_proof=proof,
                         environment=env)
    base, candidate = arms[8], arms[10]
    assert base['binary'] == candidate['binary']
    for a, b in zip(base['requests'], candidate['requests']):
        assert (a['request_sha256'], a['actual_prompt_tokens']) == (b['request_sha256'], b['actual_prompt_tokens'])
    # Intentional per-process paths and the mode are the only environment differences.
    excluded = {'XDG_CACHE_HOME', 'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT',
                'STRATA_SYCL_PREFILL_MOE_WINDOW'}
    assert {k: v for k, v in base['environment'].items() if k not in excluded} == {
        k: v for k, v in candidate['environment'].items() if k not in excluded}
    first_gain = (candidate['first'] / base['first'] - 1) * 100
    later_gain = (candidate['later_summary']['median'] / base['later_summary']['median'] - 1) * 100
    record = dict(schema='strata-moe-window10-first-subsequent-main-analysis-v1',
                  recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True,
                  complete=True, arms=arms, first_long_gain_percent=first_gain,
                  subsequent_median_gain_percent=later_gain,
                  hardware='Arc B570 10GiB; Ryzen5600X; 128GiB RAM; xe1.17.39395+14',
                  scope='Each arm is one process with one first long request and three subsequent distinct zero-reuse long requests. Same binary/input per pair; order8 then10. Fresh per-process cache root, persistentSYCLcache0, safeLZv2 environment unchanged.',
                  causal_JIT_attribution_qualified=False, hardware_bottleneck_attribution_qualified=False,
                  general_quality_qualified=False, physical262144_RESTORE_qualified=False,
                  sustained_decode_qualified=False, production_default_selected=False,
                  limitations=['First requests are one sample per process, not a repeated cold-process cohort.',
                               'Subsequent samples share one process and differ by run-tag inputs; they are not independent fresh-process replicates.',
                               'Free VRAM and engine/driver/cache state vary across requests. Speed alone does not attribute JIT or hardware utilization.',
                               '24 thinking-token completions are probes, not long-generation/task quality or sustained TG qualification.',
                               'Read-only research/review may run concurrently; no other GPU, build, benchmark or cleanup runs.'])
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    lines = ['# window10: ログなし初回・後続長文比較', '',
             'Arc B570 10GiB / Ryzen5600X / 128GiB RAM。同一binary、公式bench.sh、thinking有効、',
             '入力78453 token、context131072、INT8 KV、chunk8192、全GPU BF16 residual。',
             '各armは同一engineで4つの異なる要求を実行し、全件reuse=0。TRACE/UR layer/profileは無効。', '',
             '|MoE mode|初回PP(tok/s)|後続3回PP(tok/s)|後続中央値|後続標本SD|',
             '|---|---:|---|---:|---:|']
    for mode, value in arms.items():
        summary = value['later_summary']
        lines.append(f"|{mode}|{value['first']:.1f}|{' / '.join(f'{x:.1f}' for x in value['later'])}|{summary['median']:.1f}|{summary['stdev']:.3f}|")
    lines += ['', f'window10の改善率は初回{first_gain:.3f}%、後続中央値{later_gain:.3f}%。',
              '同一入力hash、各要求の完了、単一engineのPID/start ticksと起動前exclusive claim、',
              '正常終了、所有process消滅、新規GPU fault/dumpなしをMainが再確認した。', '',
              '初回はarmごと1標本、後続3回は一つのprocess内の標本であり、新規processの反復とは区別する。',
              'free VRAMやcache/graphの状態が変化するため、初回と後続の差をJIT時間とは断定しない。',
              '24 thinking-tokenは完了probeで、一般品質、長時間decode、物理262144消費とRESTOREは別途未検証。',
              '速度目標1000 PP / 70 TGは未達であり、最適化を継続する。', '',
              '[全個別標本・検証結果](moe-window10-first-subsequent-analysis-v1.json)']
    report.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(passed=True, first_gain_percent=first_gain, later_median_gain_percent=later_gain,
                          report=str(report))), flush=True)


if __name__ == '__main__':
    main()
