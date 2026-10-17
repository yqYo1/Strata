#!/usr/bin/env python3
"""Verify the completed matched cohort; never turn partial runs into samples."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

A = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = A / 'qsa-subgroup-engine-model-clean-cohort-v1.json'
    cohort = json.loads(path.read_text())
    assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None
    assert len(cohort['completed']) == 6
    assert {(x['window'],x['repetition']) for x in cohort['completed']} == {
        (arm,n) for arm in ['optimized','subgroup'] for n in range(3)}
    samples, environments, config_args, requests = [], [], [], []
    for item in cohort['completed']:
        assert sha(item['path']) == item['sha256']
        value = json.loads(Path(item['path']).read_text())
        assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active']
        assert value['boot_unchanged'] and not value['fresh_faults'] and not value['driver_dump_exists']
        assert value['binary'] == cohort['binary']
        assert all(c['exit_code'] == 0 and c['session_empty'] and c['direct_child_reaped'] for c in value['commands'])
        assert value['result']['server_exit_code'] == value['result']['real_engine_exit']['exit_code'] == 0
        env = value['environment']
        assert 'STRATA_TRACE' not in env and 'UR_ENABLE_LAYERS' not in env
        assert not any(k.startswith('UR_LOG_') for k in env)
        assert env['STRATA_SYCL_PREFILL_QSA_MATH'] == 'bf16-prefix-'+item['window']
        assert env['STRATA_SYCL_PREFILL_MOE_WINDOW'] == '2'
        normalized = {k:v.replace(value['root'],'<run>') for k,v in env.items()}
        normalized['STRATA_SYCL_PREFILL_QSA_MATH'] = '<arm>'
        environments.append(normalized)
        config_args.append([v.replace(value['root'],'<run>') for v in value['config']['args']])
        request, = value['benchmark_result']['results']['prefill_only']['requests']
        assert request['ok'] and request['eligible'] and request['stream_complete']
        assert request['actual_prompt_tokens'] == request['actual_read_tokens'] == 39262
        assert request['actual_reused_tokens'] == 0 and request['thinking_requested'] and request['reasoning_observed']
        assert request['generated_tokens'] == 24
        assert all(math.isfinite(request[k]) and request[k]>0 for k in ['prompt_tps','prompt_ms'])
        requests.append(request['request_sha256'])
        samples.append(dict(arm=item['window'],repetition=item['repetition'],
             receipt=dict(path=item['path'],sha256=item['sha256']),request=request,
             moe_window_lines=value['window_lines']))
    assert all(x == environments[0] for x in environments)
    assert all(x == config_args[0] for x in config_args)
    assert len(set(requests)) == 1
    summaries = {}
    for arm in ['optimized','subgroup']:
        values = [x['request']['prompt_tps'] for x in samples if x['arm']==arm]
        times = [x['request']['prompt_ms'] for x in samples if x['arm']==arm]
        summaries[arm] = dict(n=3,samples_tok_s=values,median_tok_s=statistics.median(values),
            min_tok_s=min(values),max_tok_s=max(values),mean_tok_s=statistics.mean(values),
            sample_sd_tok_s=statistics.stdev(values),median_prompt_ms=statistics.median(times))
    baseline, candidate = summaries['optimized'],summaries['subgroup']
    gain = candidate['median_tok_s']/baseline['median_tok_s']-1
    mean_gain = candidate['mean_tok_s']/baseline['mean_tok_s']-1
    separated = min(candidate['samples_tok_s']) > max(baseline['samples_tok_s'])
    result = dict(schema='strata-qsa-subgroup-engine-model-clean-comparison-v1',passed=True,complete=True,
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_cohort=dict(path=str(path),sha256=sha(path)),analysis_recipe_sha256=sha(__file__),
        binary=cohort['binary'],samples=samples,summaries=summaries,
        median_rate_gain_fraction=gain,all_candidate_samples_above_all_baseline_samples=separated,
        mean_rate_gain_fraction=mean_gain,model_improvement_demonstrated=separated and gain>0,
        normalized_environment_and_config_identical_except_QSA_selector=True,
        performance_evidence_qualified_for_this_cohort=True,performance_adopted=False,
        general_quality_qualified=False,physical262144_RESTORE_qualified=False,
        prefill600_milestone_achieved=False,prefill1000_goal_achieved=False,decode70_goal_achieved=False,
        limits=['Three independent fresh processes per arm in interleaved order estimate this observed variation only.',
                'Official bench.sh39K input/64K capacity, C8192/RGPU32768/window2/BF16/vLLM path, no reuse.',
                'No cross-arm output/token bit equality gate; fingerprints remain observations.',
                'Changed sum association can change routing: retain per-arm MoE window counts; equal input does not mean equal expert work.',
                '24 thinking generation tokens are a completion probe, not sustained decode or quality.',
                'No hardware-resource saturation or full-context/RESTORE qualification follows.'])
    dst = A / 'qsa-subgroup-engine-model-clean-comparison-v1.json'
    assert not dst.exists()
    dst.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    report = ['# QSA subgroup 実モデル clean比較','',
        '2026-10-17 Arc B570/driver1.17.39395+14/LLVM22.1。同一binaryのQSA選択だけを変え、',
        '公式bench.shの実入力39262/read39262/reuse0、thinking有効、C8192/RGPU32768/window2。',
        'optimized,subgroup,subgroup,optimized,optimized,subgroupの6独立process。詳細ログ・validationなし。','',
        '| 経路 | 各process PP(tok/s) | 中央値(tok/s) |','| --- | --- | --- |']
    for arm in ['optimized','subgroup']:
        s=summaries[arm]
        report.append(f"| {arm} | {', '.join(str(v) for v in s['samples_tok_s'])} | {s['median_tok_s']} |")
    report.extend(['',f'中央値の変化は{gain*100:.4f}%。候補全sampleが基準全sampleを上回る: {separated}。',
        f'平均の変化は{mean_gain*100:.4f}%。中央値と平均を区別し、遅いsampleも残す。',
        '実モデルの安定した改善はこのcohortでは未証明。現行経路を維持して次の検証を選ぶ。' if not separated else 'このcohortでは全candidateが全baselineを上回った。品質・文脈・採用の資格付けは別。',
        '全process正常終了・所有終了、fresh fault0、dumpなし。個別timing/環境/入力fingerprintをJSONへ保持。',
        'この条件の観測であり、他条件のばらつき全体を保証しない。24生成は完了確認で一般品質やdecode速度ではない。',
        '全physical262144モデルconsume/RESTORE、cancel/reuse、品質とデフォルト採用は未資格付け。',
        '600/1000tok/sおよびdecode70tok/sの達成は認定しない。'])
    (A/'qsa-subgroup-engine-model-clean-results-v1.ja.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(passed=True,summaries=summaries,median_rate_gain_fraction=gain,
                         all_candidate_samples_above_all_baseline_samples=separated)))


if __name__ == '__main__':
    main()
