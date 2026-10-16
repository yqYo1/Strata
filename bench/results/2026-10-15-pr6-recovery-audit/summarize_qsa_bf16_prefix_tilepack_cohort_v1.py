#!/usr/bin/env python3
"""Summarize only fully closed, same-binary officialbench individual samples."""
import hashlib,json,statistics
from pathlib import Path
A=Path(__file__).resolve().parent
cohort=A/'qsa-bf16-prefix-tilepack-clean-cohort-v1.json';d=json.loads(cohort.read_text())
assert d['passed'] and d['complete'] and d['active_controller'] is None and len(d['completed'])==6
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=dict(schema='strata-qsa-bf16-prefix-tilepack-summary-v1',passed=True,complete=True,cohort_sha256=sha(cohort),samples=[],statistics={},full_model_262144_qualified=False,RESTORE_qualified=False,general_quality_qualified=False,decode_performance_qualified=False,scope='Officialbench39K thinkingON/24output cap, independent fresh processes, same engine/context65536/chunk8192/GPU32768+RAM residuals, BF16 tensor/FP32 online-state contract unchanged. Not a long decode benchmark or task-quality test.')
for item in d['completed']:
 p=Path(item['path']);assert sha(p)==item['sha256'];v=json.loads(p.read_text())
 assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and v['boot_unchanged'] and not v['fresh_faults'] and not v['driver_dump_exists']
 req,=v['benchmark_result']['results']['prefill_only']['requests'];assert req['ok'] and req['eligible'] and req['stream_complete'] and req['actual_prompt_tokens']>=32768 and req['actual_read_tokens']==req['actual_prompt_tokens'] and req['actual_reused_tokens']==0 and req['prompt_tps_source']=='engine' and req['thinking_requested']
 assert v['result']['server_exit_code']==v['result']['real_engine_exit']['exit_code']==0
 r['samples'].append(dict(arm=item['arm'],repetition=item['repetition'],receipt=str(p),sha256=sha(p),binary_sha256=v['binary_sha256'],source_commit=v['source_commit'],request_sha256=req['request_sha256'],actual_prompt_tokens=req['actual_prompt_tokens'],prompt_tps=req['prompt_tps'],prompt_ms=req['prompt_ms'],placement=v['placement_lines'],QSA_arithmetic=v['QSA_arithmetic_lines'],fresh_faults=[],normal_native_exit=True))
for k in ['binary_sha256','source_commit','request_sha256','actual_prompt_tokens']:assert len({x[k] for x in r['samples']})==1,k
assert len({tuple(x['placement']) for x in r['samples']})==1
for arm in ['bf16-prefix-optimized','bf16-prefix-tilepack']:
 vals=[x['prompt_tps'] for x in r['samples'] if x['arm']==arm];assert len(vals)==3
 ms=[x['prompt_ms'] for x in r['samples'] if x['arm']==arm]
 r['statistics'][arm]=dict(independent_processes=3,individual_prompt_tps=vals,median_tps=statistics.median(vals),min_tps=min(vals),max_tps=max(vals),mean_tps=statistics.mean(vals),sample_sd_tps=statistics.stdev(vals),individual_prompt_ms=ms,median_prompt_ms=statistics.median(ms))
a=r['statistics']['bf16-prefix-optimized'];b=r['statistics']['bf16-prefix-tilepack']
r['throughput_gain_percent']=(b['median_tps']/a['median_tps']-1)*100
r['prompt_latency_reduction_percent']=(1-b['median_prompt_ms']/a['median_prompt_ms'])*100
mechanics=A/'own-qsa-bf16-prefix-tilepack-model-smoke-v1.json'
m=json.loads(mechanics.read_text());assert m['passed'] and m['complete'] and m['ownership_closed']
r['model_mechanics_receipt']=dict(path=str(mechanics),sha256=sha(mechanics),difference=m['difference'])
r['decision']='Prefer dense16/VNNI score33 for this scoped dedicated Mode0 workload' if b['median_tps']>a['median_tps'] else 'No speed adoption from this comparison'
r['prefill600_milestone_reached']=min(b['individual_prompt_tps'])>600
out=A/'qsa-bf16-prefix-tilepack-clean-summary-v1.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n')
text=f"""# BF16 QSA dense16/VNNI配置のモデル全体比較

同一バイナリ、公式bench.shのprefill-only拡張、thinking ON/生成上限24、実入力{r['samples'][0]['actual_prompt_tokens']} token、context/resident65536、chunk8192、GPU残差32768行＋RAM。各arm3独立processを交互順に測定。

|経路|個別prefill tok/s|中央値|標本SD|
|---|---|---:|---:|
|基準score33 row-major|{a['individual_prompt_tps']}|{a['median_tps']:.1f}|{a['sample_sd_tps']:.3f}|
|候補score33 dense16/VNNI|{b['individual_prompt_tps']}|{b['median_tps']:.1f}|{b['sample_sd_tps']:.3f}|

スループット+{r['throughput_gain_percent']:.3f}%、prompt時間-{r['prompt_latency_reduction_percent']:.3f}%。同じ入力body hash、全6回正常終了/ownership closed/新規GPU faultゼロ。部品測定6.68%の短縮をprefill全体の改善率として扱わない。

BF16 Q/K/V/P/出力とFP32の累積・max/normalizer、32セルの逐次更新順は同じ。all48合成129tokenのNLL/3箇所全語彙logitの比較は別のmodel-smoke receiptを参照。精度の採用基準は同系モデルのSGLang/vLLM契約であり、llama.cppとの完全一致ではない。一般品質、full-model256K、保存/RESTORE、長いdecodeは未認定。prefill600目標は{r['prefill600_milestone_reached']}。

次は独立した64K入力/残差GPU配置と、現在のモデルprofileに基づくGPU処理の検討。全実行はMainが直列管理する。
"""
(A/'qsa-bf16-prefix-tilepack-clean-summary-v1.ja.md').write_text(text)
print(json.dumps(dict(statistics=r['statistics'],gain_percent=r['throughput_gain_percent'],decision=r['decision'])))
