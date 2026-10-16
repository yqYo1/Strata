#!/usr/bin/env python3
"""Summarize all independent clean official benchmark samples after closure."""
import hashlib,json,statistics,math,subprocess
from pathlib import Path
A=Path(__file__).resolve().parent;P=A.parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 cohort=A/'qsa-bf16-prefix-clean-cohort-v1.json';c=json.loads(cohort.read_text());assert c['passed'] and c['complete'] and c['active_controller'] is None and len(c['completed'])==9
 dst=A/'qsa-bf16-prefix-clean-summary-v1.json';assert not dst.exists()
 groups={k:[] for k in ['current','bf16','bf16-prefix']};binary=None
 for item in c['completed']:
  p=Path(item['path']);v=json.loads(p.read_text());assert sha(p)==item['sha256'] and v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and not v['fresh_faults'] and not v['driver_dump_exists'] and v['performance_qualified']
  assert not any(k in v['environment'] for k in ['UR_ENABLE_LAYERS','STRATA_TRACE','LD_PRELOAD']);assert v['phase']=='clean'
  if binary is None:binary=v['binary_sha256']
  assert v['binary_sha256']==binary
  requests=v['benchmark_result']['results']['prefill_only']['requests'];assert len(requests)==1;r=requests[0]
  assert r['eligible'] and r['ok'] and r['actual_prompt_tokens']>=32768 and r['actual_read_tokens']==r['actual_prompt_tokens'] and r['actual_reused_tokens']==0 and r['generated_tokens']==24 and r['thinking_requested']
  assert r['prompt_tps_source']=='engine' and r['prompt_ms']>0
  groups[item['arm']].append(dict(repetition=item['repetition'],receipt=str(p),receipt_sha256=sha(p),prompt_tps=r['prompt_tps'],prompt_ms=r['prompt_ms'],input_tokens=r['actual_prompt_tokens'],reused_tokens=r['actual_reused_tokens'],request_sha256=r['request_sha256'],placement=v['placement_lines'],QSA_math=v['QSA_arithmetic_lines']))
 stats={}
 for arm,samples in groups.items():
  assert len(samples)==3 and {x['repetition'] for x in samples}=={0,1,2}
  t=[x['prompt_tps'] for x in samples];m=[x['prompt_ms'] for x in samples]
  stats[arm]=dict(n=3,median_tps=statistics.median(t),mean_tps=statistics.mean(t),sample_stdev_tps=statistics.stdev(t),min_tps=min(t),max_tps=max(t),median_prompt_ms=statistics.median(m))
 base=stats['current'];compare={}
 for arm in ['bf16','bf16-prefix']:
  v=stats[arm];compare[arm]=dict(median_tps_ratio=v['median_tps']/base['median_tps'],median_tps_gain_percent=(v['median_tps']/base['median_tps']-1)*100,median_prompt_latency_reduction_percent=(1-v['median_prompt_ms']/base['median_prompt_ms'])*100,observed_ranges_disjoint=v['min_tps']>base['max_tps'])
 rec=dict(schema='strata-bf16-prefix-clean-summary-v1',passed=True,complete=True,cohort_sha256=sha(cohort),binary_sha256=binary,order=c['order'],samples=groups,statistics=stats,comparisons=compare,scope='Official bench.sh actual32K+ input, context/resident65536, chunk8192, GPU32768 residual rows, thinking ON/24 output cap, three independent native processes per arm. Source references and individual run command/env/fault/ownership proofs reside in each pinned receipt. Read-only research may run concurrently; no CPU isolation or GPU hardware utilization claim.',precision='Current best BF16 hidden/QSA0 baseline versus SGLang-design BF16 operand QSA1 and live-prefix BF16 packing. This is performance evidence, not framework CUDA runtime, full-model quality, full physical256K or RESTORE qualification.',adoption=False,goal600_reached=min(x['prompt_tps'] for x in groups['bf16-prefix'])>600,goal1000_reached=min(x['prompt_tps'] for x in groups['bf16-prefix'])>=1000,next='Profile the faster path again to identify remaining resource bottlenecks. Keep numerical stage and full-context/RESTORE gates explicit before default adoption.')
 dst.write_text(json.dumps(rec,indent=2)+'\n')
 md=['# BF16 QSA 長大prefill比較','', '公式bench.sh、実入力32K以上、context/resident65536、chunk8192、thinking ON、各arm独立3process。', '', '|経路|中央値 tok/s|平均 ± sample SD|範囲|', '|---|---:|---:|---:|']
 for arm,v in stats.items():md.append(f"|{arm}|{v['median_tps']:.1f}|{v['mean_tps']:.1f} ± {v['sample_stdev_tps']:.1f}|{v['min_tps']:.1f}–{v['max_tps']:.1f}|")
 md+=['','全個別値・prompt時間・入力件数・request hash・配置をJSONに保存。3点はばらつきの把握であり、変動が消えた証明ではない。', 'これはprefill性能の検証。品質・SGLang/vLLM CUDA実行値・full-model256K・RESTORE・default採用は別の判定。次は新経路の資源別profile。']
 (A/'qsa-bf16-prefix-clean-summary-v1.ja.md').write_text('\n'.join(md)+'\n');print(json.dumps(stats))
if __name__=='__main__':main()
