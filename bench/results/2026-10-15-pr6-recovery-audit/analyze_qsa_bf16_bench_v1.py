#!/usr/bin/env python3
"""Verify all nine closed receipts before comparing independent process medians."""
import hashlib,json,math,statistics
from datetime import datetime,timezone
from pathlib import Path
A=Path(__file__).resolve().parent
ORDER=['fp32-0','typed-0','bf16-0','bf16-1','typed-1','fp32-1','fp32-2','bf16-2','typed-2']
def pin(p):
 return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 out=A/'qsa-bf16-matched-bench-summary-v1.json';md=A/'qsa-bf16-matched-bench-summary-v1.ja.md'
 assert not out.exists() and not md.exists()
 records={};inputs=None;binary=None;start=[];arms={}
 for case in ORDER:
  p=A/f'own-qsa-bf16-bench-{case}-v1.json';r=json.loads(p.read_text())
  assert r['passed'] and r['complete'] and not r['active'] and r['ownership_closed'] and not r['fresh_faults'] and not r['driver_dump_entries']
  assert r['native_exit_code']==0 and r['boot_unchanged'] and r['nq']==8192 and r['repetitions']==10 and not r['traced']
  assert 'UR_ENABLE_LAYERS' not in r['native_environment'] and 'UR_LOG_TRACING' not in r['native_environment']
  assert len(r['samples'])==10 and len(r['cases'])==1
  f=r['frame'];assert f['cap']=='2051' and f['workgroups']=='16384' and f['warmups']=='2'
  ident={k:v for k,v in f.items() if k!='arm'}
  if inputs is None:inputs=ident;binary=r['binary']
  assert ident==inputs and r['binary']==binary
  values=[float(s['whole_wall_ms']) for s in r['samples']]
  assert all(math.isfinite(x) and x>0 for x in values) and [int(s['sample']) for s in r['samples']]==list(range(10))
  check=dict(x.split('=',1) for x in r['cases'][0].split()[1:])
  assert all(check[k]=='PASS' for k in ['status','guards','inputs','full_finite','repeat_bits'])
  if r['arm']=='bf16':assert check['compared']=='24576' and float(check['max_budget_ratio'])<=1
  native=[c for c in r['commands'] if c['label']=='qsa-bf16'];assert len(native)==1
  start.append((native[0]['started_utc'],case))
  item=dict(case=case,receipt=pin(p),samples=r['samples'],median_whole_wall_ms=statistics.median(values),oracle_and_output=check)
  records[case]=item;arms.setdefault(r['arm'],[]).append(item)
 assert [x[1] for x in sorted(start)]==ORDER
 stats={}
 for arm,items in arms.items():
  assert len(items)==3 and len({x['oracle_and_output']['output_FNV1a64'] for x in items})==1
  med=[x['median_whole_wall_ms'] for x in items]
  stats[arm]=dict(process_medians_ms=med,median_of_process_medians_ms=statistics.median(med),process_median_min_ms=min(med),process_median_max_ms=max(med),process_median_stdev_ms=statistics.stdev(med),independent_processes=3,timed_samples=30)
 new=stats['bf16']['median_of_process_medians_ms']
 comparison={a:dict(new_to_reference_latency_ratio=new/stats[a]['median_of_process_medians_ms'],latency_reduction_percent=(1-new/stats[a]['median_of_process_medians_ms'])*100,speedup=stats[a]['median_of_process_medians_ms']/new) for a in ['fp32','typed']}
 r=dict(schema='strata-qsa-bf16-matched-summary-v1',created_utc=datetime.now(timezone.utc).isoformat(),source=pin(Path(__file__)),order=ORDER,frame_identity=inputs,binary=binary,processes=records,statistics=stats,comparisons=comparison,
  numerical_contract='New BF16 Q-scaled/K-decoded/P operands and BF16 O, FP32 online max/norm/accumulation. Four complete queries independently checked, all outputs finite/guards/inputs/repeat checked. Legacy paths prior-qualified, not this oracle.',
  qualification=dict(component_comparison=True,model_PP=False,model_quality=False,full_262144_model=False,RESTORE=False,hardware_utilization=False,physical_memory_traffic=False,production_adoption=False),
  interpretation='A matched component speed result, not whole-model tok/s or a measurement of arithmetic utilization. Changes include operand precision/scale placement/products/local payload; no single resource is causally isolated.',
  next='If faster, link as explicit opt-in long-prefill candidate, validate model state/quality and run repeated official bench.sh long32K+ before default adoption. Retire successful fullAPI traces after compact facts are committed.')
 out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 lines=['# BF16 operand QSA matched component comparison','',r['interpretation'],'','| Path | Process medians (ms) | Median (ms) |','| --- | --- | --- |']
 for arm in ['fp32','typed','bf16']:
  s=stats[arm];lines.append('| '+arm+' | '+', '.join(f'{x:.6f}' for x in s['process_medians_ms'])+f" | {s['median_of_process_medians_ms']:.6f} |")
 lines.extend(['','各経路3プロセス×10 samples。全プロセスでQ/K/V/scales/IDs/steps・VRAM確保量を一致。QはBF16で表現可能な同一値。各processの2 warmup後、割当・転送・検証・ログの書込みを計測窓に入れない。実行順: '+','.join(ORDER)+'.',''])
 for arm in ['fp32','typed']:
  c=comparison[arm];lines.append(f"新BF16 / {arm}: latency {c['latency_reduction_percent']:.3f}% reduction, {c['speedup']:.4f}x.")
 lines.extend(['','新BF16はquery 0/1/2/lastの24,576要素を独立BF16契約で検証。旧2経路は既存の資格付けを比較の前提とし、旧FP32出力との一致をBF16採用条件にしない。全出力有限・guard・全入力不変・最終repeat一致、各armの独立3process出力fingerprint一致、正常終了・所有終了・新GPU fault無しを確認。','', 'これはQSA単体。モデル全体のPP、品質、物理256K context、RESTORE、実メモリ帯域・occupancy・算術利用率は未認定。デフォルト採用はしていない。新経路はscale/丸め位置・積数・SLM構成も変更しており、個々の速度寄与は分離していない。','',r['next']])
 md.write_text('\n'.join(lines)+'\n');print(json.dumps(dict(statistics=stats,comparisons=comparison),indent=2))
if __name__=='__main__':main()
