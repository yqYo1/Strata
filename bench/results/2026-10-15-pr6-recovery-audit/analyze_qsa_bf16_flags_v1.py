#!/usr/bin/env python3
"""Verify closed matched before/after diagnostic aggregation samples."""
import hashlib,json,math,statistics
from pathlib import Path
A=Path(__file__).resolve().parent
ORDER=['base-0','late-0','late-1','base-1','base-2','late-2']
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 dst=A/'qsa-bf16-late-flags-summary-v1.json';md=A/'qsa-bf16-late-flags-summary-v1.ja.md';assert not dst.exists() and not md.exists()
 processes={};frame=None;output=None;starts=[];stats={};host_source=None
 for case in ORDER:
  p=A/f'own-qsa-bf16-flags-bench-{case}-v1.json';r=json.loads(p.read_text())
  assert r['passed'] and r['complete'] and not r['active'] and r['ownership_closed'] and not r['fresh_faults'] and not r['driver_dump_entries'] and r['native_exit_code']==0 and r['boot_unchanged']
  assert not r['traced'] and r['nq']==8192 and r['repetitions']==10 and len(r['samples'])==10
  assert 'UR_ENABLE_LAYERS' not in r['native_environment'] and 'UR_LOG_TRACING' not in r['native_environment']
  f=r['frame'];assert f['arm']=='bf16' and f['cap']=='2051' and f['workgroups']=='16384'
  if frame is None:frame=f
  assert f==frame
  check=dict(x.split('=',1) for x in r['cases'][0].split()[1:])
  assert all(check[k]=='PASS' for k in ['status','guards','inputs','full_finite','repeat_bits']) and check['compared']=='24576' and float(check['max_budget_ratio'])<=1
  if output is None:output=check['output_FNV1a64']
  assert check['output_FNV1a64']==output
  build=json.loads(Path(r['build']['path']).read_text());source=[v for v in build['frozen_files'].values() if Path(v['path']).name=='qsa_bf16_bench.cpp'];assert len(source)==1
  if host_source is None:host_source=source[0]
  assert source[0]==host_source
  values=[float(s['whole_wall_ms']) for s in r['samples']];assert all(math.isfinite(x) and x>0 for x in values)
  assert [int(s['sample']) for s in r['samples']]==list(range(10))
  native=[c for c in r['commands'] if c['label']=='qsa-bf16'];assert len(native)==1;starts.append((native[0]['started_utc'],case))
  processes[case]=dict(receipt=pin(p),binary=r['binary'],samples=r['samples'],median_ms=statistics.median(values),oracle_and_output=check)
 assert [x[1] for x in sorted(starts)]==ORDER
 for arm in ['base','late']:
  vals=[processes[x]['median_ms'] for x in ORDER if x.startswith(arm+'-')]
  stats[arm]=dict(process_medians_ms=vals,median_ms=statistics.median(vals),stdev_ms=statistics.stdev(vals),min_ms=min(vals),max_ms=max(vals),processes=3,samples=30)
 ratio=stats['late']['median_ms']/stats['base']['median_ms']
 rec=dict(schema='strata-qsa-bf16-late-flags-summary-v1',source=pin(Path(__file__)),order=ORDER,frame=frame,common_host_fixture=host_source,processes=processes,statistics=stats,latency_reduction_percent=(1-ratio)*100,speedup=1/ratio,all_six_initial_final_output_fingerprints_identical=True,
  change='Same BF16 operand/FP32 online arithmetic and matrix geometry. Diagnostic-only workgroup collectives: three per 32-cell tile -> two after all tiles. Explicit local accessor payload 28168 -> 28160 bytes; actual compiled hardware barriers/allocation are not measured.',
  numerical='Each variant checks four full queries against independent BF16 contract; full output finite/guards/repeat and all inputs immutable. Full late-flags boundary/diagnostic suite passed separately. Cross-variant exact output is a control result for this fixture, not an adoption precision requirement.',
  limits='Component host timing only. No model PP/quality, full-model 262144 context, RESTORE, production selection, GPU arithmetic utilization/occupancy or physical traffic qualification.',
  decision='Retain as experimental scheduling result; defer production selection until whole-prefill gate. This particular diagnostic aggregation does not explain most component latency. Next discriminate repeated per-query KV decode/local transpose from one-time GPU BF16 pack, charging pack time and extra memory; use pinned SGLang prefix branch as design reference, not a speed claim.')
 dst.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
 lines=['# BF16 QSA: diagnostic aggregation scheduling control','',rec['change'],'','| Path | Independent process medians (ms) | Median (ms) |','| --- | --- | --- |']
 for arm in ['base','late']:
  v=stats[arm];lines.append('| '+arm+' | '+', '.join(f'{x:.6f}' for x in v['process_medians_ms'])+f" | {v['median_ms']:.6f} |")
 lines.extend(['',f"Latency reduction: {(1-ratio)*100:.4f}%; speedup: {1/ratio:.5f}x. 各3process×10 samples、2 warmups。順序: {','.join(ORDER)}.",'',rec['numerical'],'',rec['limits'],'',rec['decision']]);md.write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(statistics=stats,latency_reduction_percent=(1-ratio)*100,speedup=1/ratio),indent=2))
if __name__=='__main__':main()
