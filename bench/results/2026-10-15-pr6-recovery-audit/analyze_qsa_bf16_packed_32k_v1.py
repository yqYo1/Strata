#!/usr/bin/env python3
"""Compare only the six closed32K runs; preserve the failed256K gate separately."""
import hashlib,json,math,statistics
from pathlib import Path
A=Path(__file__).resolve().parent
ORDER=['base-32768-0','packed-32768-0','packed-32768-1','base-32768-1','base-32768-2','packed-32768-2']
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 dst=A/'qsa-bf16-packed-32k-summary-v1.json';md=A/'qsa-bf16-packed-32k-summary-v1.ja.md';assert not dst.exists() and not md.exists()
 cases={};frame=None;binary=None;fingerprint=None;starts=[];stats={}
 for case in ORDER:
  p=A/f'own-qsa-bf16-packed-bench-{case}-v1.json';r=json.loads(p.read_text());assert r['passed'] and r['complete'] and not r['active'] and r['ownership_closed'] and not r['fresh_faults'] and not r['driver_dump_entries'] and r['native_exit_code']==0
  assert r['cells']==32768 and r['nq']==8192 and r['repetitions']==10 and not r['traced']
  assert len(r['samples'])==10 and 'UR_ENABLE_LAYERS' not in r['native_environment']
  f={k:v for k,v in r['frame'].items() if k not in ['arm','pack_cost']};assert f['global_selected_cells']=='32768' and f['cap']=='2051' and f['workgroups']=='16384'
  if frame is None:frame=f;binary=r['binary']
  assert f==frame and r['binary']==binary
  check=dict(p.split('=',1) for p in r['cases'][0].split()[1:]);assert check['status']=='PASS' and check['compared']=='24576' and float(check['max_budget_ratio'])<=1
  for k in ['guards','inputs','full_finite','repeat_bits']:assert check[k]=='PASS'
  if fingerprint is None:fingerprint=check['output_FNV1a64']
  assert fingerprint==check['output_FNV1a64']
  vals=[float(s['whole_wall_ms']) for s in r['samples']];assert all(math.isfinite(v) and v>0 for v in vals)
  native=[c for c in r['commands'] if c['label']=='qsa-bf16'];assert len(native)==1;starts.append((native[0]['started_utc'],case))
  cases[case]=dict(receipt=pin(p),samples=r['samples'],median_ms=statistics.median(vals),checks=check)
 assert [s[1] for s in sorted(starts)]==ORDER
 for arm in ['base','packed']:
  v=[cases[c]['median_ms'] for c in ORDER if c.startswith(arm+'-')];stats[arm]=dict(process_medians_ms=v,median_ms=statistics.median(v),stdev_ms=statistics.stdev(v),processes=3,samples=30)
 ratio=stats['packed']['median_ms']/stats['base']['median_ms']
 rec=dict(schema='strata-qsa-bf16-packed32k-summary-v1',source=pin(Path(__file__)),order=ORDER,frame=frame,binary=binary,cases=cases,statistics=stats,latency_reduction_percent=(1-ratio)*100,speedup=1/ratio,all_final_output_fingerprints_match=True,packed_extra_payload_bytes=int(frame['packed_scratch_payload_bytes']),
  scope='8192-query QSA component with 2051 selected cells/query. Per-query rotated selections actually cover all32768 logical/physical cells; all input/guard checks and four-query BF16 contract oracle. Same initialized allocation footprint both arms. Whole physical BF16 pack included in every packed timed call. No CPU tensor work or host transfer within timed call.',
  caveats=['Full32K pool is not a32K whole-model prompt/bench.sh result; model PP/quality/physical262144-context/RESTORE/production selection remain unqualified.','Minimum payload/footprint is not measured physical GPU traffic or device utilization. No CPU exclusivity claim while read-only research workers ran.','The benchmark repeat check compares float values, so it does not distinguish signed-zero bits. Cross-process final FNV fingerprints preserve bit distinctions but are not a retained full-byte comparator. The separate component fixture repeats uint16 BF16 outputs exactly.'],
  full256k=dict(status='FAILED_NUMERICAL_GATE_BEFORE_TIMING',original_receipt='own-qsa-bf16-packed-bench-base-262144-0-v1.json',original_status_unchanged=True,physical_closure='qsa-bf16-packed-256k-numerical-stop-v1.json',diagnostics=['own-qsa-bf16-packed-diagnostic-base-v1.json','own-qsa-bf16-packed-diagnostic-packed-v1.json'],facts='Both paths fail the same10 elements at query2/head13 and have matching full-output FNV fingerprints. Diagnostic executions completed normally and closed without fresh GPU faults. Cause of host/GPU numerical difference not yet proved; no256K speed result.'),
  decision='Continue investigating the numerical boundary and real-session initialized/live extents before model integration. The32K component signal is stable and substantial; it does not qualify whole-prefill adoption.')
 dst.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
 lines=['# GPU BF16 KV pack:32K physical-pool comparison','',rec['scope'],'','| Path | Independent process medians(ms) | Median(ms) |','| --- | --- | --- |']
 for arm in ['base','packed']:
  s=stats[arm];lines.append('| '+arm+' | '+', '.join(f'{x:.6f}' for x in s['process_medians_ms'])+f" | {s['median_ms']:.6f} |")
 lines.extend(['',f"Pack込みlatency reduction {(1-ratio)*100:.4f}%, speedup {1/ratio:.5f}x; 3process×10samples per arm. Extra decoded payload {int(frame['packed_scratch_payload_bytes'])} bytes; both benchmark arms allocate it.",'','256K比較は最初の既存経路で数値gateに失敗し、速度samples無し。独立diagnosticでは両経路が同じ10要素(query2/head13)で参照から外れ、全出力FNV一致。プロセス正常終了、freshfault無し。旧失敗statusは変更していない。原因未確定なので256K速度やモデル採用を認定しない。','',*rec['caveats'],'',rec['decision']]);md.write_text('\n'.join(lines)+'\n');print(json.dumps(dict(statistics=stats,latency_reduction_percent=(1-ratio)*100,speedup=1/ratio),indent=2))
if __name__=='__main__':main()
