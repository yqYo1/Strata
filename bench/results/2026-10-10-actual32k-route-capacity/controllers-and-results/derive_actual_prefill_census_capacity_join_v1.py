"""Pure host evidence join; no workload, GPU query, performance/capacity claim."""
from pathlib import Path
import collections,datetime,fcntl,hashlib,json
B=Path(__file__).parent;W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
RUN=B/'owned-prefill-route-census-v3-code32k-diagnostic-r1';PACK=W/'bench/results/2026-10-10-current-native-pack-census/receipt.json'
DQ=W/'bench/results/2026-10-10-gpu-iq-dequant-capacity/three-process-summary.json';OUT=B/'actual-prefill-census-capacity-join-v1.json'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not OUT.exists();run=json.loads((RUN/'record.json').read_text());assert run['healthy'] and run['math_gate_passed'] and run['census_gate_passed'] and not run['active'] and run['exit_code']==0
 pack=json.loads(PACK.read_text());assert pack['passed'] and ident(PACK)['sha256']=='902283e324b339127a9eb197954260ab606e32767b4e9acaf1d17e2463d31f84'
 assert ident(Path(pack['pack_manifest']['path']))['sha256']==pack['pack_manifest']['sha256']
 shard=Path(pack['primary']['path']);st=shard.stat()
 for k,v in [('device',st.st_dev),('inode',st.st_ino),('bytes',st.st_size),('mtime_ns',st.st_mtime_ns),('ctime_ns',st.st_ctime_ns)]:assert pack['primary'][k]==v
 with shard.open('rb') as f:assert hashlib.sha256(f.read(pack['primary']['header_plus_padding_bytes'])).hexdigest()==pack['primary']['header_plus_padding_sha256']
 census=RUN/'census.jsonl';assert ident(census)['sha256']==run['census_file']['sha256']
 rows=[json.loads(s) for s in census.read_text().splitlines()];layers=rows[:-1];receipt=rows[-1];assert receipt==run['census_receipt']
 totals=collections.Counter();by_pair=collections.defaultdict(collections.Counter);buckets=collections.defaultdict(collections.Counter);by_source=collections.defaultdict(collections.Counter);flat=[]
 for row in layers:
  p=pack['per_layer'][row['layer']];roles={x['role']:x for x in p['roles']}
  assert (row['gu_type'],row['down_type'],row['blob_bytes'],row['packed_layer_offset'])==(p['gu_type'],p['down_type'],p['packed_blob_bytes'],p['packed_offset'])
  assert row['gu_row']*row['FF']==row['up_off']==roles['gate']['bytes_per_expert']==roles['up']['bytes_per_expert'] and row['down_row']*row['H']==roles['down']['bytes_per_expert'] and row['down_off']==2*row['up_off']
  assert row['layout_version']==3 and row['native'] and row['branch']==1 and not row['use_mmq'] and not row['mmq_built'] and not row['fused_l'] and not row['fused_nat']
  assert row['K']==10 and row['H']==2560 and row['FF']==640 and row['n_expert']==512 and row['rows']==row['T']*row['K'] and row['local_rows']==row['rows'] and row['peer_rows']==0 and row['peer_calls']==0
  assert row['generic_GU_calls']==row['generic_Down_calls']==row['calls']==sum(x['calls'] for x in row['bins'])
  assert sum(x['routed_rows'] for x in row['bins'])==row['rows']
  pair=f"{row['gu_type']}/{row['down_type']}";count=collections.Counter({k:row[k] for k in ['calls','rows','transfer_calls','transfer_bytes','generic_GU_calls','generic_Down_calls']});totals.update(count);by_pair[pair].update(count)
  for z in row['bins']:
   assert 0<z['ne']<=row['T'] and z['routed_rows']==z['ne']*z['calls'] and z['logical_packed_bytes']==z['calls']*row['blob_bytes']
   name='1-16' if z['ne']<=16 else '17-80' if z['ne']<=80 else '81-160' if z['ne']<=160 else '161-640' if z['ne']<=640 else '641-8192'
   cc=collections.Counter({k:z[k] for k in ['calls','routed_rows','logical_packed_bytes']});buckets[name].update(cc);by_source[str(z['source'])].update(cc)
   flat.append(dict(chunk=row['chunk_p0'],layer=row['layer'],pair=pair,**z))
 for k in ['calls','transfer_calls','transfer_bytes']:assert totals[k]==receipt[k]
 assert totals['rows']==receipt['routed_rows']==32767*10*48
 dq=json.loads(DQ.read_text());r=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,model_route_join_qualified=True,performance_eligible=False,full256k_qualified=False,controller=ident(__file__),model_receipt=ident(RUN/'record.json'),census=ident(census),pack_receipt=ident(PACK),dequant_capacity=ident(DQ),scope='Actual onefresh32768 request native genericIQ census joined to current unchanged manifest/header/144 descriptors and source metadata. Measured route counts and logical byte amounts; isolated capacities are attained estimates, never exclusive model critical path or absolute ceilings.',totals=dict(totals),paired_formats={k:dict(v) for k,v in sorted(by_pair.items())},ne_buckets={k:dict(v) for k,v in buckets.items()},source_buckets={k:dict(v) for k,v in by_source.items()},minimum_ne=min(z['ne'] for z in flat),maximum_ne=max(z['ne'] for z in flat),weighted_mean_rows_per_expert=totals['rows']/totals['calls'],generic_gu_and_down_expanded_output_bytes=totals['calls']*9830400,attained_HostUSM_H2D_GBs=6.446833643341082,attained_pageable_H2D_GBs=4.605214183505477,external_Gen4x4_encoded_GBs=7.876923076923077,H2D_payload_service_seconds_at_attained_USM=totals['transfer_bytes']/6.446833643341082e9,H2D_payload_service_seconds_at_attained_pageable=totals['transfer_bytes']/4.605214183505477e9,H2D_payload_seconds_at_encoded_external_link=totals['transfer_bytes']/7.876923076923077e9,census_host_accumulation_seconds=receipt['host_accumulation_ns']/1e9,census_host_serialization_seconds=receipt['host_serialization_ns']/1e9,source_buckets_semantics='Header sourceenum: 1resident;3copied. transfer_source detail in lossless census. Source inferred from admitted caller accounting, not physical DMA/DRAM counters.',limits=['Diagnostic wall/DONE times excluded from clean speed.','Counts/bytes do not establish DMA active durations, overlap or CPU exclusive critical path.','No arbitrary nearest-ne interpolation: capacity fixtures do not time every admitted row size.','Standalone sevenpair DQ O2 AOT differs from production O3 compile policy; use as candidate scale, not model latency fact.','No whole54.8GB GGUF payload hash; descriptor/header/manifest identity and actual loadedMeta paths only.','No full262144 candidate physical context, defaultOFF runtime parity or production adoption yet.'])
 OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['passed','totals','ne_buckets','source_buckets','minimum_ne','maximum_ne','weighted_mean_rows_per_expert','H2D_payload_service_seconds_at_attained_USM','H2D_payload_seconds_at_encoded_external_link']},indent=2))
