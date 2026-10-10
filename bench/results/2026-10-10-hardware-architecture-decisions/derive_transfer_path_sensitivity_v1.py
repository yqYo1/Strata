"""Pure conditional arithmetic; this program performs no hardware measurement."""
from pathlib import Path
import datetime,fcntl,hashlib,json
A=Path(__file__).parent
def load(name):return json.loads((A/name).read_text())
def ident(p):
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def calculate():
 matrix=load('component-capacity-matrix-v2.json');correction=load('residual-dimension-correction.json');topology=load('physical-bandwidth-reference.json')
 pack=A.parent/'2026-10-10-current-native-pack-census/receipt.json';packed=json.loads(pack.read_text())
 assert packed['passed'] and packed['expert_layers']==48 and packed['experts_per_layer']==512
 def rate(cell):return next(x['value'] for x in matrix['rows'] if x['cell']==cell)
 h_usm=rate(['transfer','H2D_host_USM',268435456]);d_usm=rate(['transfer','D2H_host_USM',268435456]);h_page=rate(['transfer','H2D_pageable',268435456]);d_page=rate(['transfer','D2H_pageable',268435456]);ideal=topology['external_path_Gen4x4_encoded_upper_GBps']
 residual=next(x for x in correction['corrected']['residual_scenarios'] if x['hypothetical_positions']==32768)
 hbytes=residual['all_RAM_residual_H2D_payload_bytes'];dbytes=residual['all_RAM_residual_D2H_payload_bytes'];weight=packed['total_all_layer_packed_bytes']
 assert hbytes==dbytes==47*32768*10240*4 and weight==50292326400
 cases=[]
 for name,h,d,wh in [('attained_pageable',h_page,d_page,h_usm),('conditional_direct_host_USM',h_usm,d_usm,h_usm),('ideal_encoded_external_link',ideal,ideal,ideal)]:
  rs=hbytes/(h*1e9)+dbytes/(d*1e9);ws=weight/(wh*1e9)
  cases.append(dict(case=name,residual_H2D_GBps=h,residual_D2H_GBps=d,packed_H2D_GBps=wh,residual_seconds=rs,packed_seconds=ws,serial_transfer_seconds=rs+ws,seconds_remaining_for_all_other_work_at_1000_tokens_s=32.768-rs-ws))
 return dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Conditional transfer-path arithmetic only. No model/hardware execution, speed result, allocation/pinning admission or architecture adoption.',controller=ident(Path(__file__)),sources={n:ident(A/n) for n in ['component-capacity-matrix-v2.json','residual-dimension-correction.json','physical-bandwidth-reference.json']},packed_census=ident(pack),positions=32768,residual_width_FP32=10240,layer_boundaries=47,residual_H2D_payload_bytes=hbytes,residual_D2H_payload_bytes=dbytes,one_unique_packed_population_bytes=weight,formula='residual_H2D/(H2D_GBps*1e9)+residual_D2H/(D2H_GBps*1e9)+packed/(packed_H2D_GBps*1e9)',cases=cases,limitations=['All cases serialize copies and exclude computation, dequant, attention, staging, protocol/other traffic, scheduling and allocation overhead.','USM rates were measured with 256MiB buffers; direct placement/registration of the residual plane is not implemented or admitted. A staging implementation must add its actual CPU copies and memory peak.','Ideal encoded link rate excludes packet/protocol overhead and does not assert full-duplex concurrency. It is not an attained payload rate.','Pageable-rate violation of a serial budget is conditional on that transfer path; it does not prove all-RAM architectures cannot fit at other rates. Full262144 all-GPU residual exceeds entire recorded device capacity independently.'])
if __name__=='__main__':
 lock=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/owned-v0141-measurement.lock')
 with lock.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);p=A/'transfer-path-sensitivity-v1.json';assert not p.exists();r=calculate();p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['cases'],indent=2))
