#!/usr/bin/env python3
"""Bounded read-only GGUF metadata extraction; never read tensor payloads."""
import hashlib,json,struct
from pathlib import Path
A=Path(__file__).resolve().parent
MODEL=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
LIMIT=64<<20
SCALARS={0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}
def main():
 out=A/'gdn-target-gguf-metadata-v1.json';assert not out.exists()
 before=MODEL.stat();omitted={};values={}
 with MODEL.open('rb') as f:
  def read(n):
   assert 0<=n<=LIMIT and f.tell()+n<=LIMIT and f.tell()+n<=before.st_size
   b=f.read(n);assert len(b)==n;return b
  def u32():return struct.unpack('<I',read(4))[0]
  def u64():return struct.unpack('<Q',read(8))[0]
  def string(keep=True):
   n=u64();b=read(n);return b.decode('utf8') if keep else None
  def value(kind,keep=True,depth=0):
   assert depth<4
   if kind in SCALARS:
    fmt,n=SCALARS[kind];b=read(n);return struct.unpack('<'+fmt,b)[0] if keep else None
   if kind==8:return string(keep)
   assert kind==9,kind
   element=u32();n=u64();assert n<=1<<22
   if not keep and element in SCALARS:
    read(n*SCALARS[element][1]);return None
   if keep:assert n<=4096
   result=[]
   for _ in range(n):
    x=value(element,keep,depth+1)
    if keep:result.append(x)
   return result if keep else None
  assert read(4)==b'GGUF';version=u32();assert version in [2,3]
  tensors=u64();count=u64();assert count<=65536
  for _ in range(count):
   name=string();assert name not in values and name not in omitted;kind=u32()
   keep=not name.startswith('tokenizer.')
   result=value(kind,keep)
   if keep:values[name]=result
   else:omitted[name]={'value_type':kind,'reason':'Tokenizer payload is unrelated to GDN head/layout/activation contract.'}
  end=f.tell();f.seek(0);header=f.read(end);assert len(header)==end
 after=MODEL.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 receipt=dict(schema=1,passed=True,source=str(Path(__file__).resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_path=str(MODEL),file_bytes=before.st_size,file_mtime_ns=before.st_mtime_ns,gguf_version=version,tensor_count=tensors,metadata_count=count,metadata_prefix_bytes=end,metadata_prefix_sha256=hashlib.sha256(header).hexdigest(),metadata=values,omitted=omitted,tensor_data_read=False,scope='Actual loaded model shard metadata only; full GGUF/tensor identity remains the model-cohort file stat binding. Metadata cannot prove an absent converter permutation or missing HF output-gate configuration. No GPU, process inspection or numerical claim.')
 out.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'receipt':str(out),'prefix_bytes':end,'model_metadata':values},ensure_ascii=False))
if __name__=='__main__':main()
