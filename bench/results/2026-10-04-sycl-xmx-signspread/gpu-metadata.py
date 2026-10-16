import struct,re,json,sys
from pathlib import Path
data=Path(sys.argv[1]).read_bytes();prefix=Path(sys.argv[2]);offset=-1;images=0;docs=[];kernels=[]
while True:
 offset=data.find(b'\x7fELF',offset+1)
 if offset<0:break
 try:
  h=struct.unpack_from('<16sHHIQQQIHHHHHH',data,offset)
  if h[0][4:6]!=b'\x02\x01' or h[2]!=205 or h[11]!=64:continue
  sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+h[6]+i*h[11]) for i in range(h[12])]
  ns=sections[h[13]];names=data[offset+ns[4]:offset+ns[4]+ns[5]]
  for s in sections:
   name=names[s[0]:].split(b'\0',1)[0].decode()
   if name!='.ze_info':continue
   ze=data[offset+s[4]:offset+s[4]+s[5]].rstrip(b'\0').decode();docs.append(ze);images+=1
   ze=re.split(r'\n[a-z_]+:',ze.split('\nkernels:',1)[1],maxsplit=1)[0]
   for part in ze.split('\n  - name:')[1:]:
    name=part.splitlines()[0].strip();reg=re.search(r'\n      grf_count:\s+(\d+)',part)
    k=dict(name=name,grf=int(reg[1]) if reg else None,has_scratch='per_thread_memory_buffers:' in part)
    match=re.search(r'launchILi(\d+)ELi(\d+)ELb([01])ELb([01])E',name)
    if match:k.update(type=int(match[1]),tile=int(match[2]),exact=match[3]=='1',packed=match[4]=='1')
    kernels.append(k)
 except (struct.error,IndexError,UnicodeError):continue
prefix.with_suffix('.ze-info.txt').write_text('\n'.join(docs))
xmx=[k for k in kernels if 'type' in k]
result=dict(executable=sys.argv[1],images=images,kernels=len(kernels),unique_names=len({k['name'] for k in kernels}),xmx=xmx)
prefix.with_suffix('.registers.json').write_text(json.dumps(result,indent=2)+'\n')
print('images',images,'kernels',len(kernels),'XMX',len(xmx),'GRFs',sorted({k['grf'] for k in xmx}),'XMX scratch',sum(k['has_scratch'] for k in xmx))
