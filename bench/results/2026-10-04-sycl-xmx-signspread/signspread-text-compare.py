import struct,hashlib,re,json
from pathlib import Path
def kernels(path):
 data=Path(path).read_bytes();o=-1;result={}
 while True:
  o=data.find(b'\x7fELF',o+1)
  if o<0:break
  try:
   h=struct.unpack_from('<16sHHIQQQIHHHHHH',data,o)
   if h[0][4:6]!=b'\x02\x01' or h[2]!=205:continue
   sh=[struct.unpack_from('<IIQQQQIIQQ',data,o+h[6]+i*h[11]) for i in range(h[12])]
   ns=sh[h[13]];names=data[o+ns[4]:o+ns[4]+ns[5]]
   for s in sh:
    name=names[s[0]:].split(b'\0',1)[0].decode()
    if name.startswith('.text.'):
     result[name]=hashlib.sha256(data[o+s[4]:o+s[4]+s[5]]).hexdigest()
  except (struct.error,IndexError,UnicodeError):continue
 return result
old=kernels('/tmp/strata-sycl-goal-signspread-baseline-aot');new=kernels('build-sycl-aot/strata')
result={'old_count':len(old),'new_count':len(new),'same_names':old.keys()==new.keys(),'changed':[n for n in old if old[n]!=new.get(n)]}
Path('/tmp/strata-sycl-goal-signspread-text-compare.json').write_text(json.dumps(result,indent=2)+'\n')
print('old/new kernels',len(old),len(new),'changed',len(result['changed']))
for n in result['changed']:
 m=re.search(r'launchILi(\d+)ELi(\d+)ELb([01])ELb([01])E',n)
 if m:print('type/tile/exact/packed',m.groups())
 else:print('other',n[:150])
