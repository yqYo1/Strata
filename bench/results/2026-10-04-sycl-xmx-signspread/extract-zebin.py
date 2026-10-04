import struct,sys
from pathlib import Path
data=Path(sys.argv[1]).read_bytes();needle=sys.argv[2].encode();offset=-1
while True:
 offset=data.find(b'\x7fELF',offset+1)
 if offset<0:raise SystemExit('matching device image not found')
 try:
  h=struct.unpack_from('<16sHHIQQQIHHHHHH',data,offset)
  if h[0][4:6]!=b'\x02\x01' or h[2]!=205 or h[11]!=64:continue
  sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+h[6]+i*h[11]) for i in range(h[12])]
  ns=sections[h[13]];names=data[offset+ns[4]:offset+ns[4]+ns[5]]
  found=False
  for s in sections:
   name=names[s[0]:].split(b'\0',1)[0]
   if name==b'.ze_info' and needle in data[offset+s[4]:offset+s[4]+s[5]]:found=True
  if found:
   extent=max([h[6]+h[11]*h[12],h[8]]+[s[4]+s[5] for s in sections if s[1]!=8])
   Path(sys.argv[3]).write_bytes(data[offset:offset+extent]);print('extracted',extent,'bytes');break
 except (struct.error,IndexError):continue
