from pathlib import Path
import subprocess,hashlib,json,re,difflib
out=Path(__file__).resolve().parent;previous=out.parent/'prefill-canonical-dense-bank-v2'
base='5942f1853f006b0df45e15fd4f1c60bc37b7b3b3'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
patch=(previous/'source.diff').read_text();assert sha(patch)=='a4a8258bc087a8e79fbce233866385f36dd6cd9941c1f0a7498acffe77ed77cc'
lines=patch.splitlines(True);i=0;files={};orig={};ledger=[]
while i<len(lines):
 assert lines[i].startswith('--- '),lines[i]
 oldpath=lines[i][4:].strip();path=lines[i+1][6:].strip();i+=2
 source='' if oldpath=='/dev/null' else subprocess.check_output(['git','show',base+':'+path]).decode();orig[path]=source
 while i<len(lines) and lines[i].startswith('@@'):
  header=lines[i].strip();i+=1;old=[];new=[]
  while i<len(lines) and not lines[i].startswith(('@@','--- ')):
   line=lines[i];i+=1
   if line.startswith((' ','-')):old.append(line[1:])
   if line.startswith((' ','+')):new.append(line[1:])
  a,b=''.join(old),''.join(new)
  if a:
   assert source.count(a)==1,(path,header,source.count(a))
   source=source.replace(a,b,1)
  else:assert not source;source=b
  ledger.append({'path':path,'old_hunk':header,'old_context_sha256':sha(a),'new_context_sha256':sha(b),'exact_unique_context':True})
 files[path]=source
result='';pins={}
for path in sorted(files):
 a,b=orig[path],files[path]
 assert not re.search(r'DecodeService|DecodeRouteLedger|decode_service|cpu_gu_reuse|CpuGuReuse|MistVVK|XeStrata',b),path
 result+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+path if a else '/dev/null',tofile='b/'+path))
 pins[path]={'base_sha256':sha(a) if a else None,'proposed_sha256':sha(b)}
 # Main is the sole named consumer; copies are temporary, clean-base only.
 q=out/'main-review-consumer'/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(b)
(out/'source.diff').write_text(result)
(out/'proposal-manifest.json').write_text(json.dumps({'immutable_base':base,'source_diff_sha256':sha(result),'parent_diff_sha256':sha(patch),'status':'source-only-unapplied-uncompiled-untested','files':pins},indent=2)+'\n')
(out/'hunk-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(sha(result));print('files',len(files),'hunks',len(ledger))
