from pathlib import Path
import json,re,hashlib,difflib
base=Path(__file__).parent;root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');out=base/'upstream-e8ca-refresh-20261007'
mapping=json.loads((base/'observed-cooperative-source-targets.json').read_text());targets={}
for row in mapping['rows']:
 c=row['source_candidates'][0];targets.setdefault(c['path'],set()).update(c['classes'])
targets['sycl/src/prefill/kernels.dp.cpp'].update(['gdn_out_norm_keyhead_kernel_43e92c','to_f16_kernel_c8575f'])
targets['sycl/src/kernels/cuda/verify_kernels.dp.cpp'].add('add_streams_broadcast_kernel_7de8f5')
targets['sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp'].update(['attn_chunk_kernel_5c04f7','attn_merge_kernel_338175'])
# Upstream re-migration renamed these launch classes and split type specializations.
renames={'sycl/src/prefill/kernels.dp.cpp':('route_kernel_4f80f3',['route_kernel_ceb296','route_kernel_4c3133']),
'sycl/src/kernels/cuda/iq_kernels.dp.cpp':('dequant_flat_kernel_d891c8',['dequant_flat_kernel_c7cec2']),
'sycl/src/kernels/cuda/verify_kernels.dp.cpp':('gather_rows_kernel_7960b2',['gather_rows_kernel_b3ae5a','gather_rows_kernel_f256ac','gather_rows_kernel_c5b171'])}
for path,(old,new) in renames.items():
 targets[path].remove(old);targets[path].update(new)
flag='sycl::ext::oneapi::experimental::use_root_sync';rows=[]
for relative,names in targets.items():
 p=root/relative;s=p.read_text();decls=list(re.finditer(r'(?:const\s+)?auto\s+(\w+)\s*=\s*sycl::ext::oneapi::experimental::properties\s*\{([^{}]*)\}',s));spans={}
 assert not re.search(r'get_root_group|root_group_barrier',s)
 for name in sorted(names):
  matches=list(re.finditer(r'\bclass\s+'+re.escape(name)+r'\b',s));assert matches,(relative,name)
  for match in matches:
   decl=[d for d in decls if d.end()<match.start()][-1];between=s[decl.end():match.start()]
   assert len(re.findall(r'parallel_for\s*<',between))==1 and len(between)<2000,(relative,name)
   assert decl.group(2).count(flag)==1
   at=decl.start(2)+decl.group(2).index(flag);spans.setdefault(at,[]).append(name)
 t=s
 for at in sorted(spans,reverse=True):t=t[:at]+t[at+len(flag):]
 restored=t
 for offset,at in reversed(list(enumerate(sorted(spans)))):
  pos=at-offset*len(flag);restored=restored[:pos]+flag+restored[pos:]
 assert restored==s
 p.write_text(t)
 delta=''.join(difflib.unified_diff(s.splitlines(keepends=True),t.splitlines(keepends=True),fromfile=relative,tofile=relative))
 d=out/'regular-launch-review'/relative;d.parent.mkdir(parents=True,exist_ok=True);d.with_suffix(d.suffix+'.patch').write_text(delta)
 rows.append(dict(path=relative,before_sha256=hashlib.sha256(s.encode()).hexdigest(),after_sha256=hashlib.sha256(t.encode()).hexdigest(),changed_classes=sorted(names),removed_properties=[dict(line=s.count('\n',0,at)+1,classes=spans[at]) for at in sorted(spans)]))
record=dict(scope='Selective ordinary launch properties on reviewed independent element/row/local-reduction kernels; no kernel-body, range, name, subgroup, arithmetic or queue-order edits. Persistent/global coordination kernels retain their properties. Prior zero-cooperative long jobs still stalled; this is not a hang-resolution claim.',sources=rows,removed_property_count=sum(len(r['removed_properties']) for r in rows))
(out/'regular-launch-review.json').write_text(json.dumps(record,indent=2)+'\n');print('reviewed regular launch properties',record['removed_property_count'])
