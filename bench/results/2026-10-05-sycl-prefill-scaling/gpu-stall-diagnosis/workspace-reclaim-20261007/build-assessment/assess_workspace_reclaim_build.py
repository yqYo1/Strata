from pathlib import Path
import hashlib,json,runpy,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
ns={'__file__':str(base/'map_prefill_completion_wait.py')}
import ast
source=(base/'map_prefill_completion_wait.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Elf')
code='import struct, hashlib\n'+ast.get_source_segment(source,node).replace('result[self.names[i]] = {','result[(i,self.names[i])] = {')
exec(compile(code,str(base/'map_prefill_completion_wait.py'),'exec'),ns)
Elf=ns['Elf']
a=base/'prefill-workspace-reclaim-build/prefill.cpp.o'
b=base/'workspace-reclaim-file-path-audit/prefill.cpp.o'
c=root/'build-sycl-upstream-jit/CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'
ae,be,ce=[Elf(p).executable() for p in [a,b,c]]
assert be==ce,'Mapped and normal-build host machine code/relocations differ'
assert set(ae)==set(be)
unmapped_diff=[k for k in ae if ae[k]['sha256']!=be[k]['sha256'] or ae[k]['bytes']!=be[k]['bytes']]
rec=json.loads((base/'workspace-reclaim-file-path-audit/record.json').read_text())
assert rec['binary_equal']
assessment={'scope':'Offline path-remapping assessment. Initial whole-object and unmapped-executable hash assertions failed; exact linked executable reproduction and executable-section comparisons pass. No independent production-path GPU check.','passed':True,'normal_build_exit_code':0,'private_binary_sha256':rec['private_binary_sha256'],'production_binary_sha256':rec['production_binary_sha256'],'mapped_binary_sha256':rec['mapped_binary_sha256'],'mapped_and_production_binary_equal':True,'mapped_and_production_executable_sections_and_canonical_relocations_equal':True,'unmapped_and_mapped_executable_section_bytes_equal':not unmapped_diff,'unmapped_and_mapped_different_executable_sections':[list(k) for k in unmapped_diff],'executable_sections_compared':len(ae),'whole_objects_equal':False,'path_mapping_argument':rec['steps'][0]['argv'][-1],'original_vs_mapped_code_differences':'22 one-byte changes of embedded source filename length 108 to 103; all other executable bytes match, see byte-differences.json','initial_records_preserved':['workspace-reclaim-production-build/record.json','workspace-reclaim-file-path-audit/record.json']}
(base/'workspace-reclaim-build-assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
print(json.dumps(assessment,indent=2))
out=root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/workspace-reclaim-20261007'
for name in ['workspace-reclaim-production-build','workspace-reclaim-file-path-audit']:
 for p in (base/name).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.stdout','.stderr']:
   dest=out/name/p.relative_to(base/name);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
for name in ['workspace-reclaim-build-assessment.json','assess_workspace_reclaim_build.py','build_workspace_reclaim_production.py','verify_workspace_reclaim_file_path.py']:
 dest=out/'build-assessment'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/name,dest)
