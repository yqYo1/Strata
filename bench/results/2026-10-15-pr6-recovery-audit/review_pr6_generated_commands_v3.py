#!/usr/bin/env python3
"""Review generated recipes and private header; do not execute compile/link commands."""
from pathlib import Path
import hashlib,json,shlex,subprocess
A=Path(__file__).resolve().parent

def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
 dst=A/'own-pr6-generated-command-review-v3.json';assert not dst.exists()
 o=json.loads((A/'own-pr6-harness-configure-v2.json').read_text());n=json.loads((A/'own-pr6-harness-configure-v3.json').read_text())
 assert o['passed'] and n['passed'] and n['ownership_closed']
 old=Path(o['build_path']);new=Path(n['build_path']);C=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-pr6-prefill-restore')
 oc=json.loads((old/'compile_commands.json').read_text());nc=json.loads((new/'compile_commands.json').read_text())
 om={r['file']:r for r in oc};nm={r['file']:r for r in nc}
 probe=str(C/'validation/pr6-restore/harness/provider_probe.cpp')
 assert len(oc)==111 and len(nc)==112 and set(nm)-set(om)=={probe} and not set(om)-set(nm)
 def normalize(row,root):return [v.replace(str(root),'BUILD') for v in shlex.split(row['command'])]
 for f in om:assert normalize(om[f],old)==normalize(nm[f],new),f
 args=normalize(nm[probe],new)
 for flag in ['-fsycl','-std=c++20','-ffp-model=precise','-fsycl-device-code-split=per_kernel','-include']:
  assert flag in args,flag
 assert '-fno-sycl' not in args and args[args.index('-include')+1]=='BUILD/pr6-checked-dpct.hpp'
 assert '-fsycl-default-sub-group-size=32' in args,args
 header=new/'pr6-checked-dpct.hpp';assert pin(header)['sha256']=='e418eb84b52fe8516739b4559c612c72311236f344b88a970adcb5e4c5bf0c1f'
 proof=json.loads((new/'pr6-llvm-shuffle64-proof.json').read_text());assert proof['checked_header_sha256']==pin(header)['sha256']
 assert proof['generated_util_body_sha256']=='cb93a6d8ecb064bf8d693fad0c9c5e64ce753d6f4b126f5e086969d9b1228609'
 links={}
 for label,build,exe in [('old_harness',old,'pr6_prefill_api'),('new_harness',new,'pr6_prefill_api'),('provider',new,'pr6_provider_probe')]:
  argv=['/usr/bin/ninja','-C',str(build),'-t','commands',exe]
  p=subprocess.run(argv,env=n['commands'][0]['environment'],capture_output=True,text=True,timeout=15)
  assert p.returncode==0,(p.returncode,p.stderr);assert len(p.stdout.encode())<8<<20
  line=p.stdout.splitlines()[-1];links[label]=dict(argv=argv,exit_code=p.returncode,command=line,full_output_sha256=hashlib.sha256(p.stdout.encode()).hexdigest(),stderr=p.stderr)
 assert links['old_harness']['command'].replace(str(old),'BUILD')==links['new_harness']['command'].replace(str(new),'BUILD')
 provider_link=links['provider']['command']
 assert '-fsycl' in provider_link
 assert not any(v in provider_link for v in ['libstrata_','libggml','mkl_sycl','mkl_intel','-fno-sycl'])
 rec=dict(schema='strata-pr6-generated-command-review-v3',passed=True,complete=True,GPU_executed=False,
  configure_receipts=[pin(A/'own-pr6-harness-configure-v2.json'),pin(A/'own-pr6-harness-configure-v3.json')],
  old_compile_count=len(oc),new_compile_count=len(nc),all_old111_commands_byte_equal_after_build_root_normalization=True,
  complete_historical84_TUs_preserved=True,provider_compile_argv=args,provider_only_new_TU=True,final_link_recipes=links,
  existing_final_harness_link_unchanged=True,provider_no_historical_libraries_linked=True,
  checked_header=pin(header),shuffle_proof=proof,
  limits=['Recipe and deterministic private-header proof only, no compiler/runtime execution.',
          'Same arithmetic and source closure do not prove emitted kernel equivalence or historical performance.',
          'OFF-mode header and signed/unsigned native64bit shuffled values still need separate qualification.'])
 dst.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(dict(receipt=pin(dst),old111_unchanged=True,new_provider=True,GPU_executed=False)))
if __name__=='__main__':main()
