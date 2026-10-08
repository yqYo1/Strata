from pathlib import Path
import json,hashlib,shutil
b=Path(__file__).parent;r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');out=b/'upstream-e8ca-refresh-20261007';changes=[]
p=r/'sycl/src/core/verify.cpp';s=p.read_text();a=s.index('bool one_token_self_commit() {');z=s.index('\n// True when the GPU',a);old=s[a:z];new='''// The SYCL conv kernel does not implement the fused history commit. Keep
// the separate commit graph, including one-token windows at the context end.
bool one_token_self_commit() { return false; }
''';t=s[:a]+new+s[z:];p.write_text(t);changes.append(dict(path=str(p.relative_to(r)),before_sha256=hashlib.sha256(s.encode()).hexdigest(),after_sha256=hashlib.sha256(t.encode()).hexdigest(),old=old,new=new))
p=r/'sycl/include/strata/sycl_execution_policy.hpp';s=p.read_text();old='for (const char* name : {"STRATA_SH_STREAM", "STRATA_MTP_SHARED_BRANCH"})';new='for (const char* name : {"STRATA_SH_STREAM", "STRATA_MTP_SHARED_BRANCH", "STRATA_ONE_TOKEN_COMMIT"})';assert s.count(old)==1;t=s.replace(old,new).replace('=1 needs SYCL cross-queue graph validation first; use =0','=1 is unavailable in the validated SYCL execution path; use =0');p.write_text(t);changes.append(dict(path=str(p.relative_to(r)),before_sha256=hashlib.sha256(s.encode()).hexdigest(),after_sha256=hashlib.sha256(t.encode()).hexdigest(),old=old,new=new))
(out/'one-token-commit-reconciliation.json').write_text(json.dumps(changes,indent=2)+'\n')
d=out/'build-before-gpu-refusal';d.mkdir()
for n in ['build-record.json','configure.stdout','configure.stderr','build.stdout','build.stderr','binary-receipt.json']:shutil.copy2(out/n,d/n)
s=(b/'run_owned_upstream_e8ca_refresh_short.py').read_text().replace("'owned-upstream-e8ca-refresh-short'","'owned-upstream-e8ca-refresh-short-retry'").replace("'post-observed-regular-launch-2048-retained-health/record.json'","'post-upstream-e8ca-short-api-refusal-health/record.json'")
(b/'run_owned_upstream_e8ca_refresh_short_retry.py').write_text(s)
print('separate one-token commit restored; initial model refusal/build preserved')
