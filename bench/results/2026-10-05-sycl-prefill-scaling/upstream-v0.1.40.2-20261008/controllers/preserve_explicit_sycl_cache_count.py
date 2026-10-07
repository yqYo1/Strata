from pathlib import Path
import json,hashlib,shutil
b=Path(__file__).parent;r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');out=b/'upstream-e8ca-refresh-20261007';p=r/'sycl/src/program/generate.cpp';s=p.read_text()
old='    if (native_pack && o.expert_cache > 0 && o.expert_cache_per_layer) {'
new='''    // Explicit SYCL counts retain the same CPU/GPU expert placement as before.
    // The new mixed-size expansion is used when the caller asks for auto sizing.
    if (native_pack && o.expert_cache > 0 && o.expert_cache_per_layer && auto_cache) {''';assert s.count(old)==1;t=s.replace(old,new);p.write_text(t)
(out/'explicit-cache-count-reconciliation.json').write_text(json.dumps(dict(path=str(p.relative_to(r)),before_sha256=hashlib.sha256(s.encode()).hexdigest(),after_sha256=hashlib.sha256(t.encode()).hexdigest(),old=old,new=new,reason='Matched numerical comparisons require identical residency; upstream #369 expands explicit N-largest-blob budget to a different number of per-layer slots. Keep existing explicit count semantics on SYCL, retain new auto sizing.'),indent=2)+'\n')
d=out/'build-before-cache-contract';d.mkdir()
for n in ['build-record.json','configure.stdout','configure.stderr','build.stdout','build.stderr','binary-receipt.json']:shutil.copy2(out/n,d/n)
s=(b/'probe_after_upstream_e8ca_short_unbatched.py').read_text().replace('owned-upstream-e8ca-refresh-short-unbatched/record.json','owned-upstream-e8ca-refresh-short-legacy-quant-kv/record.json').replace('post-upstream-e8ca-short-unbatched-health','post-upstream-e8ca-short-legacy-quant-kv-health');(b/'probe_after_upstream_e8ca_short_legacy_quant_kv.py').write_text(s)
s=(b/'run_owned_upstream_e8ca_refresh_short_retry.py').read_text().replace('owned-upstream-e8ca-refresh-short-retry','owned-upstream-e8ca-refresh-short-matched-cache').replace('post-upstream-e8ca-short-api-refusal-health/record.json','post-upstream-e8ca-short-legacy-quant-kv-health/record.json');(b/'run_owned_upstream_e8ca_refresh_short_matched_cache.py').write_text(s)
print('explicit cache count restored')
