"""Add the cold-decode adoption decision and existing-host-counter analysis."""
from pathlib import Path
import hashlib
import json
import shutil

b=Path(__file__).parent
target=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-server-env-v0141-2026-10-09/bench/results/2026-10-05-sycl-prefill-scaling/native-expert-copy-v0141-20261009/decode-repeat')
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
initial=target/'manifest.json'
assert digest(initial)=='8710accd2c229bc2f817d976bd925eaf20d803c313249066c98ac7d15c8c4338'
preserved=b/'native-copy-decode-repeat-v0141-initial-public-manifest-v1.json'
assert not preserved.exists();shutil.copyfile(initial,preserved)
manifest=json.loads(initial.read_text())
for item in manifest['files']:assert digest(target/item['file'])==item['sha256']
summary=json.loads((target/'summary.json').read_text())
cold=summary['secondary_fresh32k_first_decode_comparisons']['nativeon_vs_baseline']
assert cold['speed_change_95_percent_ci'][1]<0
copy=b/'native-copy-decode-repeat-v0141-first-decode-host-counter-summary-v1.json'
shutil.copyfile(copy,target/copy.name)
shutil.copyfile(__file__,target/Path(__file__).name)
readme=target/'README.md';text=readme.read_text()
marker='An interval including zero does not establish equivalence or rule out a smaller regression.'
addition='''The fresh32K result is a measured regression in this workload: all six ON-versus-baseline paired speed changes are negative, and its95% interval is entirely below zero. ON versus OFF also has an entirely negative interval. The candidate as a whole is held back. The qualified decoder remains the reference; faster prefill is not used to offset this loss. The opt-in prefill work stays a candidate until the transition into decode is isolated and measured again.

Existing host counters put the first-decode CPU expert work near123.78 ms/window for baseline and125.22 for ON, and GPU-reach waits near2.76 and2.96. The input-stage category is3.23 versus10.10 ms/window. These are diagnostic counters, not physical GPU busy times. Source review finds that `Verifier::run` counts the PLE gather/copy interval both explicitly and inside the enclosing input-stage interval; therefore its stage number cannot be treated as a time fraction. Graph capture occurs before that stage clock starts, so the stage increase is not a direct graph-capture measurement. The data points to checking input staging and phase resources; it does not yet identify the cause. See [host-counter means](native-copy-decode-repeat-v0141-first-decode-host-counter-summary-v1.json).

'''
assert text.count(marker)==1;readme.write_text(text.replace(marker,addition+marker))
parent=target.parent/'README.md';parent_text=parent.read_text()
parent_text+='\nThe repeated fresh32K comparison finds an ON-versus-baseline first-decode regression of7.985% (95% speed-change interval −12.806% to −2.897%). The candidate is held back despite its prefill gain; phase separation is required before adopting it.\n'
parent.write_text(parent_text)
files=[]
for path in sorted(target.rglob('*')):
    if path.is_file() and path.name!='manifest.json':files.append({'file':str(path.relative_to(target)),'sha256':digest(path),'bytes':path.stat().st_size})
manifest['files']=files
manifest['supplement_sources']=[{'source':str(copy),'file':copy.name,'sha256':digest(copy)},
                                {'source':str(Path(__file__)),'file':Path(__file__).name,'sha256':digest(__file__)}]
manifest['initial_manifest_sha256']=digest(preserved)
initial.write_text(json.dumps(manifest,indent=2)+'\n')
proof={'passed':True,'active':False,'initial_archive_proof':str(b/'native-copy-decode-repeat-v0141-public-archive-proof-v1.json'),
       'preserved_initial_manifest':str(preserved),'final_manifest_sha256':digest(initial),'files':len(files)+1,
       'cold_decode_regression_confirmed':True,'candidate_adopted':False,'decoder_kept_qualified':True,
       'engine_source_changed':False,'defaults_changed':False,'destination':str(target)}
(b/'native-copy-decode-repeat-v0141-public-archive-proof-v2.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
