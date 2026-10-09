"""Archive completed phase-specific decode measurements without model data or API logs."""
from pathlib import Path
import csv
import hashlib
import io
import json
import shutil
import subprocess

b=Path(__file__).parent
rw=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-server-env-v0141-2026-10-09')
parent=rw/'bench/results/2026-10-05-sycl-prefill-scaling/native-expert-copy-v0141-20261009'
target=parent/'decode-repeat'
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
summary_path=b/'native-copy-decode-repeat-v0141-summary-v1.json'
result=json.loads(summary_path.read_text());assert result['passed'] and not result['active']
sequence_path=b/'native-copy-decode-repeat-v0141-comparison-sequence-v1/record.json'
assert digest(sequence_path)==result['sequence_receipt_sha256']
seq=json.loads(sequence_path.read_text());assert seq['passed'] and not seq['active']
assert len(seq['steps'])==18 and all(x['exit_code']==0 and not x['active'] for x in seq['steps'])
assert result['all18_actual_info_geometry_equal']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rw,text=True).strip()=='0ed619e5239bebc87f018cd2f9b8443e37105c72'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=rw,text=True).strip()
assert not target.exists()
staging=b/'native-copy-decode-repeat-v0141-public-archive-stage-v1'
assert not staging.exists();staging.mkdir(mode=0o700)
copied=[]
def copy(src,destination):
    src=Path(src);p=staging/destination;p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,p);assert digest(src)==digest(p)
    copied.append({'file':destination,'source':str(src),'sha256':digest(p),'bytes':p.stat().st_size})
for name in ['run_owned_native_copy_decode_repeat_v0141_v1.py','prepare_native_copy_decode_repeat_v0141_v1.py',
             'run_native_copy_decode_repeat_sequence_v0141_v1.py','summarize_native_copy_decode_repeat_v0141_v1.py',
             'archive_native_copy_decode_repeat_v0141_v1.py','native-copy-decode-repeat-v0141-controller-preparation-v1.json',
             'native-copy-v0141-phase-scope-source-review-v1.json']:
    copy(b/name,name)
copy(summary_path,'summary.json');copy(sequence_path,'sequence/record.json')
diag=b/'owned-native-copy-decode-repeat-v0141-nativeon-diagnostic-r1'
dr=json.loads((diag/'record.json').read_text())
assert dr['healthy'] and dr['math_gate_passed'] and not dr['active'] and not dr['performance_eligible']
assert digest(diag/'record.json')==seq['first_diagnostic_receipt_sha256']
for name in ['record.json','project-messages.txt','protocol.stdout.raw']:
    copy(diag/name,'first-diagnostic/'+name)
private=[{'purpose':'First-use API log, excluded from performance','file':str(diag/'debugger/inferior.stderr'),
          'bytes':dr['engine_log_bytes'],'sha256':dr['engine_log_sha256'],'included_in_git':False},
         {'purpose':'Qualified common32K saved numerical state','file':dr['checkpoint_identity']['file'],
          'bytes':dr['checkpoint_identity']['bytes'],'sha256':dr['checkpoint_identity']['sha256'],'included_in_git':False}]
for step in seq['steps']:
    path=Path(step['receipt']);assert digest(path)==step['receipt_sha256']
    r=json.loads(path.read_text());assert r['healthy'] and r['math_gate_passed'] and not r['active']
    sub=f'block{step["block"]}/{step["mode"]}'
    for name in ['record.json','project-messages.txt','protocol.stdout.raw']:
        copy(path.parent/name,sub+'/'+name)
    for name in ['stdout','stderr']:
        copy(step[name],sub+'/controller.'+name)
(staging/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
table=io.StringIO();writer=csv.writer(table)
writer.writerow(['block','order_position','mode','trial','decode_ms','generated_tokens','mtp_accepted','mtp_offered','resume_tokens','timing_category'])
for step in seq['steps']:
    r=json.loads(Path(step['receipt']).read_text())
    for trial in r['requests']:
        category='fresh32k-first' if trial['phase_kind']=='prefill-prime' else ('restored-warm-excluded' if trial['warmup_decode'] else 'restored-measured')
        writer.writerow([step['block'],step['position'],step['mode'],trial['name'],trial['measurement']['decode_ms'],64,*trial['mtp_counts'],max(trial['resume_tokens']),category])
(staging/'measurements.csv').write_text(table.getvalue())
rows=[]
for mode,label in [('baseline','Qualified baseline869'),('nativeoff','Candidate272, prefill queue OFF'),('nativeon','Candidate272, prefill queue ON')]:
    r=result['summary'][mode];f=result['secondary_fresh32k_first_decode_summary'][mode]
    rows.append(f'| {label} | {r["decode_tok_s"]:.4f} | {r["mean_decode_ms"]:.2f} | {r["process_mean_cv_percent"]:.3f}% | {f["decode_tok_s"]:.4f} |')
comparisons=[]
for key,c in result['comparisons'].items():
    lo,hi=c['speed_change_95_percent_ci']
    comparisons.append(f'- `{key}`: {c["geometric_speed_change_percent"]:+.3f}% speed, paired process-block95% interval [{lo:+.3f}%, {hi:+.3f}%]. {c["interpretation"]}.')
fresh=[]
for key,c in result['secondary_fresh32k_first_decode_comparisons'].items():
    lo,hi=c['speed_change_95_percent_ci']
    fresh.append(f'- `{key}`: {c["geometric_speed_change_percent"]:+.3f}%, interval [{lo:+.3f}%, {hi:+.3f}%].')
readme='''# Separate decode comparison

On2026-10-09 JST, Arc B57010 GiB / Ryzen5 5600X /128 GiB RAM, six independent processes per condition completed twelve measured decode repeats each:72 measured requests per condition,216 total. All output IDs, logprobs and MTP counts match the qualified reference. All18 processes exit normally, with no GPU faults or forced cleanup. The first logged protocol check also exits normally; its timings are excluded.

Prefill and decode are assessed separately. Faster prefill is not a reason to accept slower decode. The earlier24-read comparison has only two independent processes per condition and mixes two prompts with different MTP counts. Its per-read ranges are not an estimate of timing noise or proof of decoder equivalence.

Each process first runs the same fresh32768-token prompt and64-output decode, creating its actual prefill workspace and queue. Before every subsequent decode it restores the same verified32831-token numerical state and sends32832 input tokens for64 outputs. All repeated decodes reuse32831 tokens, execute no PP chunks, and have46 accepted of51 offered MTP drafts. One warm decode is reported separately; twelve later repeats are measured. Restoring and priming times are excluded from these repeated-decode rates. The three conditions run in all six order permutations; every condition occupies each position twice.

| Condition | Repeated decode tokens/s | Mean64-token decode ms | CV of six process means | First decode after fresh32K tokens/s |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Uncertainty uses six paired independent process blocks, a log-latency Student t interval with five degrees of freedom. Twelve repeats improve each process estimate; they are not twelve independent process samples. Positive speed changes mean faster decoding.

'''+ '\n'.join(comparisons)+'''

The first64 output tokens immediately after fresh32K prefill are a separate secondary comparison: six samples per condition,41 accepted of66 offered MTP drafts. These are not pooled with the restored continuation workload.

'''+ '\n'.join(fresh)+'''

An interval including zero does not establish equivalence or rule out a smaller regression. Repetition does not remove physical variance. These results cover one qualified32K coding continuation and the separate first decode after its fresh prefill, not every prompt or longer generation.

The measured engine's decoder source files are byte-identical to baseline869; only prefill.cpp and the common queue header change. The selector is called only during prefill initialization. That source scope does not prove zero runtime effect: the prefill queue/workspace survives into decode and the common header changes the build. Candidate OFF versus baseline checks the common build effect; ON versus OFF checks native queue selection and its retained resources. This experiment does not adopt a decoder implementation or change defaults.

Actual startup geometry matches in all18 processes: context262144, int8 KV resident32768, expert cache128 slots/325 MiB, workers5, MTP4, PCIe fraction0, free VRAM1565 MiB. LP5 scoring is included. Normal timings have no API tracing, GPU profiler, timestamps or new synchronization. Existing unconditional host counters are printed once per request in all conditions. Their CPU/GPU-wait categories are host wall times and can overlap other work.

The raw reused-request receipts inherit a generic `prefill_tok_s` calculation over all input tokens. That value is not a prefill rate because32831 tokens are reused; it is excluded from these summaries and decisions. Tensor data, model files, the619 MB checkpoint and the logged API trace remain private. The prior [full physical262144 lifecycle proof](../full256k/README.md) qualifies this unchanged binary; this timing experiment does not replace it.

[Summary and process-block intervals](summary.json), [all252 requests by category](measurements.csv), [source phase review](native-copy-v0141-phase-scope-source-review-v1.json), [sequence receipt](sequence/record.json).
'''
(staging/'README.md').write_text(readme)
files=[]
for p in sorted(staging.rglob('*')):
    if p.is_file():files.append({'file':str(p.relative_to(staging)),'sha256':digest(p),'bytes':p.stat().st_size})
manifest={'files':files,'copied_source_identities':copied,'data_included':False}
(staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
target.mkdir();shutil.copytree(staging,target,dirs_exist_ok=True)
for item in files:assert digest(target/item['file'])==item['sha256']
old=(parent/'README.md').read_text()
old=old.replace('Decode variation does not establish a small speed change; logprob scoring is included in these timings.',
                'The two-process-per-condition decode comparison does not establish equivalence; its mixed-prompt ranges are not a noise estimate. Logprob scoring is included in these timings.')
old+='\n[Separate repeated decode comparison](decode-repeat/README.md) uses six independent process blocks and72 fixed-state measured decodes per condition. Prefill and decode adoption are judged separately; a confidence interval including zero is not proof of equivalence.\n'
(parent/'README.md').write_text(old)
proof={'passed':True,'active':False,'source_sequence_receipt_sha256':digest(sequence_path),'source_summary_sha256':digest(summary_path),
       'destination':str(target),'manifest_sha256':digest(target/'manifest.json'),'files':len(files)+1,'engine_source_changed':False,'defaults_changed':False}
(b/'native-copy-decode-repeat-v0141-public-archive-proof-v1.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
