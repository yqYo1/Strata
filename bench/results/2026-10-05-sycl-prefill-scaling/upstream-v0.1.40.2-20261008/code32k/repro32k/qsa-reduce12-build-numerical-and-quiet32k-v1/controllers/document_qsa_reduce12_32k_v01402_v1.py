"""Publish the actual one-object build, numerical gate and sixteen complete quiet reads."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil, subprocess

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root/'bench/results/2026-10-05-sycl-prefill-scaling'
upstream = parent/'upstream-v0.1.40.2-20261008'
code = upstream/'code32k'
repro = code/'repro32k'
out = repro/'qsa-reduce12-build-numerical-and-quiet32k-v1'
s = json.loads((out/'summary.json').read_text())
assert s['passed'] and s['fresh_full_reads'] == 16 and s['candidate_engine_compiled']
assert s['minimum_performance_input_tokens'] == 32768 and not s['physical256k_sequence_passed'] and not s['adopted']
assert s['qsa_initial_four_fresh32k_and_actual_disk_continuation_passed']
assert 1.3 < s['effects']['subsequent_full_reads']['prefill_tok_s_change_percent'] < 1.4
assert s['effects']['subsequent_full_reads']['decode_tok_s_change_percent'] < 0
shutil.copy2(__file__, out/'controllers'/Path(__file__).name)
for directory in [repro, code, upstream, parent]:
    p = directory/'README.md'
    text = p.read_text()
    link = str((out/'README.md').relative_to(directory))
    assert link not in text
    text += '\nThe [later QSA twelve-head build, numerical gate and matched 32K comparison]('+link+') replaces one kernel archive member on DD5. Four fresh 32768-token numerical reads and actual disk continuation passed, then all sixteen quiet A/B/A/B reads matched IDs/logprobs/MTP and exited normally. Initial process PP was 392.213 versus 395.194 tok/s; later full-read PP was 427.237 versus 432.984 tok/s (+1.345%). Later TG was 16.730 versus 16.410 tok/s (-1.916% in this small sample); no decode gain is claimed. Logged numerical durations are excluded. The QSA binary remains unadopted pending its own complete physical 256K lifecycle, which cannot be inherited from DD5. Source-only HC/HIP knob applicability checks do not enable unsupported SYCL flags.\n'
    p.write_text(text)
def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
manifests = []
for directory in [out, repro, code, upstream, parent]:
    p = directory/'manifest.json'
    d = json.loads(p.read_text()) if p.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_commit':s['source_commit']}
    d.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), revision_reason='Actual QSA one-object build, corrected logged32K numerical gate, and16 quiet full32768 comparisons. Later PP +1.345%; no decode gain, physical256K qualification or adoption claim.')
    files = {str(x.relative_to(directory)):x for x in sorted(directory.rglob('*')) if x.is_file() and x != p}
    d['files'] = {name:{'bytes':x.stat().st_size,'sha256':digest(x)} for name,x in files.items()}
    p.write_text(json.dumps(d,indent=2)+'\n')
    assert {str(x.relative_to(directory)) for x in directory.rglob('*') if x.is_file() and x != p} == set(d['files'])
    assert all(x.stat().st_size == d['files'][n]['bytes'] and digest(x) == d['files'][n]['sha256'] for n,x in files.items())
    manifests.append({'path':str(p),'files':len(files),'sha256':digest(p)})
    print(p.relative_to(root),len(files),flush=True)
for p in out.rglob('*.py'): ast.parse(p.read_text(),filename=str(p))
for p in out.rglob('*.json'): json.loads(p.read_text())
validation = {'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'summary_sha256':digest(out/'summary.json'),'all_archived_controllers_ast_passed':True,
              'exact_file_sets_sizes_and_hashes_validated':True,'minimum_performance_input_tokens':32768,
              'qsa_gpu_numerical_passed':True,'quiet16_passed':True,'qsa_full256k_passed':False,'adopted':False,'manifests':manifests}
p = base/'qsa-reduce12-32k-v01402-archive-validation-v1.json'
assert not p.exists()
p.write_text(json.dumps(validation,indent=2)+'\n')
