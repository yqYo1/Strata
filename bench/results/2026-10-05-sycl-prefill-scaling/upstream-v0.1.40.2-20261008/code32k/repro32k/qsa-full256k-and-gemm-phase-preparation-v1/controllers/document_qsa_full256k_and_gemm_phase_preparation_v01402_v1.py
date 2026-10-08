"""Document qualified QSA lifecycle and refresh exact archive manifests."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil, subprocess

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling'
upstream=parent/'upstream-v0.1.40.2-20261008';code=upstream/'code32k';repro=code/'repro32k'
out=repro/'qsa-full256k-and-gemm-phase-preparation-v1'
s=json.loads((out/'summary.json').read_text())
assert s['passed'] and s['all_saved_state_and_kv_bytes_equal_qualified_dd5']
assert not s['gemm_engine_compiled'] and not s['existing_compute_phase_profile_executed']
shutil.copy2(Path(__file__),out/'controllers'/Path(__file__).name)
updates=[(repro/'README.md','qsa-full256k-and-gemm-phase-preparation-v1/README.md'),
         (code/'README.md','repro32k/qsa-full256k-and-gemm-phase-preparation-v1/README.md'),
         (upstream/'README.md','code32k/repro32k/qsa-full256k-and-gemm-phase-preparation-v1/README.md'),
         (parent/'README.md','upstream-v0.1.40.2-20261008/code32k/repro32k/qsa-full256k-and-gemm-phase-preparation-v1/README.md'),
         (repro/'qsa-reduce12-build-numerical-and-quiet32k-v1/README.md','../qsa-full256k-and-gemm-phase-preparation-v1/README.md'),
         (repro/'uniform-full256k-and-qsa-source-v1/README.md','../qsa-full256k-and-gemm-phase-preparation-v1/README.md')]
for p,link in updates:
    text=p.read_text();assert 'qsa-full256k-and-gemm-phase-preparation-v1/README.md' not in text
    text+='\nThe [later complete QSA physical 256K sequence]('+link+') passed two fresh full reads through cell 262143, all 13 KV layers with 262144 cells, every saved state/KV tensor byte against DD5 and on repetition and actual disk restoration, a clipped two-token tail, both capacity refusals and a later fresh 32768-token input. All twelve request gates, six session operations and normal owned exit passed without a new GPU fault. Diagnostic durations are excluded. Earlier statements about uncompiled or capacity-pending QSA describe their respective archive boundaries. The later archive also preserves separate source-only GEMM dispatch and unexecuted compute-phase profiling preparations, including CPU rejection checks; neither inherits QSA qualification. Every performance comparison uses at least 32768 input tokens with first/later full reads separate.\n'
    p.write_text(text)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
directories=[out,repro/'qsa-reduce12-build-numerical-and-quiet32k-v1',repro/'uniform-full256k-and-qsa-source-v1',repro,code,upstream,parent]
manifests=[]
for directory in directories:
    p=directory/'manifest.json'
    d=json.loads(p.read_text()) if p.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':source_commit}
    d.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Complete QSA physical256K lifecycle; separately preserve uncompiled GEMM and unexecuted phase profiling preparation. No new speed or stall-cause claim.')
    files={str(p.relative_to(directory)):p for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'manifest.json'}
    d['files']={n:{'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in files.items()}
    p.write_text(json.dumps(d,indent=2)+'\n')
    assert set(d['files'])=={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
    for n,f in files.items():assert f.stat().st_size==d['files'][n]['bytes'] and sha(f)==d['files'][n]['sha256']
    manifests.append({'path':str(p),'files':len(files),'sha256':sha(p)})
for p in (out/'controllers').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for p in out.rglob('*.json'):json.loads(p.read_text())
authored=[p.relative_to(root).as_posix() for p in root.rglob('README.md') if p in [x[0] for x in updates]+[out/'README.md']]
subprocess.run(['git','diff','--check','--',*authored],cwd=root,check=True)
validation={'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'controller_sha256':sha(__file__),'summary_sha256':sha(out/'summary.json'),'exact_file_sets_sizes_hashes_validated':True,
            'all_archived_controllers_AST_passed':True,'all_archived_JSON_parsed':True,'authored_README_whitespace_passed':True,
            'minimum_performance_input_tokens':32768,'QSA_full256k_passed':True,'gemm_engine_compiled':False,
            'existing_compute_phase_profile_executed':False,'adopted':False,'manifests':manifests}
p=base/'qsa-full256k-and-gemm-phase-preparation-v01402-archive-validation-v1.json'
assert not p.exists();p.write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'passed':True,'manifests':[{'files':d['files'],'sha256':d['sha256']} for d in manifests],'archive_controllers_AST_and_JSON_passed':True}))
