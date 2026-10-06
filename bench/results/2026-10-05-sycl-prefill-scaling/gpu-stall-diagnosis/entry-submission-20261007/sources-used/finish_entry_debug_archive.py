"""Add build receipts and verify every archived source/file digest."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/entry-submission-20261007'
def copy(source, target):
    target=out/target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
for name in ['prefill-layer-trace-build','prefill-layer-trace-build-system-archiver','prefill-layer-trace-build-unset-library-path','prefill-layer-trace-build-mkl-path']:
    for p in (base/name).iterdir():
        if p.is_file():copy(p,Path(name)/p.name)
for name in ['build_prefill_layer_trace.py','build_prefill_layer_trace_system_archiver.py','build_prefill_layer_trace_unset_library_path.py','build_prefill_layer_trace_mkl_path.py']:
    copy(base/name,Path('sources-used')/name)
copy(base/'submit-stall-sources/loaded-module-build-id.json',Path('upstream-source-receipts/loaded-module-build-id.json'))
copy(root/'sycl/src/prefill/prefill.cpp',Path('sources-used/prefill.cpp'))
copy(root/'sycl/src/program/generate.cpp',Path('sources-used/generate.cpp'))
old_commit='fbd16bf8cd647a3842cfe7ed8ad5d906dd0cb8b6'
old=subprocess.run(['/usr/bin/git','show',old_commit+':sycl/src/prefill/prefill.cpp'],cwd=root,capture_output=True,check=True).stdout
(out/'sources-used/prefill-before-layer-trace.cpp').write_bytes(old)
copy(Path(__file__),Path('sources-used')/Path(__file__).name)

def info(p):
    data=p.read_bytes();return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
files={str(p.relative_to(out)):info(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
by_digest={}
for name,d in files.items():by_digest.setdefault(d['sha256'],[]).append(name)
source_checks=[]
for p in out.rglob('record.json'):
    d=json.loads(p.read_text())
    entries={}
    if isinstance(d.get('source_sha256'),dict):entries.update(d['source_sha256'])
    elif isinstance(d.get('source_sha256'),str):entries['source_sha256']=d['source_sha256']
    for k in ['helper_sha256','controller_sha256']:
        if isinstance(d.get(k),str):entries[k]=d[k]
    for source,digest in entries.items():
        assert digest in by_digest,(str(p),source,digest)
        source_checks.append({'record':str(p.relative_to(out)),'recorded_source':source,'sha256':digest,'matching_artifacts':by_digest[digest]})
for name in ['owned-gdb-pty-cpu-fixed-path','owned-gdb-with-tty-cli-cpu']:
    d=json.loads((out/name/'record.json').read_text());assert d['passed']
    assert all(x['passed'] and not x['cleanup']['inferior_survived'] and not x['cleanup']['gdb_survived'] for x in d['cases'])
for name in ['api-entry-profile-health','post-entry-stall-health','post-entry-csr-health']:
    assert json.loads((out/name/'record.json').read_text())['healthy']
assert json.loads((out/'prefill-layer-trace-build-mkl-path/record.json').read_text())['passed']
manifest={'scope':'Compact Level Zero logging; one EAGAIN-stalled full-input diagnostic and one progressing bounded repeat, both incomplete; healthy follow-up GPU probes; real CPU/GDB PTY and CLI checks; layer-range trace build. No capacity, prevention or throughput claim.',
          'original_engine_commit':old_commit,
          'original_prefill_source':info(out/'sources-used/prefill-before-layer-trace.cpp'),
          'files':files,'verified_record_sources':source_checks}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'verified_record_sources':len(source_checks)},indent=2))
