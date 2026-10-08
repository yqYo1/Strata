"""Repair the indexer spare after replaying only an accepted verifier prefix."""
from pathlib import Path
import datetime,difflib,hashlib,json,os,shlex,shutil,signal,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
original=build.parent/'source/sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp'
candidate=root/'build-sycl-indexer-spare-commit-v1-20261008';candidate.mkdir()
(candidate/'source').mkdir();source=candidate/'source/native_qsa_indexer.dp.cpp'
out=base/'indexer-spare-commit-v01402-build-v1';out.mkdir(mode=0o700)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
text=original.read_text();changed=text
anchor='    const int d = item_ct1.get_local_id(2);\n    for (int step_idx = 0; step_idx < n_steps; ++step_idx) {'
assert changed.count(anchor)==1
changed=changed.replace(anchor,anchor.replace('    for (int step_idx', '    int last_valid_pos = -1;\n    for (int step_idx'))
anchor='        if (pos >= 0 && pos < max_cells) {\n            const float* raw_step'
assert changed.count(anchor)==1
changed=changed.replace(anchor,'        if (pos >= 0 && pos < max_cells) {\n            last_valid_pos = pos;\n            const float* raw_step')
anchor='        if (n_steps > 1) item_ct1.barrier();\n    }\n}\n// ---- C-2:'
assert changed.count(anchor)==1
changed=changed.replace(anchor,'''        if (n_steps > 1) item_ct1.barrier();
    }
    // A speculative window may complete this block before commit replays
    // fewer valid positions. Restore its spare key from the live first key;
    // trailing negative positions must not retain the rejected block key.
    // No completed block is changed, and the existing final barrier orders
    // the append stores before this per-lane copy.
    if (last_valid_pos >= 0 && d < D)
        pooled[(std::size_t(last_valid_pos) + 1) / R * D + d] = dead[d];
}
// ---- C-2:''')
source.write_text(changed)
(out/'candidate.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile=str(original),tofile=str(source))))
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record=dict(active=True,passed=False,gpu_tested=False,adopted=False,steps=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=digest(__file__),original_source_sha256=digest(original),candidate_source_sha256=digest(source),scope='Only native indexer append restores its live spare after valid-prefix replay. Keep arithmetic, local memory, existing barriers, subgroup32, root properties, ND ranges, queues and all other kernels. Link fdca visible-output commit, v3 KV and direct pointer assignments. No polling policy change. Compilation is not GPU correctness or speed evidence.')
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=600):
    step=dict(label=label,argv=list(map(str,argv)));record['steps'].append(step);save();begin=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as a,(out/(label+'.stderr')).open('wb') as b:
        p=subprocess.Popen(step['argv'],cwd=build,env=env,stdout=a,stderr=b,start_new_session=True);step['pid']=p.pid;save()
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-begin);save();assert rc==0,label
save()
try:
    previous_path=base/'owned-full-kv-access-v01402-full256k-diagnostic-r4/record.json';previous=json.loads(previous_path.read_text())
    assert previous['completed'] and previous['healthy'] and not previous['active'] and not previous['math_gate_passed']
    assert previous['exit_code']==0 and not previous['new_fault_messages'] and not previous['cleanup']['forced']
    assert not previous['cleanup']['inferior_survived'] and not previous['cleanup']['gdb_survived']
    record['full_capacity_rejection_receipt_sha256']=digest(previous_path)
    prior_path=base/'visible-commit-v01402-build-v1/record.json';prior=json.loads(prior_path.read_text());assert prior['passed'] and not prior['active']
    assert digest(prior['candidate_binary'])==prior['candidate_binary_sha256']=='fdca351f73bd2433d45d9db7962fde7ad1a4b83f5e254751d27782fe9f5f75c6'
    record['baseline_binary_sha256']=prior['candidate_binary_sha256'];record['visible_commit_build_receipt_sha256']=digest(prior_path)
    kv_path=base/'kv-stream-safe-v01402-build-v3/record.json';kv=json.loads(kv_path.read_text());assert kv['passed'] and not kv['active']
    archive=Path(kv['candidate_binary']).parent/'libstrata_kernels.a';assert digest(archive)==kv['candidate_archive_sha256']
    commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=build,env=env,text=True,timeout=30).splitlines()
    compiles=[shlex.split(x) for x in commands if ' -c ' in x and Path(shlex.split(x)[-1]).resolve()==original.resolve()];assert len(compiles)==1
    argv=compiles[0];record['original_compile_argv']=argv.copy();obj=candidate/'native_qsa_indexer.dp.cpp.o'
    for flag,value in [('-c',source),('-o',obj),('-MT',obj),('-MF',candidate/'native_qsa_indexer.dp.cpp.o.d')]:
        assert argv.count(flag)==1;argv[argv.index(flag)+1]=str(value)
    run('compile',argv)
    private=candidate/'libstrata_kernels.a';shutil.copy2(archive,private);record['original_archive_sha256']=digest(archive)
    members=subprocess.check_output(['/usr/bin/ar','t',str(archive)],text=True).splitlines();assert members.count(obj.name)==1 and len(members)==len(set(members))
    run('archive-replace',['/usr/bin/ar','r',str(private),str(obj)]);run('archive-index',['/usr/bin/ranlib',str(private)])
    record['archive_members']=[]
    for n in members:
        a=subprocess.check_output(['/usr/bin/ar','p',str(archive),n]);b=subprocess.check_output(['/usr/bin/ar','p',str(private),n])
        assert b==obj.read_bytes() if n==obj.name else a==b
        record['archive_members'].append(dict(member=n,replaced=n==obj.name,before_sha256=hashlib.sha256(a).hexdigest(),after_sha256=hashlib.sha256(b).hexdigest()))
    argv=list(prior['steps'][-1]['argv']);assert prior['steps'][-1]['label']=='link' and argv.count(str(archive))==1
    inputs={str((build/p).resolve()):digest((build/p).resolve()) for p in argv if p.endswith(('.a','.o')) and (build/p).is_file()};record['link_input_sha256']=inputs
    argv=[str(private) if p==str(archive) else p for p in argv];binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary);run('link',argv)
    assert digest(original)==record['original_source_sha256'] and digest(prior['candidate_binary'])==record['baseline_binary_sha256']
    assert all(digest(p)==s for p,s in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),candidate_object_sha256=digest(obj),candidate_archive_sha256=digest(private))
except BaseException as e:record['error']=repr(e);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','elapsed_seconds','error','candidate_binary_sha256','baseline_inputs_unchanged']},indent=2))
