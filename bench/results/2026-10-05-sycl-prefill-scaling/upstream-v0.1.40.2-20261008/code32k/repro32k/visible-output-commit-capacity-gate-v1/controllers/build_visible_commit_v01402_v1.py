"""Commit only the prefix whose predictions are actually emitted."""
from pathlib import Path
import datetime,difflib,hashlib,json,os,shlex,signal,subprocess,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
compiled=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
build=root/'build-sycl-event-ack-registered-copy-v3-20261008/build'
original=compiled/'sycl/src/program/generate.cpp'
candidate=root/'build-sycl-visible-commit-v1-20261008';candidate.mkdir()
(candidate/'source').mkdir();source=candidate/'source/generate.cpp'
out=base/'visible-commit-v01402-build-v1';out.mkdir(mode=0o700)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
helper='''// A verifier may accept past max_new or an EOS. Only the predictions that
// reach the caller may advance the saved running state; the final emitted
// prediction remains the next input and is not itself consumed.
int visible_output_prefix(int accepted, int64_t remaining, const int32_t* targets,
                          const std::vector<int64_t>& eos_ids) {
    if (accepted < 0 || remaining < 1 || targets == nullptr) return 0;
    int keep = (int) std::min<int64_t>((int64_t) accepted + 1, remaining);
    for (int i = 0; i < keep; ++i) {
        if (std::find(eos_ids.begin(), eos_ids.end(), (int64_t) targets[i]) != eos_ids.end()) {
            keep = i + 1;
            break;
        }
    }
    return keep;
}

'''
text=original.read_text();changed=text
anchor='}  // namespace\n\nint main(int argc, char **argv) try {'
assert changed.count(anchor)==1
changed=changed.replace(anchor,helper+anchor)
pipeline='while (a < A.T - 1 && A.tok[a + 1] == outp[(size_t) a]) ++a;'
assert changed.count(pipeline)==1
changed=changed.replace(pipeline,pipeline+'\n                    a = visible_output_prefix(a, max_new - produced_n, outp.data(), o.eos_ids) - 1;')
serial='while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;'
assert changed.count(serial)==2
first=changed.index(serial)
changed=changed[:first]+changed[first:].replace(serial,serial+'\n                a = visible_output_prefix(a, max_new - produced_n, outv.data(), o.eos_ids) - 1;',1)
second=changed.rindex(serial)
changed=changed[:second]+changed[second:].replace(serial,serial+'\n            a = visible_output_prefix(a, max_new - (int64_t) produced.size(), outv.data(), o.eos_ids) - 1;',1)
source.write_text(changed)
(out/'candidate.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile=str(original),tofile=str(source))))
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',MKLROOT='/opt/intel/oneapi/mkl/2026.1',LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
record={'active':True,'passed':False,'gpu_tested':False,'adopted':False,'steps':[],
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_sha256':digest(__file__),'original_source_sha256':digest(original),'candidate_source_sha256':digest(source),
        'scope':'Only output-limit/EOS commit prefix on serial serve, pipeline serve and CLI speculative loops. Verifier T, kernels, output/logprob rows, queues and allocation remain unchanged. Accepted counts now count the emitted/committed prefix. Includes prior direct pointer assignment and v3 KV/no-root prefill candidates.'}
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def run(label,argv,timeout=600):
    step={'label':label,'argv':list(map(str,argv))};record['steps'].append(step);save();begin=time.monotonic()
    with (out/(label+'.stdout')).open('wb') as a,(out/(label+'.stderr')).open('wb') as b:
        p=subprocess.Popen(step['argv'],cwd=build,env=env,stdout=a,stderr=b,start_new_session=True);step['pid']=p.pid;save()
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    step.update(exit_code=rc,elapsed_seconds=time.monotonic()-begin);save();assert rc==0,label
save()
try:
    # Compile the actual helper on CPU: length/EOS boundaries must be applied
    # before commit, not merely before formatting the answer.
    cpu=out/'prefix-check.cpp'
    cpu.write_text('#include <algorithm>\n#include <cstdint>\n#include <vector>\n#include <cassert>\n'+helper+'''int main() {
      const int32_t t[] = {11,22,33,44};
      assert(visible_output_prefix(3,4,t,{}) == 4);
      assert(visible_output_prefix(3,1,t,{}) == 1);
      assert(visible_output_prefix(3,2,t,{}) == 2);
      assert(visible_output_prefix(3,4,t,{11}) == 1);
      assert(visible_output_prefix(3,4,t,{22}) == 2);
      assert(visible_output_prefix(3,4,t,{44}) == 4);
      assert(visible_output_prefix(3,2,t,{33}) == 2);
      assert(visible_output_prefix(0,INT64_MAX,t,{}) == 1);
      assert(visible_output_prefix(3,0,t,{}) == 0);
      assert(visible_output_prefix(-1,4,t,{}) == 0);
      assert(visible_output_prefix(3,4,nullptr,{}) == 0);
      // Reproduced32K limit: two visible predictions in the final accepted
      // four-row window must commit two, leaving32831 consumed, not32833.
      assert(32829 + visible_output_prefix(3,2,t,{}) == 32831);
    }\n''')
    run('cpu-compile',['/usr/bin/c++','-std=c++20','-O2','-fsanitize=address,undefined',str(cpu),'-o',str(out/'prefix-check')])
    run('cpu-run',[str(out/'prefix-check')]);record['cpu_cases_passed']=12
    prior_path=base/'kv-access-safe-v01402-build-v2/record.json';prior=json.loads(prior_path.read_text())
    assert prior['passed'] and not prior['active'] and prior['baseline_inputs_unchanged']
    record['baseline_binary_sha256']=digest(prior['candidate_binary']);assert record['baseline_binary_sha256']==prior['candidate_binary_sha256']
    record['access_build_receipt_sha256']=digest(prior_path)
    commands=subprocess.check_output(['/usr/bin/ninja','-t','commands','strata'],cwd=build,env=env,text=True,timeout=30).splitlines()
    compiles=[shlex.split(x) for x in commands if ' -c ' in x and Path(shlex.split(x)[-1]).resolve()==original.resolve()]
    assert len(compiles)==1
    argv=compiles[0];record['original_compile_argv']=argv.copy();obj=candidate/'generate.cpp.o'
    for flag,value in [('-c',source),('-o',obj),('-MT',obj),('-MF',candidate/'generate.cpp.o.d')]:
        assert argv.count(flag)==1;argv[argv.index(flag)+1]=str(value)
    run('compile',argv)
    argv=list(prior['steps'][-1]['argv']);assert prior['steps'][-1]['label']=='link'
    assert argv.count('CMakeFiles/strata.dir/src/program/generate.cpp.o')==1
    argv=[str(obj) if v=='CMakeFiles/strata.dir/src/program/generate.cpp.o' else v for v in argv]
    inputs={str((build/p).resolve()):digest((build/p).resolve()) for p in argv if p.endswith(('.a','.o')) and (build/p).is_file()}
    record['link_input_sha256']=inputs
    binary=candidate/'strata';argv[argv.index('-o')+1]=str(binary);run('link',argv)
    assert digest(original)==record['original_source_sha256'] and digest(prior['candidate_binary'])==record['baseline_binary_sha256']
    assert all(digest(p)==s for p,s in inputs.items())
    record.update(passed=True,baseline_inputs_unchanged=True,candidate_binary=str(binary),candidate_binary_sha256=digest(binary),candidate_object_sha256=digest(obj))
except BaseException as e:record['error']=repr(e);raise
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['passed','elapsed_seconds','error','candidate_binary_sha256','cpu_cases_passed']},indent=2))
