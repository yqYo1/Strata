"""Root-owned bounded, serialized actual pack input preparation and independent bundle admission."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-static-quota-oracle-v0141-20261010')
T = W / 'sycl/tools'
PINS = {
    T / 'prepare_independent_prompt_tokens_v2.py': '829f6335ff7b47d587832c40b9f30b9aafddce112e7b980095aa17225b0940cd',
    T / 'test_prepare_independent_prompt_tokens_v2.py': 'a6e36b7da9bc8d7928bc7417a6eb4d4277f87f72ae435dda95afe3191c68a729',
    B / 'research-20261009/implementation-independent-prompt-preparer-v2.txt': '5fd23cbb593fde95d25a6e9141575b45432ee4317efaed67295d5f8a12f8c8ef',
    B / 'independent-code-corpus-manifest-v2.json': '4881239ef30f643a55ad13c1bff17c49bcc04abb921139bc97975b5eacbf8e40',
    B / 'independent-prompt-actual-pack-cpu-validation-v2/record.json': 'bdb1f524b16312c5996bb55da12c0023ea283eceaba0fe38c72942fb905957d4',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert __debug__
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, expected in PINS.items():
            assert digest(path) == expected, str(path)
        out = B / 'independent-prompt-actual-pack-cpu-validation-v3'
        out.mkdir(mode=0o700)
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
        bundle=out/'bundle'
        manifest_path=B/'independent-code-corpus-manifest-v2.json'
        manifest=json.loads(manifest_path.read_text())
        for entries in manifest['corpora'].values():
            for entry in entries: assert digest(Path(entry['path']))==entry['sha256'],'document input pin'
        assets=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/tokenizer')
        for name,expected in manifest['tokenizer_asset_sha256'].items(): assert digest(assets/name)==expected,'asset input pin'
        python=Path('/home/yayoi/.local/share/strata-sycl/venv/bin/python')
        argv=[str(python),str(T/'prepare_independent_prompt_tokens_v2.py'),'--self-sha256',PINS[T/'prepare_independent_prompt_tokens_v2.py'],'--manifest',str(manifest_path),'--manifest-sha256',PINS[manifest_path],'--source-root','/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-cache-route-pairs-v0141-20261010','--tokenizer-dir',str(assets),'--output',str(bundle)]
        record = dict(active=True, complete=False, passed=False,
                      scope='Actual frozen whole-document input preparation using existing host venv and exact production Python tokenizer/template assets; no model payload or inference',
                      controller_sha256=digest(Path(__file__)), pins={str(p): h for p, h in PINS.items()},
                      argv=argv, cwd=str(T), environment=env, deadline_seconds=180, text_budget_bytes=1048576,
                      started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      gpu_work_submitted=False, model_opened=False, actual_pack_tokenized=False,
                      inference_run=False, adopted=False, performance_eligible=False, full_lifecycle_passed=False,
                      cleanup=[], survivors=[], resource_limits=dict(AS_bytes=1073741824,RSS_poll_bytes=805306368,CPU_soft_seconds=120,CPU_hard_seconds=121,FSIZE_bytes=33554432,NOFILE=64,CORE_bytes=0),peak_rss_bytes=0,python_executable_resolved=str(python.resolve()),python_executable_sha256=digest(python.resolve()))
        rp = out / 'record.json'
        def save():
            rp.write_text(json.dumps(record, indent=2) + '\n')
        save()
        begin = time.monotonic()
        proc = None
        stdout, stderr = out / 'preparation.stdout', out / 'preparation.stderr'
        try:
            with stdout.open('wb') as so, stderr.open('wb') as se:
                def child_limits():
                    for kind,limits in ((resource.RLIMIT_AS,(1073741824,1073741824)),(resource.RLIMIT_CPU,(120,121)),(resource.RLIMIT_FSIZE,(33554432,33554432)),(resource.RLIMIT_NOFILE,(64,64)),(resource.RLIMIT_CORE,(0,0))):
                        resource.setrlimit(kind,limits)
                proc = subprocess.Popen(argv, cwd=T, env=env, stdout=so, stderr=se,preexec_fn=child_limits,start_new_session=True)
                ticks = int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                record['owner'] = dict(pid=proc.pid, start_ticks=ticks)
                save()
                while proc.poll() is None:
                    assert time.monotonic() - begin < 180, 'CPU deadline'
                    assert stdout.stat().st_size + stderr.stat().st_size <= 1048576, 'CPU log budget'
                    try:
                        status=Path(f'/proc/{proc.pid}/status').read_text()
                    except FileNotFoundError:
                        status=''
                    for line in status.splitlines():
                        if line.startswith('VmRSS:'):
                            rss=int(line.split()[1])*1024
                            record['peak_rss_bytes']=max(record['peak_rss_bytes'],rss)
                            assert rss<=805306368,'CPU RSS budget'
                    time.sleep(.05)
            record['exit_code'] = proc.returncode
            assert proc.returncode == 0, 'fixture child failed'
            assert stdout.stat().st_size+stderr.stat().st_size<=1048576,'final log budget'
            marker=json.loads(stdout.read_text())
            candidate_path=bundle/'record.json'
            assert marker==dict(preparation_returned=True,output=str(bundle),record_sha256=digest(candidate_path)),'post-close return marker'
            candidate=json.loads(candidate_path.read_text())
            record['actual_pack_tokenized']=True
            assert candidate['completed'] is False and candidate['ready_for_owner_validation'] is True
            assert candidate['self_sha256']==PINS[T/'prepare_independent_prompt_tokens_v2.py']
            assert candidate['manifest_sha256']==PINS[manifest_path]
            assert candidate['render_options']==dict(enable_thinking=True,reasoning_effort='xhigh')
            assert candidate['tokenizer_asset_sha256']==manifest['tokenizer_asset_sha256']
            assert candidate['GPU_used'] is False and candidate['inference_run'] is False
            assert (bundle.stat().st_mode & 0o777)==0o700
            expected_names={'manifest.json','record.json','train.rendered.txt','validation.rendered.txt','train.tokens.txt','validation.tokens.txt'}
            assert {p.name for p in bundle.iterdir()}==expected_names
            output_files={}
            for path in bundle.iterdir():
                st=path.lstat()
                assert path.is_file() and not path.is_symlink() and st.st_uid==__import__('os').getuid() and st.st_nlink==1
                assert (st.st_mode & 0o777)==0o600
                output_files[path.name]=dict(bytes=st.st_size,sha256=digest(path))
            assert (bundle/'manifest.json').read_bytes()==manifest_path.read_bytes()
            sequences={}
            for role in ('train','validation'):
                q=candidate['prompts'][role]
                selected=q['selected_documents'];assert selected==manifest['corpora'][role][:len(selected)] and 1<=len(selected)<=len(manifest['corpora'][role])
                text=(bundle/q['rendered_path']).read_text()
                body=manifest['wrapper']['user_prefix']+'\n\n'.join(Path(e['path']).read_text() for e in selected)+manifest['wrapper']['user_suffix']
                assert text.count(body.strip())==1,'whole source body after exact template content trim'
                record['template_content_normalization']='Pinned chat_template.jinja render_content(message.content,true)|trim; outer whitespace only, no document/token slicing'
                ids_bytes=(bundle/q['tokens_path']).read_bytes()
                ids=list(map(int,ids_bytes.decode('ascii').split()))
                assert ids_bytes==(' '.join(map(str,ids))+'\n').encode('ascii'),'canonical ID serialization'
                assert 32768<=len(ids)<=65536 and len(ids)==q['token_count'] and all(0<=i<248320 for i in ids)
                assert digest(bundle/q['rendered_path'])==q['rendered_text_sha256'] and digest(bundle/q['tokens_path'])==q['token_ids_sha256']
                sequences[role]=ids
            a={tuple(sequences['train'][i:i+16]) for i in range(len(sequences['train'])-15)}
            z={tuple(sequences['validation'][i:i+16]) for i in range(len(sequences['validation'])-15)}
            shared=len(a.intersection(z));denom=min(len(a),len(z));assert shared*5<denom*4
            assert candidate['overlap']['shared_shingles']==shared and candidate['overlap']['containment_denominator']==denom
            for path,expected in PINS.items(): assert digest(path)==expected,'input changed'
            for entries in manifest['corpora'].values():
                for entry in entries: assert digest(Path(entry['path']))==entry['sha256'],'document changed'
            for name,expected in manifest['tokenizer_asset_sha256'].items(): assert digest(assets/name)==expected,'asset changed'
            record.update(complete=True,passed=True,actual_pack_tokenized=True,owner_admission_passed=True,
                          candidate_record_sha256=digest(candidate_path),output_files=output_files,
                          prompt_summaries=candidate['prompts'],overlap=candidate['overlap'],
                          provenance='Distinct whole source files/components within one frozen codebase; no independent authorship or population claim',
                          retained_consumer='Root future >=32K source-disjoint inference and quota/route validation',
                          retained_owner='/root',retained_review_point='After this workload comparison closes or fixtures superseded')
        except BaseException as error:
            record['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                record['cleanup'].append('TERM owned CPU fixture')
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    record['cleanup'].append('KILL owned CPU fixture')
                    proc.wait(timeout=5)
            if proc is not None and proc.poll() is None:
                record['survivors'].append(record['owner'])
            record['logs'] = {str(p): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in (stdout, stderr) if p.exists()}
            record.update(active=False, elapsed_seconds=time.monotonic() - begin,
                          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save()
        print(json.dumps(dict(record=str(rp), sha256=digest(rp), passed=record['passed'], elapsed_seconds=record['elapsed_seconds'])))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
