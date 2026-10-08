"""Run an owned, finite ABBA sequence of four full32K quiet jobs."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
controller=base/'run_owned_poll_matched_quiet32k_v01402_v1.py'
assert hashlib.sha256(controller.read_bytes()).hexdigest()=='e681237f10cdb6cfd27e6a6ef848ac26519def4809a74125584054f56ecc3ff6'
out=base/'poll-matched-quiet32k-v01402-sequence-v1';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'job_controller_sha256':hashlib.sha256(controller.read_bytes()).hexdigest(),
        'scope':'Matched DD5/44a ABBA, four fresh32768 inputs A/B/A/B and64 output IDs/logprobs per process. No performance conclusion until all16 full reads, output/MTP/freshness and normal exit gates pass. First process read separate from later reads; no global cache clearing, physical256K/adoption or stall-fix claim.',
        'minimum_comparison_input_tokens':32768,'steps':[]}
started=time.monotonic()
def save():
    record['elapsed_seconds']=time.monotonic()-started
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def complete(path):
    d=json.loads(path.read_text())
    assert not d['active'] and d['healthy'] and d['completed'] and d['math_gate_passed']
    assert d['quiet_four_fresh_reads_completed'] and not d['full_capacity_sequence_completed']
    assert d['exit_code']==0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
    assert len(d['requests'])==4 and all(v['math_gate_passed'] and v['input_tokens']==32768 and v['measurement']['generated_tokens']==64 and v['resume_tokens'] and all(n==0 for n in v['resume_tokens']) for v in d['requests'])
    for key in ['inferior','debugger']:
        old=d[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
    return d
save()
try:
    plan=[('dd5-control',1),('dd5-poll-backoff',1),('dd5-poll-backoff',2),('dd5-control',2)]
    for ordinal,(mode,repetition) in enumerate(plan,1):
        argv=[sys.executable,str(controller),mode,'quiet',str(repetition)]
        step={'ordinal':ordinal,'mode':mode,'repetition':repetition,'argv':argv,
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        record['steps'].append(step);record['active_step']=ordinal;save()
        print('START',ordinal,mode,repetition,flush=True)
        with (out/f'{ordinal}-{mode}-r{repetition}.stdout').open('wb') as stdout:
            child=subprocess.Popen(argv,stdout=stdout,stderr=subprocess.STDOUT)
            step['owned_controller']=process_identity(child.pid);save()
            while child.poll() is None:
                now=process_identity(child.pid)
                assert now and now['start_ticks']==step['owned_controller']['start_ticks']
                # Each child owns the model/debugger and its finite1200s
                # deadline/cleanup. The supervisor does not kill the owner
                # merely because an observation interval expired.
                time.sleep(1);save()
        step.update(exit_code=child.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        assert child.returncode==0
        path=base/f'owned-poll-matched-{mode}-v01402-code32k-quiet-r{repetition}/record.json'
        d=complete(path)
        step.update(receipt=str(path),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    binary_sha256=d['binary_sha256'],measurements=[v['measurement'] for v in d['requests']])
        save();print('FINISH',ordinal,mode,repetition,flush=True)
    record['active_step']=None;record['passed']=True
except BaseException as error:
    record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({'passed':record['passed'],'elapsed_seconds':record['elapsed_seconds'],'error':record.get('error')},indent=2))
if not record['passed']:raise SystemExit(1)
