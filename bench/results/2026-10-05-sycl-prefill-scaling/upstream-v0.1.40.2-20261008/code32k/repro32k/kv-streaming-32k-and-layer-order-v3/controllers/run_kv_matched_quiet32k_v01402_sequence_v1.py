"""Matched ABBA processes, each with three complete >=32K rereads."""
from pathlib import Path
import datetime,hashlib,json,signal,subprocess,sys,time
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
out=base/'kv-matched-quiet32k-v01402-sequence-v1';out.mkdir(mode=0o700)
controller=base/'run_owned_kv_matched_quiet32k_v01402_v1.py'
record=dict(active=True,passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='ABBA matched configurations, fresh process each, three full32768 reads (ckpt0/prompt-cache0) and64 greedy output tokens. Initial prompt JIT/capture versus reads2/3 separate. No API/state/head/trace/profile/payload verification. Correct64IDs/logprobs/MTP43/66 and orderly exit/noXE required.',
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
            minimum_comparison_tokens=32768,steps=[])
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    for mode,repetition in [('kv-stream32k-quiet',1),('kv-layer-major32k-quiet',1),('kv-layer-major32k-quiet',2),('kv-stream32k-quiet',2)]:
        argv=[sys.executable,str(controller),mode,'quiet',str(repetition)]
        step=dict(argv=argv,mode=mode,repetition=repetition,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        record['steps'].append(step);save();print('START',mode,repetition,flush=True)
        with (out/f'{mode}-r{repetition}.stdout').open('wb') as f:
            p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT)
            step['controller_process']=process_identity(p.pid);save()
            try:rc=p.wait(timeout=2000)
            except subprocess.TimeoutExpired:
                # Ask the exact owned Python controller to run its finally cleanup.
                identity=process_identity(p.pid);old=step['controller_process']
                if identity and identity['start_ticks']==old['start_ticks']:p.send_signal(signal.SIGINT)
                rc=p.wait(timeout=90);raise RuntimeError('outer deadline reached; exact controller interrupted')
        step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
        path=base/f'owned-{mode}-v01402-code32k-quiet-r{repetition}/record.json'
        r=json.loads(path.read_text());assert r['healthy'] and r['completed'] and r['math_gate_passed'] and not r['active']
        assert len(r['requests'])==3 and r['exit_code']==0 and not r['new_fault_messages']
        assert not any(r['cleanup'].values())
        for key in ['inferior','debugger']:
            old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
        for q in r['requests']:
            assert q['math_gate_passed'] and q['mtp_counts']==[43,66] and q['no_prompt_reuse']
            assert q['measurement']['prompt_tokens']==32768 and q['measurement']['generated_tokens']==64
        assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) for k in r['environment'])
        assert not any(k in r['environment'] for k in ['STRATA_TRACE','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_CACHE_VERIFY','STRATA_PREFILL_DRAFT_VERIFY','STRATA_PROFILE','STRATA_VERIFY_PROFILE','STRATA_PREFILL_SYNC','STRATA_PREFILL_TRANSFER_TIMING'])
        step.update(record_file=str(path),record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),measurements=[q['measurement'] for q in r['requests']]);save()
        print('FINISH',mode,repetition,flush=True)
    record['passed']=True
except BaseException as e:record['error']=repr(e)
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
