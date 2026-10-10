"""One corrected/serial-issuer two-chunk diagnostic, never a speed sample."""
import fcntl,importlib.util,json,math,os,re,time,types
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
s=importlib.util.spec_from_file_location('qualified_owner',OWNER);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
HELPER=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011/sycl/tools/recover-xe.sh')
HELPER_SHA='51f5ba140613ae6ee81905eeeea153243a6a41742488ac161d76bad241944457'
LINK=B/'xe-prefill-failclosed-linked-root-v7/record.json'
HEALTH=B/'postreboot-status-health-root-r3/record.json'
OLD=B/'owned-xestrata39-hostusm4k-code32k-diagnostic-v8-r1/record.json'
DUMP=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')
RUNTIME_ERROR=re.compile(r"UR_RESULT_ERROR_[A-Z_]+|ZE_RESULT_ERROR_[A-Z_]+|\[(?:ERROR|FATAL)\]|GPU completion unconfirmed|terminate called|Assertion .*failed|CAT error|fault response|out of resources",re.I)
def runtime_findings(text):
    return [v for v in text.splitlines() if RUNTIME_ERROR.search(v)]
def validate_trace(text):
    markers=[int(v) for v in re.findall(r'strata trace: prompt chunk (\d+) of 8192',text)]
    m.require(markers==[0,4096],'exact two chunk-start trace markers')
    m.require(not runtime_findings(text),'no severe runtime error in model logs')
    return dict(chunk_start_positions=markers,runtime_errors=[],trace_enabled=True)
def validate(lines):
    pp=[v.split() for v in lines if v.startswith('PP ')]
    m.require([int(v[1]) for v in pp]==[4096,8192],'two complete batched chunks')
    m.require(all(len(v)==5 and int(v[2])==8193 and all(math.isfinite(float(x)) and float(x)>=0 for x in v[3:]) for v in pp),'PP identity/finite')
    done=[v.split() for v in lines if v.startswith('DONE ')];m.require(len(done)==1 and len(done[0])>=16,'one full DONE')
    d=done[0];m.require(int(d[1])==1 and int(d[2])==8193 and int(d[8])==0 and int(d[14])==8193,'complete fresh prompt and one output')
    m.require(d[5] in ('stop','length') and all(math.isfinite(float(d[i])) and float(d[i])>0 for i in (3,4)),'completed phases')
    m.require([v for v in lines if v.startswith('RESUME ')]==['RESUME 0'] and [v for v in lines if v.startswith('REUSED ')]==['REUSED 0'],'fresh state')
    outputs=[v for v in lines if v.startswith(('T ','LP '))];m.require(len(outputs)==2 and outputs[0].startswith('T ') and outputs[1].startswith('LP '),'one T/LP pair')
    token=int(outputs[0].split()[1]);m.require(0<=token<248320,'token domain')
    lp=outputs[1].split();m.require(len(lp)==7 and math.isfinite(float(lp[1])),'LP5')
    for v in lp[2:]:
        t,p=v.split(':');m.require(0<=int(t)<248320 and math.isfinite(float(p)),'finite LP fields')
    return dict(prompt_tokens=8193,actual_batched_positions=8192,completed_chunks=[4096,8192],generated_tokens=1,token_id=token,finite_output=True)

def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-prefill-two-chunk-serial-root-v3';m.require(not out.exists(),'new output');out.mkdir(mode=0o700)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    record=dict(active=True,complete=False,passed=False,adopted=False,fork_pristine=False,performance_eligible=False,
                math_equivalence=False,quality_qualified=False,full262144_qualified=False,retry_count=0,
                reset_executed=False,model_launch_attempted=False,started_utc=m.utc(),boot_id=boot,
                controller_sha256=m.sha(__file__),helper_sha256=HELPER_SHA,
                scope='corrected Xe, issuer OFF, 2x4096 batched positions plus terminal prompt token, one generated token')
    def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);record['commands']=owner.commands
    base=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    base['XDG_RUNTIME_DIR']='/run/user/'+str(os.getuid())
    def utility(label,args):
        e,so,se=owner.run(label,args,base,HERE,wall=10,text_cap=8<<20,rss_cap=1<<30,total_cap=96<<20,cpu=10)
        m.require(m.closed(e) and e['exit_code']==0,'normal utility '+label);return so.read_text()
    save();cursor=None;helper=None
    try:
        m.require(os.geteuid()!=0 and not DUMP.exists(),'unprivileged; no known GPU dump')
        qual=json.loads((HERE/'cpu-qualification.json').read_text());m.require(qual['passed'] and qual['controller_sha256']==m.sha(__file__),'actual controller CPU qualification')
        m.require(m.sha(HELPER)==HELPER_SHA,'new reset-aware guard')
        helper=types.ModuleType('read_only_helper');exec(compile(HELPER.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(HELPER),'exec'),helper.__dict__)
        link=json.loads(LINK.read_text());health=json.loads(HEALTH.read_text());old=json.loads(OLD.read_text())
        m.require(link['passed'] and link['complete'] and not link['active'] and link['fork_pristine'] is False,'corrected actual build only')
        m.require(health['passed'] and not health['active'] and health['boot_id']==boot and old['boot_id']!=boot,'new boot passed integer/kernel health')
        cpu=json.loads(Path(link['cpu_qualification']['path']).read_text())
        m.require(m.sha(link['cpu_qualification']['path'])==link['cpu_qualification']['sha256'] and cpu['passed'] and cpu['terminal_cleanup_expected_negative']['passed'],'terminal cleanup qualification')
        for p,v in link['sources'].items():m.require(m.sha(p)==v['sha256'],'actual corrected source pin')
        binary=Path(link['binary']['path']);m.require(m.sha(binary)==link['binary']['sha256'] and m.sha(binary)!=old['binary_sha256'],'changed executable required')
        record['binary']=link['binary'];record['source_pins']=link['sources'];record['link_receipt_sha256']=m.sha(LINK);record['health_before_sha256']=m.sha(HEALTH)
        fixture=B/'clean64k-task-fixture-root-r3/coding-review-65536-tokens.txt'
        m.require(m.sha(fixture)=='5284d8fa53d1ef28f6ba916a629faf901c890bc0c669bf986ee04ed72aa2c3ff','canonical input identity')
        all_ids=[int(v) for v in re.split(r'[\s,]+',fixture.read_text().strip())]
        m.require(len(all_ids)==65536 and all(0<=v<248320 for v in all_ids),'input domain/count')
        tokens=all_ids[:8144]+all_ids[-49:];record['input']=dict(parent_path=str(fixture),parent_sha256=m.sha(fixture),count=len(tokens),purpose='two-chunk correctness/fault exercise; no independent ability or performance claim')
        (out/'tokens.txt').write_text(','.join(map(str,tokens))+'\n');record['input']['sha256']=m.sha(out/'tokens.txt')
        argv=list(old['argv']);argv[0]=str(binary)
        target=dict(old['environment']);target['STRATA_PREFILL_ISSUER']='0'
        record['argv']=argv;record['environment']=target
        m.require(utility('embedding-state',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState']).strip()=='ActiveState=inactive','embedding inactive')
        text=utility('kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager']);match=re.search(r'^-- cursor: (.+)$',text,re.M)
        m.require(match is not None,'fresh journal cursor');cursor=match[1];record['kernel_cursor_before']=cursor
        pre=utility('kernel-boot-before',['/usr/bin/journalctl','-k','-b','--no-pager','-o','json'])
        rows=[json.loads(v) for v in pre.splitlines() if v.startswith('{')]
        relevant=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))]
        record['kernel_preflight_relevant_entries']=relevant
        m.require(not any(helper.FAULT.search(v['MESSAGE']) for v in relevant) and not DUMP.exists(),'fault-free current boot immediately before model')
        probe=B/'postfault-status-health-v2/health';m.require(m.sha(probe)=='59d2392ea53b87e6f565737953f7846537de340d20755f8561f2f0ff0e328d53','health binary')
        e,hs,he=owner.run('health-immediate-before',[str(probe),'0000:05:00.0'],helper.diagnostic_environment(target),HERE,wall=30,text_cap=8<<20,rss_cap=4<<30,total_cap=64<<20,cpu=30)
        expected=['STATUS 0000:05:00.0 0','QUEUE 0000:05:00.0','H2D 65536 complete','KERNEL 16384 complete','D2H 65536 complete','PASS 0000:05:00.0: 1 round, 16384 exact words']
        m.require(m.closed(e) and e['exit_code']==0 and hs.read_text().splitlines()==expected,'immediate exact integer health')
        record['health_immediate_before_passed']=True;health_finished=time.monotonic()
        interval=utility('kernel-after-prehealth',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'])
        rows=[json.loads(v) for v in interval.splitlines() if v.startswith('{')]
        relevant=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))]
        record['kernel_prehealth_entries']=relevant
        m.require(not any(helper.FAULT.search(v['MESSAGE']) for v in relevant) and not DUMP.exists(),'prehealth fresh fault gate')
        stderr_path=out/'commands/model.stderr'
        def interaction(io,entry):
            # Stop on a fresh dump during owned-process polling, not just at the end.
            original_poll=io.poll
            def poll():
                original_poll();m.require(not DUMP.exists(),'fresh GPU dump: stop owned model')
            io.poll=poll
            m.require(time.monotonic()-health_finished<30,'bounded health admission age')
            record['health_age_at_model_start_seconds']=time.monotonic()-health_finished
            startup=[];record['startup']=startup
            while True:
                v=io.line(120);startup.append(v);m.require(len(startup)<1000 and not v.startswith('ERR'),'startup protocol')
                if v.startswith('READY '):m.require(v.split()[1]=='262144','READY context');break
            pid=entry['direct_identity']['pid'];exe=Path('/proc',str(pid),'exe')
            m.require(exe.resolve()==binary.resolve() and m.sha(exe)==record['binary']['sha256'],'actual process executable')
            actual=dict(v.decode().split('=',1) for v in Path('/proc',str(pid),'environ').read_bytes().split(b'\0') if b'=' in v)
            m.require(actual==target,'explicit actual environment')
            settings=dict(v.split('=',1) for v in next(v for v in startup if v.startswith('INFO ')).split()[1:] if '=' in v)
            m.require(all(settings.get(k)==v for k,v in {'context':'262144','kv_resident':'32768','expert_slots':'144','pool_workers':'5','spec':'4'}.items()),'effective parameters')
            record['startup_vram_free_mib']=int(settings.get('vram_free_mib','-1'));m.require(record['startup_vram_free_mib']>=512,'VRAM reserve before GEN')
            m.require(not runtime_findings(stderr_path.read_text()),'no severe runtime error before GEN')
            lines=[];record['request_protocol']=lines;record['chunk_progress']=[];gen_started=time.monotonic();io.send(('GEN 1 ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\n').encode())
            while True:
                v=io.line(90);lines.append(v);m.require(len(lines)<2000 and not v.startswith('ERR'),'request protocol')
                if v.startswith('PP '):
                    record['chunk_progress'].append(dict(protocol=v,elapsed_seconds=time.monotonic()-gen_started,stderr_bytes=stderr_path.stat().st_size))
                    print('corrected Xe diagnostic: '+v,flush=True);save()
                if v.startswith('DONE '):break
            record['validation']=validate(lines);save();io.send(b'QUIT\n');io.exit(30)
        record['model_launch_attempted']=True;save()
        e,so,se=owner.run('model',argv,target,binary.parent,interaction,wall=210,text_cap=64<<20,rss_cap=96<<30,total_cap=96<<20,cpu=220)
        record['model_result']=dict(exit_code=e.get('exit_code'),owned_closed=m.closed(e));m.require(m.closed(e) and e['exit_code']==0,'normal model close')
        logs=se.read_text();record['trace_validation']=validate_trace(logs);record['model_runtime_error_lines']=runtime_findings(so.read_text()+logs)
        m.require(not record['model_runtime_error_lines'],'no severe model runtime error')
        record['model_stderr_sha256']=m.sha(se);record['model_stderr_bytes']=se.stat().st_size
        for p,v in link['sources'].items():m.require(m.sha(p)==v['sha256'],'source stable after model')
        m.require(m.sha(binary)==record['binary']['sha256'] and m.sha(HELPER)==HELPER_SHA and m.sha(LINK)==record['link_receipt_sha256'],'executable/helper/link stable')
        record['source_and_executable_stable']=True
        record['complete']=True
    except BaseException as exc:record['error']=repr(exc)
    finally:
        if owner.active is None and cursor and helper:
            try:
                rows=[json.loads(v) for v in utility('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if v.startswith('{')]
                relevant=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))]
                record['kernel_device_entries']=relevant;record['new_fault_messages']=[v['MESSAGE'] for v in relevant if helper.FAULT.search(v['MESSAGE'])]
                m.require(not record['new_fault_messages'] and not DUMP.exists(),'fault-free kernel/dump gate')
                record['kernel_gate_passed']=True
                if record['complete']:
                    probe=B/'postfault-status-health-v2/health';m.require(m.sha(probe)=='59d2392ea53b87e6f565737953f7846537de340d20755f8561f2f0ff0e328d53','health probe pin')
                    env=helper.diagnostic_environment(target)
                    e,so,se=owner.run('health-after',[str(probe),'0000:05:00.0'],env,HERE,wall=30,text_cap=8<<20,rss_cap=4<<30,total_cap=96<<20,cpu=30)
                    m.require(m.closed(e) and e['exit_code']==0 and so.read_text().endswith('PASS 0000:05:00.0: 1 round, 16384 exact words\n'),'post-model exact integer health')
                    rows=[json.loads(v) for v in utility('kernel-after-health',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if v.startswith('{')]
                    relevant=[v for v in rows if '0000:05:00.0' in v.get('MESSAGE','') or re.search(r'\bxe\b',v.get('MESSAGE',''))]
                    record['kernel_device_entries']=relevant;record['new_fault_messages']=[v['MESSAGE'] for v in relevant if helper.FAULT.search(v['MESSAGE'])]
                    m.require(not record['new_fault_messages'] and not DUMP.exists(),'post-health fresh kernel/dump gate');record['health_after_passed']=True
            except BaseException as exc:record['kernel_or_health_error']=repr(exc)
        record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
        record['active']=owner.active is not None;record['passed']=bool(record['complete'] and record.get('kernel_gate_passed') and record.get('health_after_passed') and record.get('source_and_executable_stable') and record['boot_unchanged'] and not record['active'] and not record.get('error') and not record.get('kernel_or_health_error'))
        record['finished_utc']=m.utc();save()
    print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'),kernel_or_health_error=record.get('kernel_or_health_error'))))
    return 0 if record['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
