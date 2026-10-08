"""Freeze a full-capacity controller without editing earlier executed evidence."""
from pathlib import Path
import ast
import datetime
import difflib
import hashlib
import json

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = base / 'run_owned_kv_stream32k_v01402_v2.py'
target = base / 'run_owned_full_kv_access_v01402_v1.py'
assert not target.exists()
original = parent.read_text()
text = original

def replace(a, b):
    global text
    assert text.count(a) == 1, a[:100]
    text = text.replace(a, b)

replace("assert mode=='kv-stream32k' and phase in ['diagnostic','state'] and repetition in [1,2,3]",
        "assert mode=='full-kv-access' and phase=='diagnostic' and repetition==1")
replace("out = base/f'owned-{mode}-v01402-code32k-{phase}-r{repetition}'", "out = base/f'owned-{mode}-v01402-full256k-{phase}-r{repetition}'")
replace("'deadline_seconds':1800,'protocol_timeout_seconds':600,'log_limit_bytes':20*1024**3",
        "'deadline_seconds':10800,'protocol_timeout_seconds':5400,'log_limit_bytes':128*1024**3")
replace("binary=root/'build-sycl-kv-stream-safe-v3-20261008/strata'", """binary=root/'build-sycl-kv-access-safe-v2-20261008/strata'
    access_receipt=base/'kv-access-safe-v01402-build-v2/record.json'
    access_build=json.loads(access_receipt.read_text())
    assert access_build['passed'] and not access_build['active'] and access_build['baseline_inputs_unchanged']
    assert access_build['baseline_binary_sha256']=='1441ad556cb1d8ecddf42e525a1462cc53f9251fae4734c008046e10e8b0e599'
    assert access_build['candidate_binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
    assert access_build['candidate_source_sha256']==hashlib.sha256((binary.parent/'source/layer.cpp').read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in access_build['link_input_sha256'].items())
    record['access_build_receipt_sha256']=hashlib.sha256(access_receipt.read_bytes()).hexdigest()
    record['access_source_sha256']=access_build['candidate_source_sha256']""")
replace("assert kv_build['candidate_binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()", "assert kv_build['candidate_binary_sha256']==access_build['baseline_binary_sha256']\n    assert access_build['kv_build_receipt_sha256']==hashlib.sha256(kv_receipt.read_bytes()).hexdigest()")
replace("kv_source=binary.parent/'source/kv_stream.dp.cpp'", "kv_source=Path(kv_build['candidate_binary']).parent/'source/kv_stream.dp.cpp'")
replace("record['candidate_build_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()", "record['candidate_build_receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()\n        assert access_build['prefill_build_receipt_sha256']==record['candidate_build_receipt_sha256']")
replace("list(base.glob('owned-kv-stream32k-v01402-code32k-*/record.json')):", "list(base.glob('owned-kv-stream32k-v01402-code32k-*/record.json'))+list(base.glob('owned-kv-layer-major32k-*-*/record.json'))+list(base.glob('owned-kv-stream32k-quiet-*/record.json'))+list(base.glob('owned-full-kv-access-*/record.json')):")
replace("args+=['--kv','int8','--kv-resident','32768','--prompt-cache','0']", "args+=['--kv','int8','--kv-resident','32768','--prompt-cache','1','--prompt-cache-every','0','--prompt-cache-root','0']")
replace("    record['scope']='KV streaming candidate,", "    record['scope']='Full256K occupancy/repeat/session restore/clipped verify tail/refusal/later32K correctness gate. All inputs >=32768. Diagnostic timing excluded from speed. KV streaming candidate,")
replace("    record['configuration_note']='Context262144/kv-resident32768", "    record['configuration_note']='PC1/ckpt1 for real SAVE/RESTORE; no periodic/root checkpoints; fresh full reads must report RESUME0 after an interposed different32K prompt. Context262144/kv-resident32768")
start = text.index('    for request_index in range(3):')
end = text.index("    current=None;send(b'QUIT", start)
loop = r'''
    from read_saved_session_v01402_v1 import read_session, full_kv_gate
    fixture=base/'coding-review-32k-tokens.txt'
    full_fixture=base/'full-context-copy-off/coding-context-256k-tokens.txt'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
    assert hashlib.sha256(full_fixture.read_bytes()).hexdigest()=='cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a'
    control=list(map(int,fixture.read_text().split()))[:32768]
    full_source=list(map(int,full_fixture.read_text().split()))
    suffix=[248046,198,248045,74455,198,248068,198,248069,271]
    def full_prompt(n):
        assert 32768<=n<=262144
        ids=full_source[:n-len(suffix)]+suffix
        assert len(ids)==n
        return ids
    full=full_prompt(262140)
    common=next(i for i,(a,b) in enumerate(zip(control,full)) if a!=b)
    assert common<32767, 'interposed control must clear the full-prefix checkpoint'
    record['fixtures']={'control_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        'full_sha256':hashlib.sha256(full_fixture.read_bytes()).hexdigest(),
                        'common_prefix_tokens':common}
    baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]
    stderr=out/'debugger/inferior.stderr'
    record['sessions']=[]
    math_ok=True
    assert int(record['startup'][-1].split()[1])==262144

    def captures(name, required):
        result={}
        head=out/'first-head.bin'
        if head.exists():
            kept=out/f'{name}.head.bin';assert not kept.exists();head.rename(kept)
            data=kept.read_bytes();values=array.array('f');values.frombytes(data)
            assert len(values)==248320 and all(map(math.isfinite,values))
            result['first_head']={'file':str(kept),'sha256':hashlib.sha256(data).hexdigest(),'floats':len(values),'finite':True}
        state=out/'prefill-state.bin'
        if state.exists():
            kept=out/f'{name}.state.bin';assert not kept.exists();state.rename(kept)
            parts=[]
            with kept.open('rb') as stream:
                while header:=stream.read(8):
                    assert len(header)==8
                    size=struct.unpack('=Q',header)[0];offset=stream.tell();left=size;digest=hashlib.sha256()
                    while left:
                        data=stream.read(min(left,1048576));assert data;digest.update(data);left-=len(data)
                    parts.append({'index':len(parts),'offset':offset,'bytes':size,'sha256':digest.hexdigest()})
            assert len(parts)==66 and parts[0]['bytes']==8
            result['prefill_state']={'file':str(kept),'bytes':kept.stat().st_size,'parts':parts}
        if required:
            assert 'first_head' in result and 'prefill_state' in result, 'missing fresh full-read capture'
        return result

    def request(name,tokens,new,expected=None,fresh=False,allow=True,tail=False):
        global current,math_ok
        current=name
        assert len(tokens)>=32768
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        start_offset=stderr.stat().st_size
        command=f'GEN {new} ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\n'
        (out/f'{name}.input-tokens.txt').write_text(' '.join(map(str,tokens))+'\n')
        result={'name':name,'input_tokens':len(tokens),'max_new':new,'ids':[],'logprobs':[],'protocol':[],
                'request_sha256':hashlib.sha256(command.encode()).hexdigest(),'allowed':allow}
        record['requests'].append(result);save();send(command.encode());event('request',name=name)
        request_started=time.monotonic()
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in value.split()[1:])
                result['logprobs'].append(value)
            if value.startswith(('DONE ','ERR ')):break
        result['diagnostic_wall_seconds']=time.monotonic()-request_started
        windows=[]
        with stderr.open('rb') as stream:
            stream.seek(start_offset)
            for item in stream:
                if item.startswith(b'strata trace: window '):
                    match=re.search(rb'window (-?\d+) (-?\d+)',item);assert match
                    windows.append(list(map(int,match.groups())))
        result['verify_windows']=windows
        assert all(pos>=0 and count>0 and pos+count<=262144 for pos,count in windows)
        if not allow:
            assert value.startswith('ERR prompt') and not result['ids'] and not result['logprobs'] and not windows
            assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
            result['math_gate_passed']=True;save();return result
        assert value.startswith('DONE '),value
        fields=value.split()
        assert int(fields[2])==len(tokens) and int(fields[1])==new and len(result['ids'])==new and len(result['logprobs'])==new
        result['mtp_counts']=list(map(int,fields[6:8]))
        result['measurement']={'generated_tokens':int(fields[1]),'prompt_tokens':int(fields[2]),
                               'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),'purpose':'diagnostic only; excluded from performance'}
        result['resume_tokens']=[int(v.split()[1]) for v in result['protocol'] if v.startswith(('RESUME ','REUSED '))]
        assert result['resume_tokens']
        if fresh:assert all(n==0 for n in result['resume_tokens']), 'a repeated full read was silently reused'
        result.update(captures(name,fresh))
        result['math_gate_passed']=True
        if expected is not None:
            comparison={'ids_equal':result['ids']==expected['ids'],
                        'logprobs_equal':result['logprobs']==expected['logprobs']}
            if fresh:
                comparison['first_head_equal']=result['first_head']['sha256']==expected['first_head']['sha256']
                state_cmp=compare_states(result['prefill_state'],expected['prefill_state'],len(tokens)-1)
                result['live_prefill_comparison']=state_cmp
                comparison['all_live_state_equal']=not state_cmp['different_live_parts']
                comparison['mtp_counts_equal']=result['mtp_counts']==expected.get('mtp_counts',[43,66])
            elif 'first_head' in result and 'first_head' in expected:
                comparison['first_head_equal']=result['first_head']['sha256']==expected['first_head']['sha256']
            result['comparison']=comparison
            result['math_gate_passed']=all(comparison.values())
        if tail:
            assert windows and max(pos+count for pos,count in windows)==262144
            result['last_executed_physical_cell']=262143
            if new==2:
                assert windows[-1][1]==2, 'expected a clipped two-token verification window'
                result['clipped_verify_tail']=2
        math_ok=math_ok and result['math_gate_passed'];save()
        return result

    def session(command,name,path,expected_tokens):
        global current
        current=name
        if command=='SAVE':assert not path.exists()
        result={'name':name,'command':command,'file':str(path),'protocol':[]}
        record['sessions'].append(result);save();send(f'{command} {path}\n'.encode())
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith(('SAVED ','RESTORED ','SERR ')):break
        assert value.startswith('SAVED ' if command=='SAVE' else 'RESTORED '),value
        fields=value.split();assert int(fields[1])==expected_tokens and int(fields[2])==path.stat().st_size
        result['tokens']=int(fields[1]);result['bytes']=int(fields[2]);result['diagnostic_ms']=float(fields[3])
        if command=='SAVE':
            result['image']=read_session(path)
            assert result['image']['semantic']['live']['ids']['count']==expected_tokens
        else:result['engine_checksum_and_compatibility_validated']=True
        result['passed']=True;save();return result

    def reject():
        raise ValueError('mathematical gate rejected; no timing/adoption')

    # A mismatch leaves the engine waiting for input, so QUIT can close it normally.
    try:
        control_first=request('control32k-before',control,64,baseline,fresh=True)
        if not math_ok:reject()
        control_file=out/'control32k.session.bin'
        control_saved=session('SAVE','save-control32k',control_file,32831)
        continuation=control+control_first['ids'];assert len(continuation)==32832
        resume_reference=request('resume32k-reference',continuation,64)
        assert max(resume_reference['resume_tokens'])==32831
        full_first=request('full256k-first',full,4,fresh=True,tail=True)
        full_file=out/'full256k-first.session.bin'
        full_saved=session('SAVE','save-full256k-first',full_file,262143)
        full_saved['full_capacity_gate']=full_kv_gate(full_saved['image']);save()
        request('control32k-between-full-reads',control,64,baseline,fresh=True)
        if not math_ok:reject()
        full_repeat=request('full256k-repeat',full,4,full_first,fresh=True,tail=True)
        if not math_ok:reject()
        repeat_saved=session('SAVE','save-full256k-repeat',out/'full256k-repeat.session.bin',262143)
        repeat_saved['full_capacity_gate']=full_kv_gate(repeat_saved['image'])
        repeat_saved['all_saved_state_and_kv_bytes_equal']=repeat_saved['image']['semantic_sha256']==full_saved['image']['semantic_sha256']
        math_ok=math_ok and repeat_saved['all_saved_state_and_kv_bytes_equal'];save()
        if not math_ok:reject()
        clipped=full+full_first['ids'][:2];assert len(clipped)==262142
        clip_reference=request('full256k-clipped-reference',clipped,2,tail=True)
        assert max(clip_reference['resume_tokens'])==262139
        request('control32k-before-restore',control,64,baseline,fresh=True)
        if not math_ok:reject()
        session('RESTORE','restore-control32k',control_file,32831)
        resume_restored=request('resume32k-restored',continuation,64,resume_reference)
        assert max(resume_restored['resume_tokens'])==32831
        if not math_ok:reject()
        session('RESTORE','restore-full256k',full_file,262143)
        roundtrip=session('SAVE','save-full256k-restored',out/'full256k-restored.session.bin',262143)
        roundtrip['full_capacity_gate']=full_kv_gate(roundtrip['image'])
        roundtrip['all_saved_state_and_kv_bytes_equal']=roundtrip['image']['semantic_sha256']==full_saved['image']['semantic_sha256']
        math_ok=math_ok and roundtrip['all_saved_state_and_kv_bytes_equal'];save()
        if not math_ok:reject()
        clip_restored=request('full256k-clipped-restored',clipped,2,clip_reference,tail=True)
        assert max(clip_restored['resume_tokens'])==262139
        if not math_ok:reject()
        request('no-room',full_prompt(262144),1,allow=False)
        request('one-too-many',full_prompt(262142),3,allow=False)
        request('control32k-after-refusals',control,64,baseline,fresh=True)
        record['capacity_sequence_completed']=True
    except ValueError as rejected:
        if not math_ok:
            record['mathematical_rejection']=str(rejected)
        else:raise
'''
text = text[:start] + loop + '\n' + text[end:]
replace("record['math_gate_passed']=len(record['requests'])==3 and all(req['math_gate_passed'] for req in record['requests'])",
        "record['math_gate_passed']=math_ok and bool(record.get('capacity_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])")
replace("assert record['math_gate_passed'], 'reject before timing: repeated live-state/head/output mismatch'", "# A normal mathematical rejection is terminal evidence, not a GPU-health failure.")
replace("record['healthy']=record['completed'] and", "record['healthy']=record['completed'] and")
replace("if not record['healthy']:raise SystemExit(1)", "if not record['healthy'] or not record.get('math_gate_passed'):raise SystemExit(1)")
ast.parse(text)
target.write_text(text)
out = base/'full-kv-access-v01402-source-review-v1'
out.mkdir(mode=0o700)
compiled=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
files=['sycl/src/core/conversation_state.cpp','sycl/src/core/conversation_snapshot.cpp',
       'sycl/src/program/generate.cpp','src/core/conversation_file.cpp','include/strata/core/conversation_cache.hpp']
receipt={'active':False,'passed':True,'gpu_tested':False,'adopted':False,
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'controller_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
         'parent_controller_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
         'session_reader_sha256':hashlib.sha256((base/'read_saved_session_v01402_v1.py').read_bytes()).hexdigest(),
         'actual_compiled_restore_sources':{f:hashlib.sha256((compiled/f).read_bytes()).hexdigest() for f in files},
         'reviewed':['Snapshot restore validates first, drains queues, restores authoritative host KV and then checkpoint state. Checkpoint restore returns sync, so ring/map-reset operations finish before API return.',
                     'SAVE requires PC>0 and a completed checkpointed request. ckpt0 alone does not disable reuse. This gate uses PC1/ckpt1 with periodic/root checkpoints disabled; every claimed full fresh read must report zero reuse.',
                     'Disk image live.ids records consumed tokens and excludes the final emitted bonus token: 262140+4 output yields262143 consumed, but speculative verifier physically reaches cell262143, and every13-layer image contains262144 rounded cells.',
                     'All saved tensor/state bytes are compared, including rounded last physical cells and the pooled spare row. Only checkpoint LRU stamps and derived representation checksums are omitted from semantic equality. Engine RESTORE validates original checksums before writes.',
                     'Both restored32K continuation and restored256K-prefix clipped tail generate real tokens, and compare complete output IDs/logprobs. No artificial fresh-prefill substitute is used.',
                     'All initial/full/refusal/later input token lists are >=32768. First diagnostic run uses flushed UR/L0 logging; timings are not clean performance evidence.'],
         'limitations':['CPU/source preparation is not full-capacity correctness proof.', 'No reset/rebind/reboot or service changes are performed.']}
(out/'record.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'controller.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(parent),tofile=str(target))))
print(json.dumps({'passed':True,'controller':str(target),'sha256':receipt['controller_sha256']},indent=2))
