from pathlib import Path
import datetime, hashlib, json, re, shutil, sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k';out=repro/'main-cache-release-and-restore'
sys.path.insert(0,str(root/'sycl/tools'));from owned_gdb import process_identity
def digest(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def load(name):return json.loads((base/name/'record.json').read_text())
states=load('main-cache-release-v01402-state-sequence');clean=load('main-cache-release-v01402-clean-sequence')
assert states['passed'] and not states['active'] and len(states['steps'])==12
assert clean['passed'] and not clean['active'] and len(clean['steps'])==8
assert (out/'manifest.json').is_file() and not (out/'analysis/clean-sequence.json').exists()
assert digest(root/'build-sycl-e8ca-refresh-20261007/strata')=='c88f94d81bfb22227310ea00d09ecc8ab21a6e670aee9556307746530f5af714'
def copy(p,dst):
    assert p.is_file() and p.stat().st_size<20*1024**2,p
    q=out/dst
    if q.exists():assert digest(q)==digest(p);return
    q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
private=json.loads((out/'private-artifacts.json').read_text());known={r['file'] for r in private};jobs=[]
baseline=load('owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1')['requests'][0]
counts=baseline['protocol'][-1].split()[6:8]
modes=['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot']
for mode in modes:
    for phase,rep in [('diagnostic',1),('state',1),('state',2),('clean',1),('clean',2)]:
        name=f'owned-{mode}-v01402-code32k-{phase}-r{rep}';d=base/name;r=load(name)
        assert r['healthy'] and r['completed'] and not r['active'] and r['exit_code']==0 and not r['new_fault_messages']
        assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
        for key in ['inferior','debugger']:
            old=r[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
        req=r['requests'][0];c=req['comparison_to_logged_control'] if phase=='clean' else req['comparison_to_default_counter_control']
        assert req['measurement']['prompt_tokens']==32768 and req['measurement']['generated_tokens']==64 and c['ids_equal'] and c['logprobs_equal']
        if phase!='clean':assert c['first_head_equal'] and not c['different_prefill_state_parts']
        assert req['protocol'][-1].split()[6:8]==counts
        jobs.append({'name':name,'prompt_tokens':32768,'captured_main_state_and_head_equal':phase!='clean','ids_and_logprobs_equal':True,'mtp_counts':counts,'normal_exit_and_owned_cleanup':True,'no_new_xe_fault':True})
        for f in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','input-tokens.txt']:
            p=d/f
            if p.exists():copy(p,'runs/'+name+'/'+f)
        for p in (d/'probes').glob('*'):
            if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
        for f in ['inferior-argv.json','inferior-environment.json']:
            p=d/'debugger'/f
            if p.exists():copy(p,'runs/'+name+'/debugger/'+f)
        for f in ['debugger/inferior.stderr','first-head.bin','prefill-state.bin']:
            p=d/f
            if p.exists() and str(p) not in known:
                private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)});known.add(str(p))
    copy(base/(mode+'-v01402-state-sequence')/'record.json','analysis/'+mode+'-state-sequence.json')
copy(base/'main-cache-release-v01402-state-sequence/record.json','analysis/state-sequence.json')
copy(base/'main-cache-release-v01402-clean-sequence/record.json','analysis/clean-sequence.json')
copy(Path(__file__),'controllers/'+Path(__file__).name)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'all20-controls.json').write_text(json.dumps({'scope':'Twelve complete main-state/head captures and eight clean32K timings; all outputs/logprobs and MTP43/66 counts equal. Counts are not all internal MTP tensor bytes.','jobs':jobs},indent=2)+'\n')
releases=[]
pattern=r'^strata prefill cache release: (\d+) physical bytes, (\d+) restored bytes, source (\w+), suspend ([\d.]+) ms, restore ([\d.]+) ms, graph ([\d.]+) ms, same_address (\d), slots restored$'
for mode in modes:
    for rep in [1,2]:
        lines=(base/f'owned-{mode}-v01402-code32k-clean-r{rep}/project-messages.txt').read_text().splitlines()
        rows=[s for s in lines if s.startswith('strata prefill cache release:')];assert len(rows)==1
        m=re.fullmatch(pattern,rows[0]);assert m and m[7]=='1'
        releases.append({'mode':mode,'repetition':rep,'physical_bytes':int(m[1]),'logical_tail_bytes':int(m[2]),'source':m[3],'suspend_ms':float(m[4]),'restore_ms':float(m[5]),'graph_ms':float(m[6]),'same_address':True,'message':rows[0],'mtp_messages':[s for s in lines if s.startswith(('strata mtp decode release:','strata mtp decode restore:'))]})
(out/'clean-release-messages.json').write_text(json.dumps(releases,indent=2)+'\n')
table='| Setting | Run | Prefill token/s | Decode token/s |\n| --- | --- | ---: | ---: |\n'
labels={'main-vmm-kept-ram':'Kept backing','main-vmm-half-ram':'Half / RAM','main-vmm-full-ram':'All / RAM','main-vmm-full-snapshot':'All / snapshot'}
for mode in modes:
    for s in clean['steps']:
        if s['argv'][2]==mode:table+=f"| {labels[mode]} | {s['argv'][4]} | {s['measurement']['prefill_tok_s']:.3f} | {s['measurement']['decode_tok_s']:.3f} |\n"
    mean=clean['clean_mean'][mode];table+=f"| {labels[mode]} | mean | {mean['prefill_tok_s']:.3f} | {mean['decode_tok_s']:.3f} |\n"
p=out/'README.md';text=p.read_text()
text=text.replace('This checkpoint retains completed\ncorrectness controls; clean timing remains pending. No speed claim or\nproduction adoption follows from diagnostic/state-capture durations.','This result retains twelve complete\ncorrectness controls followed by eight clean32K timing jobs. Diagnostic and\nstate-capture durations are excluded; the candidate remains unadopted.')
start=text.index('Completed three-run arms at this checkpoint: ');end=text.index('\nEach has a first',start)
text=text[:start]+'Completed three-run arms: '+', '.join(modes)+'.'+text[end:]
old='Full-RAM/snapshot and the clean matched\nkept/half-RAM/full-RAM/full-snapshot/reverse comparison are pending unless\ntheir completed controls appear above. No captured duration enters a speed\ncomparison. Graph recapture overhead will be included in clean prefill time.'
assert old in text
text=text.replace(old,'The full-snapshot arm also passes all\nthree state/head/output captures and complete payload verification. No captured\nduration enters a speed comparison. Graph recapture overhead is included in\nclean prefill time.\n\nEight fresh clean jobs run kept/half-RAM/full-RAM/full-snapshot and reverse.\nPayload checks, validation, debug/API logs, state/head dumps, profiler,\ntransfer profiling and extra waits are absent. The owned GDB/PTY observer\nis common. All20 jobs match output and MTP counts, exit normally and record\nno new xe fault.\n\n'+table+'\nThe [clean sequence](analysis/clean-sequence.json) preserves durations, means\nand relative changes. The [release messages](clean-release-messages.json)\nseparate physical backing, logical payload, suspend/restore and graph time.\nTwo repetitions per setting do not establish small differences beyond\nobserved variation or attribute decode variation to a prefill-only setting.\nThese measurements hold fixed the8192-token chunk and residual geometry;\nrelease alone does not test the benefit of enlarging chunks with freed VRAM.')
p.write_text(text)
for directory in [out,repro,parent/'code32k',parent]:
    p=directory/'manifest.json';r=json.loads(p.read_text());r.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Complete12 gated32K main-cache lease captures and8 clean matched timing jobs.',files={str(f.relative_to(directory)):{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(directory.rglob('*')) if f.is_file() and f!=p});p.write_text(json.dumps(r,indent=2)+'\n')
    for name,v in r['files'].items():
        f=directory/name;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256']
    print(directory.name,len(r['files']))
