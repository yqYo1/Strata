from pathlib import Path
import datetime, fcntl, hashlib, json, subprocess, types, traceback
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010')
O=B/'ple-offline-census-v2-owned'
parent=B/'run_gdn_gate_factor_probe_v2.py'
def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not O.exists();O.mkdir()
    assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='495369cf4dc6e6da564095a1c26504dd3694e3f2'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    source=parent.read_text().split('\ndef parse_probe(',1)[0].replace('assert rss <= 2 << 30','assert rss <= 1 << 30')
    m=types.ModuleType('plecensusowner');exec(compile(source,str(parent),'exec'),m.__dict__);m.W=O;o=m.Owner(O)
    binary=O/'ple-ngram-oracle';obj=B/'ple-offline-census-source-object-build-v1/ngram.cpp.o'
    assert ident(obj)['sha256']=='0e56a389046c1825d66ce82d7fec2f60cefa6725e6ec36a235b79396fa3141e3'
    files=[B/x for x in ('ple_ngram_oracle_v2.cpp','ple_offline_census_v2.py','qualify_ple_census_v2.py','run_ple_census_v2.py')]
    r=dict(active=True,complete=False,passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller=ident(Path(__file__)),parent_owner=ident(parent),modified_owner_sha256=hashlib.sha256(source.encode()).hexdigest(),commands=o.commands,sources={str(p):ident(p) for p in files},original_ngram_object=ident(obj),scope='CPU-only independently checked offline PLE request geometry. No hardware/performance/GPU/cache/shard data or fullcontext inference claim.',gpu_work_submitted=False,model_inference=False,adopted=False,limits=dict(wall_per_process_seconds=360,RSS_session_bytes=1<<30,CPU_each_seconds=[120,121],AS_each_bytes=16<<30,output_file_bytes=32<<20,text_per_process_bytes=4<<20))
    def save():(O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    o.persist=save;env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    def run(label,args,wall=360,expected=0):
        cmd,so,se=o.run(label,args,env,wall=wall,text_cap=4<<20,file_cap=32<<20)
        assert m.completed(cmd) and cmd['normal_exit'] and cmd['exit_code']==expected,(label,cmd)
        return so,se
    try:
        run('compile',['/usr/bin/g++','-std=c++20','-O2','-Wall','-Wextra','-Wpedantic','-ffunction-sections','-fdata-sections','-I'+str(W/'include'),str(B/'ple_ngram_oracle_v2.cpp'),str(obj),'-Wl,--gc-sections','-o',str(binary)],120)
        so,_=run('dependencies',['/usr/bin/readelf','-d',str(binary)],30);r['dependencies']=so.read_text();assert not any(x in r['dependencies'] for x in ('libsycl','libze_loader','libur_adapter','libcuda','libhip'))
        r['binary']=dict(path=str(binary),**ident(binary));save()
        so,_=run('qualify',['/usr/bin/python3',str(B/'qualify_ple_census_v2.py'),str(binary),str(O/'qualification')],120)
        r['qualification']=json.loads(so.read_text());assert r['qualification']['passed'];save()
        cases=[('actual32K',B/'coding-review-32k-tokens.txt','32k-insert198',32768,True,['--insert198']),('context262144',B/'full-context-copy-off/coding-context-256k-tokens.txt','256k',262143,True,[])]
        r['cases']=[]
        for label,input_,fixture,prefix,tail,extra in cases:
            rows=O/(label+'.rows.bin')
            so,_=run(label+'-oracle',[str(binary),str(input_),str(prefix+int(tail)),str(rows),*extra]);oracle=json.loads(so.read_text());assert oracle['success'] and oracle['row_count']==(prefix+int(tail))*16
            args=['/usr/bin/python3',str(B/'ple_offline_census_v2.py'),'--input',str(input_),'--fixture',fixture,'--prefix',str(prefix),'--oracle-rows',str(rows),'--oracle-bin',str(binary),'--include-tail']
            so,_=run(label+'-census',args);result=json.loads(so.read_text());assert result['success'] and result['all_oracle_rows_compared']==(prefix+1)*16 and result['pp_tokens']==prefix
            assert result['totals_sum_of_independent_chunks']['row_requests']==prefix*16 and result['chunk_count']==(prefix+8191)//8192
            r['cases'].append(dict(label=label,oracle=oracle,oracle_rows=dict(path=str(rows),**ident(rows)),census_path=str(so),census_identity=ident(so),summary=result['totals_sum_of_independent_chunks'],chunk_count=result['chunk_count'],all_rows_compared=result['all_oracle_rows_compared'],first_chunk=result['chunks'][0],last_chunk=result['chunks'][-1],tail=result['separate_optional_tail']));save()
        assert all(ident(p)==r['sources'][str(p)] for p in files)
        r.update(passed=True,complete=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as e:r['error']=type(e).__name__+': '+str(e);r['traceback']=traceback.format_exc()
    finally:r['active']=o.active is not None;save();print(json.dumps({k:r.get(k) for k in ('active','complete','passed','error','binary','qualification')},separators=(',',':')))
    if not r['passed']:raise SystemExit(1)
