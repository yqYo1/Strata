import ast, copy, fcntl, hashlib, json, os, runpy, sys, time
from pathlib import Path
B=Path(__file__).parent
Q=B/'xestrata-diagnostic-contract-root-v3'
OUT=B/'xestrata-diagnostic-v8-cpu-root-v4'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ident(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir(mode=0o700)
    source=Q/'run_first_diagnostic_v8.py';compile(source.read_text(),str(source),'exec')
    m=runpy.run_path(str(source)); root,binary,argv,tokens,baseline,profile=m['preflight'](Q/'build-receipt-adapter.json')
    v4=ast.parse((Q/'run_first_diagnostic_v4.py').read_text());v5=ast.parse(source.read_text())
    get=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert '4096, 8192, 12288, 16384, 20480, 24576, 28672, 32767' in source.read_text()
    baseline=copy.deepcopy(baseline)
    baseline['requests'][0]['protocol']=[x for x in baseline['requests'][0]['protocol'] if not x.startswith('PP ')]
    baseline['requests'][0]['protocol'][:0]=[f'PP {p} 32768 100 100.0' for p in [4096,8192,12288,16384,20480,24576,28672,32767]]
    old=(B/'prepare_xestrata_diagnostic_contract_v3.py').read_text()
    block=old.split('    valid = copy.deepcopy',1)[1].split('    record = dict(',1)[0]
    block='    valid = copy.deepcopy'+block
    import textwrap
    ns=dict(copy=copy,baseline=baseline,m=m);exec(textwrap.dedent(block),ns)
    tests=ns['tests'];tests[0]['case']='synthetic4K-PP-geometry-plus-existing-finite-LP-template'
    original={k:os.environ.get(k)for k in ['UNLISTED_SECRET','PYTHONPATH','LD_PRELOAD','STRATA_TRACE','UR_LOG_LOADER','ONEAPI_DEVICE_SELECTOR']}
    try:
        os.environ.update({k:'unlisted-value-must-not-survive'for k in original})
        base,runtime=m['explicit_environments']()
        assert all(k not in base for k in original)
        assert all(k not in runtime for k in original if k!='ONEAPI_DEVICE_SELECTOR')
        assert runtime['ONEAPI_DEVICE_SELECTOR']=='level_zero:gpu'
        tests.append(dict(case='explicit-env-secret-and-runtime-injection',passed=True))
    finally:
        for k,v in original.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    sys.path.insert(0,str(m['OBSERVER']))
    from owned_gdb import OwnedGdb,process_identity
    script=OUT/'cpu_target.py';script.write_text('import time\nprint("READY",flush=True)\ntime.sleep(20)\n')
    record=dict(active=True,passed=False,complete=False,gpu_executed=False,model_executed=False,controller=ident(source),tests=tests)
    g=None; cleanup=None
    try:
        g=OwnedGdb(['/usr/bin/python3','-S',str(script)],OUT/'gdb',base)
        g.command('-gdb-set may-call-functions off');g.run()
        end=time.monotonic()+10
        while g.inferior is None and time.monotonic()<end:g.poll(.02)
        assert g.inferior
        actual=dict(x.decode().split('=',1)for x in Path('/proc',str(g.inferior['pid']),'environ').read_bytes().split(b'\0')if b'='in x)
        # Only report key names for differences, never inherited values.
        assert actual==base, 'CPU GDB initial environment differs: '+str(sorted(set(actual)^set(base)))
        tests.append(dict(case='real-owned-gdb-exact-initial-environment',passed=True))
        # Controlled CPU inferior is stopped through qualified owned close.
    except BaseException as e:
        record['error']=repr(e)
    finally:
        if g:cleanup=g.close()
        record['cleanup']=cleanup;record['active']=bool(not cleanup or cleanup['inferior_survived'] or cleanup['gdb_survived'])
        record['complete']=not record['active'];record['passed']=record['complete'] and not record.get('error')
        (OUT/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    assert record['passed']
    print(json.dumps(dict(passed=True,cases=len(tests),gpu_executed=False,record=str(OUT/'record.json'))))
