"""Root closed actual115-object cache-pairs/H production compile proof; no compile or GPU."""
from pathlib import Path
import datetime, hashlib, json, shlex, subprocess, sys, fcntl
B=Path(__file__).parent
lock=(B/'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
OLD=B/'native-dispatch-histogram-v0141-private-build-v1/record.json'
NEW=B/'cache-route-pairs-v0141-private-build-v1/record.json'
OUT=B/'cache-route-pairs-v0141-uniform-build-flags-v1.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert len(sys.argv)==2 and len(sys.argv[1])==64
assert sha(NEW)==sys.argv[1] and not OUT.exists()
assert sha(OLD)=='7f59396de97d54760539d88e97dac453fd4b1ccfa3ff1e923abcb4b1b72928b6'
old,new=[json.loads(p.read_text()) for p in (OLD,NEW)]
assert all(d['passed'] and not d['active'] and d['compiled_engine'] for d in (old,new))
assert old['source_head']=='96bd5bb4e054f6ddcf677fd96e161b499a013fd7'
assert new['source_head']=='66aaec3a66ad9ed12e6d042cffe743c47e364432'
assert new['complete'] and not new['cleanup'] and not new['survivors']
assert sha(new['binary'])==new['binary_sha256']
def inspect(d):
    root=Path(d['root']);build=Path(d['binary']).parent
    database=build/'compile_commands.json';assert sha(database)==d['compile_commands_sha256']
    assert sha(build/'build.ninja')==d['build_ninja_sha256']
    def paths(value):return value.replace(str(build),'<build>').replace(str(root),'<source>')
    def normalize(args):
        kept=[];i=0
        while i<len(args):
            arg=args[i]
            if arg in ['-MT','-MF']:
                assert i+1<len(args);i+=2;continue
            if arg in ['-MD','-MMD']:
                i+=1;continue
            kept.append(paths(arg));i+=1
        return kept
    declared={}
    rows=json.loads(database.read_text())
    for item in rows:
        args=item.get('arguments') or shlex.split(item['command'])
        assert args.count('-o')==1
        output=args[args.index('-o')+1]
        assert output==item['output']
        key=(paths(item['file']),paths(output))
        assert key not in declared,'unique source/object target pair required'
        declared[key]=normalize(args)
    text=subprocess.check_output(['/usr/bin/ninja','-C',str(build),'-t','commands','strata'],text=True,timeout=15)
    assert len(text)<3*1024**2
    reachable={};objects=[]
    for line in text.splitlines():
        args=shlex.split(line)
        if '-c' not in args:continue
        assert args.count('-c')==args.count('-o')==1
        source=args[args.index('-c')+1];output=args[args.index('-o')+1]
        key=(paths(source),paths(output));assert key not in reachable
        normalized=normalize(args)
        assert declared[key]==normalized,'reachable Ninja command must match declared flags'
        reachable[key]=normalized;objects.append(output)
    log=build/'.ninja_log';entries=[line.split('\t')[3] for line in log.read_text().splitlines()[1:] if line]
    compiled=[value for value in entries if value.endswith('.o')]
    assert len(compiled)==len(set(compiled))==len(objects)==115
    assert set(compiled)==set(objects),'fresh build must actually compile every reachable object'
    assert len({key[0] for key in reachable})==114
    proof={'declared_database_rows':len(rows),'reachable_compile_count':len(reachable),'compiled_object_count':len(compiled),'compile_commands_sha256':sha(database),'build_ninja_sha256':sha(build/'build.ninja'),'ninja_log_sha256':sha(log),'reachable_command_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'unbuilt_declarations':len(rows)-len(reachable)}
    return reachable,proof

a,old_proof=inspect(old);z,new_proof=inspect(new)
missing=sorted(set(a)^set(z))
differences=[{'source':key[0],'object':key[1],'old':a[key],'new':z[key]} for key in sorted(a.keys()&z.keys()) if a[key]!=z[key]]
assert len(a)==len(z)==115 and not missing and not differences
record={'active':False,'passed':True,'gpu_executed':False,'runtime_tested':False,'adopted':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checker_sha256':sha(__file__),'old_build_receipt_sha256':sha(OLD),'build_receipt_sha256':sha(NEW),'old_compile_count':len(a),'candidate_compile_count':len(z),'candidate_unique_source_count':len({key[0] for key in z}),'old_target_proof':old_proof,'candidate_target_proof':new_proof,'missing_sources':missing,'different_flags':differences,'candidate_binary_sha256':new['binary_sha256'],'normalization':'Only source/build prefixes and dependency outputs; every compiler, ISA, math, include, macro, source and object flag retained.'}
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['passed','candidate_compile_count','candidate_unique_source_count','missing_sources','different_flags','candidate_binary_sha256']}))
