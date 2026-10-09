"""Compare all 115 objects (114 sources) reachable from the strata target."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess

B=Path(__file__).parent
OLD=B/'native-expert-copy-v0141-private-build-v1/record.json'
NEW=B/'repeat-capture-v0141-private-build-v1/record.json'
OUT=B/'repeat-capture-v0141-uniform-build-flags-v1.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(OLD)=='74b48ad90996c942fd74fa826d22dfecf76145d8dec82a40cfa0d35dc9b65fad'
assert not OUT.exists()
old,new=[json.loads(p.read_text()) for p in (OLD,NEW)]
assert all(d['passed'] and not d['active'] and d['compiled_engine'] for d in (old,new))
assert old['source_head']=='6caa1421f9212750a425fe1729139ffdde6e9f9a'
assert new['source_head']=='9c2ebde89e5157c81a9c9ae135719452a0244ca3'
assert sha(NEW)=='f7d8b0b1efec0aa7ad21fb937d92166db608fe0c8a9253a6998c9a7e24f5b166'

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
qualified_path=B/'native-expert-copy-v0141-uniform-build-flags-v1.json'
assert sha(qualified_path)=='54a1a491a7f00874a90af1b7776d0654b8b001e9aea3da99f6e0b1dfe0a83e53'
qualified=json.loads(qualified_path.read_text());q={key.replace('<root>','<source>'):[token.replace('<root>','<source>') for token in value] for key,value in qualified['candidate_commands'].items()}
assert set(q)=={key[0] for key in a} and all(any(value==expected for key,value in a.items() if key[0]==source) for source,expected in q.items()),'legacy114-source proof must match corresponding old compiled object flags'
duplicate_sources={source:[key[1] for key in a if key[0]==source] for source in q if sum(key[0]==source for key in a)>1}
assert len(duplicate_sources)==1 and next(iter(duplicate_sources)).endswith('/sycl/src/core/vmm.cpp')
record={'active':False,'passed':True,'gpu_executed':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All115 actual strata-target compilation objects from114 distinct source files, not all159 configured declarations. vmm.cpp has two separate core/engine objects with intentionally different Native macro definitions, unchanged between builds. Reachable Ninja flags agree with compile_commands and every object appears once in the fresh build log. No compilation command executed by this checker. Legacy114-source proof projects one vmm object; this proof compares both.','old_build_receipt_sha256':sha(OLD),'build_receipt_sha256':sha(NEW),'checker_sha256':sha(Path(__file__)),'old_compile_count':len(a),'candidate_compile_count':len(z),'old_unique_source_count':len(q),'candidate_unique_source_count':len({key[0] for key in z}),'duplicate_source_objects':duplicate_sources,'old_target_proof':old_proof,'candidate_target_proof':new_proof,'missing_sources':missing,'different_flags':differences,'normalization':'Only worktree/build prefixes and -MT/-MF argument pairs plus -MD/-MMD dependency switches; -o object target and every compiler/source flag retained.','original_qualified_flags_receipt_sha256':sha(qualified_path),'candidate_binary_sha256':new['binary_sha256'],'adopted':False}
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['passed','gpu_executed','candidate_compile_count','candidate_unique_source_count','missing_sources','different_flags','candidate_binary_sha256']}))
