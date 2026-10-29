#!/usr/bin/env python3
"""Prospective exact-executable linker-input proof for the pair model build.

Own source,following this project's existing resolved-link qualification.
No native execution. Trace GNU ld inputs and retained driver-generated inputs;
reuse numerical gates only for the exact original engine/fixture bytes.
"""
import ast
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
import build_qsa_pair_model as b
P,B,ROOT,RESULT,KEY=b.P,b.B,b.ROOT,b.RESULT,b.KEY
OUT=ROOT/'link-proof'
require,digest,write=b.require,b.digest,b.write


def compact(row):return {k:v for k,v in row.items() if k not in ['controller_inputs','commands_executed']}


def inputs(revision):
    pins=b.sources(revision);name=str(Path(__file__).resolve().relative_to(P))
    raw=b.model.git('show',revision+':'+name);require((P/name).read_bytes()==raw,'commit exact link controller')
    pins[name]=b.model.sha(raw);return pins


def compiled():
    record=b.runtime.validation.committed(RESULT);row=json.loads((ROOT/'record.json').read_text())
    require(b.compact(row)==record[KEY]['build'] and row['passed'] and row['ownership_closed'] and
        not row['active'] and all(b.runtime.supervision.closed(c) for c in row['commands_executed']),
        'closed source-bound compile required')
    allowed=b.admit(row)
    artifact=row['actual_compiler_inputs'];require(digest(artifact['path'])==artifact['sha256'],'compiler dependency evidence changed')
    for path,sha in json.loads(Path(artifact['path']).read_text()).items():require(digest(path)==sha,'compiled input changed:'+path)
    require(digest(ROOT/'build/compile_commands.json')==row['compile_database_sha256'],'effective compilation changed')
    for asset in [*row['binaries'].values(),row['fixture']]:require(digest(asset['path'])==asset['sha256'],'final executable changed')
    return row,allowed


def recipes():
    raw=subprocess.check_output(['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','commands',*b.TARGETS],text=True,timeout=20)
    result={}
    for line in raw.splitlines():
        args=shlex.split(line)
        if '-o' not in args or args[args.index('-o')+1] not in b.TARGETS:continue
        require(args[:2]==[':','&&'] and args[-2:]==['&&',':'],'unexpected CMake link shell')
        args=args[2:-2];name=args[args.index('-o')+1];require(name not in result,'duplicate link target');result[name]=args
    require(set(result)==set(b.TARGETS),'missing target link recipe')
    original=json.loads((ROOT/'record.json').read_text())
    result['fixture']=next(argv for name,argv in original['commands'] if name=='fixture-link')
    return result


def admit(row):
    old,allowed=compiled()
    require(inputs(row['source_commit'])==row['controller_inputs'],'link controller changed')
    require(recipes()==row['original_link_commands'],'effective link recipe changed')
    require(digest(row['inventory']['path'])==row['inventory']['sha256'],'prospective link inventory changed')
    frozen=json.loads(Path(row['inventory']['path']).read_text())
    require(len(frozen)==row['inventory']['files'],'link inventory count changed')
    for path,sha in frozen.items():require(digest(path)==sha,'pre-link object/tool/provider changed:'+path)
    allowed.update(frozen)
    for path,sha in row['build_rules'].items():require(digest(path)==sha,'generated compile/link rule changed')
    return old,allowed


def prepare():
    old,allowed=compiled();record=b.runtime.validation.committed(RESULT)
    require(not OUT.exists() and 'link_contract' not in record[KEY],'fresh link proof only')
    revision=b.model.git('rev-parse','HEAD').decode().strip();pins=inputs(revision)
    inventory={str(p.resolve()):digest(p) for pattern in ['*.o','*.a'] for p in (ROOT/'build').rglob(pattern)}
    inventory.update({str(p):digest(p) for p in [ROOT/'selected.o',ROOT/'fixture.o']})
    require(len(inventory)>100,'incomplete compiled object inventory')
    sdk=Path(old['compiler']['path']).parents[1]
    inventory.update({str(p.resolve()):digest(p) for p in (sdk/'bin').iterdir() if p.is_file()})
    rules={str(p):digest(p) for p in (ROOT/'build').rglob('*.ninja')}
    rules.update({str(ROOT/'build'/n):digest(ROOT/'build'/n) for n in ['CMakeCache.txt','compile_commands.json']})
    original=recipes();commands=[]
    for name,args in original.items():
        argv=list(args);argv[argv.index('-o')+1]=str(OUT/'output'/name)
        argv[1:1]=['-save-temps=obj','-v','-Wl,-t'];commands.append([name,argv])
    OUT.mkdir(mode=0o700);(OUT/'output').mkdir();(OUT/'tmp').mkdir();write(OUT/'inventory.json',inventory)
    row=dict(stage='prepared',source_commit=revision,controller_inputs=pins,
        inventory=dict(path=str(OUT/'inventory.json'),sha256=digest(OUT/'inventory.json'),files=len(inventory)),
        original_link_commands=original,build_rules=rules,commands=commands,
        environment=dict(old['environment'],TMPDIR=str(OUT/'tmp')),
        budgets=dict(wall=600,cpu=1800,rss_cap=10<<30,text_cap=32<<20,file_cap=512<<20,total_cap=4<<30),
        passed=False,active=False,ownership_closed=True,GPU_executed=False,
        criteria='Frozen original effective recipes,all compiled objects/archives,SDKtools and precompile providers. Relink three engine/runtime executables and fixture using ld trace and save-temps. Every consumed file is frozen or a retained actual producer output from pinned owned compiler tools. All four final ELF hashes must match original build exactly. No native/model qualification.',
        retention='Temporary same-generation link proof only;retain compact input/identity result,retire duplicate ELFs/generated products and successful driver traces after source/recipe recovery.')
    admit(row);write(OUT/'record.json',row);record[KEY]['link_contract']=compact(row);write(RESULT,record)
    print('Link proof prepared;commit before execution.',flush=True)


def compiler_products(stderr,allowed):
    produced=set();lists=set()
    for line in stderr.splitlines():
        if not line.startswith(' "'):continue
        args=shlex.split(line)
        if not args or not Path(args[0]).is_absolute():continue
        tool=Path(args[0]).resolve();require(allowed.get(str(tool))==digest(tool),'unfrozen executed compiler/link tool:'+str(tool))
        outputs=[args[args.index('-o')+1]] if '-o' in args else []
        outputs+=[a.split('=',1)[1] for a in args if a.startswith('-output=')]
        for a in args:
            if a.startswith('--out-file-list='):
                outputs.append(a.split('=',1)[1]);lists.add(Path(a.split('=',1)[1]))
        for output in outputs:
            path=Path(output);path=(path if path.is_absolute() else ROOT/'build'/path).resolve()
            if path.is_file():
                require(path.is_relative_to(OUT) or path.parent==Path('/home/yayoi') and
                    re.fullmatch(r'-[0-9a-f]{6}(?:_\d+)?\.(?:bc|table|txt|spv)',path.name),'compiler output escapes owned scope:'+str(path))
                produced.add(path)
    for listing in lists:
        if not listing.is_file():continue
        require(listing in produced and listing.stat().st_size<1<<20,'unowned or oversized compiler file list')
        for line in listing.read_text().splitlines():
            path=Path(line.strip())
            if not path.is_file():continue
            require(path.is_absolute() and (path.is_relative_to(OUT) or
                path.parent==listing.parent and path.stem.startswith(listing.stem+'_')),'compiler file-list output escapes scope')
            produced.add(path.resolve())
    require(len(produced)<4096 and sum(p.stat().st_size for p in produced)<4<<30,'generated input budget')
    return produced


def run():
    record=b.runtime.validation.committed(RESULT);row=json.loads((OUT/'record.json').read_text())
    require(row['stage']=='prepared' and compact(row)==record[KEY]['link_contract'] and not (OUT/'owned').exists(),'fresh committed link proof required')
    old,allowed=admit(row);owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None);write(OUT/'record.json',dict(row,commands_executed=owner.commands if owner else []))
    try:
        owner=b.runtime.supervision.Owner(OUT/'owned',persist);outputs={};links={}
        for name,argv in row['commands']:
            print('CPU link proof:'+name,flush=True)
            entry,out,err=owner.run(name,argv,row['environment'],ROOT/'build',**row['budgets'])
            require(b.runtime.supervision.closed(entry) and entry['exit_code']==0,'link proof failed:'+err.read_text(errors='replace')[-3000:])
            produced=compiler_products(err.read_text(),allowed);observed={};generated={}
            for line in out.read_text().splitlines():
                line=line.strip()
                if not line:continue
                path=Path(line);path=(path if path.is_absolute() else ROOT/'build'/path).resolve()
                require(path.is_file(),'unrecognized linker input:'+line);sha=digest(path)
                if str(path) in allowed:
                    require(allowed[str(path)]==sha,'resolved provider differs');observed[str(path)]=sha
                else:
                    require(path in produced and path.is_relative_to(OUT),'unfrozen actual linker input:'+str(path));generated[str(path)]=sha
            require(len(observed)>8,'incomplete actual input trace')
            path=OUT/'output'/name;expected=old['fixture'] if name=='fixture' else old['binaries'][name]
            outputs[name]=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,matches_original=digest(path)==expected['sha256'])
            require(outputs[name]['matches_original'],'relinked ELF differs;new native qualification required')
            links[name]=dict(frozen_inputs=observed,compiler_generated_inputs=generated,uncovered_inputs=0,
                owned_compiler_products={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(produced)})
        admit(row);write(OUT/'resolved-inputs.json',links)
        row.update(stage='closed',passed=True,outputs=outputs,
            actual_links=dict(path=str(OUT/'resolved-inputs.json'),sha256=digest(OUT/'resolved-inputs.json'),
                frozen_provider_counts={n:len(v['frozen_inputs']) for n,v in links.items()},uncovered_inputs=0))
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row['ownership_closed']=owner is None or owner.active is None and all(b.runtime.supervision.closed(c) for c in owner.commands)
        row['passed']=bool(row['passed'] and row['ownership_closed']);persist();record[KEY]['link_contract']=compact(row);write(RESULT,record)
    print(json.dumps({k:row.get(k) for k in ['stage','passed','ownership_closed','error']}),flush=True)
    require(row['passed'],'actual-link proof did not qualify')


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);{'prepare':prepare,'run':run}[sys.argv[1]]()
