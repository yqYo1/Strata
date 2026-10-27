#!/usr/bin/env python3
"""CPU-only paired SYCL/IGC lowering probe. Never execute a device program."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import measurement_guards as guards
import measurement_owner as supervision

P=Path(__file__).resolve().parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT=B/'qsa-exp2-lowering-v1'
RESULT=P/'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY='qsa_exp2_lowering'
CPP=P/'bench/sycl/qsa_exp2_lowering.cpp'
NATIVE='6e40afc1beddd059ee518617f6c1efce1da14aca'
require=guards.require


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=P,timeout=60)
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def pins(revision):
    names=['bench/sycl/qsa_exp2_lowering.cpp','tools/sycl/check_qsa_exp2_lowering.py',
        'tools/sycl/measurement_owner.py','tools/sycl/measurement_guards.py']
    result={}
    for name in names:
        raw=git('show',revision+':'+name)
        require(raw==(P/name).read_bytes(),'commit probe source first: '+name)
        result[name]=hashlib.sha256(raw).hexdigest()
    return result


def baseline():
    row=json.loads((B/'main-native-gdn-engine/record.json').read_text())
    require(row.get('binary_origin_source_commit',row['source_commit'])==NATIVE,'selected6e40 required')
    artifacts=[row['binary'],*row['fixtures'].values()]
    require(len(artifacts)==13,'selected fixture inventory differs')
    for a in artifacts:require(digest(a['path'])==a['sha256'],'selected native artifact changed')
    return dict(native_source=NATIVE,binary=row['binary'],compiler=row['compiler'],artifacts_verified=13)


def admission(row):
    require(pins(row['source_commit'])==row['inputs'] and baseline()==row['baseline'],'probe source/baseline changed')
    manifest=row['dependencies']
    require(digest(manifest['path'])==manifest['sha256'],'compiler dependency manifest changed')
    deps=json.loads(Path(manifest['path']).read_text());require(len(deps)==manifest['files'],'dependency count changed')
    for dep in deps:require(digest(dep['path'])==dep['sha256'],'compiler/header/library changed: '+dep['path'])
    require(digest('/usr/bin/ocloc')==row['tools']['ocloc']['sha256'],'offline compiler changed')


def persist_scope(row):
    container=json.loads(RESULT.read_text());container[KEY]=row;write(RESULT,container);write(ROOT/'record.json',row)


def prepare():
    original=RESULT.read_bytes();container=json.loads(original)
    require(original==git('show','HEAD:'+str(RESULT.relative_to(P))) and KEY not in container and not ROOT.exists(),
        'fresh committed lowering probe only')
    require(container['complete'] and container['passed'] and container['selected_qsa_service']['analysis_passed'] and
        container['overhead']['passed'],'source-reconciled service/cost gate required')
    selected=baseline();compiler=Path(selected['compiler']['path']);sdk=compiler.parent.parent
    require(digest(compiler)==selected['compiler']['sha256'],'qualified compiler bytes changed')
    source=git('rev-parse','HEAD').decode().strip();source_pins=pins(source)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','native','asm'): (ROOT/name).mkdir()
    flags=[str(compiler),'-std=c++20','-O3','-DNDEBUG','-fsycl','-fsycl-device-only',
        '-fsycl-id-queries-range=size_t','-fno-fast-math','-ffp-contract=off']
    env=dict(PATH=str(compiler.parent)+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C',LC_ALL='C',
        LD_LIBRARY_PATH=str(sdk/'lib')+':/opt/intel/oneapi/mkl/2026.1/lib',
        CPLUS_INCLUDE_PATH=str(sdk/'include'),PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),
        SYCL_CACHE_PERSISTENT='0')
    row=dict(stage='preparing',source_commit=source,inputs=source_pins,baseline=selected,
        environment=env,tools=dict(compiler=selected['compiler'],
            translator=dict(path=str(compiler.parent/'llvm-spirv'),sha256=digest(compiler.parent/'llvm-spirv')),
            ocloc=dict(path='/usr/bin/ocloc',sha256=digest('/usr/bin/ocloc'))),
        flags=flags,commands=dict(
            preprocess=flags+['-M','-MT','qsa_exp2','-MF',str(ROOT/'headers.d'),str(CPP)],
            llvm_text=flags+['-S','-emit-llvm',str(CPP),'-o',str(ROOT/'exp2.ll')],
            llvm_bitcode=flags+['-c','-emit-llvm',str(CPP),'-o',str(ROOT/'exp2.bc')],
            spirv=[str(compiler.parent/'llvm-spirv'),str(ROOT/'exp2.bc'),'-o',str(ROOT/'exp2.spv')],
            native=['/usr/bin/ocloc','compile','-file',str(ROOT/'exp2.spv'),'-spirv_input','-device','0xe20c',
                '-output','exp2','-out_dir',str(ROOT/'native')]),
        original_source_flags={name:hashlib.sha256(git('show',NATIVE+':'+name)).hexdigest()
            for name in ('CMakeLists.txt','cmake/sycl_backend.cmake','src/kernels/sycl/qsa_bf16_prefix.cpp')},
        criteria='Compile the same3kernels in one TU:regular exp2,native exp2,and conditional negative-domain seam. Compare actual LLVM/SPIRV and B570 offline ISA,with no fast-math/FTZ global change. If native and regular lower identically,do not select a math rewrite. A difference alone qualifies neither supported error/domain nor speed.',
        limits='CPU-only compiler evidence. No queue/device program/model execution,numerical/range/FTZ/performance/adoption or physical-context qualification. The bounded native seam is hypothetical,not enabled in inference.',
        scope='The independently installed Intel SDK under an old Xe checkout is a toolchain dependency;no Xe application source copied or used.',
        budgets=dict(wall=180,cpu=180,rss_cap=4<<30,text_cap=4<<20,file_cap=32<<20,total_cap=64<<20),
        retention='Current6e40+12fixtures remain untouched. This tiny offline module and ISA serve only this named lowering comparison;retire generated binaries/IR/asm/logs after recording useful source/flags/opcodes/hashes. No runnable project generation is built.',
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False)
    owner=None
    def persist():write(ROOT/'prepare-record.json',dict(row,active=bool(owner and owner.active),owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'prepare-owned',persist)
    try:
        entry,_,_=owner.run('preprocess',row['commands']['preprocess'],env,P,**row['budgets'])
        require(supervision.closed(entry) and entry['exit_code']==0,'probe preprocessing failed')
        raw=(ROOT/'headers.d').read_text().replace('\\\n',' ').split(':',1)[1]
        require('\\' not in raw,'unexpected dependency escaping')
        paths={Path(n).resolve() for n in raw.split()}
        binaries=[compiler,compiler.parent/'llvm-spirv',Path('/usr/bin/ocloc'),
            Path('/usr/lib/x86_64-linux-gnu/libocloc.so'),Path('/usr/lib/x86_64-linux-gnu/libigc.so.2'),
            Path('/usr/lib/x86_64-linux-gnu/libigdfcl.so.2')]
        for pattern in ('libopencl-clang.so*','libiga64.so*'):
            binaries.extend(Path('/usr/lib/x86_64-linux-gnu').glob(pattern))
        for binary in binaries:
            binary=binary.resolve();require(binary.is_file(),'compiler library missing: '+str(binary));paths.add(binary)
            text=subprocess.check_output(['/usr/bin/ldd',str(binary)],text=True,env=env,timeout=15)
            require('not found' not in text,'compiler dependency missing')
            paths.update(Path(n).resolve() for n in re.findall(r'(/[^\s()]+)',text))
        deps=[dict(path=str(p),sha256=digest(p)) for p in sorted(paths)]
        manifest=ROOT/'compiler-inputs.json';write(manifest,deps)
        row.update(stage='prepared',dependencies=dict(path=str(manifest),files=len(deps),sha256=digest(manifest)))
        admission(row)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        persist();persist_scope(row)
    print(json.dumps({k:row.get(k) for k in ('stage','ownership_closed','dependencies','error')}),flush=True)
    return 0 if row['stage']=='prepared' and row['ownership_closed'] else 1


def run():
    raw=RESULT.read_bytes();container=json.loads(raw);row=container[KEY]
    require(raw==git('show','HEAD:'+str(RESULT.relative_to(P))) and row==json.loads((ROOT/'record.json').read_text()) and
        row['stage']=='prepared' and not (ROOT/'execution.json').exists(),'fresh committed prepared probe only')
    admission(row);owner=None
    def persist():write(ROOT/'execution.json',dict(row,active=bool(owner and owner.active),owned_commands=owner.commands if owner else []))
    owner=supervision.Owner(ROOT/'owned',persist);row['stage']='running';persist()
    try:
        for name in ('llvm_text','llvm_bitcode','spirv','native'):
            entry,_,_=owner.run(name,row['commands'][name],row['environment'],P,**row['budgets'])
            require(supervision.closed(entry) and entry['exit_code']==0,'offline phase failed: '+name)
        binaries=list((ROOT/'native').glob('*.bin'));require(len(binaries)==1,'one generated device binary required')
        require(binaries[0].read_bytes().startswith(b'\x7fELF'),'offline output is not an ELF device module')
        command=['/usr/bin/ocloc','disasm','-file',str(binaries[0]),'-dump',str(ROOT/'asm'),'-device','0xe20c']
        row['generated_native_input']=dict(path=str(binaries[0]),sha256=digest(binaries[0]));row['disasm_command']=command;persist()
        entry,_,_=owner.run('disasm',command,row['environment'],P,**row['budgets'])
        require(supervision.closed(entry) and entry['exit_code']==0,'offline disassembly failed')
        files=sorted((ROOT/'asm').rglob('*.asm'));require(files,'no actual ISA emitted')
        row['assembly']=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
        text=(ROOT/'exp2.ll').read_text()
        row['llvm_exp2_calls']=sorted(set(re.findall(r'@([^\s(]*exp2[^\s(]*)\(',text)))
        require(row['llvm_exp2_calls'],'no exp2 LLVM lowering found')
        admission(row);row.update(stage='closed',passed=True,complete=True)
    except BaseException as error:row.update(stage='failed',error=repr(error))
    finally:
        row['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        row['passed']=bool(row['passed'] and row['ownership_closed']);persist();persist_scope(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','ownership_closed','llvm_exp2_calls','assembly','error')}),flush=True)
    return int(not row['passed'])


if __name__=='__main__':
    require(sys.argv[1:] in (['prepare'],['run']),'prepare|run')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        sys.exit(prepare() if sys.argv[1]=='prepare' else run())
