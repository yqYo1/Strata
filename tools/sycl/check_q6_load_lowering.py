#!/usr/bin/env python3
"""Compile-only Q6 packed-load discriminator; repository LICENSE.

Generate both whole translation units from the measured native source, changing
only Q6 packed-word reads. No product edit, linking, GPU work or speed claim.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
import build_verify_window_model as generation
import measurement_owner as supervision

P, B, RESULT, base = generation.P, generation.B, generation.RESULT, generation.base
ROOT = B / 'q6-byte-load-lowering-v1'
KEY = 'q6_byte_load_lowering_v1'
CORE = 'src/kernels/sycl/native_mmvq.cpp'
SELF = Path(__file__).resolve()
require, digest, write = generation.require, generation.digest, generation.write


def alternative(raw):
    text = raw.decode()
    require('#include <bit>' not in text and text.count('#include <cstdint>') == 1,
            'unexpected native include layout')
    text = text.replace('#include <cstdint>', '#include <bit>\n#include <cstdint>')
    helper = '''inline int load_q6_word(const void *ptr, int i32) {
  static_assert(sizeof(int) == sizeof(uint32_t));
  uint32_t word = 0;
  __builtin_memcpy(&word, static_cast<const unsigned char *>(ptr) + 4 * i32,
                   sizeof(word));
  return std::bit_cast<int>(word);
}
'''
    require(text.count('inline int load_int_b2(') == 1, 'one inherited packed-load helper')
    text = text.replace('inline int load_int_b2(', helper + 'inline int load_int_b2(')
    begin, end = text.index('struct Q6KTraits {'), text.index('int signed_grid_word(')
    body = text[begin:end]
    require(body.count('load_int_b2(') == 2 and body.count(
        'reinterpret_cast<const int *>(x[r.bq8_offset + 2 * i].qs)[iqs % 8]') == 1,
        'unexpected Q6 load sites')
    body = body.replace('load_int_b2(', 'load_q6_word(').replace(
        'reinterpret_cast<const int *>(x[r.bq8_offset + 2 * i].qs)[iqs % 8]',
        'load_q6_word(x[r.bq8_offset + 2 * i].qs, iqs % 8)')
    return (text[:begin] + body + text[end:]).encode()


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('controller_inputs','commands_executed')}
    pins=row['controller_inputs']
    out['controller_inputs']=dict(files=len(pins),aggregate_sha256=hashlib.sha256(
        json.dumps(pins,sort_keys=True).encode()).hexdigest())
    return out


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=compact(row);write(RESULT,data)


def inputs(row):
    require(base.controls(row['source_commit'],controller=SELF)==row['controller_inputs'],
            'commit and freeze comparison controls')
    native=json.loads((generation.ROOT/'record.json').read_text())
    require(base.committed(RESULT)[generation.KEY]==generation.compact(native) and
            digest(generation.ROOT/'record.json')==row['native_build_record_sha256'] and
            native['passed'] and native['complete'] and native['ownership_closed'],
            'qualified frozen phase build changed')
    allowed=generation.inputs(native)
    original=base.basis.build.git(P,'show',generation.phase.Q6+':'+CORE)
    for arm,raw in (('reference',original),('candidate',alternative(original))):
        path=ROOT/arm/'native_mmvq.cpp'
        require(path.read_bytes()==raw and digest(path)==row['generated_sources'][arm]['sha256'],
                'generated native arm changed')
        allowed[str(path)]=digest(path)
    return allowed


def prepare():
    data=base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(),'one fresh source-only discriminator')
    phase=data['verify_window_phase_device_v1']
    native=json.loads((generation.ROOT/'record.json').read_text())
    require(phase['passed'] and phase['complete'] and phase['ownership_closed'] and
            phase['fresh_faults']==[] and native['stage']=='built-not-executed' and
            data[generation.KEY]==generation.compact(native),'close the prior phase before preparation')
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    controls=base.controls(revision,controller=SELF)
    original=base.basis.build.git(P,'show',generation.phase.Q6+':'+CORE)
    require(hashlib.sha256(original).hexdigest()==native['native_sources'][CORE],
            'discriminator must use the actual measured native translation unit')
    database=json.loads(Path(native['compile_database']['path']).read_text())
    hits=[x for x in database if x['file']==str(generation.ROOT/'source'/CORE)]
    require(len(hits)==1,'one actual production compile recipe')
    command=shlex.split(hits[0]['command']);args=[];skip=False
    for arg in command:
        if skip:skip=False;continue
        if arg in ('-o','-MF','-MT','-MQ'):skip=True;continue
        if arg in ('-c','-MD','-MMD','-MP',hits[0]['file']):continue
        args.append(arg)
    require(not skip and '-O3' in args and '-std=c++20' in args and '-fsycl' in args,
            'preserve actual compiler,optimization,language and SYCL recipe')
    row=dict(stage='prepared',source_commit=revision,controller_inputs=controls,
        native_source_commit=native['source_commit'],q6_source_commit=generation.phase.Q6,
        native_build_record_sha256=digest(generation.ROOT/'record.json'),generated_sources={},
        environment=dict(native['recipe']['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache')),
        commands=[],budgets=dict(wall=240,cpu=300,rss_cap=6<<30,text_cap=2<<20,
            file_cap=48<<20,total_cap=192<<20),
        plan='Compile the full actual Q6 translation unit,reference versus local byte-based4B builtin memcpy plus bit_cast only in Q6KTraits,for host and spir64 with the actual -O3 C++20 recipe. No linking/execution. Require compiler/dependency/source closure and little-endian target data layouts. Compare complete optimized IR after replacing only declared source-path/ModuleID identities;retain raw/normalized hashes and load/TBAA counts. Any lowering difference needs a separate prospective numerical plan;equivalent IR does not prove surrounding Block producer lifetime or runtime/model correctness.',
        retention=dict(owner='Main',byte_budget=192<<20,
            consumer='Immediate source-contract and actual lowering decision before a Q6 load change',
            review_point='Retire successful compiler logs and IR after compact comparison and needed load evidence are recorded;retain a pair only for an explicit unresolved lowering question.'),
        passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False,
        model_executed=False,inference_adopted=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache','output','reference','candidate'):(ROOT/name).mkdir()
    for arm,raw in (('reference',original),('candidate',alternative(original))):
        source=ROOT/arm/'native_mmvq.cpp';source.write_bytes(raw)
        row['generated_sources'][arm]=dict(path=str(source),sha256=digest(source))
        for mode in ('host','device'):
            label=mode+'-'+arm;out=ROOT/'output'/(label+'.ll');dep=out.with_suffix('.d')
            row['commands'].append([label,[*args,'-fsycl-'+mode+'-only','-S','-emit-llvm',
                '-MD','-MF',str(dep),str(source),'-o',str(out)]])
    inputs(row);save(row)
    print('Prepared full-source host/SPIR load discriminator;commit generated plan before compile.',flush=True)


def normalized(path,source):
    text=path.read_text()
    return '\n'.join(x for x in text.replace(source,'<native_mmvq.cpp>').splitlines()
                     if not x.startswith('; ModuleID =') and not x.startswith('source_filename ='))


def compile_only():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY]==compact(row) and row['stage']=='prepared' and
            not (ROOT/'owned').exists(),'one committed compiler-only comparison')
    allowed=inputs(row);owner=supervision.Owner(ROOT/'owned')
    owner.persist=lambda:write(ROOT/'execution.json',dict(commands=owner.commands))
    try:
        results={}
        for label,argv in row['commands']:
            print('Load IR:',label,flush=True)
            entry,_,err=owner.run(label,argv,row['environment'],generation.ROOT/'build',**row['budgets'])
            require(supervision.closed(entry) and entry['exit_code']==0,'compile-only failure: '+err.read_text()[-3000:])
            dep=ROOT/'output'/(label+'.d')
            used={str(Path(x).resolve()) for x in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])}
            require(len(used)>100 and all(allowed.get(p)==digest(p) for p in used),'uncovered actual compiler input')
            arm=label.split('-',1)[1];ir=ROOT/'output'/(label+'.ll')
            text=normalized(ir,row['generated_sources'][arm]['path'])
            require('target datalayout = "e-' in text and
                ('target triple = "spir64-unknown-unknown"' in text if label.startswith('device')
                 else 'target triple = "x86_64-' in text),'unexpected target/byte order')
            results[label]=dict(raw_sha256=digest(ir),normalized_sha256=hashlib.sha256(text.encode()).hexdigest(),
                compiler_inputs=len(used),bytes=ir.stat().st_size,
                scalar_i16_loads=len(re.findall(r'\bload i16\b',text)),
                scalar_i32_loads=len(re.findall(r'\bload i32\b',text)),tbaa_mentions=text.count('!tbaa'))
        inputs(row)
        row.update(stage='closed',passed=True,complete=True,results=results,
            same_normalized_IR={mode:results[mode+'-reference']['normalized_sha256']==
                results[mode+'-candidate']['normalized_sha256'] for mode in ('host','device')})
    except BaseException as error:row.update(stage='failed',error=repr(error));raise
    finally:
        row.update(commands_executed=owner.commands,ownership_closed=owner.active is None and
            all(supervision.closed(x) for x in owner.commands));save(row)
    print('Compiler-only comparison closed;no numerical,model or performance qualification.',flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','compile'),'use prepare|compile')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        {'prepare':prepare,'compile':compile_only}[sys.argv[1]]()
