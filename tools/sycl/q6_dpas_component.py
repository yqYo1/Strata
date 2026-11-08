#!/usr/bin/env python3
"""Owned compressed-Q6 C4 DPAS component; Main-authored, repository LICENSE.

No inference dispatch is changed. Compare the existing public column entry
against the new packed entry through ordinary production graphs. Keep the
failed model observer and closed row-copy decisions; neither proves a current
physical bottleneck. All stages require committed frozen source/recipes.
"""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import select
import shlex
import statistics
import struct
import subprocess
import sys
import tarfile
sys.dont_write_bytecode = True
import benchmark_runtime as python_runtime
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation
import build_compat_profile as selected
P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'q6-dpas-component-v5'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'q6_dpas_component_v6'
TARGET = 'sycl_q6_dpas_component'
HELPERS = ('tools/sycl/q6_dpas_component.py','tools/sycl/test_q6_dpas_component.py',
    'tools/sycl/benchmark_runtime.py','tools/sycl/measurement_guards.py',
    'tools/sycl/measurement_owner.py','tools/sycl/ur_validation.py',
    'AGENTS.md','docs/ARTIFACT_RETENTION.md')
UNITS = ('src/sycl/runtime.cpp','src/sycl/compat.cpp','src/sycl/command_profile.cpp',
    'src/kernels/sycl/native_mmvq.cpp','src/kernels/sycl/native_q5_head.cpp',
    'src/kernels/sycl/native_q3_esimd.cpp','src/kernels/sycl/native_q4_esimd.cpp',
    'src/kernels/sycl/native_q2_esimd.cpp','src/kernels/sycl/native_iq4xs_esimd.cpp',
    'src/kernels/sycl/native_iq_esimd.cpp','src/kernels/sycl/native_mmq_xmx.cpp',
    'src/kernels/sycl/iq.cpp','src/kernels/sycl/q6_dpas_component.cpp','bench/sycl/q6_dpas.cpp')
PACKAGES = ('libc6','libc6-dev','linux-libc-dev','libstdc++6',
            'libstdc++-13-dev','libgcc-s1','libgcc-13-dev','binutils')
FIELDS = ('wall_ns','cpu_ns','submit_wall_ns','submit_cpu_ns')
require, sha = guards.require, owner.sha


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)

def signature(pins):
    return dict(files=len(pins), aggregate_sha256=hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode()).hexdigest())

def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'runtime_pins', 'dependency_pins', 'commands')}
    out['source_inputs'] = signature(row['sources'])
    out['runtime_inputs'] = signature(row['runtime_pins'])
    out['dependency_inputs'] = signature(row['dependency_pins'])
    return out

def save(row):
    (ROOT / 'record.json').write_text(json.dumps(row, indent=2) + '\n')
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def packages():
    return subprocess.check_output(['/usr/bin/dpkg-query', '-W',
        '-f=${binary:Package} ${Version}\n', *PACKAGES], text=True, timeout=15).splitlines()

def dependency_pins(roots):
    pending, visited, pins, total, entries = list(map(Path, roots)), set(), {}, 0, 0
    while pending:
        root = pending.pop().resolve()
        if root in visited: continue
        require(root.is_dir(), 'dependency include/config directory missing')
        visited.add(root)
        for path in root.rglob('*'):
            entries += 1
            require(entries <= 32768, 'dependency traversal budget')
            if path.is_symlink() and path.is_dir():
                pending.append(path.resolve())
            elif path.is_file():
                name = str(path.resolve())
                if name in pins: continue
                total += path.stat().st_size
                require(len(pins) < 16384 and total <= 256 << 20, 'finite dependency inventory')
                pins[name] = sha(path)
    return pins

KERNEL_ENTRY = '_ZTSN6strata7kernels9component12_GLOBAL__N_18Q6DpasC4E'
CONTROLLER_FILES = ('tools/sycl/q6_dpas_component.py', 'tools/sycl/test_q6_dpas_component.py')


def embedded_dpas(blob):
    """Read exact linked ELF images and count reachable SPIR-V DPAS calls.

    This is a bounded format/callgraph check,not a full SPIR-V validator or
    evidence of final native ISA,hardware utilization or dynamic call count.
    """
    require(64 <= len(blob) <= 64 << 20, 'finite ELF extent')
    def span(offset, size):
        require(0 <= offset <= len(blob) and 0 <= size <= len(blob)-offset,
                'ELF file span')
        return blob[offset:offset+size]
    def name(table, at):
        require(0 <= at < len(table), 'ELF/SPIR-V string offset')
        end=table.find(b'\0',at)
        require(at <= end <= at+4096, 'bounded terminated name')
        return table[at:end].decode('utf-8')
    h=struct.unpack('<16sHHIQQQIHHHHHH',span(0,64))
    require(h[0][:6]==b'\x7fELF\x02\x01' and h[1] in (2,3) and h[2]==62
        and h[8]==64 and h[11]==64 and 0<h[12]<=4096 and h[13]<h[12],
        'supported ELF64 little-endian native executable')
    sections=[struct.unpack('<IIQQQQIIQQ',span(h[6]+64*i,64)) for i in range(h[12])]
    strings=sections[h[13]]
    require(strings[1]==3, 'ELF section-name string table')
    section_names=span(strings[4],strings[5])
    ids={i for i,q in enumerate(sections)
        if name(section_names,q[0]).startswith('__CLANG_OFFLOAD_BUNDLE__sycl-spir64')}
    require(ids, 'linked SYCL SPIR-V section')
    seen=set();matches=[]
    for table in sections:
        if table[1]!=2:continue
        require(table[9]==24 and table[5]%24==0 and table[5]//24<=1000000
            and table[6]<len(sections), 'finite ELF symbol table')
        string_section=sections[table[6]]
        require(string_section[1]==3, 'ELF symbol-name string table')
        symbol_names=span(string_section[4],string_section[5])
        for at in range(table[4],table[4]+table[5],24):
            symbol,_,_,section,value,size=struct.unpack('<IBBHQQ',span(at,24))
            if section not in ids or size==0:continue
            q=sections[section]
            require(q[3]<=value and size<=q[5] and value-q[3]<=q[5]-size,
                    'embedded image belongs to its ELF section')
            offset=q[4]+value-q[3];key=(offset,size)
            image=span(offset,size)
            if key in seen or image[:4]!=b'\x03\x02\x23\x07':continue
            seen.add(key)
            require(len(seen)<=1024, 'finite linked image inventory')
            if KERNEL_ENTRY.encode()+b'\0' not in image:continue
            require(20<=size and size%4==0, 'SPIR-V module extent')
            words=struct.unpack('<'+str(size//4)+'I',image)
            require(words[0]==0x07230203 and 0x10000<=words[1]<=0x10600
                and 0<words[3]<=10000000 and words[4]==0, 'SPIR-V module header')
            labels={};entries=[];functions=set();calls=[];function=None
            cursor=5;instructions=0
            while cursor<len(words):
                count,op=words[cursor]>>16,words[cursor]&65535
                require(count>0 and count<=len(words)-cursor, 'SPIR-V instruction extent')
                a=words[cursor+1:cursor+count];instructions+=1
                require(instructions<=1000000, 'finite SPIR-V instruction inventory')
                if op==5:
                    require(len(a)>=2 and 0<a[0]<words[3], 'SPIR-V OpName')
                    text=name(struct.pack('<'+str(len(a)-1)+'I',*a[1:]),0)
                    require(a[0] not in labels or labels[a[0]]==text, 'unambiguous SPIR-V names')
                    labels[a[0]]=text
                elif op==15:
                    require(len(a)>=3 and 0<a[1]<words[3], 'SPIR-V OpEntryPoint')
                    text=name(struct.pack('<'+str(len(a)-2)+'I',*a[2:]),0)
                    entries.append((a[0],a[1],text))
                elif op==54:
                    require(len(a)==4 and function is None and 0<a[1]<words[3]
                        and a[1] not in functions, 'SPIR-V function declaration')
                    function=a[1];functions.add(function)
                elif op==56:
                    require(not a and function is not None, 'SPIR-V function end')
                    function=None
                elif op==57:
                    require(len(a)>=3 and function is not None and 0<a[2]<words[3],
                            'SPIR-V function call')
                    calls.append((function,a[2]))
                cursor+=count
            require(function is None, 'closed SPIR-V function')
            if not any(text==KERNEL_ENTRY for _,_,text in entries):continue
            require(len(entries)==1 and entries[0][0]==6 and entries[0][1] in functions,
                    'one exact C4 kernel entrypoint')
            reachable=set();pending=[entries[0][1]];edges={}
            for caller,callee in calls:edges.setdefault(caller,[]).append(callee)
            while pending:
                f=pending.pop()
                if f in reachable:continue
                reachable.add(f);pending.extend(edges.get(f,()))
            dpas=[(caller,callee) for caller,callee in calls if caller in reachable and
                labels.get(callee,'').startswith(('llvm.genx.dpas.','llvm.genx.dpas2.'))]
            require(len(dpas)>=2 and all(callee in functions for _,callee in dpas),
                    'at least two reachable SPIR-V DPAS calls')
            matches.append(dict(image_symbol=name(symbol_names,symbol),image_offset=offset,
                image_bytes=size,image_sha256=hashlib.sha256(image).hexdigest(),
                entrypoint=KERNEL_ENTRY,reachable_functions=len(reachable),
                dpas_static_calls=len(dpas),dpas_callees=sorted({labels[v] for _,v in dpas})))
    require(len(matches)==1, 'one exact linked C4 device image')
    return dict(matches[0],device_images=len(seen),
        scope='Reachable static GenX DPAS calls in exact linked ELF SPIR-V;not final native ISA,utilization or dynamic call count.')


def prepare():
    """Prospectively admit the unchanged,unexecuted V5 native build for V6."""
    data=validation.committed(RESULT)
    require(KEY not in data, 'fresh V6 admission only')
    previous=data['q6_dpas_component_v5']
    require(previous['stage']=='failed' and not previous['passed'] and not previous['complete']
        and previous['ownership_closed'] and not previous['active'] and not previous['GPU_executed'],
        'preserve the failed V5 text-output gate')
    row=json.loads((ROOT/'record.json').read_text())
    require(row['stage']=='failed' and row['ownership_closed'] and not row['active']
        and not row['GPU_executed'] and all(owner.closed(c) for c in row['commands']),
        'original native owner must be closed')
    require([(c['label'],c['exit_code']) for c in row['commands']]==
        [('configure',0),('build',0),('codegen',0),('ir-text',1)],
        'only the original unexecuted successful native build is reusable')
    recipe_commit='3db30d9201f163dbadbc83f4c801890227d04043'
    original=json.loads(git('show',recipe_commit+':'+str(RESULT.relative_to(P))))['q6_dpas_component_v5']
    require(all(row[k]==original[k] for k in ('source_commit','configure','build','environment',
        'source_generation','tools','tool_versions','dependency_roots','system_packages')),
        'original native source/recipe/toolchain identity')
    require(signature(row['sources'])==original['source_inputs'], 'original native source inventory')
    archive=git('archive','--format=tar',row['source_commit'],*sorted(row['sources']))
    require(len(archive)<=64<<20, 'finite original Git archive')
    recovered={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isdir():continue
            require(item.isfile() and item.name in row['sources'], 'original regular native inputs')
            recovered[item.name]=hashlib.sha256(source.extractfile(item).read()).hexdigest()
    require(recovered==row['sources'] and
        all(sha(ROOT/'source'/n)==h for n,h in recovered.items()), 'exact original private native source')
    revision=git('rev-parse','HEAD').decode().strip()
    helpers={n:sha(P/n) for n in HELPERS}
    require(all((P/n).read_bytes()==git('show',revision+':'+n) for n in HELPERS),
            'commit exact new admission/reader controls')
    deltas={n:dict(native_snapshot_sha256=h,controller_sha256=sha(P/n))
        for n,h in row['sources'].items() if sha(P/n)!=h}
    require(set(deltas)<=set(CONTROLLER_FILES) and
        all(v['controller_sha256']==helpers[n] for n,v in deltas.items()),
        'only explicitly pinned controller/test files may differ from native source')
    native=previous['unexecuted_native_build']
    require(native['actual_inputs']['source_commit']==row['source_commit'] and
        sha(native['path'])==native['sha256'] and
        sha(native['actual_inputs']['path'])==native['actual_inputs']['sha256'],
        'original final ELF and actual compiler census')
    census_inputs=json.loads(Path(native['actual_inputs']['path']).read_text())
    require(native['actual_inputs']['uncovered']==0 and not census_inputs['generated']
        and all(sha(p)==h for k in ('compiled','links') for p,h in census_inputs[k].items())
        and {p for p in census_inputs['compiled'] if Path(p).suffix in ('.cpp','.cc','.c','.cu')}==
            {str((ROOT/'source'/n).resolve()) for n in UNITS}, 'all14 original compiled units and link bytes')
    database_path=ROOT/'build/compile_commands.json'
    database=json.loads(database_path.read_text())
    commands=[c for c in database if TARGET+'.dir' in c['command'] or 'strata_sycl_runtime.dir' in c['command']]
    require(len(commands)==14 and {str(Path(c['file']).relative_to(ROOT/'source')) for c in commands}==set(UNITS)
        and all('-fsycl' in shlex.split(c['command']) and '-fsycl-id-queries-range=size_t' in shlex.split(c['command']) for c in commands)
        and all('-fno-fast-math' in shlex.split(c['command']) and '-ffp-contract=off' in shlex.split(c['command'])
                for c in commands if TARGET+'.dir' in c['command']), 'original numerical/target flags')
    _,identity=selected.basis()
    require(identity==row['selected_native'], 'same selected90f production control')
    # Keep original native source/recipe and ELF. Pin the new controller revision
    # separately;none of the V5 host/device/performance gates is inherited.
    for key in ('commands','codegen_command','ir_text_command'):row.pop(key,None)
    row.update(controller_commit=revision,helpers=helpers,controller_only_deltas=deltas,
        binary={k:native[k] for k in ('path','sha256','bytes')},actual_inputs=native['actual_inputs'],
        compile_database=dict(path=str(database_path),sha256=sha(database_path),translation_units=14),
        native_reuse=dict(native_source_commit=row['source_commit'],native_recipe_commit=recipe_commit,
            original_failed_evidence_commit='8162122aa73602eb91a113e7533c2bee602efb98',
            native_inputs_unchanged=True,previous_host_device_qualification_inherited=False,
            reason='V5 native compile/link succeeded;only optional LLVM-text output failed. New embedded-SPIR-V gate binds actual linked device code.'),
        stage='prepared',results=[],timing=[],active=False,ownership_closed=True,passed=False,
        complete=False,GPU_executed=False,model_executed=False)
    row['provenance']['device_ir_reference']='ELF64 System V ABI and Khronos SPIR-V core OpName/OpEntryPoint/OpFunction/OpFunctionCall definitions;Main-authored bounded linked-image reader,no external implementation code.'
    old='Codegen must retain>=2DPAS intrinsic calls;this is compiler-IR evidence,not native ISA or utilization.'
    require(old in row['plan']['acceptance'], 'preserve original gate intent explicitly')
    row['plan']['acceptance']=row['plan']['acceptance'].replace(old,
        'The exact linked ELF C4 SPIR-V image must contain>=2reachable GenX DPAS call instructions from its sole kernel entry;this is packaged device-code evidence,not final native ISA or utilization.')
    row['plan']['order']='16CPU census/timing/ELF-SPIR-V reader controls;admit the unchanged original14-unit native build and exact linked DPAS callgraph;12CPU pack/shape controls;small traced3shape/2seed;full2shape/2seed;three fresh timing processes0/1/2.Commit each closed stage;stop first failure,no retries.'
    row['retention']='One reused component native build,not another engine generation. Keep useful samples,numeric/setup/callgraph/source/recipe decisions;retire successful raw/build output when the component/model-proposal consumer ends.'
    save(row)
    print('Prepared prospective V6 admission of exact unchanged V5 native ELF;commit before execution.',flush=True)

def current(row):
    require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact component state')
    require(all((P / n).read_bytes() == git('show', row['controller_commit'] + ':' + n)
                and sha(P / n) == h for n, h in row['helpers'].items()), 'component control drift')
    require(all(sha(ROOT / 'source' / n) == h for n, h in row['sources'].items()), 'native snapshot drift')
    deltas=row['controller_only_deltas']
    require(set(deltas)<=set(CONTROLLER_FILES) and all(
        (sha(P/n)==h if n not in deltas else
         deltas[n]['native_snapshot_sha256']==h and
         sha(P/n)==deltas[n]['controller_sha256']==row['helpers'][n])
        for n,h in row['sources'].items()), 'live native/controller source drift')
    _, identity = selected.basis()
    require(identity == row['selected_native'], 'selected production provenance drift')
    external = row['external_inputs']
    require(sha(external['path']) == external['sha256'] and
        all(sha(p) == h for p, h in json.loads(Path(external['path']).read_text()).items()),
        'external compiler/link input drift')
    require(all(sha(n) == h for n, h in row['runtime_pins'].items()), 'runtime provider drift')
    require(dependency_pins(row['dependency_roots']) == row['dependency_pins'], 'compiler/header/config dependency drift')
    require(all(str(Path(t['path']).resolve()) == t['realpath'] and sha(t['path']) == t['sha256']
                for t in row['tools'].values()) and packages() == row['system_packages'], 'toolchain drift')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot drift')
    if 'binary' in row:
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'component executable drift')
        actual = row['actual_inputs']
        require(sha(actual['path']) == actual['sha256'], 'actual component input census drift')
        census = json.loads(Path(actual['path']).read_text())
        require(all(sha(p) == h for k in ('compiled', 'generated', 'links') for p, h in census[k].items()),
            'actual component compiler/link bytes changed')
    database=row['compile_database']
    require(sha(database['path'])==database['sha256'], 'original compile database drift')
    if 'codegen' in row:
        require(embedded_dpas(Path(row['binary']['path']).read_bytes())==row['codegen'],
                'linked C4 SPIR-V callgraph drift')


def census(row, dependencies, link_text):
    """Reject actual compiler/link files outside the precommitted inventories."""
    allowed = {str((ROOT / 'source' / p).resolve()): h for p, h in row['sources'].items()}
    allowed.update({str(Path(p).resolve()): h for p, h in row['dependency_pins'].items()})
    allowed.update({str(Path(p).resolve()): h for p, h in
        json.loads(Path(row['external_inputs']['path']).read_text()).items()})
    allowed.update({v['realpath']: v['sha256'] for v in row['tools'].values()})
    compiled, generated, links, owned = {}, {}, {}, set()
    for line in dependencies.splitlines():
        if not line.startswith('    ') or not line.strip(): continue
        p = Path(line.strip()); p = (p if p.is_absolute() else ROOT / 'build' / p).resolve()
        value = sha(p)
        if p.is_relative_to(ROOT / 'build'): generated[str(p)] = value
        else:
            require(allowed.get(str(p)) == value, 'unfrozen actual compiler input: ' + str(p))
            compiled[str(p)] = value
    expected = {str((ROOT / 'source' / p).resolve()) for p in UNITS}
    observed = {p for p in compiled if Path(p).suffix in ('.cpp', '.c', '.cc', '.cu')}
    require(observed == expected and len(compiled) > 100 and not generated,
        'exact component-unit compiler census with no uncovered/generated input required')
    for line in link_text.splitlines():
        name = line.strip(); p = Path(name)
        if not name or any(c.isspace() for c in name) or not (
            p.is_absolute() or p.suffix in ('.o', '.a') or '.so' in p.name): continue
        p = (p if p.is_absolute() else ROOT / 'build' / p).resolve()
        if any(p.is_relative_to(ROOT / n) for n in ('build', 'tmp')):
            owned.add(str(p)); continue
        require(p.is_file() and allowed.get(str(p)) == sha(p), 'unfrozen actual linker input: ' + str(p))
        links[str(p)] = sha(p)
    require(len(links) > 8, 'actual external link census incomplete')
    path = ROOT / 'actual-inputs.json'
    path.write_text(json.dumps(dict(compiled=compiled, generated=generated, links=links,
        owned_link_products=sorted(owned)), indent=2) + '\n')
    return dict(path=str(path), sha256=sha(path), compiler_files=len(compiled),
        generated_files=len(generated), external_link_files=len(links), uncovered=0,
        translation_units=UNITS, source_commit=row['source_commit'])

def budget(wall, cpu, *, file_cap):
    # Owner includes the private source/build tree,not just raw output.
    return dict(wall=wall, cpu=cpu, rss_cap=8 << 30, text_cap=16 << 20,
                file_cap=file_cap, total_cap=512 << 20)


def collect_timing(text, index):
    lines=text.splitlines()
    clocks=[v for v in lines if v.startswith('CLOCK ')]
    require(len(clocks)==1 and re.fullmatch(r'CLOCK median_stamp_ns=\d+ floor_ns=10000000 process='+str(index),clocks[0]),'timing calibration')
    timer=int(clocks[0].split()[1].split('=')[1])
    samples=[v for v in lines if v.startswith('SAMPLE ')]
    order=(1,0) if index==1 else (0,1)
    expected=[(n,arm,s) for n in (640,248320) for arm in order for s in range(5)]
    require(len(samples)==len(expected)==20 and lines.count('TIMING passed=1 samples=20 model_speed=0 physical_traffic=0')==1,'complete timing extent')
    cells={}
    for line,(n,arm,s) in zip(samples,expected):
        pairs=[v.split('=',1) for v in line.split()[1:]];f=dict(pairs)
        require(len(f)==len(pairs) and set(f)=={'process','arm','k','n','columns','sample','launches','graph_launches','nodes_per_graph','route',*FIELDS,'wait_wall_ns','wait_cpu_ns'},'timing fields')
        require(f['route']=='compat_graph','actual graph route differs')
        ints={k:int(v) for k,v in f.items() if k!='route'}
        batch=4096 if n==640 else 256
        require(all(re.fullmatch(r'\d+',f[k]) for k in ints) and all(ints[k]==v for k,v in
            dict(process=index,arm=arm,k=2560,n=n,columns=4,sample=s,launches=batch,graph_launches=batch,nodes_per_graph=1).items()),'actual timing shape/order/replay count')
        require(ints['wall_ns']>=max(10000000,300*timer),'timing floor/overhead')
        require(ints['submit_wall_ns']+ints['wait_wall_ns']==ints['wall_ns'] and ints['submit_cpu_ns']+ints['wait_cpu_ns']==ints['cpu_ns'],'timing conservation')
        cells.setdefault(str(n),{}).setdefault(str(arm),[]).append([ints[k] for k in FIELDS])
    numeric=[v for v in lines if v.startswith('NUMERIC ')]
    require(len(numeric)==4 and all(v.startswith('NUMERIC passed=1 ') and v.endswith(' guards=1') for v in numeric),'complete timing numerical checks')
    return dict(index=index,timer_stamp_median_ns=timer,cells=cells)


def decision(timing):
    require(len(timing)==3 and [v['index'] for v in timing]==[0,1,2],'three fresh timing processes')
    cells={}
    for n in ('640','248320'):
        ratios=[statistics.median(a[0] for a in v['cells'][n]['1'])/statistics.median(a[0] for a in v['cells'][n]['0']) for v in timing]
        cells[n]=dict(candidate_over_reference_median_wall=ratios,component_gain_qualified=all(r<1 for r in ratios))
    return dict(cells=cells,head_component_gain=cells['248320']['component_gain_qualified'],model_speed_qualified=False,
        inference_adopted=False,physical_traffic_qualified=False,next='Only a winning full-head cell permits a new decode-only model proposal;placement/setup,phase preservation,task/agent and physical262144 remain unqualified.')


def execute(mode):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    expected={'controls':'prepared','build':'cpu-controlled','host':'built','check-small':'host-checked','check':'small-checked'}
    if mode.startswith('time'):
        index=int(mode[-1]);require(row['stage'] in ('checked','timing') and len(row['timing'])==index,'fresh ordered timing only')
    else:require(row['stage']==expected[mode],'fresh ordered component stage only')
    scope=None;cursor=None;outcome=dict(mode=mode,passed=False,host_probe_accepted=False)
    def persist():
        (ROOT/'execution.json').write_text(json.dumps(dict(mode=mode,active=bool(scope and scope.active),commands=scope.commands if scope else []),indent=2)+'\n')
    try:
        scope=owner.Owner(ROOT/('v6-owned-'+mode),persist)
        if mode=='controls':
            e,out,err=scope.run('reader-controls',row['host_command'],row['environment'],P,**row['host_budget'])
            text=out.read_text()+err.read_text()
            require(owner.closed(e) and e['exit_code']==0 and 'Ran 16 tests' in text and '\nOK\n' in text and 'skipped' not in text,'all16CPU reader controls')
            outcome.update(tests=16,skipped=0,native_execution_attempted=False);row['stage']='cpu-controlled'
        elif mode=='build':
            # Native compilation is original44715a/3db30d920. The new gate
            # inspects that unchanged final ELF and never refreshes its inputs.
            row['codegen']=embedded_dpas(Path(row['binary']['path']).read_bytes())
            outcome.update(native_recompiled=False,native_execution_attempted=False,
                translation_units=14,compiler_files=row['actual_inputs']['compiler_files'],
                external_link_files=row['actual_inputs']['external_link_files'],
                dpas_static_calls=row['codegen']['dpas_static_calls'])
            row['stage']='built'
        elif mode=='host':
            e,out,err=scope.run('host-pack',[row['binary']['path'],'--host'],row['environment'],ROOT,**row['host_budget'])
            require(owner.closed(e) and e['exit_code']==0 and out.read_text().splitlines().count('HOST passed=1 cases=12 GPU=0')==1,'all12CPU pack/shape controls')
            outcome.update(cases=12,native_execution_attempted=True,GPU_executed=False);row['stage']='host-checked'
        else:
            env=dict(row['environment'],STRATA_Q6_DPAS_GATE='1',LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/(mode+'-validation.log')))
            if mode!='check-small':env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
            args=['--time',str(index)] if mode.startswith('time') else ['--'+mode]
            cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines() if v.startswith('-- cursor: '))
            def interact(io,command):
                while True:
                    line=io.line(600)
                    if line.startswith('STRATA_Q6_DPAS_GATE '):break
                identity=owner.identity(command['pid'])
                require(identity and identity['pid']==identity['sid']==command['pid'] and all(identity[k]==command['direct_identity'][k] for k in ('pid','start_ticks','sid'))
                    and line=='STRATA_Q6_DPAS_GATE pid='+str(command['pid']),'actual native owner')
                require(str(Path('/proc',str(command['pid']),'exe').resolve())==row['binary']['path'],'actual executable')
                actual=dict(v.split('=',1) for v in Path('/proc',str(command['pid']),'environ').read_text().split('\0') if '=' in v)
                require(actual==env,'immutable native environment')
                mapped=set(python_runtime.code_maps(command['pid']));required={str(Path(env[k]).resolve()) for k in ('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER','UR_ADAPTERS_FORCE_LOAD','ZE_ENABLE_ALT_DRIVERS')}
                require(required<=mapped<=set(row['runtime_pins'])|{row['binary']['path']},'actual provider mappings')
                outcome.update(native_identity=identity,mapped_provider_count=len(mapped));row['GPU_executed']=True
                io.send(b'continue\n');io.proc.stdin.close()
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
                io.exit(1)
            outcome['native_execution_attempted']=True
            if not row['GPU_executed']:row['GPU_executed']=None
            e,out,err=scope.run('native',[row['binary']['path'],*args],env,ROOT,interaction=interact,**row['native_budget'])
            require(owner.closed(e) and e['exit_code']==0,'owned native component failed')
            outcome['native_exit_code']=0;text,diagnostic=out.read_text(),err.read_text()
            outcome['validation']=validation.preload_evidence(diagnostic,e['pid'])
            require(outcome['validation']['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'native validation provider')
            log=ROOT/(mode+'-validation.log');require(not log.exists() or log.stat().st_size==0,'parameter validation diagnostic')
            require(any(v.startswith('DEVICE name=') and 'B570' in v for v in text.splitlines()),'actual device')
            outcome['setup']=[v for v in text.splitlines() if v.startswith(('SETUP ','ADMISSION '))]
            outcome['numerical']=[v for v in text.splitlines() if v.startswith(('NUMERIC ','C1_CONTROL '))]
            if mode.startswith('time'):
                row['timing'].append(collect_timing(text,index));row['stage']='timing'
                if index==2:row.update(stage='closed',decision=decision(row['timing']),complete=True)
            else:
                small=mode=='check-small';cases=6 if small else 4
                require(text.splitlines().count('CHECK passed=1 small='+str(int(small))+' cases='+str(cases)+' route=compat_graph columns=4')==1
                    and sum(v.startswith('NUMERIC passed=1 ') for v in text.splitlines())==cases
                    and text.splitlines().count('C1_CONTROL passed=1 dpas_launched=0')==(3 if small else 2),'complete numeric/negative-control checks')
                if small:require('urEnqueueCommandBufferExp' in diagnostic and 'Q6DpasC4' in diagnostic,'actual small DPAS kernel graph trace')
                row['stage']='small-checked' if small else 'checked'
            outcome['host_probe_accepted']=True
        current(json.loads((ROOT/'record.json').read_text()));outcome['passed']=True
    except BaseException as error:
        outcome['error']=repr(error);row.update(stage='failed',passed=False,complete=False)
    finally:
        if cursor is not None and scope is not None:
            validation.finish(outcome,scope,cursor,row['boot'])
            if not outcome['passed'] or outcome.get('driver_after')!=row['driver'] or 'error' in outcome:
                outcome['passed']=False;row.update(stage='failed',passed=False,complete=False)
        row['results'].append(outcome);row['commands']=scope.commands if scope else []
        row['active']=bool(scope and scope.active);row['ownership_closed']=scope is None or (scope.active is None and all(owner.closed(e) for e in scope.commands))
        row['passed']=bool(row['complete'] and row['ownership_closed'] and outcome['passed']);persist();save(row)
    print(json.dumps(dict(mode=mode,stage=row['stage'],passed=outcome['passed'],error=outcome.get('error'))),flush=True)
    require(outcome['passed'],outcome.get('error','component stage failed'))


if __name__=='__main__':
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','controls','build','host','check-small','check','time0','time1','time2'),'component stage argument')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        prepare() if sys.argv[1]=='prepare' else execute(sys.argv[1])
