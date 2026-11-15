#!/usr/bin/env python3
"""Build/attest the selected model plus owned GDN16 integration. Repository MIT.

This CPU-only constructor owns a complete generation (engine and same-build
fixtures). GPU validation and model qualification are separate consumers.
"""
import copy
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import time

import retained_prefill_candidate as retained
import gdn_model_sources as source

base, P, B, RESULT = retained.base, retained.P, retained.B, retained.RESULT
require, digest, write = base.require, base.digest, base.write
ROOT = B/'gdn16-model-v1'
KEY = 'gdn16_model_build_v1'
COMPONENT = 'sycl_gdn_storage_pipeline_bench'
CONTROLLER = Path(__file__).resolve()


def save(row):
    write(ROOT/'record.json',row)
    data=json.loads(RESULT.read_text());data[KEY]=base.compact(row);write(RESULT,data)


def native_inputs(revision):
    full,native=source.sources(revision)
    for name,raw in native.items():
        path=ROOT/'source'/name
        require(path.is_file() and not path.is_symlink() and path.read_bytes()==raw,
                'frozen projected source changed: '+name)
    return full,native


def controls(row, *, reuse_revision=None):
    changes=('tools/sycl/gdn_model_build.py',) if reuse_revision is not None else ()
    require(base.controls(row['source_commit'],controller=CONTROLLER,
            reuse_revision=reuse_revision,controller_changes=changes)==row['controller_inputs'],
            'GDN model build controls changed')
    _,native=native_inputs(row['source_commit'])
    require({n:hashlib.sha256(v).hexdigest() for n,v in native.items()}==row['native_sources'] and
            source.input_pins(row['source_commit'])==row['source_inputs'], 'projected native inputs changed')


def allowed_inputs(row, *, retained_parent=True):
    if retained_parent:
        controls(row);current,providers,_=retained.verify()
        require(current['binaries']==row['retained_binaries'] and current['source_commit']==retained.NATIVE,
                'selected engine/fixtures changed')
    else:
        # Reuse this closed generation after its parent executables retire.
        # Only this committed Python controller differs from the build inputs;
        # projected native/compiler/link inputs and all nine ELFs still match.
        revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
        controls(row,reuse_revision=revision)
        data=base.committed(RESULT);gate=data['gdn16_model_component_device_v1']
        require(gate['passed'] and gate['complete'] and not gate['active'] and
                gate['ownership_closed'] and not gate['fresh_faults'] and
                all(v['passed'] for v in gate['processes']) and
                gate['native_source_commit']==row['source_commit'] and
                gate['binary']==row['binaries'][COMPONENT],
                'closed same-build production-wrapper qualification required')
        providers=retained.runtime.inventory(gate['runtime_manifest'])
        require(retained.runtime.qualified_probe(data,providers)==row['native_validation_probe'] and
                gate['native_validation_probe']==row['native_validation_probe'],
                'same-generation qualified native probe changed')
        require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot']==gate['boot'] and
                base.guards.driver_guard()==row['driver']==gate['driver'],
                'qualified generation boot/driver changed')
    allowed=base.sdk()
    allowed.update(retained.runtime.inventory(row['external_manifest']));allowed.update(providers)
    full,_=native_inputs(row['source_commit'])
    for name,sha in row['native_sources'].items():allowed[str((ROOT/'source'/name).resolve())]=sha
    for group in full['dependencies']:
        for name,sha in group['source_sha256'].items():
            path=(Path(group['root'])/name).resolve()
            require(digest(path)==sha,'external dependency changed: '+str(path));allowed[str(path)]=sha
    allowed.update(row['generated_expected'])
    require(base.host_link_inventory.inventory(base.COMPILER,row['environment'])==row['host_link_inputs'],
            'actual compiler host linker inputs changed')
    require(subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',
        'intel-oneapi-dpcpp-cpp-2026.2','intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
        'intel-oneapi-umf-1.2'],text=True,timeout=15).splitlines()==row['packages'], 'compiler packages changed')
    return allowed


def parser_proof():
    key='gdn_kernel_trace_host_v4';path=B/'gdn-kernel-trace-host-v4/record.json'
    row=json.loads(path.read_text())
    require(base.committed(RESULT)[key]==base.compact(row) and row['passed'] and row['tests']==6 and
            row['ownership_closed'] and not row['active'] and all(base.supervision.closed(v)
                for v in row['commands_executed']), 'closed original parser tests required')
    names=['tools/sycl/kernel_trace.py',row['test']['path']]
    for name in names:
        raw=base.basis.build.git(P,'show',row['source_commit']+':'+name)
        require(raw==(P/name).read_bytes(),'parser test/import changed: '+name)
    require(digest(row['interpreter']['path'])==row['interpreter']['sha256'],'parser interpreter changed')
    return dict(source_commit=row['source_commit'],files={n:digest(P/n) for n in names},tests=6,
        scope='Reuse byte-identical unittest file and its sole local parser import;original interpreter/process closure preserved. Build/measurement-controller changes are outside that CPU test;no new test or GPU qualification claimed.')


def prepare():
    data=base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(),'one fresh model generation')
    before,_,probe=retained.verify()
    old=data['gdn16_cast_component_retirement']
    require(old['original_inputs_recovered'] and old['deleted_verified'] and
            not Path(old['retired_binary']['path']).exists(),'retire closed raw-MKL component before build')
    decision=data['gdn16_pipeline_speed_v3']
    require(decision['passed'] and decision['complete'] and decision['ownership_closed'] and
            all(v['follow_up_eligible'] for v in decision['component_decisions']),
            'typed raw component has not passed prospective speed screens')
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    full,native=source.sources(revision)
    parser=parser_proof()
    env=dict(before['environment'],TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache'))
    targets=[*before['targets'],COMPONENT]
    cfg=[v.replace(str(retained.ROOT),str(ROOT)) for v in before['configure_argv']]
    ROOT.mkdir(mode=0o700)
    for part in ('source','tmp','cache'):(ROOT/part).mkdir()
    for name,raw in native.items():
        path=ROOT/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    row=dict(stage='prepared',source_commit=revision,controller_inputs=base.controls(revision,controller=CONTROLLER),
        native_sources={n:hashlib.sha256(v).hexdigest() for n,v in native.items()},source_inputs=source.input_pins(revision),
        source_basis=dict(selected=retained.NATIVE,owned_delta_before=source.BEFORE,typed_kernel_source=source.TYPED,
            independent_from_XeStrata=True,unchanged_unlisted_source_files=len(native)-8),parser_proof=parser,
        **{k:before[k] for k in ('compiler','sdk','external_manifest','host_link_inputs','packages','dependencies',
            'native_validation_probe','boot','driver')},
        retained_binaries=before['binaries'],environment=env,configure_argv=cfg,targets=targets,
        generated_expected={p.replace(str(retained.ROOT),str(ROOT)):h for p,h in before['generated_expected'].items()},
        commands=[['configure',cfg],['compile',['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',*targets,'-j','4']],
            ['dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps']]],
        budgets=dict(wall=1800,cpu=7200,rss_cap=12<<30,text_cap=16<<20,file_cap=512<<20,total_cap=4<<30),
        whole_wall_seconds=3600,commands_executed=[],active=False,ownership_closed=True,passed=False,
        GPU_executed=False,model_executed=False,inference_adopted=False,
        plan=dict(scope='CPU-only full generation;selected two-profile engine plus opt-in GDN3 FP16 vendor output and typed consumers. Actual production wrappers used by the same-build component fixture.',
            storage='STRATA_SYCL_PREFILL_GDN_FP16=1 only selects the native16 long layer-major lease. Default0,TTFT/short/decode paths retained. No FP32 restore before conv/recurrence;state/history,gates and raw/RMS scratch remain FP32. Legacy and typed views alias the shared arena;global reservation remains max of all required layouts,not a claimed VRAM saving.',
            safety='Qualified immutable preexec environment unchanged. No model execution from compilation. No unsupported device-IR output step. Keep selected f6cea2eight-ELF generation until qualification;this is the sole new comparison generation,legacy14holds separate.',
            next='Commit recipe;CPU compile;fresh traced/validated small production-wrapper component before general/timing. Only then real-model range,handoff,thinking32K+ PP/TTFT,agent/long generation,independent decode,physical262144fill/tails/RESTORE. No inherited model qualification.'))
    controls(row);save(row);print('Prepared557-file model/source recipe;commit before CPU build.',flush=True)


def build():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY]==base.compact(row) and row['stage']=='prepared' and
            not (ROOT/'owned-build').exists(),'one frozen fresh model build')
    allowed=allowed_inputs(row);frozen=copy.deepcopy(row);owner=None
    deadline=time.monotonic()+row['whole_wall_seconds']
    def monitor(io,entry):
        next_size=0.
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic()>=next_size:
                require(base.supervision.artifact_bytes(ROOT)<=row['budgets']['total_cap'],'whole generation byte budget');next_size=time.monotonic()+1
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=base.supervision.Owner(ROOT/'owned-build')
        for label,argv in row['commands']:
            print('CPU build:',label,flush=True)
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,interaction=monitor,
                **dict(row['budgets'],wall=min(row['budgets']['wall'],deadline-time.monotonic())))
            require(base.supervision.closed(entry) and entry['exit_code']==0,
                    label+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-5000:])
        allowed_inputs(frozen)
        found={str((Path(v.strip()) if Path(v.strip()).is_absolute() else ROOT/'build'/v.strip()).resolve())
            for v in out.read_text().splitlines() if v.startswith('    ') and v.strip()}
        require(len(found)>=1400 and {str(ROOT/'source'/n) for n in
            ('src/prefill/prefill.cpp','src/prefill/sycl/gemm.cpp',source.CAST,source.FIXTURE)}<=found,
            'actual model/fixture compiler closure incomplete')
        require(all(allowed.get(p)==digest(p) and '/MistVVK/' not in p for p in found),
                'unplanned actual compiler input')
        write(ROOT/'compiler-inputs.json',{p:allowed[p] for p in sorted(found)})
        db=ROOT/'build/compile_commands.json';commands=json.loads(db.read_text());attested=[]
        for name in ('src/prefill/prefill.cpp','src/prefill/sycl/gemm.cpp',source.KERNEL,source.FIXTURE):
            candidates=[v for v in commands if v['file']==str(ROOT/'source'/name)]
            if name==source.KERNEL:candidates=[v for v in candidates if '-o CMakeFiles/strata_kernels.dir/' in v['command']]
            require(len(candidates)==1,'one actual production compile target required: '+name)
            argv=shlex.split(candidates[0]['command'])
            require(argv[0]==base.COMPILER and '-O3' in argv and '-fsycl' in argv and
                    not any(v in argv for v in ('-ffast-math','-Ofast','-ffp-contract=fast')),
                    'production compile options changed: '+name)
            if name in (source.KERNEL,source.FIXTURE):
                require('-fno-fast-math' in argv and '-ffp-contract=off' in argv,'GDN/fixture math flags changed')
            attested.append(dict(source=name,argv_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest()))
        row.update(stage='built-not-executed',passed=True,
            compiled_inputs=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(found),uncovered=0),
            actual_link_inputs=base.traced_link_inputs(allowed,ROOT),
            compile_database=dict(path=str(db),sha256=digest(db),units=attested),
            binaries={n:dict(path=str(ROOT/'build'/n),sha256=digest(ROOT/'build'/n),bytes=(ROOT/'build'/n).stat().st_size) for n in row['targets']})
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active),commands_executed=owner.commands if owner else [],
            ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)))
        require(owner is None or owner.active is None,'build ownership unresolved');save(row)
    print(json.dumps({k:row.get(k) for k in ('stage','passed','error')}),flush=True)
    require(row['passed'],'GDN model build failed')


def verify():
    row=json.loads((ROOT/'record.json').read_text())
    require(base.committed(RESULT)[KEY]==base.compact(row) and row['passed'] and row['stage']=='built-not-executed' and
            row['ownership_closed'] and not row['active'] and all(base.supervision.closed(v) for v in row['commands_executed']),
            'closed complete model build required')
    allowed=allowed_inputs(row,retained_parent=False);asset=row['compiled_inputs']
    require(digest(asset['path'])==asset['sha256'],'actual compiler map changed')
    found=json.loads(Path(asset['path']).read_text())
    require(len(found)==asset['files'] and all(allowed.get(p)==sha==digest(p) for p,sha in found.items()),
            'actual compiler inputs changed')
    require(digest(row['compile_database']['path'])==row['compile_database']['sha256'] and
            row['actual_link_inputs']['uncovered_external_inputs']==0 and all(allowed.get(p)==sha==digest(p)
            for p,sha in row['actual_link_inputs']['frozen_inputs'].items()), 'actual compile/link recipe changed')
    require(not any(Path(p).is_relative_to(retained.ROOT) for p in
            (*found,*row['actual_link_inputs']['frozen_inputs'],
             *retained.runtime.inventory(row['external_manifest']))),
            'current generation still consumes a parent build input')
    require(set(row['binaries'])==set(row['targets']) and all(digest(a['path'])==a['sha256'] and
            Path(a['path']).stat().st_size==a['bytes'] for a in row['binaries'].values()), 'same-build engine/fixtures changed')
    return row
