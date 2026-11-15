#!/usr/bin/env python3
"""Build the measured engine plus two prefill strategies/GDN3; repository LICENSE.

Main owns this CPU-only build. Preserve current official seven-ELF reference;
retire the superseded selected6e40 binaries after source/consumer verification.
Compilation does not qualify GPU execution, either profile, or model performance.
"""
import fcntl
import copy
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_official_model as base
import measure_gdn_scheduling as component
import prefill_strategy_sources as source

P, B, RESULT = base.P, base.B, base.RESULT
ROOT = B / 'official-prefill-strategies-v1'
OLD_KEY, KEY = 'official_prefill_strategies_build_v1', 'official_prefill_strategies_build_v2'
RECORD = ROOT / 'record-v2.json'
FAILED_COMMIT = 'dd94c45dbcb70cfda4e6308f916c2e2e7c446d72'
TARGETS = (*base.TARGETS, 'sycl_gdn_columns_bench')
require, digest, write = base.require, base.digest, base.write
COMPONENT_EVIDENCE = 'c170d64631a262c8bfdac59f982a52882c736fac'


def controls(revision, **reuse):
    return base.controls(revision, controller=Path(__file__).resolve(), **reuse)


def compact(row):
    return base.compact(row)


def save(row):
    write(RECORD, row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def retained():
    data = base.committed(RESULT)
    row = json.loads((base.ROOT / 'record.json').read_text())
    require(data[base.KEY] == base.compact(row) and row['passed'] and row['ownership_closed']
            and not row['active'], 'closed official reference required')
    for item in row['binaries'].values():
        require(digest(item['path']) == item['sha256'], 'current official executable changed')
    providers = component.inventory(row['runtime_basis'])
    probe = component.qualified_probe(data, providers)
    archive = json.loads(base.basis.build.git(P, 'show', COMPONENT_EVIDENCE + ':' + str(RESULT.relative_to(P))))
    done = archive['official_gdn_barriers3_v1']
    require(done['passed'] and done['complete'] and done['ownership_closed'] and not done['active']
            and done['performance']['advance'] and len(done['processes']) == 7
            and all(p['passed'] and not p['fresh_faults'] for p in done['processes'])
            and done['processes'][-1]['events'][0]['tokens'] == 262144,
            'closed component performance/context proposal required')
    return row, providers, probe


def prepare():
    data = base.committed(RESULT)
    require(KEY not in data and not ROOT.exists(), 'fresh candidate only')
    prior, providers, probe = retained()
    full, native = source.sources()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = controls(revision)
    sdk = base.sdk(); external = component.inventory(prior['external_manifest'])
    env = dict(prior['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require(base.host_link_inventory.inventory(base.COMPILER, env) == prior['host_link_inputs'],
            'official host linker inputs changed')
    cfg = [v.replace(str(base.ROOT), str(ROOT)) for v in prior['configure_argv']]
    ROOT.mkdir(mode=0o700)
    for part in ('source', 'tmp', 'cache'): (ROOT / part).mkdir()
    for name, raw in native.items():
        path = ROOT / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    row = dict(stage='prepared', source_commit=revision, implementation_commit=source.IMPLEMENTATION,
        controller_inputs=pins, native_sources={n:hashlib.sha256(raw).hexdigest() for n,raw in native.items()},
        source_basis=dict(official_native=prior['source_commit'], official_aggregate=source.BASE_AGGREGATE,
            implementation=source.IMPLEMENTATION, component_evidence=COMPONENT_EVIDENCE,
            unrelated_decode_sources_unchanged=True, fully_submitted_MoE_jobs_per_checkpoint=8),
        compiler=prior['compiler'], sdk=prior['sdk'], external_manifest=prior['external_manifest'],
        host_link_inputs=prior['host_link_inputs'], packages=prior['packages'], dependencies=prior['dependencies'],
        runtime_basis=prior['runtime_basis'], native_validation_probe=probe, environment=env,
        configure_argv=cfg, targets=list(TARGETS),
        generated_expected={p.replace(str(base.ROOT),str(ROOT)):h for p,h in prior['generated_expected'].items()},
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=base.guards.driver_guard(),
        retained_current_binaries=prior['binaries'], selected_fallback_retirement=dict(complete=False,
            next_action='Recover exact selected6e40 source/recipe,verify no live binary/fixture consumer,retain only consumed provenance receipt,then retire older executable generation before compile.'),
        budgets=dict(wall=1800,cpu=7200,rss_cap=12<<30,text_cap=16<<20,file_cap=512<<20,total_cap=4<<30),
        whole_wall_seconds=3600,commands_executed=[],active=False,ownership_closed=True,passed=False,
        GPU_executed=False,inference_adopted=False,model_qualified=False,
        retention='Current official seven-ELF reference plus this imminent eight-ELF candidate. Retire selected6e40 executable/fixtures before compile after exact source and consumer review;keep its consumed provenance receipt. Retire rejected/new extra after comparison. Legacy14holds remain separate.',
        plan=dict(scope='CPU build only;typed startup strategy plus prefill-only GDN3 on exact official source.',
            next='After source review: CPU CLI parser/default/override checks,then fresh same-build traced submission/GDN17,exact tails and context. Audit long TTFT public-GU progression before any such model run. Fresh bounded small model before32K+ PP/TTFT/agent/thinking/decode/handoff/physical262144/RESTORE evidence.',
            performance='No speed samples or adoption from compilation. Model A/B and input-length/TTFT plans must be committed before those runs;both explicit profiles share one ELF and preserve decode settings.'))
    save(row)
    print('Prepared555-file two-strategy model source/recipe;retirement and commit required before compile.',flush=True)


def current():
    row=json.loads(RECORD.read_text())
    require(base.committed(RESULT)[KEY] == compact(row), 'commit exact preceding build plan/result')
    return verify_inputs(row)


def verify_inputs(row):
    admission=row.get('controller_admission')
    reuse=dict(reuse_revision=admission['revision'],controller_changes=tuple(admission['changed_paths'])) if admission else {}
    require(controls(row['source_commit'],**reuse) == row['controller_inputs'], 'candidate controls changed')
    prior, providers, probe=retained()
    require(prior['binaries']==row['retained_current_binaries'] and probe==row['native_validation_probe'],
            'current reference/probe changed')
    full,native=source.sources()
    require({n:hashlib.sha256(b).hexdigest() for n,b in native.items()}==row['native_sources'],
            'candidate source projection changed')
    allowed=base.sdk();allowed.update(component.inventory(row['external_manifest']));allowed.update(providers)
    require(base.host_link_inventory.inventory(base.COMPILER,row['environment'])==row['host_link_inputs'],
            'host linker changed')
    require(subprocess.check_output(['dpkg-query','-W','-f=${Package} ${Version}\n',
        'intel-oneapi-dpcpp-cpp-2026.2','intel-oneapi-compiler-dpcpp-cpp-runtime-2026.2',
        'intel-oneapi-umf-1.2'],text=True,timeout=15).splitlines()==row['packages'],'official package versions changed')
    for name,raw in native.items():
        p=ROOT/'source'/name;require(p.is_file() and not p.is_symlink() and p.read_bytes()==raw,
                                  'frozen native source changed: '+name)
        allowed[str(p.resolve())]=row['native_sources'][name]
    for group in full['dependencies']:
        for name,sha in group['source_sha256'].items():
            p=(Path(group['root'])/name).resolve();require(digest(p)==sha,'dependency changed: '+str(p));allowed[str(p)]=sha
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==row['boot']
            and base.guards.driver_guard()==row['driver'],'boot/driver changed')
    allowed.update(row['generated_expected'])
    for item in row.get('binaries',{}).values():require(digest(item['path'])==item['sha256'],'candidate executable changed')
    if row.get('compiled_inputs'):component.inventory(row['compiled_inputs'])
    return row,allowed


def attest_outputs(row,allowed,out):
    found={str((Path(s.strip()) if Path(s.strip()).is_absolute() else ROOT/'build'/s.strip()).resolve())
           for s in out.read_text().splitlines() if s.startswith('    ') and s.strip()}
    require(len(found)>=1000 and str(ROOT/'source/include/strata/prefill/strategy.hpp') in found
            and all(allowed.get(p)==digest(p) for p in found),'unplanned/missing actual compiler input')
    db=ROOT/'build/compile_commands.json';attestation=[]
    for entry in json.loads(db.read_text()):
        argv=entry.get('arguments') or shlex.split(entry['command'])
        path=Path(entry['file'])
        if not path.is_relative_to(ROOT/'source'):continue
        name=str(path.relative_to(ROOT/'source'))
        if name not in ('src/prefill/prefill.cpp','src/program/generate.cpp','src/kernels/sycl/fused_gdn.cpp','bench/sycl/gdn_columns.cpp'):continue
        required=['-fsycl','-DSTRATA_ENABLE_SYCL=1','-O3','-DNDEBUG','-std=c++20']
        if name!='src/program/generate.cpp':required+=['-fno-fast-math','-ffp-contract=off']
        require(Path(argv[0]).resolve()==Path(base.COMPILER).resolve() and all(a in argv for a in required)
                and not any(a in argv for a in ('-ffast-math','-Ofast','-ffp-contract=fast')),'candidate compile options differ: '+name)
        attestation.append(dict(source=name,output=argv[argv.index('-o')+1],required_options=required,
            argv_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest()))
    require(len(attestation)==5 and sum(a['source']=='src/kernels/sycl/fused_gdn.cpp' for a in attestation)==2,
            'production/fixture GDN and policy command attestation missing')
    write(ROOT/'compiler-inputs.json',{p:allowed[p] for p in sorted(found)})
    row.update(stage='built-not-executed',passed=True,
        compiled_inputs=dict(path=str(ROOT/'compiler-inputs.json'),sha256=digest(ROOT/'compiler-inputs.json'),files=len(found)),
        compile_command_attestation=dict(database_sha256=digest(db),units=attestation),
        actual_link_inputs=base.traced_link_inputs(allowed,ROOT),
        binaries={n:dict(path=str(ROOT/'build'/n),sha256=digest(ROOT/'build'/n),bytes=(ROOT/'build'/n).stat().st_size) for n in TARGETS})


def compile_candidate():
    row,allowed=current()
    require(row['stage']=='prepared' and not (ROOT/'owned-build').exists(),'one fresh build only')
    require(row['selected_fallback_retirement']['complete'] and
            not Path(source.original.original.ORIGIN.parent/'build/strata').exists(),
            'recover and retire superseded selected6e40 binaries before creating another engine generation')
    deadline=time.monotonic()+row['whole_wall_seconds'];owner=None
    def persist():
        write(RECORD,dict(row,commands_executed=owner.commands if owner else [],
                                    active=bool(owner and owner.active)))
    def monitor(io,entry):
        next_size=0.
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic()>=next_size:
                require(base.supervision.artifact_bytes(ROOT)<=row['budgets']['total_cap'],'whole build size budget');next_size=time.monotonic()+1
            if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
        io.exit(1)
    try:
        owner=base.supervision.Owner(ROOT/'owned-build',persist)
        commands=[('configure',row['configure_argv']),('compile',['/usr/bin/cmake','--build',str(ROOT/'build'),'--target',*TARGETS,'-j','4']),('dependencies',['/usr/bin/ninja','-C',str(ROOT/'build'),'-t','deps'])]
        for label,argv in commands:
            print('CPU build:',label,flush=True)
            entry,out,err=owner.run(label,argv,row['environment'],ROOT,interaction=monitor,
                **dict(row['budgets'],wall=min(deadline-time.monotonic(),row['budgets']['wall'])))
            require(base.supervision.closed(entry) and entry['exit_code']==0,
                    label+': '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-4000:])
        current()
        attest_outputs(row,allowed,out)
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:
        require(owner is None or owner.active is None,'CPU build owner remains unresolved')
        row.update(commands_executed=owner.commands if owner else [],active=False,
            ownership_closed=bool(owner and all(base.supervision.closed(v) for v in owner.commands)));save(row)
    require(row['passed'] and row['ownership_closed'],row.get('error','build did not close'))
    print('Eight candidate executables built;GPU/model/two-profile qualification remains pending.',flush=True)


def prepare_attestation_recovery():
    data=base.committed(RESULT)
    require(KEY not in data and not RECORD.exists(),'fresh attestation recovery only')
    oldpath=ROOT/'record.json';old=json.loads(oldpath.read_text())
    require(data[OLD_KEY]==compact(old) and old['stage']=='failed' and not old['passed']
            and old['ownership_closed'] and not old['active'] and not old['GPU_executed']
            and len(old['commands_executed'])==3 and all(base.supervision.closed(v) and v['exit_code']==0
                for v in old['commands_executed']) and old['error'].startswith('ValueError('),
            'unchanged closed metadata failure after three successful build commands required')
    archived=json.loads(base.basis.build.git(P,'show',FAILED_COMMIT+':'+str(RESULT.relative_to(P))))
    require(archived[OLD_KEY]==compact(old),'original failure archive differs')
    revision=base.basis.build.git(P,'rev-parse','HEAD').decode().strip()
    row=copy.deepcopy(old)
    row['controller_admission']=dict(revision=revision,changed_paths=['tools/sycl/build_prefill_candidate.py'],
        native_sources_recipe_and_build_commands_unchanged=True)
    verify_inputs(row)
    names=[name for name in old['commands_executed'][-1]['logs'] if name.endswith('.stdout')]
    require(len(names)==1,'one original dependency output required')
    matches=list((ROOT/'owned-build').rglob(names[0]));require(len(matches)==1,'original dependency log path ambiguous')
    dependencies=matches[0];require(digest(dependencies)==old['commands_executed'][-1]['logs'][names[0]]['sha256'],
                                   'original dependency output changed')
    row.update(stage='pending-attestation',passed=False,
        original_build_failure=dict(commit=FAILED_COMMIT,key=OLD_KEY,source_commit=old['source_commit'],error=row.pop('error'),
            record=dict(path=str(oldpath),sha256=digest(oldpath))),
        attestation_freeze=dict(dependency_stdout=dict(path=str(dependencies),sha256=digest(dependencies)),
            compile_database=dict(path=str(ROOT/'build/compile_commands.json'),sha256=digest(ROOT/'build/compile_commands.json')),
            binaries={n:dict(path=str(ROOT/'build'/n),sha256=digest(ROOT/'build'/n),bytes=(ROOT/'build'/n).stat().st_size) for n in TARGETS}),
        attestation_plan='CPU-only replay of complete actual compiler/link input and five production/fixture option attestations. Filter external dependency commands before project-relative path conversion. Frozen555native inputs,recipe,eight ELFs,dependency output and database must match;no recompilation or GPU execution. Original V1failure receipt stays unchanged. Stop on first mismatch;no hash refresh.')
    save(row);print('Prepared separate controller-only attestation plan;commit before replay.',flush=True)


def attest():
    row,allowed=current();require(row['stage']=='pending-attestation','fresh pending attestation only')
    frozen=row['attestation_freeze']
    for item in [row['original_build_failure']['record'],frozen['dependency_stdout'],frozen['compile_database'],*frozen['binaries'].values()]:
        require(digest(item['path'])==item['sha256'],'frozen recovery artifact changed')
    try:
        attest_outputs(row,allowed,Path(frozen['dependency_stdout']['path']))
        require(row['binaries']==frozen['binaries'],'attestation changed native executable identities')
        row['attestation_replayed_without_recompile']=True
    except BaseException as error:row.update(stage='failed',passed=False,error=repr(error))
    finally:save(row)
    require(row['passed'],row.get('error','CPU attestation failed'))
    print('Original compiled outputs independently attested;V1failure preserved;GPU execution pending.',flush=True)


if __name__=='__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.guards.no_model_owners([])
        if sys.argv[1:]==['prepare']:prepare()
        elif sys.argv[1:]==['build']:compile_candidate()
        elif sys.argv[1:]==['prepare-attestation']:prepare_attestation_recovery()
        elif sys.argv[1:]==['attest']:attest()
        else:raise RuntimeError('build_prefill_candidate.py prepare|build|prepare-attestation|attest')
