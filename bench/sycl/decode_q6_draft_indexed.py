"""Build and compare the draft-only projection on the original B570 packed head."""

import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_prefill_mixed_gather as fixture
import prefill_large_gather_model_source as parent
from check_dflash2_model import model_identity
from check_q6_head_model import run_owner_fixture

bank, require, sha, write = fixture.bank, fixture.require, fixture.sha, fixture.write
owner, guards = bank.builder.ownership, bank.builder.guards
P, B = fixture.PROJECT, bank.B
ROOT = B / 'decode-q6-draft-indexed-v1'
SOURCE, BUILD = parent.MODEL_ROOT / 'source', parent.MODEL_ROOT / 'build'
CANDIDATE = 'a0ede5973e20f65ed224b93172a1413f781abe97'
REFERENCE = 'd758639ed307c297ab6fa6d3e7de09fd15b8809a'
CONTROLLER = str(Path(__file__).resolve().relative_to(P))
KERNEL = 'src/kernels/sycl/q6_dpas_component.cpp'
HEADER = 'src/kernels/sycl/q6_dpas_component.hpp'
TEST = 'tests/sycl/native_head_dpas.cpp'
ARCHIVE, TARGET = 'libstrata_kernels.a', 'sycl_q6_head_owner_test'
VOCAB_SHA = 'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'
RECIPE_SHA = '19ea3964f3ae0b6d9e0c552e5e999f9585bd195eeecef8843349b8968b649b97'
NINJA_SHA = '5965527e09fe2b3787772aa4f711d6a36b393e7f2fcaa744a7a96c5a4ddf59cb'
ORDER = (('smoke', None), ('0', 'rc'), ('1', 'cr'), ('2', 'rc'))
BUDGET = dict(wall=240,cpu=360,rss_cap=8<<30,text_cap=16<<20,file_cap=32<<20,total_cap=64<<20)
BOOT = Path('/proc/sys/kernel/random/boot_id')
PROBE_STAMP = 'DRAFT_INDEX_CHECK rows=106299 exact_cases=24 masked_cases=3 guards=PASS input_immutable=PASS'


def inputs():
    native = bank.previous.read(parent.MODEL_ROOT / 'record.json')
    expected = bank.basis.validation.committed(bank.RESULT, (parent.KEY,))
    require(expected == bank.builder.compact(native) and native['native_source_commit'] == parent.DISPATCH_COUNTS
            and native['stage'] == 'built' and native['complete'] and native['passed'] and native['ownership_closed']
            and not native['active'] and native['host_control']['passed'], 'closed coherent parent required')
    require(len(native['configured_files']) == 4 and
            all(sha(p) == h for p,h in native['configured_files'].items()), 'changed original configured recipe')

    overlay = {}
    for path in (KERNEL, HEADER, TEST):
        original = fixture.git_file(parent.DISPATCH_COUNTS, path)
        candidate = fixture.git_file(CANDIDATE, path)
        require(sha(SOURCE / path) == native['source_files'][path] == hashlib.sha256(original).hexdigest()
                and (P / path).read_bytes() == candidate, 'original or indexed source changed: '+path)
        overlay[path] = candidate
    declaration = overlay[HEADER]
    start = declaration.index(b'// Decode draft C1 on the same packed head.')
    end = declaration.index(b'// Prospective component:', start)
    require(declaration[:start]+declaration[end:] == fixture.git_file(parent.DISPATCH_COUNTS, HEADER),
            'a reused translation unit would see changed existing definitions')

    commands, objects = [], {}
    database = json.loads((BUILD/'compile_commands.json').read_bytes())
    for name, target in ((KERNEL, 'strata_kernels'), (TEST, TARGET)):
        unit, = [c for c in database if c['file']==str(SOURCE/name) and
                 c['output'].startswith('CMakeFiles/'+target+'.dir/')]
        argv = shlex.split(unit['command'])
        require(Path(unit['directory'])==BUILD and argv[-4:]==['-o',unit['output'],'-c',unit['file']]
                and {'-fno-fast-math','-ffp-contract=off','-fsycl'}<=set(argv), 'changed production C++ recipe')
        objects[name] = unit['output']
        product = ROOT/Path(unit['output']).name
        commands.append([*argv[:-4],'-MD','-MF',str(product)+'.d','-o',str(product),'-c',str(ROOT/name)])

    ar, ninja = '/usr/bin/ar', '/usr/bin/ninja'
    external = native['external_inputs']
    require(sha(external['path'])==external['sha256'], 'original external-tool inventory changed')
    external_files = json.loads(Path(external['path']).read_bytes())
    require(external_files.get(str(Path(ar).resolve()))==sha(ar), 'retained archiver identity changed')
    require(sha(ninja)==NINJA_SHA, 'configured-recipe reader identity changed')
    archived = native['actual_inputs']['produced']
    require(sha(BUILD/ARCHIVE)==archived[str(BUILD/ARCHIVE)], 'original kernels archive changed')
    members = subprocess.check_output([ar,'t',ARCHIVE],cwd=BUILD,text=True,timeout=15).splitlines()
    retained_objects = {Path(p).name:p for p in native['actual_inputs']['objects']
                        if Path(p).is_relative_to(BUILD/'CMakeFiles/strata_kernels.dir')}
    member = Path(objects[KERNEL]).name
    require(len(members)==len(set(members))==len(retained_objects) and set(members)==set(retained_objects)
            and retained_objects[member]==str(BUILD/objects[KERNEL]), 'archive membership changed')
    retained_objects[member] = str(ROOT/member)
    commands.append([ar,'rcsD',str(ROOT/ARCHIVE),*(retained_objects[n] for n in members)])

    line = subprocess.check_output([ninja,'-t','commands',TARGET],cwd=BUILD,text=True,timeout=15).splitlines()[-1]
    words = shlex.split(line)
    require(words[:2]==[':','&&'] and words[-2:]==['&&',':'] and '-Wl,-t' in words
            and words[words.index('-o')+1]==TARGET, 'frozen linked owner fixture required')
    replacement = {ARCHIVE:str(ROOT/ARCHIVE), TARGET:str(ROOT/TARGET),
                   objects[TEST]:str(ROOT/Path(objects[TEST]).name)}
    require(all(words.count(word)==1 for word in replacement), 'missing expected direct linker input')
    commands.append([words[2],'-v',*(replacement.get(word,word) for word in words[3:-2])])
    proof = dict(original_native=parent.DISPATCH_COUNTS,
                 original_build=bank.signature(expected), candidate_source=CANDIDATE,
                 recipe_sha256=bank.signature(commands), archive_members=len(members),
                 overlay_sha256={p:hashlib.sha256(raw).hexdigest() for p,raw in overlay.items()})
    allowed = fixture.build_inputs(native,ROOT,overlay,commands,source_directory=SOURCE)
    # The component consumes these retained archive members and link archives.
    # Other old controllers or decoded fixtures are not compilation inputs.
    reused = {str((BUILD/arg).resolve()) for arg in (*commands[2][3:],*commands[-1])
              if arg.endswith(('.a','.o')) and not Path(arg).is_relative_to(ROOT)}
    require(reused and all(allowed.get(path)==sha(path) for path in reused),
            'changed retained object or link archive')
    proof['retained_inputs'] = dict(count=len(reused),sha256=bank.signature({p:allowed[p] for p in sorted(reused)}))
    return native, overlay, commands, allowed, proof


def checked_archive(commands, allowed):
    ar, _, archive, *members = commands[2]
    order = subprocess.check_output([ar,'t',archive],text=True,timeout=15).splitlines()
    require(order==[Path(p).name for p in members], 'archive changed its member order')
    payload = memoryview(subprocess.check_output([ar,'p',archive],timeout=15))
    offset, hashes = 0, {}
    for name in members:
        obj = Path(name)
        stop = offset+obj.stat().st_size
        expected = sha(obj) if obj.is_relative_to(ROOT) else allowed[name]
        require(hashlib.sha256(payload[offset:stop]).hexdigest()==expected,
                'archive contains unexpected object bytes: '+obj.name)
        hashes[obj.name] = expected
        offset = stop
    require(offset==len(payload), 'archive member byte extent differs')
    return dict(count=len(hashes),sha256=bank.signature(hashes))


def model_inputs(native, proof):
    historical = json.loads(fixture.git_file(REFERENCE, str(bank.RESULT.relative_to(P))))
    pinned = historical['decode_expert_profile_generation_v1']
    first, = pinned['processes']
    require(first['passed'] and first['ownership_closed'] and first['native_exit_code']==0 and
            not first['fresh_faults'] and pinned['binary']==native['binaries']['strata'] and
            pinned['native_build_sha256']==proof['original_build'], 'missing same-build stock benchmark reference')
    live = bank.previous.read(B/'decode-expert-profile-generation-v1/record.json')
    require(all(live[k]==pinned[k] for k in ('runtime_manifest','native_validation_probe','binary','boot','driver'))
            and bank.signature(live['model_assets'])==pinned['model_assets_sha256']
            and bank.signature(live['environment_template'])==pinned['environment_template_sha256'],
            'changed immutable speed-reference assets, environment or providers')
    models = {p:v for p,v in live['model_assets'].items() if p.endswith(('.gguf','/draft_vocab.bin'))}
    shards = sorted(p for p in models if p.endswith('.gguf'))
    vocab, = (p for p in models if p.endswith('/draft_vocab.bin'))
    manifest = pinned['runtime_manifest']
    require(len(shards)==2 and models[vocab]['sha256']==VOCAB_SHA and sha(manifest['path'])==manifest['sha256']
            and sha(vocab)==VOCAB_SHA, 'shards, exact ordered vocabulary or runtime inventory differ')
    runtime = {str(Path(p).resolve()):h for p,h in json.loads(Path(manifest['path']).read_bytes()).items()}
    for v in pinned['native_validation_probe'].values():
        runtime[str(Path(v['path']).resolve())] = v['sha256']
    require(all(sha(p)==h for p,h in runtime.items()), 'current runtime bytes differ from qualification')
    return live, models, shards, vocab, runtime


def pinned_controller():
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True,timeout=20).strip()
    files = [CONTROLLER, str(Path(fixture.__file__).relative_to(P)),
             str(Path(parent.__file__).relative_to(P)), 'bench/sycl', 'tools/sycl', 'AGENTS.md',
             'docs/ARTIFACT_RETENTION.md']
    pins = bank.stage.source.pin(P, files)
    require(pins['revision']==revision and all(name in pins['source_sha256'] for name in files[:3]),
            'commit the complete experimental comparison inputs first')
    return dict(revision=revision,sha256=bank.signature(pins['source_sha256']))


def check_assets(models, full=False):
    deadline = time.monotonic()+300
    for name, record in models.items():
        path = Path(name)
        require(model_identity(path)==record['identity'], 'model file identity changed: '+name)
        if full:
            digest = hashlib.sha256()
            with path.open('rb',buffering=0) as stream:
                while chunk := stream.read(8<<20):
                    require(time.monotonic()<deadline, 'full model hash budget')
                    digest.update(chunk)
            require(digest.hexdigest()==record['sha256'] and model_identity(path)==record['identity'],
                    'model bytes changed: '+name)


def parse(text, diagnostic, order):
    lines = text.splitlines()
    require(lines[-2:]==[PROBE_STAMP,'HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified']
            and lines.count(PROBE_STAMP)==1, 'missing exact indexed-case and owner completion receipt')
    selected = [v for v in lines if v.startswith('DRAFT_INDEX_TIME ')]
    result = dict(exact_cases=24, masked_cases=3, guards=True, immutable=True)
    if order is None:
        require(not selected and 'Q6DpasDraftIndexed' in diagnostic and 'Q6DpasColumns' in diagnostic
                and 'urEnqueueKernelLaunch' in diagnostic, 'both actual head kernels must execute in the traced probe')
        return result
    timer, = [v for v in lines if v.startswith('DRAFT_INDEX_TIMER ')]
    overhead = int(timer.removeprefix('DRAFT_INDEX_TIMER overhead_ns='))
    require(0<overhead<=100000 and len(selected)==10, 'invalid timing counter or timer floor')
    timings = []
    for position, line in enumerate(selected):
        m = re.fullmatch(r'DRAFT_INDEX_TIME sample=(\d+) arm=([01]) rounds=128 elapsed_ns=(\d+)',line)
        require(m,'invalid draft-indexed timing row')
        pair, arm, elapsed = map(int,m.groups())
        first = int(order=='cr') ^ (pair & 1)
        require(pair==position//2 and arm==(first^(position%2)) and elapsed>=100000000,
                'changed timing order or ineligible interval')
        timings.append(dict(pair=pair,arm=arm,elapsed_ns=elapsed))
    medians = [statistics.median(t['elapsed_ns'] for t in timings if t['arm']==arm) for arm in (0,1)]
    return dict(result,order=order,timer_ns=overhead,timings=timings,median_ns=medians,
                candidate_time_ratio=medians[1]/medians[0])


def run_case(action, build, live, models, shards, vocab, runtime, proof, allowed, commands):
    require(build['complete'] and build['ownership_closed'] and build['proof']==proof and
            build['controller']==pinned_controller() and build['commands']==commands and
            sha(build['binary']['path'])==build['binary']['sha256'] and
            build['archive_proof']==checked_archive(commands,allowed), 'compiled fixture and current pins differ')
    require(all(sha(p)==h for group in ('compiled','linked') for p,h in build[group].items()) and
            all(allowed.get(p)==h for p,h in build['compiled'].items()) and
            all(sha(ROOT/p)==h for p,h in build['driver_inputs'].items()), 'actual compiled or linked inputs differ')
    out_path = ROOT/'samples.json'
    report = json.loads(out_path.read_bytes()) if out_path.is_file() else dict(
        candidate=CANDIDATE,controller=build['controller'],binary=build['binary'],proof=proof,cases=[])
    cases = report['cases']
    require(report['binary']==build['binary'] and report['proof']==proof and
            [c['stage'] for c in cases]==[s for s,_ in ORDER[:len(cases)]] and
            all(c['passed'] and c['ownership_closed'] for c in cases)
            and len(cases)<len(ORDER) and action==ORDER[len(cases)][0],
            'previous stage, original response, or execution order differs')
    _, order = ORDER[len(cases)]
    stage = ROOT/('case-'+action)
    stage.mkdir(mode=0o700)
    for d in ('tmp','cache'):(stage/d).mkdir(mode=0o700)
    original_env = live['environment_template']
    env = {k:v.replace(original_env['STRATA_MEASUREMENT_ROOT'],str(stage)) for k,v in original_env.items()}
    probe = live['native_validation_probe']
    env.pop('STRATA_TRACE',None)
    env.pop('UR_LOG_TRACING',None)
    env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2',STRATA_Q6_HEAD_OWNER_GATE='1',
               LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
               UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(stage/(action+'-validation.log')))
    if order is None:
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']==env['NEOReadDebugKeys']=='1' and
            env['EnableDirectSubmission']=='0' and env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1',
            'qualified safety settings differ')
    args = ['--draft-indexed'] if order is None else ['--draft-indexed-timing',order]
    row = dict(binary=build['binary'],boot=live['boot'],driver=live['driver'],
               runtime_pins=runtime,native_budget=BUDGET)
    scope = owner.Owner(stage/'owned',lambda:write(stage/'execution.json',scope.commands))
    case = dict(stage=action,order=order,environment_sha256=bank.signature(env))
    def read_output(text,diagnostic):
        case['owner']=parse(text,diagnostic,order)
    try:
        if action=='smoke':
            check_assets(models,full=True)
        run_owner_fixture(row,scope,case,stage,action,env,[*args,vocab,*shards],
                          lambda:check_assets(models),read_output)
        if action==ORDER[-1][0]:
            check_assets(models,full=True)
    except BaseException as error:
        case.update(passed=False,error=repr(error))
        raise
    finally:
        case['GPU_executed']=bool(row.get('GPU_executed'))
        case['ownership_closed']=scope.active is None and all(owner.closed(c) for c in scope.commands)
        cases.append(case)
        if len(cases)==len(ORDER) and all(c['passed'] and c['ownership_closed'] for c in cases):
            identities = {(c['native_identity']['pid'],c['native_identity']['start_ticks']) for c in cases}
            require(len(identities)==len(ORDER), 'independent native processes required')
            ratios = [c['owner']['candidate_time_ratio'] for c in cases[1:]]
            report['decision'] = dict(process_ratios=ratios,advance=all(r<1 for r in ratios),
                                      model_executed=False,adopted=False)
        write(stage/'samples.tmp',report)
        (stage/'samples.tmp').replace(out_path)


def main(action):
    native, overlay, commands, allowed, proof = inputs()
    require(proof['recipe_sha256']==RECIPE_SHA, 'recover the exact prospectively committed compile/link recipe')
    live, models, shards, vocab, runtime = model_inputs(native,proof)
    check_assets(models)
    require(live['environment_template']['STRATA_SYCL_DECODE_Q6_DPAS_HEAD']=='2' and
            guards.driver_guard()==live['driver'] and BOOT.read_text().strip()==live['boot'],
            'approved packed owner or B570 boot/driver changed')
    if action=='check':
        print(json.dumps(dict(proof,controller=pinned_controller(),matched_model_files=len(models),
                              matched_runtime_files=len(runtime)),indent=2))
        return
    if action=='build':
        fixture.build_fixture(ROOT,TARGET,native,overlay,commands,allowed,
                              dict(proof=proof,controller=pinned_controller()),build_directory=BUILD)
        build = bank.previous.read(ROOT/'record.json')
        try:
            build['archive_proof'] = checked_archive(commands,allowed)
        except BaseException as error:
            build.update(complete=False,error=repr(error))
            raise
        finally:
            write(ROOT/'record.json',build)
    else:
        build = bank.previous.read(ROOT/'record.json')
        run_case(action,build,live,models,shards,vocab,runtime,proof,allowed,commands)


if __name__=='__main__':
    action, = sys.argv[1:]
    require(action in ('check','build',*(stage for stage,_ in ORDER)), 'use check, build, smoke, 0, 1, or 2')
    if action=='check':
        main(action)
    else:
        with (B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            guards.no_model_owners([])
            main(action)
