#!/usr/bin/env python3
"""Main-only resident-MTP diagnostic build from the selected original Git inputs.

Repository LICENSE. No XeStrata application source. This does not select an
engine or execute a GPU/model. Generated source is a disposable build input,
not another source archive. Commit each prepared operation before executing it.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import build_engine as build
import build_full_context as prior
import check_full_context_source as selected
import measurement_guards as guards
import measurement_owner as supervision
import build_manifest

P, B = build.P, build.B
ROOT = B / 'mtp-resident-control-v3'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/mtp-resident-control-current.json'
BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
INPUTS = ['tools/sycl/mtp_resident_control.py', 'tools/sycl/test_mtp_resident_control.py',
          'tools/sycl/build_engine.py', 'tools/sycl/build_full_context.py',
          'tools/sycl/check_full_context_source.py', 'tools/sycl/measurement_guards.py',
          'tools/sycl/measurement_owner.py', 'tools/sycl/measurement_environment.json',
          'tools/sycl/build_manifest.py', 'tests/core/test_build_manifest.py']
GUARD_ANCHOR = '    strata::core::set_coupled_draft(o.coupled_draft);\n'
GUARD = '''    // Private resident-MTP control: no reuse of generation MTP state.
    if (!o.serve || o.spec != 4 || (o.mtp_max_t != 0 && o.mtp_max_t != 1) || o.mtp.empty() ||
        o.prompt_cache != 0 || o.conversation_cache_mib != 0 || o.suffix_draft != 0 ||
        o.coupled_draft || !o.spec_oracle.empty() || o.vision || !o.layer_split.empty() ||
        o.prefill_chunk != 8192 || o.prefill_auto || o.expert_cache != 1256 || !o.expert_cache_per_layer ||
        o.max_context != 131072 || o.kv_resident != 131072 ||
        o.peer_device >= 0 || o.gpu_stages || o.graph_only || o.gpu_only_full || !o.no_token_graph ||
        std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(), [](auto n) { return n > 0; })) {
        std::fprintf(stderr, "resident MTP control requires fresh single-GPU serve, spec4, mtp-max-t0/1, "
                             "prefill8192,1256 per-layer slots,resident131072; no caches,suffix,coupled/oracle,graphs/split\\n");
        return 2;
    }
'''
OBSERVER = '''#ifdef STRATA_ENABLE_SYCL
    using Observation = strata::sycl_backend::MtpGraphObserver;
    auto observation = graph_observer_.window(T, p, a, coupled_active_);
#endif
'''
CAP_ANCHOR = '    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs\n'
CAP_ZERO = '''    // Diagnostic cap zero omits generation catch-up and proposal computation.
    // Prompt precompute, MTP weights/KV and the target verifier stay resident.
    if (max_drafts_ == 0) {
        if (!drafts) { err = "mtp control: missing draft output"; return false; }
        for (int j = 0; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
        if (n_drafts) *n_drafts = 0;
        return true;
    }
'''
FOOTER_ANCHOR = '''#ifdef STRATA_ENABLE_SYCL
            const auto copy1 = ver.expert_copy_stats();'''
FOOTER = '''            std::fprintf(stderr, "strata resident MTP control: mtp_max_t=%d cumulative_draft_rounds=%lld\\n",
                         o.mtp_max_t, (long long) mtp.rounds);
'''


def write(path, row):
    path.write_text(json.dumps(row, indent=2) + '\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replace_once(text, anchor, replacement):
    guards.require(text.count(anchor) == 1, 'source anchor missing or ambiguous')
    return text.replace(anchor, replacement, 1)


def transformed(name, raw):
    if name not in ('src/core/mtp.cpp', 'src/program/generate.cpp'):
        return raw
    text = raw.decode()
    if name == 'src/core/mtp.cpp':
        # All device/head/expert/argument checks precede the shortcut. The
        # observer must not open an uncompleted window on this CPU-only return.
        text = replace_once(text, OBSERVER, '')
        text = replace_once(text, CAP_ANCHOR, CAP_ZERO + OBSERVER + CAP_ANCHOR)
    elif name == 'src/program/generate.cpp':
        text = replace_once(text, GUARD_ANCHOR, GUARD + GUARD_ANCHOR)
        text = replace_once(text, FOOTER_ANCHOR, FOOTER + FOOTER_ANCHOR)
    return text.encode()


def committed_inputs(revision):
    result = {}
    for name, raw in build.blobs(P, revision, INPUTS):
        guards.require((P / name).read_bytes() == raw, 'commit exact input: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def recover_sources(record):
    total = 0
    for s in [record['project'], *record['dependencies']]:
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                     build.blobs(Path(s['root']), s['revision'], s['source_sha256'])}
        guards.require(recovered == s['source_sha256'], 'original linked source cannot recover')
        total += len(recovered)
    return total


def project_sources(original):
    # CMake generates every declared target, including unrequested diagnostics.
    # The original linked-input receipt correctly excludes their uncompiled
    # sources; a standalone source tree still needs them to configure.
    result=dict(original['project']['source_sha256'])
    backend=build.git(P,'show',BASE+':cmake/sycl_backend.cmake').decode()
    names=sorted(set(re.findall(r'tools/sycl/[A-Za-z0-9_./-]+\.(?:cpp|hpp|c|h)\b',backend)))
    for n,raw in build.blobs(P,BASE,names):result[n]=hashlib.sha256(raw).hexdigest()
    return result


def compact(row):
    return {k: v for k, v in row.items() if k not in ('project_inputs', 'generated_inputs',
        'dependency_inputs', 'external_inputs', 'environment', 'commands', 'compiler_inputs',
        'actual_compiled_inputs')}


def prepare():
    guards.require(not ROOT.exists() and RESULT.read_bytes() == build.git(P,'show',
        'HEAD:'+str(RESULT.relative_to(P))), 'fresh root after committed failure closure only')
    previous=json.loads(RESULT.read_text())
    guards.require(previous['stage']=='build_failed' and previous['ownership_closed'] and
        not previous['active'] and not previous['GPU_executed'] and not previous['passed'] and
        previous['retire_generation']['complete'] and
        not previous.get('assets') and not (B/'mtp-resident-control-v2/build/strata').exists(),
        'do not retry or promote the closed configure failure')
    revision = build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = committed_inputs(revision)
    original, proof = selected.baseline()
    guards.require(original['source_commit'] == BASE, 'selected base changed')
    old = json.loads(prior.ENGINE_RECORD.read_text())
    old_result = json.loads(selected.RESULT.read_text())[prior.ENGINE_KEY]
    retired=old_result.pop('artifact_retirement')
    guards.require(old_result == prior.summary(old) and old['stage'] == 'closed_link' and
        old['passed'] and old['complete'] and old['ownership_closed'] and not old['active'],
        'unselected generation not closed')
    speed = json.loads((RESULT.parent / 'phase-prefill-speed-current.json').read_text())
    guards.require(speed['stage'] == 'closed' and speed['selection']['candidate_adopted'] is False and
        speed['selection']['selected_source'] == BASE and speed['artifact_retirement']['complete'],
        'close prior prefill comparison before replacing its unselected generation')
    expected_old_env = dict(original['environment'], TMPDIR=str(prior.ROOT/'tmp'),
                            XDG_CACHE_HOME=str(prior.ROOT/'cache'))
    guards.require(recover_sources(old) == 9853 and old['environment'] == expected_old_env,
        'old source/environment recovery differs')
    retirement=previous['retire_generation']
    guards.require(retirement['receipt_sha256']==build.digest(prior.ENGINE_RECORD) and
        all(retired[k]==v for k,v in retirement.items()) and
        build.digest(retirement['manifest_path'])==retirement['manifest_sha256'] and
        not Path(retirement['root']).exists() and
        all(not Path(v['path']).exists() for v in retirement['assets']),
        'completed retirement evidence changed;never repeat deletion')
    remote = build.git(P, 'ls-remote', 'origin', 'refs/heads/feature/sycl').decode().split()
    guards.require(len(remote) == 2 and remote[1] == 'refs/heads/feature/sycl', 'missing retained remote')
    build.git(P, 'merge-base', '--is-ancestor', old['source_commit'], remote[0])
    build.git(P, 'merge-base', '--is-ancestor', BASE, remote[0])
    for tool in old['tools'].values():
        guards.require(build.digest(tool['path']) == tool['sha256'], 'build tool changed')
    sdk = prior.sdk()
    guards.require(all(build.digest(p) == h for p, h in old['external_inputs'].items()), 'MKL inputs changed')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT/name).mkdir()
    source = ROOT / 'source'; source.mkdir()
    project=project_sources(original);generated = {}
    for name, raw in build.blobs(P, BASE, project):
        new = transformed(name, raw)
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(new)
        generated[name] = hashlib.sha256(new).hexdigest()
    changed = sorted(n for n, h in generated.items() if h != project[n])
    guards.require(changed == ['src/core/mtp.cpp', 'src/program/generate.cpp'], 'component selection changed')
    argv = list(original['configure_argv'])
    argv[argv.index('-S') + 1] = str(source)
    argv[argv.index('-B') + 1] = str(ROOT / 'build')
    argv.append('-DCMAKE_MAKE_PROGRAM:FILEPATH='+old['tools']['ninja']['path'])
    row = dict(schema='strata-resident-mtp-control-v3', stage='retired_previous', source_commit=revision,
        input_sha256=pins, selected_original=proof, base_native_source=BASE,
        project_inputs=project, generated_inputs=generated,
        configure_extra_paths=sorted(set(project)-set(original['project']['source_sha256'])),
        generated_source_files=len(generated), generated_source_fingerprint=fingerprint(generated),
        changed_native_files=changed, dependency_inputs=original['dependencies'],
        compiler_inputs=dict(sdk=sdk, tools=old['tools']), compiler=original['compiler'],
        external_inputs=old['external_inputs'], environment=dict(original['environment'],
            TMPDIR=str(ROOT/'tmp'), XDG_CACHE_HOME=str(ROOT/'cache')),
        environment_reference=dict(receipt=proof['receipt'],
            replacements={'TMPDIR':str(ROOT/'tmp'),'XDG_CACHE_HOME':str(ROOT/'cache')}),
        configure_argv=argv, targets=original['targets'],
        build_command=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *original['targets'], '-j', '4'],
        dependency_command=['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'],
        generated_header_sha256={str(ROOT / 'build' / 'ggml/src/ggml-version.h'):
            next(iter(old['actual_dependency_closure']['generated_sha256'].values()))},
        retire_generation=retirement,
        prior_failed_operation=dict(archive_commit=revision,path=str(RESULT.relative_to(P)),
            source_commit=previous['source_commit'],stage=previous['stage'],
            cause='Original CMake references9 unrequested diagnostic .cpp inputs absent from the linked-source receipt. CPU8+6 checks passed;configure generation failed,normal owned closure,no engine compilation or GPU execution.',
            original_v1_archive=dict(commit='afb9b33ed',path=str(RESULT.relative_to(P)),
                outcome='V1 source/admission failures remain failed;original deletion manifest remains retained.'),
            outcome='New committed source/root/plan includes original uncompiled configuration inputs;no failed operation is retried or promoted.'),
        question='Measure resident MTP4 versus resident-no-decode-draft under unchanged original prefill,target1256 slots and safety. This is an attribution control,not full MTP-off or hardware saturation.',
        criteria='Build from exact selected6e40 source plus only two declared generated TUs. Native startup refuses standalone,caches,suffix,coupled/oracle,graphs/split. Cap0 returns only after owner/T/a checks,clears outputs and avoids graph observation/staging/submission. Default cap3 retains original math. Before models require owned build/actual dependency closure,CPU refusal checks,bounded traced device and short model checks. Repeated thinking-on code/prose,long/agent-session and independent prefill preservation remain necessary for adoption.',
        budgets=dict(wall=1200,cpu=7200,rss_cap=16<<30,text_cap=8<<20,file_cap=256<<20,total_cap=4<<30),
        passed=False, complete=False, active=False, ownership_closed=True, GPU_executed=False,
        production_selected=False, model_adoption_qualified=False,
        retention='One selected original13-asset fallback plus this imminent diagnostic generation. Retire old unselected7 assets/build only after original source/recipe/remote/consumer checks. Generated source is a disposable current compile input;remove it with closed build output after qualification/comparison. Keep compact measurements and original inputs in Git.')
    row['metadata_sha256']=build_manifest.pins(row)
    write(ROOT / 'record.json', row); write(RESULT, compact(row))
    print('Prepared fresh v3 original-source/configuration diagnostic;prior failures stay closed. Commit before build.', flush=True)


def admission(row):
    guards.require(RESULT.read_bytes() == build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))),
        'commit the exact prepared operation')
    frozen=json.loads(RESULT.read_text())
    guards.require(frozen==compact(row), 'private operation differs from committed plan')
    build_manifest.verify(row,frozen['metadata_sha256'])
    guards.require(committed_inputs(row['source_commit']) == row['input_sha256'], 'controller changed')
    original,proof=selected.baseline()
    guards.require(proof == row['selected_original'] and row['base_native_source']==BASE and
        row['project_inputs']==project_sources(original) and
        row['dependency_inputs']==original['dependencies'] and
        row['environment']==dict(original['environment'],TMPDIR=str(ROOT/'tmp'),
            XDG_CACHE_HOME=str(ROOT/'cache')), 'original source/dependencies/effective environment changed')
    generated={n:hashlib.sha256(transformed(n,raw)).hexdigest() for n,raw in
        build.blobs(P,BASE,row['project_inputs'])}
    guards.require(row['generated_inputs']==generated and len(generated)==row['generated_source_files'] and
        fingerprint(generated)==row['generated_source_fingerprint'] and
        {str(p.relative_to(ROOT/'source')) for p in (ROOT/'source').rglob('*') if p.is_file()}==set(generated),
        'generated source map/transform/extent changed')
    tools=row['compiler_inputs']['tools']
    guards.require(row['configure_argv'][0]==row['build_command'][0]==tools['cmake']['path'] and
        row['dependency_command'][0]==tools['ninja']['path'] and
        '-DCMAKE_MAKE_PROGRAM:FILEPATH='+tools['ninja']['path'] in row['configure_argv'] and
        '-DCMAKE_CXX_COMPILER='+row['compiler']['path'] in row['configure_argv'],
        'actual command-to-tool binding changed')
    guards.require(prior.sdk() == row['compiler_inputs']['sdk'], 'SDK changed')
    for v in row['compiler_inputs']['tools'].values():
        guards.require(build.digest(v['path']) == v['sha256'], 'build tool changed')
    guards.require(build.digest(row['compiler']['path']) == row['compiler']['sha256'], 'compiler changed')
    for n, h in row['generated_inputs'].items():
        guards.require(build.digest(ROOT / 'source' / n) == h, 'generated native input changed')
    for source in row['dependency_inputs']:
        guards.require(build.pin(Path(source['root']), ['.']) == source, 'dependency changed')
    for n, h in row['external_inputs'].items():
        guards.require(build.digest(n) == h, 'external input changed')


def dependency_closure(row, text):
    expected = {str((ROOT/'source'/n).resolve()):h for n,h in row['generated_inputs'].items()}
    for source in row['dependency_inputs']:
        expected.update({str((Path(source['root'])/n).resolve()):h for n,h in source['source_sha256'].items()})
    expected.update({str(Path(v['path']).resolve()):v['sha256'] for v in json.loads(prior.SDK.read_text())})
    expected.update(row['external_inputs']); expected.update(row['generated_header_sha256'])
    paths={Path(line.strip()) for line in text.splitlines() if line.startswith('    ') and line.strip()}
    paths={(p if p.is_absolute() else ROOT/'build'/p).resolve() for p in paths}
    guards.require(len(paths)>100 and (ROOT/'source/src/program/generate.cpp').resolve() in paths and
        (ROOT/'source/src/core/mtp.cpp').resolve() in paths, 'actual compilation census incomplete')
    uncovered=sorted(str(p) for p in paths if str(p) not in expected)
    guards.require(not uncovered, 'uncovered compiled input: '+repr(uncovered[:12]))
    for p in paths: guards.require(build.digest(p)==expected[str(p)], 'compiled input changed: '+str(p))
    closure={str(p):expected[str(p)] for p in sorted(paths)}
    row['actual_compiled_inputs']=closure
    return dict(files=len(paths),uncovered_files=0,sha256=fingerprint(closure),
        generated_headers=row['generated_header_sha256'])


def compile_engine():
    row=json.loads((ROOT/'record.json').read_text());admission(row)
    guards.require(row['stage']=='retired_previous' and row['retire_generation']['complete'] and
        not (ROOT/'owned').exists(), 'fresh build after closed retirement only')
    owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(row,commands=owner.commands if owner else []))
    try:
        owner=supervision.Owner(ROOT/'owned',persist)
        for label,argv in [('host-controls',['/usr/bin/python3','-B',str(P/INPUTS[1])]),
            ('metadata-controls',['/usr/bin/python3','-B',str(P/'tests/core/test_build_manifest.py')]),
            ('configure',row['configure_argv']),('build',row['build_command'])]:
            entry,out,err=owner.run(label,argv,row['environment'],P,**row['budgets'])
            guards.require(supervision.closed(entry) and entry['exit_code']==0,
                label+' failed: '+(out.read_text(errors='replace')+err.read_text(errors='replace'))[-8192:])
            admission(row)
        assets={n:dict(path=str(ROOT/'build'/n),bytes=(ROOT/'build'/n).stat().st_size,
            sha256=build.digest(ROOT/'build'/n)) for n in row['targets']}
        entry,out,err=owner.run('dependency-census',row['dependency_command'],row['environment'],P,**row['budgets'])
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'dependency census failed')
        row['dependency_closure']=dependency_closure(row,out.read_text());row['assets']=assets
        row.update(stage='closed_link',passed=True,complete=True)
    except BaseException as error:
        row.update(stage='build_failed',passed=False,complete=False,error=repr(error))
    finally:
        row['ownership_closed']=owner is None or (owner.active is None and all(supervision.closed(e) for e in owner.commands))
        row['passed']=bool(row['passed'] and row['ownership_closed']);row['complete']=row['passed']
        persist();write(RESULT,compact(row))
    print(json.dumps({k:row.get(k) for k in ('stage','passed','complete','ownership_closed','error')}),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    guards.require(sys.argv[1:] in (['prepare'],['build']), 'prepare|build')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([])
        if sys.argv[1]=='prepare':prepare()
        else:sys.exit(compile_engine())
