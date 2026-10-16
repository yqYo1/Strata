#!/usr/bin/env python3
# Authored for Strata; repository LICENSE (MIT) applies.
"""SOURCE ONLY / UNRUN. Main alone executes under the shared measurement lock."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
A = PACKAGE.parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
SCHEMA = 'strata-r816-vllm-xe2-qualification-v1'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|FIXTURE_FAILED|CHECKED_DRAIN_FAILED|R816_EVIDENCE_FAILED|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed')
MODEL = re.compile(r'(^|/)(strata(?:[-_.][^/]*)?|sycl_[^/]*|paired_gemm|strata_vllm_xe2_numerics|bench\.sh|serve\.py|engine[-_]wrapper\.py|run_server_job\.py)$')
ROUTES = ['old_f32_direct', 'old_f32_plus_gpu_bf16_cast', 'native_bf16_output', 'vllm_compact16x64_bf16_output', 'vllm_wide128x128_bf16_output']
SHAPES = [[160,1280,2560], [160,2560,640], [8192,1280,2560]]
CHECKS = ['guards','edges','chain','full']
TEXT_CAP = 64 << 20


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32 << 20, 'bounded JSON input: ' + str(path))
    return json.loads(path.read_text())


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def pin(path):
    path = Path(path)
    return dict(sha256=sha(path), bytes=path.stat().st_size, realpath=str(path.resolve()))


def pinned(path, expected):
    require(pin(path) == {k: expected[k] for k in ('sha256','bytes','realpath')}, 'file identity changed: ' + str(path))


def tree_pin(root):
    """Bounded read-only complete header-root snapshot, including resolved paths.

    Digest is SHA256 of sorted compact JSON rows [relative path,realpath,bytes,
    sha256] plus LF. No preprocessor/tool invocation and no environment changes.
    Directory aliases are hashed as path/resolution rows; each inode is read once.
    """
    root = Path(root)
    require(root.is_dir(), 'existing include root: ' + str(root))
    paths, directories, seen = [], [], set()
    for parent, dirs, files in os.walk(root, followlinks=True):
        status = Path(parent).stat()
        identity = (status.st_dev, status.st_ino)
        directories.append([str(Path(parent).relative_to(root)),str(Path(parent).resolve()),'directory'])
        require(len(directories) <= 65536, 'header directory budget')
        dirs.sort()
        if identity in seen:
            dirs[:] = []  # record alias binding, do not walk a cycle twice
            continue
        seen.add(identity)
        for name in files:
            path = Path(parent)/name
            require(path.is_file(), 'nonregular header input: ' + str(path))
            paths.append(path)
            require(len(paths) <= 100000, 'header file budget')
    digest, total = hashlib.sha256(), 0
    rows = list(directories)
    for path in sorted(paths, key=lambda p: str(p.relative_to(root))):
        value = pin(path)
        total += value['bytes']
        require(total <= 2 << 30, 'header byte budget')
        row = [str(path.relative_to(root)), value['realpath'], value['bytes'], value['sha256']]
        rows.append(row)
    for row in sorted(rows,key=lambda row: (row[0],len(row))):
        digest.update((json.dumps(row, separators=(',',':'))+'\n').encode())
    return dict(sha256=digest.hexdigest(), files=len(paths), bytes=total, realpath=str(root.resolve()))


def driver_guard():
    require((DEVICE/'driver').resolve().name == 'xe', 'expected current xe driver')
    require((DEVICE/'vendor').read_text().strip() == '0x8086', 'expected Intel PCI device')
    dumps = list(Path('/sys/class/devcoredump').glob('devcd*'))
    require(not (DEVICE/'devcoredump').exists() and not dumps, 'existing dump; no reset/removal authorized')
    return dict(pci=str(DEVICE), driver='xe', vendor='0x8086', device=(DEVICE/'device').read_text().strip(),
                kernel=Path('/proc/sys/kernel/osrelease').read_text().strip(), devcoredump=False)


def no_model_owners(model_files):
    """Main-time read-only scan; never adopt/signal a discovered owner."""
    observed = []
    population = list(Path('/proc').iterdir())
    require(len(population) < 131072, 'finite proc population')
    for path in population:
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = [a.decode(errors='replace') for a in (path/'cmdline').read_bytes().split(b'\0') if a]
            matches = [a for a in args if not a.startswith('-') and MODEL.search(a)]
            matches += [a for a in args if str(P/'serve') in a and a.endswith('.py')]
            matches += [a for a in args if a in model_files or a.startswith('serve.') or a.endswith('.gguf')]
            if matches:
                observed.append(dict(pid=int(path.name), matching_arguments=matches))
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(not observed, 'model/fixture owners present: ' + json.dumps(observed))
    return dict(same_uid_model_owners=[], scope='bounded read-only cmdline scan; cannot prove hidden foreign GPU use absent')


def receipt_gate(module, path, expected_hash, current_boot, *, qualification, failure_path=None, failure_hash=None, reason=None, historical_build_only=False):
    path = Path(path).resolve()
    require(not path.is_relative_to(B/'prefill-gemm-only-contract-root-v1'), 'protected credential path forbidden')
    require(re.fullmatch('[0-9a-f]{64}', expected_hash or '') and sha(path) == expected_hash, 'explicit receipt SHA: ' + str(path))
    value = load(path)
    require(value.get('complete') is True and value.get('active') is False, 'receipt incomplete/active: ' + str(path))
    require(value.get('boot_id') == current_boot, 'receipt boot identity')
    if historical_build_only:
        require(path == (A/'own-native16-prefill-build-v2.json').resolve() and qualification and
                value.get('GPU_executed') is False and value.get('source_commit') == '9c1a2020897a9451ec616a6bb036a661de9d86b0',
                'only the exact qualified historical build-only receipt may omit GPU fault fields')
        require([c.get('label') for c in value.get('commands',[])] ==
                ['configure','build','host-prefill_layer_major_plan_test','host-dense_prefill_nll_host_test'],
                'historical build-only command census')
    else:
        require('fresh_faults' in value and not value['fresh_faults'] and value.get('driver_dump_exists') is False, 'receipt fault/dump state unqualified')
    require(not value.get('final_fault_check_error') and not value.get('closure_errors'), 'receipt incomplete closure/fault evidence')
    commands = value.get('commands')
    require(isinstance(commands,list) and commands, 'receipt needs all original owned commands')
    for command in commands:
        # module.closed concerns observation/reap/session cleanup, NOT exit zero.
        require(module.closed(command), 'command ownership not normally closed; hung/cleanup receipt is refused')
        require(command.get('owners') and command.get('direct_identity'), 'command lacks PID/start identities')
        owners = command['owners'] + [command['direct_identity']]
        for old in owners:
            require(isinstance(old.get('pid'),int) and isinstance(old.get('start_ticks'),int), 'recorded owner identity')
            now = module.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'prior PID/start identity remains present')
    success = value.get('passed') is True and all(c.get('exit_code') == 0 for c in commands)
    if qualification:
        require(success, 'failed correctness/build gate can never qualify: ' + str(path))
    elif not success:
        require(failure_path is not None and Path(failure_path).resolve() == path and failure_hash == expected_hash,
                'failed prior requires explicit same closed-failure receipt and SHA')
        require(isinstance(reason,str) and 16 <= len(reason.strip()) <= 2048, 'manual-reviewed continuation reason required')
        # No retry, gate substitution, result rewrite or automatic FTZ contract.
        # A normal compile exit1 without GPU work can close, unlike a hang that
        # required termination or lost ownership. Native exit1 remains a failure.
    else:
        require(failure_path is None and failure_hash is None and reason is None, 'failure continuation flags on passing prior')
    return value, dict(path=str(path), sha256=expected_hash, boot_id=current_boot, ownership_closed=True,
        original_passed=value.get('passed'), all_commands_exit_zero=all(c.get('exit_code') == 0 for c in commands),
        manual_reviewed_continuation_reason=reason if not success else None,
        original_status=value.get('status'), original_error=value.get('error'), original_GPU_executed=value.get('GPU_executed'),
        original_status_preserved=True, qualification=qualification, historical_build_only=historical_build_only,
        historical_GPU_fault_check='not recorded for build-only; independent GPU68 and current fresh windows required' if historical_build_only else 'required clean')


def parse_json_rows(text):
    rows, errors = [], []
    for number, line in enumerate(text.splitlines(),1):
        if not line.startswith('{'):
            continue
        try:
            row = json.loads(line)
            require(isinstance(row,dict), 'JSON object required')
            rows.append(row)
        except (ValueError, RuntimeError) as exc:
            errors.append(dict(line=number,error=repr(exc)))
    return rows, errors


def library_evidence(rows, manifest, phase):
    snapshots = [r for r in rows if r.get('kind') == 'libraries']
    require([s.get('phase') for s in snapshots] == ['before_runtime',phase], 'both own-process DSO snapshots required')
    paths = set()
    for snapshot in snapshots:
        require(snapshot.get('paths') and len(snapshot['paths']) <= 512, 'finite nonempty DSO inventory')
        for name in snapshot['paths']:
            require(not name.endswith(' (deleted)'), 'deleted mapped DSO')
            path = str(Path(name).resolve())
            require(path in manifest['loaded_library_pins'], 'unreviewed mapped DSO: ' + name)
            pinned(path,manifest['loaded_library_pins'][path])
            paths.add(path)
    for path in manifest['required_runtime_libraries']:
        require(path in paths, 'missing actual runtime/provider/driver mapping: ' + path)
    return dict(snapshots=snapshots, sha256_pins={p:manifest['loaded_library_pins'][p] for p in sorted(paths)},
                limitation='before/after snapshots do not inventory libraries unloaded between snapshots; no PLT dladdr or ISA inference')


def fixture_evidence(stdout, rows, suite):
    returns = [r for r in rows if r.get('kind') == 'fixture_return']
    require(len(returns) == 1 and returns[0].get('exit_code') == 0, 'original fixture return-zero marker')
    lines = stdout.splitlines()
    expected = dict(smoke=32, guards=26, edges=16, chain=8, full=120)
    if suite == 'guards':
        require(lines.count('guards PASS rejected=26 output/inputs immutable') == 1, 'all 26 original rejects and drained immutable readback')
        require(not any(x.startswith(('dyadic PASS','edges PASS','chain PASS')) for x in lines), 'unexpected guard workload')
        actual = 26
    else:
        require(lines.count('COMPONENT_SUITE_PASS only; DPAS_BINARY_PROOF/model_quality/full262144/performance UNRUN') == 1,
                'original entire component completion marker')
        prefix = 'dyadic PASS' if suite in ('smoke','full') else suite + ' PASS'
        actual = sum(line.startswith(prefix) for line in lines)
        require(actual == expected[suite], 'all original case completions: ' + suite)
        if suite in ('smoke','full'):
            combinations = set()
            for line in lines:
                if not line.startswith(prefix):
                    continue
                hit = re.fullmatch(r'dyadic PASS t=(\d+) n=(\d+) k=(\d+) ld=(\d+) (fp16|bf16)->(fp16|bf16) tile=([01])',line)
                require(hit is not None, 'original dyadic record format')
                t,n,k,ld = map(int,hit.group(1,2,3,4))
                combinations.add((t,n,k,ld,hit[5],hit[6],int(hit[7])))
            geometries = [(t,128,64,136) for t in (1,3,17,160)] if suite == 'smoke' else [
                (t,n,k,n+pad) for t in (1,3,17,160,8192) for n,k,pad in ((1280,2560,0),(2560,640,8),(640,1280,16))]
            wanted = {(t,n,k,ld,i,o,tile) for t,n,k,ld in geometries for i in ('fp16','bf16') for o in ('fp16','bf16') for tile in (0,1)}
            require(combinations == wanted, 'exact IO/tile/geometry census')
        if suite == 'edges':
            require(sum(line.startswith('edge row=') for line in lines) == 16*17, 'all edge row classifications/bits retained')
            # Strict gradual-underflow remains mandatory. No relaxed/FTZ pass.
            require(sum(line == 'edges PASS strict gradual-underflow comparison; hardware contract still requires review' for line in lines) == 16,
                    'strict denormal/rounding gate')
    return dict(suite=suite, declared_cases=expected[suite], actual_cases=actual,
                checked_drain='original PASS after CheckedDrain::finish plus original return zero and owned normal exit',
                strict_denormal_contract='unchanged; failure cannot be waived by this controller')


def benchmark_evidence(rows, shape, kind, process_index, smoke):
    def one(tag):
        values = [r for r in rows if r.get('kind') == tag]
        require(len(values) == 1, 'one benchmark ' + tag)
        return values[0]
    device = one('device')
    require(device.get('vendor_id') == 0x8086 and device.get('matrix_tuples') and device.get('in_order') is True,
            'actual Intel GPU/matrix/in-order evidence')
    plan = one('plan')
    dims = [3,128,64] if smoke else SHAPES[shape]
    require([plan.get(k) for k in ('T','N','K')] == dims and plan.get('ldy') == dims[1] and plan.get('beta') == 0,
            'same declared actual geometry/beta')
    require(plan.get('shape') == shape and plan.get('input_type') == kind and plan.get('process_index') == process_index and
            plan.get('rotation_base') == process_index and plan.get('route_order') == ROUTES and plan.get('warmups_per_route') == 3 and
            plan.get('timed_submits_per_route') == (0 if smoke else 3) and plan.get('oracle') == 'rank2_binary_exact_v1', 'exact benchmark plan')
    require(isinstance(plan.get('pid'),int), 'native process ID')
    samples = [r for r in rows if r.get('kind') == 'sample']
    validations = [r for r in rows if r.get('kind') == 'validation']
    begins = [r for r in rows if r.get('kind') == 'begin']
    expected = [(ROUTES[(j+round+process_index)%5],round<3,round%3) for round in range(3 if smoke else 6) for j in range(5)]
    require(len(samples) == len(validations) == len(begins) == len(expected), 'all original warmups/timed samples present')
    events = [r.get('kind') for r in rows if r.get('kind') in ('begin','sample','validation')]
    require(events == ['begin','sample','validation']*len(expected), 'flushed begin/sample/validation chronology')
    retained = []
    for ordinal,(begin,sample,validation,(route,warmup,round)) in enumerate(zip(begins,samples,validations,expected)):
        for row in (begin,sample,validation):
            require(row.get('ordinal') == ordinal and row.get('route') == route, 'exact ordinal/order')
        require(begin.get('warmup') == sample.get('warmup') == warmup and begin.get('sample') == sample.get('sample') == round, 'round/sample/warmup identity')
        require(sample.get('status') == 'pending_validation' and sample.get('stored_output_type') == ('fp32' if route == ROUTES[0] else 'bf16') and
                sample.get('global_fp32_intermediate') == (route == ROUTES[1]), 'stored type/global FP32 semantics')
        for field in ('submit_to_completion_wall_seconds','thread_cpu_submit_seconds','thread_cpu_wait_check_seconds'):
            require(isinstance(sample.get(field),(int,float)) and math.isfinite(sample[field]) and sample[field] >= 0, 'finite seconds: ' + field)
        require(sample['submit_to_completion_wall_seconds'] > 0, 'positive wall time')
        require(validation.get('status') == 'PASS' and validation.get('outputs_checked') == dims[0]*dims[1] and
                all(validation.get(k) is True for k in ('all_finite','closed_form_oracle','guards','inputs_immutable')), 'full independent exact output/guard/input gate')
        retained.append(dict(measurement=sample,validation=validation))
    result = one('result')
    require(result.get('status') == 'PASS' and result.get('validated_invocations') == len(expected) and result.get('samples_per_route') == (0 if smoke else 3) and
            result.get('warmups_per_route') == 3 and result.get('process_index') == process_index, 'entire benchmark process result')
    return dict(device=device, plan=plan, samples=retained, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['build','smoke','check','measure'])
    parser.add_argument('--root',type=Path,required=True,help='explicit fresh direct child of external B, named main-r816-*')
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--prior-sha256',required=True)
    parser.add_argument('--confirm-latest-closed-prior',action='store_true',required=True)
    parser.add_argument('--closed-failure-receipt',type=Path)
    parser.add_argument('--closed-failure-sha256')
    parser.add_argument('--manual-reviewed-continuation-reason')
    for name in ('build','smoke','check'):
        parser.add_argument('--'+name+'-record',type=Path)
        parser.add_argument('--'+name+'-record-sha256')
    args = parser.parse_args()
    manifest_path = PACKAGE/'manifest.json'
    manifest_sha = sha(manifest_path)
    require(re.fullmatch('[0-9a-f]{64}',args.manifest_sha256) and manifest_sha == args.manifest_sha256, 'explicit reviewed manifest SHA')
    manifest = load(manifest_path)
    require(manifest_path.stat().st_size <= 120 << 10, 'manifest <=120KiB')
    require(set(p.name for p in PACKAGE.iterdir()) == set(manifest['package_files']) | {'manifest.json','SHA256SUMS'}, 'exact sealed source package')
    require(sum(p.stat().st_size for p in PACKAGE.iterdir()) <= 240 << 10, 'source package <=240KiB')
    require(Path(sys.executable).resolve() == Path('/usr/bin/python3').resolve(), 'reviewed system Python only')
    root = args.root.absolute()
    require(root.parent.resolve() == B.resolve() and re.fullmatch('main-r816-[A-Za-z0-9_-]{1,80}',root.name), 'private root scope')
    require(not root.exists() and not root.is_symlink(), 'explicit fresh root; no overwrite/retry')
    if args.mode != 'build':
        require(args.build_record and args.build_record_sha256, 'explicit private build receipt/SHA')
    if args.mode in ('check','measure'):
        require(args.smoke_record and args.smoke_record_sha256, 'explicit original smoke receipt/SHA')
    if args.mode == 'measure':
        require(args.check_record and args.check_record_sha256, 'all component suites must PASS before benchmark')
    owner_path = Path(manifest['direct_owner']['path'])
    pinned(owner_path,manifest['direct_owner'])
    spec = importlib.util.spec_from_file_location('r816_pinned_direct_owner',owner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        cap = (2 << 30) if args.mode == 'build' else (512 << 20)
        record = dict(schema=SCHEMA,mode=args.mode,boot_id=boot(),private_root=str(root),passed=False,complete=False,active=False,
            status='FAIL',GPU_executed=False,offline_aot_may_query_device=args.mode=='build',native_launch_attempted=False,
            commands=[],fault_boundaries=[],cases=[],samples=[],manifest_sha256=manifest_sha,source_only_delivery='UNRUN',
            invocation=sys.argv,limits=manifest['limits'],frozen_archive_commit=manifest['qualified_archive_build_commit'],
            unchanged_old_68_cases=True,model_quality_qualified=False,physical_262144_qualified=False,performance_adopted=False)
        record_path = root/'record.json'
        def persist():
            record_path.write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(root/'owned',persist)
        record['commands'] = owner.commands
        persist()
        buildref = None
        before_trees = False
        preflight = False
        workflow_done = False
        cursor = None
        whole_deadline = time.monotonic() + manifest['limits'][args.mode+'_total_wall_seconds']
        def pin_all(trees=False):
            require(boot() == record['boot_id'], 'boot changed')
            require(sha(manifest_path) == manifest_sha, 'manifest drift')
            for name, value in manifest['package_files'].items():
                pinned(PACKAGE/name,value)
            for group in ('input_pins','external_pins','loaded_library_pins'):
                for path,value in manifest[group].items():
                    pinned(path,value)
            pinned(owner_path,manifest['direct_owner'])
            for path,value in record.get('dynamic_pins',{}).items():
                pinned(path,value)
            if buildref is not None:
                require(buildref['source_commit'] == manifest['qualified_archive_build_commit'], 'qualified v2 revision')
                for name,value in buildref['source_pins'].items():
                    require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'qualified relative source path')
                    require(sha(P/name) == value, 'qualified v2 source drift: ' + name)
            m810 = load(Path(manifest['r810_package'])/'manifest.json')
            for item in m810['package_files']:
                pinned(Path(manifest['r810_package'])/item['path'],dict(item,realpath=str((Path(manifest['r810_package'])/item['path']).resolve())))
            for group,base in [('strata',P),('tla_headers',Path(manifest['tla_root']))]:
                for item in m810['source_pins'][group]:
                    p = base/item['path']
                    require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'R810 transitive input pin: ' + str(p))
            if trees:
                observed = {path:tree_pin(path) for path in manifest['header_trees']}
                require(observed == manifest['header_trees'], 'transitive include-root snapshot drift')
                record.setdefault('header_tree_verifications',[]).append(dict(utc=module.utc(),trees=observed))
            record['pin_checks_completed'] = record.get('pin_checks_completed',0)+1
        def remember(path, expected_hash=None):
            value = pin(path)
            require(expected_hash is None or value['sha256'] == expected_hash, 'explicit dynamic SHA mismatch')
            record.setdefault('dynamic_pins',{})[str(Path(path).resolve())] = value
            return value
        def run(label,argv,env,*,building=False,cwd=None):
            require(time.monotonic() < whole_deadline, 'whole workflow deadline')
            pin_all()
            command,out,err = owner.run(label,list(map(str,argv)),dict(env),cwd or root,
                wall=min(manifest['limits']['build_process_wall_seconds' if building else 'native_process_wall_seconds'],whole_deadline-time.monotonic()),
                cpu=18000 if building else 1800,rss_cap=(32 if building else 8)<<30,text_cap=TEXT_CAP,total_cap=cap,file_cap=TEXT_CAP)
            pin_all()
            return command,out,err
        hostenv = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        def normal(command):
            require(module.closed(command) and command.get('exit_code') == 0, 'owned command failed; no retry: ' + command['label'])
        def journal(label,after=None):
            argv = ['/usr/bin/journalctl','-k','-b','--no-pager','-o','json','--show-cursor']
            argv += ['--after-cursor',after] if after else ['-n','0']
            command,out,err = run(label,argv,hostenv)
            normal(command)
            text = out.read_text()
            tail = re.search(r'^-- cursor: (\S+)\s*$',text,re.M)
            entries = [json.loads(line) for line in text.splitlines() if line and not line.startswith('-- cursor: ') and line != '-- No entries --']
            require(tail is not None or (after is not None and not entries), 'initial/nonempty kernel journal cursor missing')
            faults = [e for e in entries if FAULT.search(str(e.get('MESSAGE','')))]
            boundary = dict(label=label,after_cursor=after,end_cursor=tail[1] if tail else after,exit_code=0,
                boot_id=boot(),entry_count=len(entries),fault_filter=FAULT.pattern,fault_count=len(faults),fault_entries=faults,
                stdout_sha256=sha(out),stderr_sha256=sha(err),driver=driver_guard(),cursor_reused=tail is None)
            record['fault_boundaries'].append(boundary)
            persist()
            require(boot() == record['boot_id'] and not faults, 'fresh GPU fault; no reset or continuation')
            return boundary['end_cursor']
        def environment(label,building=False,traced=False):
            env = dict(manifest['build_environment'] if building else manifest['execution_environment'])
            cache,tmp = root/('cache-'+label),root/('tmp-'+label)
            cache.mkdir(); tmp.mkdir()
            env.update(XDG_CACHE_HOME=str(cache),TMPDIR=str(tmp))
            if traced:
                env.update(manifest['first_smoke_trace_environment'])
            # Never mutate os.environ or initialized native/provider state.
            return env
        def gate(path,expected_hash,mode):
            value,proof = receipt_gate(module,path,expected_hash,record['boot_id'],qualification=True)
            require(value.get('schema') == SCHEMA and value.get('mode') == mode and value.get('manifest_sha256') == manifest_sha,
                    'same reviewed controller qualification mode: ' + mode)
            remember(path,expected_hash)
            record.setdefault('gates',{})[mode] = proof
            for item,identity in value.get('dynamic_pins',{}).items():
                pinned(item,identity)
                record.setdefault('dynamic_pins',{})[item] = identity
            return value
        def native_case(label,argv,env,suite=None,shape=None,kind=None,process_index=0,smoke=False):
            nonlocal cursor
            pin_all()
            no_model_owners(manifest['known_model_files'])
            cursor = journal(label+'-before',cursor)
            record['GPU_executed'] = record['native_launch_attempted'] = True
            case = dict(label=label,suite=suite,shape=shape,input_format=kind,process_index=process_index,passed=False,argv=argv,environment=env)
            record['cases'].append(case)
            persist()
            command,out,err = run(label,argv,env)
            # Keep original stdout/stderr and every observed sample even on
            # failed exit/correctness/parse; no failed gate is promoted to PASS.
            stdout,stderr = out.read_text(errors='replace'),err.read_text(errors='replace')
            rows,decode_errors = parse_json_rows(stdout)
            case.update(command_index=record['commands'].index(command),process_owners=command.get('owners'),
                exit_code=command.get('exit_code'),stdout=stdout,stdout_sha256=sha(out),stderr_sha256=sha(err),rows=rows,json_decode_errors=decode_errors)
            for sample in rows:
                if sample.get('kind') == 'sample':
                    record['samples'].append(dict(case=label,process_index=process_index,owners=command.get('owners'),sample=sample,process_passed=False))
            persist()
            require(owner.active is None, 'native ownership still active; no next process')
            cursor = journal(label+'-after',cursor)
            normal(command)
            require(not ERROR.search(stderr) and not decode_errors, 'native/drain/runtime/JSON failure')
            phase = 'after_fixture_return' if suite else 'after_checked_drain'
            case['libraries'] = library_evidence(rows,manifest,phase)
            if suite:
                case['proof'] = fixture_evidence(stdout,rows,suite)
            else:
                case['proof'] = benchmark_evidence(rows,shape,kind,process_index,smoke)
                if record.get('actual_benchmark_device') is None:
                    record['actual_benchmark_device'] = case['proof']['device']
                require(record['actual_benchmark_device'] == case['proof']['device'], 'benchmark device/driver/tuples changed')
                require(case['proof']['plan']['pid'] == command['pid'], 'native PID matches supervised direct identity')
                for path in manifest['required_benchmark_libraries']:
                    require(path in case['libraries']['sha256_pins'], 'oneMKL actual library mapping missing')
            if suite == 'smoke' or smoke:
                require('urEnqueueKernelLaunch' in stderr, 'flushed UR trace lacks kernel submission evidence')
            case['passed'] = True
            for sample in record['samples']:
                if sample['case'] == label:
                    sample['process_passed'] = True
            persist()
        try:
            buildref = load(A/'own-native16-prefill-build-v2.json')
            pin_all(trees=True)
            before_trees = True
            _,record['prior'] = receipt_gate(module,args.prior,args.prior_sha256,record['boot_id'],qualification=False,
                failure_path=args.closed_failure_receipt,failure_hash=args.closed_failure_sha256,reason=args.manual_reviewed_continuation_reason)
            remember(args.prior,args.prior_sha256)
            for name in ('own-native16-prefill-build-v2.json','own-native16-gemm-correctness-v2.json'):
                path = A/name
                _,proof = receipt_gate(module,path,manifest['input_pins'][str(path)]['sha256'],record['boot_id'],qualification=True,
                    historical_build_only=name=='own-native16-prefill-build-v2.json')
                record.setdefault('v2_gates',[]).append(proof)
            old = load(A/'own-native16-gemm-correctness-v2.json')
            require(json.loads(old['checks']['gemm16-first']['stdout'])['numerical_cases'] == 68, 'unchanged qualified 68-case receipt')
            record['owner_scan'] = no_model_owners(manifest['known_model_files'])
            record['device_sysfs'] = driver_guard()
            cursor = journal('journal-fresh-n0')
            preflight = True
            if args.mode == 'build':
                private_build = root/'build'
                env = environment('build',building=True)
                record['build_environment'] = env
                compiler = manifest['compiler']
                configure = ['/usr/bin/cmake','-S',str(PACKAGE),'-B',str(private_build),'-G','Ninja',
                    '-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+compiler,
                    '-DR810_PACKAGE='+manifest['r810_package'],'-DV2_BUILD='+manifest['qualified_build_path'],
                    '-DSTRATA_SOURCE_DIR='+str(P),'-DSYCL_TLA_SOURCE_DIR='+manifest['tla_root'],'-DSTRATA_XE2_AOT_DEVICE=bmg-g21-a0']
                record['configure_argv'] = configure
                for label,argv in [('configure',configure),('compile-link',['/usr/bin/cmake','--build',str(private_build),'--parallel','1','--verbose'])]:
                    cursor = journal(label+'-before',cursor)
                    command,out,err = run(label,argv,env,building=True)
                    require(owner.active is None, 'build owner not closed')
                    cursor = journal(label+'-after',cursor)
                    normal(command)
                command,out,err = run('actual-recipes',['/usr/bin/ninja','-C',str(private_build),'-t','commands','paired_gemm','strata_vllm_xe2_numerics'],env)
                normal(command)
                recipes = out.read_text()
                require(all(token in recipes for token in ['-fsycl-targets=spir64_gen','bmg-g21-a0','-cl-intel-256-GRF-per-thread',
                    'SPV_INTEL_split_barrier','SPV_INTEL_2d_block_io','SPV_INTEL_subgroup_matrix_multiply_accumulate']), 'actual AOT/translator recipe proof')
                link_lines = [shlex.split(line) for line in recipes.splitlines() if '-o paired_gemm ' in line]
                require(len(link_lines) == 1, 'one actual paired final link argv')
                paired_link = link_lines[0]
                require(paired_link[:2] == [':','&&'] and paired_link[-2:] == ['&&',':'], 'single CMake paired link envelope')
                paired_link = paired_link[2:-2]
                require(paired_link[0] == compiler and '-fsycl-targets=spir64_gen' in paired_link, 'approved compiler and AOT paired linker')
                require(not any('runtime.cpp.o' in token or 'numerics_chain.cpp.o' in token or 'fixture_entry.cpp.o' in token for token in paired_link),
                        'paired process cannot contain fixture Runtime/entrypoint objects')
                expected_linked = set(manifest['linked_file_realpaths']) | {str(private_build/'r810/libstrata_vllm_xe2_native16.a')}
                actual_linked = set()
                for token in paired_link:
                    if token.endswith('.a') or (token.startswith('/') and '.so' in Path(token).name):
                        path = Path(token) if Path(token).is_absolute() else private_build/token
                        actual_linked.add(str(path.resolve()))
                require(actual_linked == expected_linked, 'exact original v2 archives/MKL plus R810 component; no rebuilt/extra libraries')
                record['actual_paired_link_argv'] = paired_link
                entries = load(private_build/'compile_commands.json')
                files = [e['file'] for e in entries]
                require(files.count(str(P/'src/sycl/runtime.cpp')) == 1 and files.count(str(PACKAGE/'paired_gemm.cpp')) == 1,
                        'standalone fixture Runtime exactly once; benchmark no duplicate Runtime')
                for entry in entries:
                    argv = entry.get('arguments') or shlex.split(entry['command'])
                    require(argv[0] == compiler and '-fsycl' in argv and '-fno-fast-math' in argv and '-ffp-contract=off' in argv,
                            'actual approved compiler/strict compile flags')
                require(len(entries) == 5, 'only component, original fixture, private fixture Runtime, wrapper and benchmark sources')
                record['actual_compile_entries'] = entries
                record['actual_compile_link_recipes'] = recipes
                # Ninja's existing depfile database is read-only; pins actual
                # consumed header paths. Whole header roots were pinned before
                # configure and must still hash identically after all work.
                command,out,err = run('actual-header-dependencies',['/usr/bin/ninja','-C',str(private_build),'-t','deps'],env)
                normal(command)
                dependencies = sorted(set(line.strip() for line in out.read_text().splitlines() if line.startswith('    ')))
                require(dependencies and len(dependencies) <= 50000, 'finite actual transitive include inventory')
                for name in dependencies:
                    path = Path(name) if Path(name).is_absolute() else private_build/name
                    require(path.is_file(), 'actual dependency absent: ' + str(path))
                    remember(path)
                products = dict(fixture=private_build/'r810/strata_vllm_xe2_numerics',benchmark=private_build/'paired_gemm',
                                component=private_build/'r810/libstrata_vllm_xe2_native16.a')
                record['products'] = {name:dict(path=str(path),**remember(path)) for name,path in products.items()}
                record['private_build'] = str(private_build)
                remember(private_build/'compile_commands.json')
                remember(private_build/'build.ninja')
                record['tool_acceptance'] = 'PASS for this exact recipe; actual ISA remains UNRUN'
                record['actual_ISA'] = 'UNRUN; no symbol-name DPAS inference'
            else:
                built = gate(args.build_record,args.build_record_sha256,'build')
                products = built['products']
                for item in products.values():
                    require(Path(item['path']).is_relative_to(Path(built['private_root'])/'build'), 'product belongs to explicit build root')
                    pinned(item['path'],item)
                record['products'] = products
                record['build_receipt_sha256'] = args.build_record_sha256
                fixture,benchmark = products['fixture']['path'],products['benchmark']['path']
                if args.mode in ('check','measure'):
                    smoked = gate(args.smoke_record,args.smoke_record_sha256,'smoke')
                    require(smoked.get('products') == products and [c.get('suite') for c in smoked['cases']] == ['smoke'] and
                            all(c.get('passed') is True for c in smoked['cases']), 'same binaries and original first smoke PASS')
                if args.mode == 'smoke':
                    native_case('r810-smoke',[fixture,'--device','0','--suite','smoke'],environment('r810-smoke',traced=True),suite='smoke')
                elif args.mode == 'check':
                    for suite in CHECKS:
                        native_case('r810-'+suite,[fixture,'--device','0','--suite',suite],environment('r810-'+suite),suite=suite)
                else:
                    checked = gate(args.check_record,args.check_record_sha256,'check')
                    require(checked.get('products') == products and [c.get('suite') for c in checked['cases']] == CHECKS and
                            all(c.get('passed') is True for c in checked['cases']), 'every required original component suite PASS; no FTZ waiver')
                    require(checked.get('gates',{}).get('smoke',{}).get('sha256') == args.smoke_record_sha256, 'same smoke-to-check chain')
                    # Tiny benchmark trace only after all component suites PASS.
                    # A failure stops here; no full benchmark follows it.
                    for kind in ('fp16','bf16'):
                        label = 'paired-tiny-'+kind
                        native_case(label,[benchmark,'--smoke',kind,'--process','0'],environment(label,traced=True),shape=-1,kind=kind,smoke=True)
                    for process_index in range(3):
                        for shape in range(3):
                            for kind in ('fp16','bf16'):
                                label = 'paired-p'+str(process_index)+'-s'+str(shape)+'-'+kind
                                native_case(label,[benchmark,'--measure',str(shape),kind,'--process',str(process_index)],environment(label),
                                            shape=shape,kind=kind,process_index=process_index)
                    record['independent_processes_per_shape_format'] = 3
                    record['timed_submits_per_route_per_process'] = 3
                    record['benchmark_independent_process_count'] = 18
            pin_all(trees=True)
            cursor = journal('journal-final',cursor)
            workflow_done = True
        except BaseException as exc:
            record['error'] = repr(exc)
        finally:
            if cursor and owner.active is None and (not record['fault_boundaries'] or record['fault_boundaries'][-1]['label'] != 'journal-final'):
                try:
                    cursor = journal('journal-final-after-failure',cursor)
                except BaseException as exc:
                    record['final_fault_check_error'] = repr(exc)
            try:
                pin_all(trees=before_trees)
                record['after_pins_verified'] = True
            except BaseException as exc:
                record['after_pin_error'] = repr(exc)
            record.update(active=owner.active is not None,finished_utc=module.utc(),boot_unchanged=boot() == record['boot_id'])
            record['fresh_faults'] = [fault for boundary in record['fault_boundaries'] for fault in boundary['fault_entries']]
            record['driver_dump_exists'] = (DEVICE/'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            record['ownership_closed'] = not record['active'] and bool(owner.commands) and all(module.closed(c) for c in owner.commands)
            record['normally_closed'] = record['ownership_closed'] and all(c.get('exit_code') == 0 for c in owner.commands)
            record['private_bytes'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            record['complete'] = bool(preflight and record['ownership_closed'] and record['boot_unchanged'] and
                record.get('after_pins_verified') and not record.get('final_fault_check_error') and record['private_bytes'] <= cap)
            record['passed'] = bool(workflow_done and record['complete'] and record['normally_closed'] and not record['fresh_faults'] and not record['driver_dump_exists'])
            record['status'] = 'PASS' if record['passed'] else 'FAIL'
            record['retention'] = dict(owner='Main R816',consumer='Main verifies compact proof and original failed status before serialized retention review',
                text_budget_bytes=TEXT_CAP,total_budget_bytes=cap,cleanup='NONE; retain raw normal traces only until compact proof verified; Main supplies later deletion manifest',
                next_review='this exact build/qualification/measurement acceptance or failure analysis; do not retain duplicate windows without question')
            persist()
        print(json.dumps(dict(passed=record['passed'],complete=record['complete'],status=record['status'],receipt=str(record_path),sha256=sha(record_path))),flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
