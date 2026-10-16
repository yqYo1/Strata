#!/usr/bin/env python3
"""UNRUN source proposal. Only Main may invoke this serialized owner controller.

No configure, archive rebuild, source extraction, environment mutation, profiler,
model launch, foreign signal, reset, service operation, or automatic retry.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
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
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
RUNTIME_ERROR = re.compile(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed')
MODEL = re.compile(r'(^|/)(strata(?:[-_.][^/]*)?|sycl_[^/]*|paired_gemm|bench\.sh|serve\.py|engine[-_]wrapper\.py|run_server_job\.py)$')
ROUTES = ['old_f32_direct', 'old_f32_plus_gpu_bf16_cast', 'native_bf16_output']
ORDER = [ROUTES, [ROUTES[1], ROUTES[2], ROUTES[0]], [ROUTES[2], ROUTES[0], ROUTES[1]]]
CASES = [(0,'fp16'), (0,'bf16'), (1,'fp16'), (1,'bf16'), (2,'fp16'), (2,'bf16')]
SHAPES = [[160,1280,2560], [160,2560,640], [8192,1280,2560]]
TEXT_CAP, JOB_CAP = 64 << 20, 512 << 20


def require(condition, why):
    if not condition:
        raise RuntimeError(why)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def pinned(path, value):
    p = Path(path)
    require(p.is_file() and sha(p) == value['sha256'], 'file pin: ' + str(p))
    require(p.stat().st_size == value['bytes'], 'file size: ' + str(p))
    if 'realpath' in value:
        require(str(p.resolve()) == value['realpath'], 'resolution pin: ' + str(p))


def no_model_owners(model_files):
    """Read-only same-UID scan. Ambiguous/unreadable same-UID owner fails closed.

    This is an additional gate to the shared measurement lock and exact prior
    identities, not proof that uncooperative foreign GPU work cannot exist.
    """
    observed = []
    paths = list(Path('/proc').iterdir())
    require(len(paths) < 131072, 'finite proc population')
    for path in paths:
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            args = [v.decode('utf-8', errors='replace') for v in (path/'cmdline').read_bytes().split(b'\0') if v]
            suspicious = [v for v in args if not v.startswith('-') and MODEL.search(v)]
            # Catch the repository's Python server entry points as well.
            suspicious += [v for v in args if str(P/'serve') in v and v.endswith('.py')]
            suspicious += [v for v in args if v in model_files or v.startswith('serve.') or v.endswith('.gguf')]
            if suspicious:
                observed.append(dict(pid=int(path.name), matching_arguments=suspicious))
        except (FileNotFoundError, ProcessLookupError):
            continue
    require(not observed, 'model/fixture owners present: ' + json.dumps(observed))
    return dict(same_uid_model_owners=[], scan='bounded cmdline and known model path arguments; no signals or adoption')


def closed_prior(module, path, current_boot, *, qualified=False):
    record = load(path)
    require(record.get('complete') and not record.get('active'), 'preceding run not complete/closed: ' + str(path))
    require(record.get('passed'), ('qualification gate failed: ' if qualified else 'preceding owner failed: ') + str(path))
    require(record.get('boot_id') == current_boot, 'preceding receipt boot differs')
    require(not record.get('fresh_faults') and not record.get('driver_dump_exists'), 'preceding GPU fault')
    require(record.get('commands'), 'preceding receipt has no owned commands')
    for command in record['commands']:
        require(module.closed(command) and command.get('exit_code') == 0, 'preceding owned command not normally closed')
        for old in command.get('owners', []):
            now = module.identity(old['pid'])
            require(now is None or now['start_ticks'] != old['start_ticks'], 'prior PID/start identity still present')
    return dict(path=str(Path(path).resolve()), sha256=sha(path), boot_id=record['boot_id'], normally_closed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build','smoke','measure'])
    parser.add_argument('--run-id', required=True, help='fresh label; never retry in a prior root')
    parser.add_argument('--prior', required=True, type=Path, help='Main must name the latest closed owner receipt, never a live run')
    parser.add_argument('--build-record', type=Path, help='this controller build receipt, required for smoke/measure')
    parser.add_argument('--smoke-record', type=Path, help='this controller smoke receipt, required for measure')
    args = parser.parse_args()
    require(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', args.run_id), 'bounded run label')
    require(args.mode == 'build' or args.build_record, 'private build receipt required')
    require(args.mode != 'measure' or args.smoke_record, 'closed first-smoke receipt required')
    manifest = load(PACKAGE/'manifest.json')
    for name, value in manifest['package_files'].items():
        pinned(PACKAGE/name,value)
    require(sum((PACKAGE/name).stat().st_size for name in manifest['package_files']) + (PACKAGE/'manifest.json').stat().st_size < 200 << 10,
            'proposal package budget')
    for path, value in manifest['receipt_pins'].items():
        pinned(path,value)
    owner_path = Path(manifest['direct_owner']['path'])
    pinned(owner_path,manifest['direct_owner'])
    spec = importlib.util.spec_from_file_location('r803_pinned_direct_owner',owner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = B/('main-r803-paired-gemm-'+args.run_id+'-'+args.mode)
    require(not root.exists(), 'fresh private root; do not overwrite any result')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Only external private outputs are created. Never write source or the
        # qualified CMake build, even while linking against its fixed archives.
        root.mkdir(mode=0o700)
        record = dict(schema='strata-r803-paired-gemm-v1',mode=args.mode,run_id=args.run_id,
            passed=False,complete=False,active=False,GPU_executed=False,commands=[],
            boot_id=boot(),manifest_sha256=sha(PACKAGE/'manifest.json'),
            base_commit=manifest['base_commit'],source_only_delivery='UNRUN',
            source_pins={},external_pins=manifest['external_pins'],receipt_pins=manifest['receipt_pins'],
            package_files=manifest['package_files'],direct_owner_sha256=manifest['direct_owner']['sha256'],
            limits=dict(check_wall_seconds=180,build_wall_seconds=600,check_rss_bytes=8<<30,
                        build_rss_bytes=32<<30,text_bytes=TEXT_CAP,job_bytes=JOB_CAP),
            shapes=SHAPES,case_order=CASES,route_order=ORDER,warmup_order=ROUTES,
            beta=0,scope='operator sanity and submit-to-completion wall only',
            model_quality_qualified=False,model_speed_qualified=False,physical_262144_qualified=False,
            adversarial_accumulator_gate_replaced=False,samples=[],fault_boundaries=[])
        def persist():
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(root/'owned',persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        build = Path(manifest['qualified_build_path'])
        buildref = load(A/'own-native16-prefill-build-v2.json')
        build_deadline = time.monotonic()+600 if args.mode=='build' else None
        def run(label,argv,env,*,building=False,native=False):
            wall = 600 if building else 180
            if build_deadline is not None:
                wall = min(wall,build_deadline-time.monotonic())
                require(wall>0,'whole build deadline exhausted')
            command,out,err = owner.run(label,list(map(str,argv)),env,build,
                wall=wall,cpu=18000 if building else 1800,
                rss_cap=(32 if building else 8)<<30,text_cap=TEXT_CAP,total_cap=JOB_CAP,file_cap=TEXT_CAP)
            if native:
                # Preserve flushed individual samples/statuses even when the
                # process fails, is killed by its Owner, or leaves a partial line.
                rows,decode_errors = [],[]
                if out.exists():
                    for number,line in enumerate(out.read_text(errors='replace').splitlines(),1):
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except (ValueError,TypeError) as exc:
                            decode_errors.append(dict(line=number,error=repr(exc)))
                record.setdefault('case_rows',{})[label]=rows
                record.setdefault('case_decode_errors',{})[label]=decode_errors
                validations = {v.get('ordinal'):v for v in rows if isinstance(v,dict) and v.get('kind')=='validation'}
                for value in rows:
                    if isinstance(value,dict) and value.get('kind')=='sample':
                        record.setdefault('observed_samples',[]).append(dict(case=label,measurement=value,
                            validation=validations.get(value.get('ordinal')),process_exit_code=command.get('exit_code')))
                persist()
            require(module.closed(command) and command.get('exit_code') == 0,
                    label+': owned process did not close with exit 0; inspect retained command/logs')
            return out,err
        def pin_everything():
            require(boot()==record['boot_id']==buildref['boot_id'], 'boot changed or v2 requalification needed')
            pinned(owner_path,manifest['direct_owner'])
            for path,value in manifest['receipt_pins'].items():
                pinned(path,value)
            for path,value in manifest['external_pins'].items():
                pinned(path,value)
            for name,value in buildref['source_pins'].items():
                require(sha(P/name)==value,'qualified v2 source changed: '+name)
            for name,value in manifest['immutable_source_reads'].items():
                require(sha(P/name)==value['sha256'],'31ad125b dependency changed: '+name)
            for name,value in manifest['package_files'].items():
                pinned(PACKAGE/name,value)
            require(sha(PACKAGE/'manifest.json')==record['manifest_sha256'],'manifest changed')
            record['source_pins'] = buildref['source_pins']
        def no_dump():
            require((DEVICE/'driver').resolve().name=='xe','expected xe driver')
            require((DEVICE/'vendor').read_text().strip()=='0x8086','expected Intel PCI device')
            dumps = list(Path('/sys/class/devcoredump').glob('devcd*'))
            require(not (DEVICE/'devcoredump').exists() and not dumps,'driver dump present; no removal permitted')
            return dict(pci=str(DEVICE),driver='xe',vendor='0x8086',devcoredump=False,
                        device=(DEVICE/'device').read_text().strip(),kernel=Path('/proc/sys/kernel/osrelease').read_text().strip())
        def journal(label,cursor=None):
            argv = ['journalctl','-k','-b','--no-pager','-o','json','--show-cursor']
            if cursor:
                argv += ['--after-cursor',cursor]
            else:
                # Current normal owner closure and present dump checks are the
                # preflight. Bound the new window without replaying days of
                # unrelated historical kernel messages into a new result.
                argv += ['-n','0']
            out,err = run(label,argv,hostenv)
            text = out.read_text()
            tail = re.search(r'^-- cursor: (\S+)\s*$',text,re.M)
            entries = [json.loads(line) for line in text.splitlines() if line and
                       not line.startswith('-- cursor: ') and line!='-- No entries --']
            # A healthy incremental window can contain no new kernel entries;
            # journalctl may then omit a new cursor. Retain the already checked
            # cursor only for an empty, exit-zero incremental read in this boot.
            require(tail is not None or (cursor is not None and not entries),
                    'nonempty/initial journal command lacks checked cursor')
            end_cursor = tail[1] if tail else cursor
            faults = [v for v in entries if FAULT.search(str(v.get('MESSAGE','')))]
            boundary = dict(label=label,after_cursor=cursor,end_cursor=end_cursor,check_exit_code=0,
                empty_window=not entries,cursor_reused=tail is None,
                fault_filter=FAULT.pattern,entry_count=len(entries),fault_count=len(faults),fault_entries=faults,
                stdout_sha256=sha(out),stderr_sha256=sha(err),boot_id=boot(),driver=no_dump())
            record['fault_boundaries'].append(boundary)
            persist()
            require(boot()==record['boot_id'],'journal boot changed')
            require(not faults,'preexisting or fresh GPU fault; stop without resets/dump cleanup')
            return end_cursor
        def environment(label,traced=False,building=False):
            env = dict(manifest['build_environment'] if building else manifest['measurement_environment'])
            cache,tmp = root/('cache-'+label),root/('tmp-'+label)
            cache.mkdir(); tmp.mkdir()
            env.update(XDG_CACHE_HOME=str(cache),TMPDIR=str(tmp))
            if traced:
                env.update(manifest['first_smoke_trace_environment'])
            # These dictionaries are complete pre-exec environments. Do not
            # modify os.environ, and never change an executing child's values.
            return env
        cursor = None
        try:
            pin_everything()
            record['prior'] = closed_prior(module,args.prior,record['boot_id'])
            for name in ('own-native16-prefill-build-v2.json','own-native16-gemm-correctness-v2.json'):
                record.setdefault('qualification_gates',[]).append(closed_prior(module,A/name,record['boot_id'],qualified=True))
            gate = load(A/'own-native16-gemm-correctness-v2.json')
            require(json.loads(gate['checks']['gemm16-first']['stdout'])['numerical_cases']==68,'original 68-case gate required')
            record['owner_scan'] = no_model_owners(manifest['known_model_files'])
            record['device_sysfs'] = no_dump()
            cursor = journal('journal-before') # Fresh cursor; current owner/device guards above.
            if args.mode == 'build':
                entries = load(build/'compile_commands.json')
                fixture_source = str(P/'tests/sycl/gemm16.cpp')
                hits = [v for v in entries if v['file']==fixture_source]
                require(len(hits)==1,'one actual v2 fixture compile entry')
                entry = hits[0]
                require(Path(entry['directory'])==build,'qualified compilation cwd')
                argv = entry.get('arguments') or shlex.split(entry['command'])
                require(argv[0]==manifest['compiler'] and '-fsycl' in argv and '-fno-fast-math' in argv and '-ffp-contract=off' in argv,'qualified strict SYCL flags')
                require(argv.count('-o')==argv.count('-c')==1,'single compile object/source')
                require(argv[argv.index('-c')+1]==fixture_source,'compile entrypoint')
                old_object = argv[argv.index('-o')+1]
                obj,exe = root/'paired_gemm.o',root/'paired_gemm'
                argv[argv.index('-c')+1]=str(PACKAGE/'paired_gemm.cpp')
                argv[argv.index('-o')+1]=str(obj)
                record['original_compile_entry']=entry
                record['actual_compile_argv']=argv
                env = environment('build',building=True)
                record['build_environment']=env
                persist()
                run('compile-private-fixture',argv,env,building=True)
                # ninja -t commands prints the configured recipe; it does not
                # execute it, refresh CMake, build targets, or mutate archives.
                out,_ = run('read-actual-link-recipe',['ninja','-C',str(build),'-t','commands','sycl_gemm16_test'],env)
                lines = out.read_text().splitlines()
                matches = [shlex.split(line) for line in lines if old_object in line and '-o sycl_gemm16_test ' in line]
                require(len(matches)==1,'one configured v2 final link command')
                tokens = matches[0]
                require(tokens[:2]==[':','&&'] and tokens[-2:]==['&&',':'],'qualified CMake shell envelope')
                link = tokens[2:-2]
                require(link[0]==manifest['compiler'] and link.count(old_object)==1 and link.count('-o')==1,'qualified fixture-only linker')
                require([v for v in link if v.endswith('.o')]==[old_object],'no extra/rebuilt source objects')
                require(not any(v in ('&&',';','|','>','<') for v in link),'link must be a single argv')
                for item in link:
                    if item.endswith('.a') or (item.startswith('/') and '.so' in Path(item).name):
                        path = str((build/item).resolve() if not Path(item).is_absolute() else Path(item).resolve())
                        require(path in manifest['linked_file_realpaths'],'unqualified linked file: '+path)
                record['original_link_argv']=list(link)
                link[link.index(old_object)]=str(obj)
                link[link.index('-o')+1]=str(exe)
                record['actual_link_argv']=link
                persist()
                run('link-private-fixture',link,env,building=True)
                record.update(binary=str(exe),binary_sha256=sha(exe),object_sha256=sha(obj))
            else:
                built = load(args.build_record)
                require(built['mode']=='build' and built['manifest_sha256']==record['manifest_sha256'],'matching private build proposal')
                record['private_build_gate']=closed_prior(module,args.build_record,record['boot_id'],qualified=True)
                exe = Path(built['binary'])
                require(exe.parent==args.build_record.resolve().parent,'private executable owned build root')
                require(sha(exe)==built['binary_sha256'],'private executable pin')
                record.update(binary=str(exe),binary_sha256=sha(exe))
                if args.mode=='measure':
                    smoke = load(args.smoke_record)
                    require(smoke['mode']=='smoke' and smoke['manifest_sha256']==record['manifest_sha256'] and
                            smoke['binary_sha256']==built['binary_sha256'],'same-source/binary first smoke')
                    record['first_smoke_gate']=closed_prior(module,args.smoke_record,record['boot_id'],qualified=True)
                    require(smoke['completed_cases']==[[-1,'fp16'],[-1,'bf16']],'both small input smoke processes closed')
                    expected_device = smoke['actual_device']
                else:
                    expected_device = None
                cases = [(-1,'fp16'),(-1,'bf16')] if args.mode=='smoke' else CASES
                for shape,kind in cases:
                    label = ('smoke' if shape<0 else 'shape'+str(shape))+'-'+kind
                    pin_everything()
                    no_model_owners(manifest['known_model_files'])
                    cursor = journal(label+'-before',cursor)
                    env = environment(label,traced=shape<0)
                    argv = [str(exe),'--smoke',kind] if shape<0 else [str(exe),'--measure',str(shape),kind]
                    record['GPU_executed']=True
                    record.setdefault('case_environments',{})[label]=env
                    persist()
                    # Owner fully reaps this native process before any journal
                    # follow-up or next process, including small -> large.
                    out,err = run(label,argv,env,native=True)
                    cursor = journal(label+'-after',cursor)
                    require(not RUNTIME_ERROR.search(err.read_text()),'native/runtime failure in stderr')
                    require(not record['case_decode_errors'][label],'malformed/partial native JSON output')
                    rows = record['case_rows'][label]
                    require(all(isinstance(v,dict) for v in rows),'native JSON rows must be objects')
                    device = [v for v in rows if v.get('kind')=='device']
                    require(len(device)==1 and device[0]['vendor_id']==0x8086 and device[0]['matrix_tuples'],'actual vendor/device/tuple record')
                    if expected_device is None:
                        expected_device=device[0]
                    require(device[0]==expected_device,'device/driver/platform/actual tuple changed')
                    record['actual_device']=expected_device
                    plans = [v for v in rows if v.get('kind')=='plan']
                    require(len(plans)==1,'one finite plan')
                    plan = plans[0]
                    require(plan['shape']==shape and plan['input_type']==kind and plan['beta']==0 and
                            plan['sample_order']==ORDER and plan['warmup_order']==ROUTES,'predeclared plan/order')
                    dims = [17,48,48] if shape<0 else SHAPES[shape]
                    require([plan['T'],plan['N'],plan['K']]==dims and plan['ldy']==dims[1],'actual bounded geometry')
                    samples = [v for v in rows if v.get('kind')=='sample']
                    validations = [v for v in rows if v.get('kind')=='validation']
                    expected = ROUTES if shape<0 else ROUTES + sum(ORDER,[])
                    require(len(samples)==len(validations)==len(expected),'all warmup/sample validations present')
                    for i,(sample,validation,route) in enumerate(zip(samples,validations,expected)):
                        require(sample['ordinal']==validation['ordinal']==i and sample['route']==validation['route']==route,'exact interleaved ordinal')
                        require(sample['warmup']==(i<3) and sample['sample']==(-1 if i<3 else (i-3)//3),'warmup/sample identity')
                        require(sample['stored_output_type']==('fp32' if route==ROUTES[0] else 'bf16'),'exact stored output type')
                        require(sample['global_fp32_intermediate']==(route==ROUTES[1]),'native has no global FP32 intermediate')
                        require(validation['status']=='PASS' and validation['outputs_checked']==dims[0]*dims[1] and
                            all(validation[k] for k in ('all_finite','closed_form_oracle','guards','inputs_immutable')),'full output sanity')
                        record['samples'].append(dict(case=label,measurement=sample,validation=validation))
                    result = [v for v in rows if v.get('kind')=='result']
                    require(len(result)==1 and result[0]['status']=='PASS' and result[0]['validated_invocations']==len(expected)
                            and result[0]['samples_per_route']==(0 if shape<0 else 3),'fixture normal result')
                    if shape<0:
                        require('urEnqueueKernelLaunch' in err.read_text(),'first trace missing GPU kernel submission evidence')
                    record.setdefault('completed_cases',[]).append([shape,kind])
                    persist()
                    require(sha(exe)==record['binary_sha256'],'private executable changed')
            pin_everything()
            cursor = journal('journal-final',cursor)
            record.update(passed=True,complete=True)
        except BaseException as exc:
            record['error']=repr(exc)
        finally:
            # Best-effort read-only fault boundary even after a failed native
            # command. Owner alone handles its own pinned processes and signals.
            if cursor and owner.active is None and record.get('fault_boundaries',[])[-1]['label']!='journal-final':
                try:
                    journal('journal-final-after-failure',cursor)
                except BaseException as exc:
                    record['final_fault_check_error']=repr(exc)
            record.update(active=owner.active is not None,finished_utc=module.utc())
            record['fresh_faults']=[v for boundary in record['fault_boundaries'] for v in boundary['fault_entries']]
            record['boot_unchanged']=boot()==record['boot_id']
            record['driver_dump_exists']=(DEVICE/'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            if (record['active'] or not all(module.closed(c) and c.get('exit_code')==0 for c in owner.commands) or
                    not record['boot_unchanged'] or record['driver_dump_exists'] or record['fresh_faults'] or
                    record.get('final_fault_check_error') or not record['fault_boundaries'] or
                    sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>JOB_CAP):
                record.update(passed=False,complete=False)
            persist()
        print(json.dumps(dict(passed=record['passed'],complete=record['complete'],active=record['active'],
            GPU_executed=record['GPU_executed'],receipt=str(root/'record.json'))),flush=True)
        return 0 if record['passed'] and record['complete'] else 1


if __name__=='__main__':
    sys.exit(main())
