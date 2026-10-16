#!/usr/bin/env python3
"""Main-only integrated-image component gates. No model or speed adoption."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
META_SHA = '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['smoke', 'check'])
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--build-sha256', required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--prior-sha256', required=True)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    package = A/'own-vllm-xe2-qualification-v1'
    meta_path = package/'manifest.json'
    if sha(meta_path) != META_SHA:
        raise RuntimeError('reviewed manifest drift')
    meta = json.loads(meta_path.read_text())
    helper = package/'main_controller.py'
    owner_path = Path(meta['direct_owner']['path'])
    if sha(helper) != meta['package_files']['main_controller.py']['sha256'] or sha(owner_path) != meta['direct_owner']['sha256']:
        raise RuntimeError('reviewed helper drift')
    h = imported('integrated_components_helpers', helper)
    o = imported('integrated_components_owner', owner_path)
    need = h.require
    root = args.root.absolute()
    need(root.parent == B and re.fullmatch('main-vllm-engine-components-'+args.mode+'-[A-Za-z0-9_-]{1,32}', root.name), 'root scope')
    need(not root.exists() and not root.is_symlink(), 'fresh root only')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        record = dict(schema='strata-vllm-engine-components-v1', mode=args.mode,
            passed=False, complete=False, active=False, boot_id=h.boot(), commands=[],
            cases=[], fresh_faults=[], fault_boundaries=[], GPU_executed=False,
            controller_sha256=sha(__file__), invocation=sys.argv,
            scope='actual integrated mixed AOT/JIT images; independent native fixture plus existing own16 and same-queue FP32 regression. Public vLLM Gemm engine hook awaits model gate.',
            original_separate_queue_regression='UNRUN here; original unresolved timeout remains FAIL',
            model_quality_qualified=False, physical_262144_qualified=False, performance_adopted=False,
            retention=dict(owner='Main', consumer='integrated model and mixed-image qualification',
                byte_budget=512<<20, review='at paired component/model closure; keep compact checks and one needed image trace only'))
        path = root/'record.json'
        def save():
            record['active'] = owner.active is not None
            path.write_text(json.dumps(record, indent=2)+'\n')
        owner = o.Owner(root/'owned', save)
        record['commands'] = owner.commands
        save()
        deadline = time.monotonic()+1800
        base = dict(meta['measurement_environment'])
        record['base_environment'] = base
        cursor = None
        build = None
        def verify():
            need(sha(__file__)==record['controller_sha256'] and h.boot()==record['boot_id'], 'controller/boot drift')
            need(sha(args.build)==args.build_sha256 and sha(args.prior)==args.prior_sha256, 'receipt drift')
            if build:
                for name, value in build['source_pins'].items():
                    need(sha(P/name)==value, 'source drift: '+name)
                for name, value in build['pins'].items():
                    h.pinned(name, value)
                for product in build['products'].values():
                    h.pinned(product['path'], product)
        def run(label, argv, env, wall=300):
            verify()
            remaining = deadline-time.monotonic()
            need(remaining>0, 'component whole deadline')
            c,out,err = owner.run(label, list(map(str,argv)), env, P,
                wall=min(wall,remaining), cpu=1200, rss_cap=12<<30,
                text_cap=64<<20, total_cap=512<<20, file_cap=64<<20)
            verify()
            need(o.closed(c) and c.get('exit_code')==0,
                 label+': '+out.read_text(errors='replace')[-2500:]+err.read_text(errors='replace')[-2500:])
            return c,out,err
        def journal(label):
            nonlocal cursor
            argv=['/usr/bin/journalctl','-k','-b','--no-pager','-o','json','--show-cursor']
            argv+=['--after-cursor',cursor] if cursor else ['-n','0']
            _,out,_=run(label,argv,base,60)
            text=out.read_text()
            match=re.search(r'^-- cursor: (\S+)\s*$',text,re.M)
            rows=[json.loads(v) for v in text.splitlines() if v and not v.startswith('-- cursor: ') and v!='-- No entries --']
            need(match or (cursor and not rows), 'journal cursor')
            faults=[v for v in rows if h.FAULT.search(str(v.get('MESSAGE','')))]
            record['fresh_faults'].extend(faults)
            record['fault_boundaries'].append(dict(label=label,after=cursor,end=match[1] if match else cursor,entries=len(rows),faults=faults))
            cursor=match[1] if match else cursor
            h.driver_guard()
            need(not faults,'fresh GPU fault')
            save()
        def case(label, target, argv, expected, *, traced=False):
            h.no_model_owners(meta['known_model_files'])
            env=dict(base, STRATA_SYCL_PREFILL_NATIVE16_BACKEND='own')
            cache=root/('cache-'+label); tmp=root/('tmp-'+label)
            cache.mkdir(); tmp.mkdir()
            env.update(XDG_CACHE_HOME=str(cache),TMPDIR=str(tmp))
            if traced:
                env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                    UR_LOG_TRACING='level:info;flush:info;output:stderr')
            journal(label+'-before')
            record['GPU_executed']=True
            c,out,err=run(label,[build['products'][target]['path'],*argv],env)
            text=out.read_text(); errors=err.read_text()
            need(not h.ERROR.search(text+'\n'+errors),'runtime/fixture failure')
            need(expected in text,'fixture explicit completion absent: '+label)
            if traced:
                need('urEnqueueKernelLaunch' in errors,'flushed launch trace absent')
            record['cases'].append(dict(label=label,target=target,environment=env,
                stdout_original=text,stderr_sha256=sha(err),stdout_sha256=sha(out),
                owned_command_label=c['label'],passed=True))
            save()
            journal(label+'-after')
        try:
            build,record['build_gate']=h.receipt_gate(o,args.build,args.build_sha256,record['boot_id'],qualification=True)
            need(build.get('schema')=='strata-vllm-prefill-build-v1','integrated engine build gate')
            prior,record['prior']=h.receipt_gate(o,args.prior,args.prior_sha256,record['boot_id'],qualification=True)
            if args.mode=='check':
                need(prior.get('schema')==record['schema'] and prior.get('mode')=='smoke' and
                     prior['build_gate']['sha256']==args.build_sha256,'same integrated first smoke gate')
            else:
                need(args.prior.resolve()==args.build.resolve(),'first smoke immediately follows integrated build')
            record['source_commit']=build['source_commit']
            record['source_pins']=build['source_pins']
            record['products']=build['products']
            h.driver_guard(); verify(); journal('journal-before')
            if args.mode=='smoke':
                case('vllm-integrated-smoke','sycl_vllm_native16_test',['--device','0','--suite','smoke'],'COMPONENT_SUITE_PASS',traced=True)
                case('qsa-local-smoke','sycl_native16_qsa_test',[],'QSA16 local component screen PASS',traced=True)
                case('attention-smoke','sycl_prompt_native16_test',['--smoke'],'fixture=PASS',traced=True)
            else:
                for suite in ['guards','edges','chain','full']:
                    case('vllm-'+suite,'sycl_vllm_native16_test',['--device','0','--suite',suite],
                         'guards PASS rejected=' if suite=='guards' else 'COMPONENT_SUITE_PASS')
                case('own16-original68','sycl_gemm16_test',['--smoke'],'"numerical_cases":68')
                case('native16-ops','sycl_native16_prefill_test',['--smoke'],'native16_prefill PASS')
                case('attention-full','sycl_prompt_native16_test',[],'fixture=PASS')
                for label,target,expected,argv in [
                    ('runtime','sycl_runtime_test','bounded completion wait and sticky asynchronous errors: PASS',[]),
                    ('compat','sycl_compat_test','and lifecycle passed',[]),
                    ('gemm-fp32-same-queue','sycl_gemm_test','SAME_QUEUE_SUBSUITE_PASS',['--same-queue-only']),
                    ('allocation-teardown-late','sycl_allocation_teardown_test','allocation_teardown late main complete',[]),
                    ('allocation-teardown-early','sycl_allocation_teardown_test','allocation_teardown early main complete',['--early'])]:
                    case(label,target,argv,expected)
            journal('journal-final'); verify(); record['passed']=True
        except BaseException as exc:
            record['error']=repr(exc)
        finally:
            if cursor and owner.active is None and record['fault_boundaries'][-1]['label']!='journal-final':
                try:
                    journal('journal-after-failure')
                except BaseException as exc:
                    record['final_fault_check_error']=repr(exc)
            record['ownership_closed']=bool(owner.commands) and owner.active is None and all(o.closed(c) for c in owner.commands)
            record['driver_dump_exists']=(h.DEVICE/'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            record['complete']=record['ownership_closed'] and h.boot()==record['boot_id'] and not record.get('final_fault_check_error')
            record['passed'] &= record['complete'] and not record['fresh_faults'] and not record['driver_dump_exists']
            record['finished_utc']=o.utc(); save()
        print(json.dumps(dict(passed=record['passed'],complete=record['complete'],receipt=str(path),sha256=sha(path))),flush=True)
        return 0 if record['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
