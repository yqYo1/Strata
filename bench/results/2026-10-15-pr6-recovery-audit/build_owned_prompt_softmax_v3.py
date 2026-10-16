#!/usr/bin/env python3
"""Main-owned R861 post-link continuation after a reference-inventory lookup error."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--prior-sha256', required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    helper = A/'own-vllm-xe2-qualification-v1/main_controller.py'
    meta_path = helper.parent/'manifest.json'
    meta = json.loads(meta_path.read_text())
    ownerpath = Path(meta['direct_owner']['path'])
    import hashlib
    def sha(p):
        with Path(p).open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    if sha(meta_path) != '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58':
        raise RuntimeError('reviewed source manifest drift')
    for path, expected in [(helper, meta['package_files']['main_controller.py']['sha256']),
                           (ownerpath, meta['direct_owner']['sha256'])]:
        if sha(path) != expected:
            raise RuntimeError('reviewed helper drift')
    def imported(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    h = imported('vllm_build_helpers', helper)
    o = imported('vllm_build_owner', ownerpath)
    need = h.require
    root = args.root.absolute()
    need(root.parent == B and re.fullmatch(r'main-prompt-softmax-build-[A-Za-z0-9_-]{1,48}', root.name), 'private root scope')
    need(not root.exists() and not root.is_symlink(), 'fresh build root')
    need(re.fullmatch('[0-9a-f]{40}', args.source_commit), 'exact source commit')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        record = dict(schema='strata-prompt-softmax-build-v1', source_commit=args.source_commit,
            boot_id=h.boot(), passed=False, complete=False, active=False, GPU_executed=False,
            offline_aot_may_query_device=True, commands=[], source_pins={}, pins={}, fresh_faults=[],
            fault_windows=[], recipe_sha256=sha(__file__), invocation=sys.argv,
            build_path=str(root/'build'), component_qualification=False, model_quality_qualified=False,
            physical_262144_qualified=False, performance_adopted=False,
            build_scope='R861 cooperative subgroup softmax opt-in; existing false path preserved. Fresh complete generic engine plus isolated vLLM AOT object; no baseline build product mutation or default adoption',
            retention=dict(owner='Main',consumer='enabled engine build/component/model qualification',
                byte_budget=8<<30,review='at this build/test closure; keep canonical binaries and compact source/command evidence, retire duplicate successful full logs'))
        receipt = root/'record.json'
        def save():
            receipt.write_text(json.dumps(record, indent=2)+'\n')
        owner = o.Owner(root/'owned', save)
        record['commands'] = owner.commands
        save()
        deadline = time.monotonic()+3600
        env = dict(meta['build_environment'])
        env.update(XDG_CACHE_HOME=str(root/'cache'), TMPDIR=str(root/'tmp'))
        (root/'cache').mkdir(); (root/'tmp').mkdir()
        record['environment'] = env
        trees = {}
        def pin(path):
            path = Path(path).resolve()
            record['pins'][str(path)] = h.pin(path)
        def verify():
            need(h.boot()==record['boot_id'] and sha(__file__)==record['recipe_sha256'], 'boot/build-controller drift')
            for name, value in record['source_pins'].items():
                need(sha(P/name)==value, 'source drift: '+name)
            for name, value in record['pins'].items():
                h.pinned(name, value)
        def run(label, argv, *, build=False):
            verify()
            need(time.monotonic()<deadline, 'whole build deadline')
            c,out,err = owner.run(label, list(map(str,argv)), env, P,
                wall=min(2400 if build else 300, deadline-time.monotonic()), cpu=18000,
                rss_cap=(40 if build else 8)<<30, text_cap=64<<20, total_cap=8<<30, file_cap=2<<30)
            verify()
            need(o.closed(c) and c.get('exit_code')==0, label+': '+out.read_text(errors='replace')[-4000:]+err.read_text(errors='replace')[-4000:])
            return out,err
        def journal(label, cursor=None):
            argv=['/usr/bin/journalctl','-k','-b','--no-pager','-o','json','--show-cursor']
            argv+=['--after-cursor',cursor] if cursor else ['-n','0']
            out,_=run(label,argv)
            text=out.read_text(); match=re.search(r'^-- cursor: (\S+)\s*$',text,re.M)
            rows=[json.loads(v) for v in text.splitlines() if v and not v.startswith('-- cursor: ') and v!='-- No entries --']
            need(match or (cursor and not rows), 'journal cursor absent')
            faults=[v for v in rows if h.FAULT.search(str(v.get('MESSAGE','')))]
            record['fresh_faults'].extend(faults)
            record['fault_windows'].append(dict(label=label,after=cursor,end=match[1] if match else cursor,entries=len(rows),faults=faults))
            h.driver_guard(); need(not faults,'fresh GPU fault')
            return match[1] if match else cursor
        cursor=None
        try:
            _,record['prior']=h.receipt_gate(o,args.prior,args.prior_sha256,record['boot_id'],qualification=True)
            failure=A/'own-prompt-softmax-build-v2.json'
            failure_sha='b341891789c5a1a4e7ea70cad28bdaf354c7005a81a3eef622928a245c3859e8'
            previous,record['original_build_failure']=h.receipt_gate(o,failure,failure_sha,record['boot_id'],
                qualification=False,failure_path=failure,failure_hash=failure_sha,
                reason='All four candidate targets linked successfully and all commands exited0; Main controller then looked for original archives in the ELF inventory rather than the already qualified public-hook linked_file_pins. Fix only that inventory reference. No product, flags, oracle or tolerance changes; original failed receipt stays unchanged.')
            need(previous['GPU_executed'] is False and previous['ownership_closed'] and
                 all(c.get('exit_code')==0 for c in previous['commands']) and
                 previous.get('error','').startswith('KeyError(') and 'libstrata_kernels.a' in previous['error'] and
                 set(previous.get('products',{}))=={'strata','sycl_prompt_native16_test','sycl_prompt_k_cellmajor_test','sycl_prompt_attention_test'},
                 'only closed post-link inventory lookup failure may reuse outputs')
            for product in previous['products'].values():h.pinned(product['path'],product)
            pin(failure)
            record['build_path']=previous['build_path']
            record['incremental_reuse']=dict(path=record['build_path'],original_receipt_sha256=failure_sha,
                scope='Closed unqualified R861 experiment build only; baseline 6354 products never modified')

            pin(args.prior); pin(helper); pin(meta_path); pin(ownerpath)
            for group in ['input_pins','external_pins','loaded_library_pins']:
                for name,value in meta[group].items():
                    h.pinned(name,value); record['pins'][name]=value
            m810=json.loads((Path(meta['r810_package'])/'manifest.json').read_text())
            for value in m810['source_pins']['tla_headers']:
                path=Path(meta['tla_root'])/value['path']
                need(sha(path)==value['sha256'] and path.stat().st_size==value['bytes'],'TLA header drift')
                pin(path)
            record['owner_scan']=h.no_model_owners(meta['known_model_files'])
            record['driver_before']=h.driver_guard()
            commit=subprocess.check_output(['/usr/bin/git','rev-parse','HEAD'],cwd=P,text=True).strip()
            need(commit==args.source_commit,'Main must commit exact integrated source before build')
            names=subprocess.check_output(['/usr/bin/git','ls-tree','-r','--name-only',commit,'--',
                'src','include','cmake','CMakeLists.txt','tests','bench/sycl'],cwd=P,text=True).splitlines()
            for name in names:
                value=sha(P/name)
                need(hashlib.sha256(subprocess.check_output(['/usr/bin/git','show',commit+':'+name],cwd=P)).hexdigest()==value,
                     'uncommitted build source: '+name)
                record['source_pins'][name]=value
            for name, expected in meta['header_trees'].items():
                actual=h.tree_pin(name)
                if Path(name)!=P/'include':
                    need(actual==expected,'SDK header tree drift: '+name)
                trees[name]=actual
            record['header_trees']=trees
            cursor=journal('journal-before')
            reference=json.loads((A/'own-native16-prefill-build-v2.json').read_text())
            options=list(reference['configure_argv'])
            options[options.index('-B')+1]=record['build_path']
            options=[v for v in options if not v.startswith('-DSTRATA_SYCL_DEVICE_ARCH:')]
            options+=['-DSTRATA_SYCL_VLLM_XE2_NATIVE16=ON','-DSTRATA_SYCL_DEVICE_ARCH=',
                '-DSYCL_TLA_SOURCE_DIR='+meta['tla_root']]
            options[0]='/usr/bin/cmake'
            record['configure_argv']=options
            run('configure',options)
            cursor=journal('journal-after-configure',cursor)
            targets=['strata','sycl_prompt_native16_test','sycl_prompt_k_cellmajor_test','sycl_prompt_attention_test']
            run('compile-link',['/usr/bin/cmake','--build',record['build_path'],'--target',*targets,'--parallel','2','--verbose'],build=True)
            cursor=journal('journal-after-build',cursor)
            build=Path(record['build_path'])
            record['products']={name:dict(path=str(build/name),**h.pin(build/name)) for name in targets}
            # Build the original FP32 fixture against the frozen ORIGINAL
            # archives, in this fresh root. Never change a qualified build tree.
            frozen_path=A/'own-vllm-unitrace-direct-v6/manifest.json'
            need(sha(frozen_path)=='1e9fa3cb57e76405deda315dbdc91b8d43cbf0a98c1d62172661fa5d31da47c0',
                 'frozen baseline dependency inventory')
            frozen=json.loads(frozen_path.read_text()); pin(frozen_path)
            original=B/'main-vllm-prefill-build-v1/build'
            libraries=[original/n for n in ('libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a')]
            library_inventory=A/'own-vllm-public-qualification-v3/manifest.json'
            need(sha(library_inventory)=='39ed2fe7da8812ad021f6ae39261459810be634acafc52f0a6f04551e9008622',
                 'already-qualified current baseline archive inventory')
            library_pins=json.loads(library_inventory.read_text())['linked_file_pins']; pin(library_inventory)
            for library in libraries:
                h.pinned(library,library_pins[str(library)]); pin(library)
            reference_source=root/'reference_prompt_k_cellmajor.cpp'
            reference_source.write_bytes(subprocess.check_output(['/usr/bin/git','show',
                '6354f0b4ff3731466795410e11ca9ef785c9d326:tests/sycl/prompt_k_cellmajor.cpp'],cwd=P))
            need(sha(reference_source)=='8779469bb00ced55c5dad5aa27f6b8485f1e5a7e07eddb502b29bd602c79bb5d',
                 'exact original reference fixture source')
            pin(reference_source)
            reference_exe=root/'reference_prompt_k_cellmajor_test'
            run('compile-original-reference-fixture',[meta['compiler'],'-std=c++20','-O3','-DNDEBUG',
                '-DSTRATA_ENABLE_SYCL=1','-DSTRATA_TEST_PROMPT_MATRIX=1',
                '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),
                '-fno-fast-math','-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t',
                '-Wall','-Wextra','-Werror',reference_source,*libraries,'-o',reference_exe],build=True)
            record['baseline_reference_products']={
                'fp32':dict(path=str(reference_exe),**h.pin(reference_exe)),
                'bf16':dict(path=str(original/'sycl_prompt_native16_test'),**h.pin(original/'sycl_prompt_native16_test'))}
            need(record['baseline_reference_products']['bf16']['sha256']==frozen['tiny']['sha256'],
                 'original BF16 reference binary unchanged')
            record['baseline_reference_scope']='Original 6354 FP32 fixture + frozen baseline archives; original BF16 binary. Current header declaration comments changed only; no replacement of old library objects.'
            record['binary']=str(build/'strata'); record['binary_sha256']=sha(build/'strata')
            record['target_pins']={name:v['sha256'] for name,v in record['products'].items()}
            record['actual_compile_entries']=json.loads((build/'compile_commands.json').read_text())
            attention=[c for c in record['actual_compile_entries'] if c['file'].endswith('/src/kernels/sycl/prompt_attention.cpp')]
            need(len(attention)==1 and '-ffp-contract=off' in attention[0]['command'] and
                 not any(flag in attention[0]['command'].split() for flag in ('-ffast-math','-Ofast','-cl-fast-relaxed-math','-funsafe-math-optimizations','-fdenormal-fp-math=positive-zero')),
                 'strict actual attention translation-unit flags; compiler defaults preserve nonfast FP')
            record['actual_attention_compile_entry']=attention[0]

            for name,value in trees.items():
                need(h.tree_pin(name)==value,'header tree changed during build')
            cursor=journal('journal-final',cursor)
            verify(); record['passed']=True
        except BaseException as exc:
            record['error']=repr(exc)
        finally:
            if cursor and owner.active is None and record['fault_windows'][-1]['label']!='journal-final':
                try: journal('journal-after-failure',cursor)
                except BaseException as exc: record['final_fault_check_error']=repr(exc)
            record['active']=owner.active is not None
            record['ownership_closed']=bool(owner.commands) and not record['active'] and all(o.closed(c) for c in owner.commands)
            record['driver_dump_exists']=(h.DEVICE/'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
            record['private_bytes']=sum(v.stat().st_size for v in root.rglob('*') if v.is_file())
            if not Path(record['build_path']).is_relative_to(root):
                record['private_bytes']+=sum(v.stat().st_size for v in Path(record['build_path']).rglob('*') if v.is_file())

            record['complete']=record['ownership_closed'] and h.boot()==record['boot_id'] and not record.get('final_fault_check_error')
            record['passed'] &= record['complete'] and not record['fresh_faults'] and not record['driver_dump_exists'] and record['private_bytes']<=8<<30
            record['finished_utc']=o.utc()
            save()
        print(json.dumps(dict(passed=record['passed'],complete=record['complete'],receipt=str(receipt),sha256=sha(receipt))))
        return 0 if record['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
