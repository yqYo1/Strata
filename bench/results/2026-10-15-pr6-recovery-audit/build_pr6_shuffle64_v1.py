#!/usr/bin/env python3
"""Main serialized model-free64bit oracle build; never runs the resulting binary."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=B/'main-pr6-shuffle64-build-v1';target=A/'own-pr6-shuffle64-build-v1.json'
    assert not root.exists() and not target.exists()
    config_path=A/'own-pr6-harness-configure-v6.json';review_path=A/'own-pr6-generated-command-review-v6.json'
    assert sha(config_path)=='aa3e36da8b2f4f26718393271ba595db227835b89738577a54b1b724289e6f27'
    assert sha(review_path)=='a65b589bb6f854e8895dbc317627c75d734e0b7faa9299abf105bff0f569fd7a'
    config=json.loads(config_path.read_text());assert config['passed'] and config['complete'] and config['ownership_closed']
    build=Path(config['build_path'])
    owner_source=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert sha(owner_source)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('pr6_harness_build_owner',owner_source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def freeze():
        for v in [*config['source_pins'].values(),*config['generated_inputs'],*config['compilers'],config['oneMKL_config']]:
            p=Path(v['path']);assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],str(p)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700);freeze()
        record=dict(schema='strata-pr6-shuffle64-main-build-v1',passed=False,complete=False,active=False,
            commands=[],GPU_executed=False,legacy_binary_executed=False,configure_receipt_sha256=sha(config_path),
            generated_command_review_sha256=sha(review_path),source_commit=config['source_commit'],build_path=str(build),
            budgets=dict(wall_seconds=900,cpu_seconds=2400,rss_bytes=24<<30,raw_text_bytes=64<<20,
                largest_build_file_bytes=512<<20,build_tree_bytes=6<<30),
            scope='Build only model-free pr6_shuffle64_probe; do not execute before ELF/runtime/Main-controller review.',
            retention=dict(owner='Main',consumer='Actual legacy compile/link and dependency qualification',next_review='At source/build failure triage or binary inspection'))
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
            if record['active']:
                total=sum(p.stat().st_size for p in build.rglob('*') if p.is_file())
                if total>6<<30:raise RuntimeError('legacy build artifact budget6GiB')
        owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
        try:
            c,out,err=owner.run('build-shuffle64-oracle',['/usr/bin/cmake','--build',str(build),'--target','pr6_shuffle64_probe','--parallel','1','--verbose'],
                config['commands'][0]['environment'],A.parents[2],wall=900,cpu=2400,rss_cap=24<<30,
                text_cap=64<<20,file_cap=512<<20,total_cap=6<<30)
            record.update(exit_code=c['exit_code'],stdout_sha256=sha(out),stderr_sha256=sha(err))
            assert m.closed(c) and c['exit_code']==0,(c['exit_code'],out.read_text()[-8192:],err.read_text()[-4096:])
            freeze();exe=build/'pr6_shuffle64_probe'
            record['binary']=dict(path=str(exe),realpath=str(exe.resolve()),bytes=exe.stat().st_size,sha256=sha(exe))
            record.update(passed=True,complete=True,binary_inspection_required=True)
        except BaseException as e:
            import traceback
            record.update(error=repr(e),traceback=traceback.format_exc())
        finally:
            record['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands)
            for c in owner.commands:
                for old in c['owners']+[c['direct_identity']]:
                    now=m.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
            persist();target.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(target),passed=record['passed'],exit_code=record.get('exit_code'),GPU_executed=False,error_type=(record.get('error') or '')[:240])))
        return 0 if record['passed'] and record['ownership_closed'] else 1


if __name__=='__main__':sys.exit(main())
