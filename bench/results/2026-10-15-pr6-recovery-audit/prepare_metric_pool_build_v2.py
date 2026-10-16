#!/usr/bin/env python3
"""Main reviewed host build of the API-only probe; no native execution."""
import difflib
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = A / 'own-metric-pool-diagnostic-v1'


def pin(p):
    p = Path(p)
    return dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    root=B/'main-metric-pool-build-v2';target=A/'own-metric-pool-build-v2.json'
    assert not root.exists() and not target.exists()
    original=(D/'metric_pool.cpp').read_text()
    assert pin(D/'source-contract.json')['sha256']=='505e40b62f957343904c051f275036165d49e770ca25c4911e384ab7390076ee'
    assert (D/'result_names.inc').read_bytes()==(A/'own-metric-enumeration-v1/result_names.inc').read_bytes()
    old='''    bool ok = true;
    if (pool_owned) ok &= tracked(pool_destroy, "zetMetricQueryPoolDestroy", [] { return zetMetricQueryPoolDestroy(pool); }, state_data) == ZE_RESULT_SUCCESS;
    if (active) ok &= tracked(deactivation, "zetContextActivateMetricGroups", [] { return zetContextActivateMetricGroups(context, device, 0, nullptr); },
        [] { return text("{\\"context\\":", ptr(context), ",\\"device\\":", ptr(device), ",\\"count\\":0,\\"groups_ptr\\":0}"); }) == ZE_RESULT_SUCCESS;
    if (context_owned) ok &= tracked(context_destroy, "zeContextDestroy", [] { return zeContextDestroy(context); }, state_data) == ZE_RESULT_SUCCESS;
'''
    new='''    bool ok = true;
    // Main correction: unknown outputs/completion cannot authorize owner teardown.
    for (const char* state : {context_state, activation_state, pool_state, pool_destroy, deactivation, context_destroy})
        if (!std::strcmp(state, "pending")) unknown = true;
    if (unknown) {
        row("{\\"kind\\":\\"cleanup\\",\\"status\\":\\"BLOCKED\\",\\"reason\\":\\"unknown_completion_or_output\\"}");
        return false;
    }
    if (pool_owned) {
        if (tracked(pool_destroy, "zetMetricQueryPoolDestroy", [] { return zetMetricQueryPoolDestroy(pool); }, state_data) != ZE_RESULT_SUCCESS) {
            row("{\\"kind\\":\\"cleanup\\",\\"status\\":\\"ERROR\\",\\"reason\\":\\"pool_destroy_failed_parent_retained\\"}");
            return false;
        }
        pool_owned = false;
    }
    if (active) {
        if (tracked(deactivation, "zetContextActivateMetricGroups", [] { return zetContextActivateMetricGroups(context, device, 0, nullptr); },
            [] { return text("{\\"context\\":", ptr(context), ",\\"device\\":", ptr(device), ",\\"count\\":0,\\"groups_ptr\\":0}"); }) != ZE_RESULT_SUCCESS) {
            row("{\\"kind\\":\\"cleanup\\",\\"status\\":\\"ERROR\\",\\"reason\\":\\"deactivation_failed_context_retained\\"}");
            return false;
        }
        active = false;
    }
    if (context_owned) {
        ok = tracked(context_destroy, "zeContextDestroy", [] { return zeContextDestroy(context); }, state_data) == ZE_RESULT_SUCCESS;
        if (ok) context_owned = false;
    }
'''
    assert original.count(old)==1
    patched=original.replace(old,new)
    owner_source=A.parents[2]/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_source)['sha256']=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('metric_pool_build_owner',owner_source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
        source=root/'metric_pool.cpp';source.write_text(patched)
        for name in ('iu.config','result_names.inc'):(root/name).write_bytes((D/name).read_bytes())
        patch=A/'metric-pool-main-cleanup-v2.patch';assert not patch.exists()
        patch.write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),
            fromfile='own-metric-pool-diagnostic-v1/metric_pool.cpp',tofile='Main-private-v2/metric_pool.cpp')))
        header=B/'main-primary-unitrace-build-root-v1/build/level_zero'
        record=dict(schema='strata-metric-pool-main-host-build-v2',passed=False,complete=False,
            active=False,commands=[],GPU_executed=False,source_package={p.name:pin(p) for p in D.iterdir() if p.is_file()},
            original_source=pin(D/'metric_pool.cpp'),actual_source=pin(source),main_patch=pin(patch),
            config=pin(root/'iu.config'),result_names=pin(root/'result_names.inc'),
            inputs=[pin('/usr/bin/g++'),pin(header/'ze_api.h'),pin(header/'zet_api.h'),pin('/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0')],
            main_review='Main read complete original CPP/README, checked exact config flag/parser sources and copied enum names; only private cleanup correction applied. No query begin, command lists, queues, kernels or explicit GPU buffers. Parent teardown stops on unknown output/completion or failed child teardown.',
            retention=dict(owner='Main',consumer='One bounded pool diagnostic and actual dependency review',next_review='At diagnostic result extraction or replacement'))
        owner=None
        def persist():
            record['active']=bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(root/'owned',persist);record['commands']=owner.commands
        env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
        try:
            binary=root/'metric-pool'
            c,out,err=owner.run('host-build',['/usr/bin/g++','-std=c++17','-O0','-g','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(header),str(source),'/usr/lib/x86_64-linux-gnu/libze_loader.so.1.32.0','-ldl','-o',str(binary)],env,A.parents[2],wall=60,cpu=60,rss_cap=2<<30,text_cap=4<<20,file_cap=16<<20,total_cap=64<<20)
            assert m.closed(c) and c['exit_code']==0,(c,out.read_text(),err.read_text())
            record['binary']=pin(binary)
            c,out,err=owner.run('host-elf-inspection',['/usr/bin/readelf','-d','-l','-n',str(binary)],env,A.parents[2],wall=30,cpu=30,rss_cap=1<<30,text_cap=4<<20,file_cap=4<<20,total_cap=64<<20)
            assert m.closed(c) and c['exit_code']==0 and not err.read_text()
            record['elf_inspection']=out.read_text()
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error)
        finally:
            record['ownership_closed']=owner.active is None and all(m.closed(c) for c in owner.commands)
            persist();target.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(target),passed=record['passed'],GPU_executed=False,error=record.get('error'))))
        return 0 if record['passed'] and record['ownership_closed'] else 1


if __name__=='__main__':
    sys.exit(main())
