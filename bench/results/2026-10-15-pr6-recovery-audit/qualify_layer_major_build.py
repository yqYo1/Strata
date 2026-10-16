#!/usr/bin/env python3
"""Correct CPU-target flag proof; reuse closed successful compiler/test commands."""
import fcntl
import json
from pathlib import Path
from check_gemm_orientation import P, B, D, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        original=D/'layer-major-build-v1.json'
        old=json.loads(original.read_text())
        assert not old['active'] and not old['passed'] and not old['complete']
        assert old['error']=='AssertionError()' and old['host_contracts_passed']
        assert [c['label'] for c in old['commands']]==[
            'configure-layer-major-host-tests','build-layer-major','host-layer-major-contracts']
        assert all(c['exit_code']==0 and c['direct_child_reaped'] and c['session_empty']
                   and not c['errors'] and not c['cleanup'] for c in old['commands'])
        build=Path(old['build_path'])
        assert all(sha(P/n)==pin for n,pin in old['source_pins'].items())
        assert all(sha(build/n)==pin for n,pin in old['binaries'].items())
        assert sha(Path(old['control_binary']['path']))==old['control_binary']['sha256']
        assert old['host_test_output']=='layer-major host arithmetic/order/lease fixtures PASS (no kernels)\n'
        commands=json.loads((build/'compile_commands.json').read_text())
        proof={}
        for n in ['src/prefill/prefill.cpp','src/core/expert_cache.cpp','src/prefill/layer_major_plan_test.cpp']:
            vals=[c for c in commands if c['file']==str(P/n)]
            assert len(vals)==1
            if n.endswith('layer_major_plan_test.cpp'):
                assert '-std=c++20' in vals[0]['command'] and '-fsycl' not in vals[0]['command']
            else:
                assert '-fsycl-id-queries-range=size_t' in vals[0]['command']
            proof[n]=vals[0]
        r={k:v for k,v in old.items() if k not in ['active','passed','complete','error','compile_flag_proof']}
        r.update(active=False,passed=True,complete=True,original_build_status=False,
            original_build_receipt_sha256=sha(original),original_build_gate_error=old['error'],
            validation_recipe_sha256=sha(Path(__file__)),commands_reused_from_original_build=True,
            no_compile_or_test_repeated=True,compile_flag_proof=proof,
            corrected_gate='The pure CPU CMake target intentionally has -std=c++20 and no -fsycl flags; the two product SYCL translation units retain the required range=size_t flag. Original failed receipt is byte-identical.',
            compile_commands_sha256=sha(build/'compile_commands.json'),
            build_logical_bytes=sum(p.stat().st_size for p in build.rglob('*') if p.is_file()))
        assert r['build_logical_bytes']<=r['build_byte_budget']
        dest=D/'layer-major-build-v2.json';assert not dest.exists()
        dest.write_text(json.dumps(r,indent=2)+'\n')
        assert sha(original)==r['original_build_receipt_sha256']
        print(json.dumps(dict(passed=True,build_and_host_commands_reused=True,GPU_executed=False,
            binaries=r['binaries'])),flush=True)


if __name__=='__main__':
    main()
