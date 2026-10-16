#!/usr/bin/env python3
"""Build a matched two-layout timing fixture; no GPU/model execution."""
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    h = load('prefix_layout_build_helpers', A/'build_qsa_bf16_prefix_parallel_pad_v1.py')
    B, L, pin = h.B, h.L, h.pin
    root = B/'main-qsa-bf16-prefix-parallel-pad-bench-build-v1'
    dst = A/'own-qsa-bf16-prefix-parallel-pad-bench-build-v1.json'
    assert not root.exists() and not dst.exists()
    prior_path = A/'own-qsa-bf16-prefix-parallel-pad-build-v1.json'
    prior = json.loads(prior_path.read_text())
    gate_path = A/'own-qsa-bf16-prefix-parallel-pad-component-v1.json'
    gate = json.loads(gate_path.read_text())
    assert all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] for v in [prior, gate])
    base_path = A/'own-qsa-bf16-prefix-build-v2.json'
    base = json.loads(base_path.read_text())
    assert base['passed'] and base['complete'] and base['ownership_closed']
    source = P/'src/kernels/sycl/qsa_bf16_prefix.cpp'
    assert base['frozen_files'][str(source)] == pin(source) == prior['base_source']
    for name in ['strata/kernels/qsa_bf16_prefix.hpp', 'strata/kernels/qsa_bf16_attention.hpp',
                 'strata/kernels/qsa_bf16_packed.hpp', 'strata/kernels/f16_bits.hpp', 'strata/sycl/launch.hpp']:
        p = P/'include'/name
        assert base['frozen_files'][str(p)] == pin(p), name
    base_obj = base['object']
    frozen = dict(prior['frozen_files'])
    fixture = P/'tests/sycl/qsa_bf16_prefix_layout_bench.cpp'
    for p in [prior_path, gate_path, base_path, Path(base_obj['path']), fixture, Path(__file__)]:
        frozen[str(p)] = pin(p)
    assert frozen[base_obj['path']] == base_obj
    owner_path = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    owner_module = load('layout_timing_build_owner', owner_path)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        (root/'tmp').mkdir()
        (root/'cache').mkdir()
        rec = dict(schema='strata-qsa-prefix-parallel-pad-bench-build-v1', passed=False, complete=False,
                   active=False, GPU_executed=False, commands=[], frozen_files=frozen,
                   base_object=base_obj, compiler=prior['compiler'], source_diff=prior['source_diff'],
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
                   production_selected=False, performance_adopted=False,
                   scope='Same binary, precision/online32cell updates, initialized inputs/allocations and prefix pack in both calls. Current versus128thread p with two extra barriers and score pitch33/+64local bytes, same K/matrix operand layout. Independent four-query oracle, complete finite/guards/inputs, exact repeat signed-zero bits. Compile only.')
        owner = None
        def persist():
            rec['active'] = bool(owner and owner.active is not None)
            (root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
        owner = owner_module.Owner(root/'owned',persist)
        rec['commands'] = owner.commands
        env = dict(PATH=str(L/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                   LD_LIBRARY_PATH=str(L/'lib'),TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),PYTHONDONTWRITEBYTECODE='1')
        flags = [prior['compiler']['path'],'-std=c++20','-O3','-DNDEBUG','-DSTRATA_ENABLE_SYCL=1',
                 '-I'+str(P/'include'),'-I'+str(P/'include/strata/sycl/compat'),'-fno-fast-math',
                 '-ffp-contract=off','-fsycl','-fsycl-id-queries-range=size_t','-Wall','-Wextra','-Werror']
        def verify():
            for path,value in frozen.items():
                assert pin(path) == value, path
        def run(label, argv):
            c,out,err = owner.run(label,argv,env,P,wall=300,cpu=900,rss_cap=8<<30,
                                  text_cap=64<<20,file_cap=128<<20,total_cap=1<<30)
            assert owner_module.closed(c) and c['exit_code'] == 0, err.read_text()[-5000:]
            return out.read_text(),err.read_text()
        try:
            verify()
            obj = root/'parallel_pad.o'
            run('compile-candidate',flags+['-Dqsa_bf16_prefix_batch=qsa_bf16_prefix_parallel_pad_batch',
                '-c',prior['candidate_source']['path'],'-o',str(obj)])
            rec['candidate_object'] = pin(obj)
            exe = root/'sycl_qsa_bf16_prefix_parallel_pad_bench'
            run('compile-link',flags+['-DSTRATA_QSA_PREFIX_CANDIDATE=qsa_bf16_prefix_parallel_pad_batch','-DSTRATA_QSA_PREFIX_CANDIDATE_NAME="combined"',str(fixture),base_obj['path'],str(obj),*prior['library_inputs'],'-o',str(exe)])
            rec['dynamic'],rec['readelf_stderr'] = run('readelf',['/usr/bin/readelf','-d',str(exe)])
            verify()
            rec.update(binary=pin(exe),passed=True,complete=True)
        except BaseException as error:
            import traceback
            rec.update(error=repr(error),traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(owner_module.closed(c) for c in owner.commands)
            persist()
            dst.write_bytes((root/'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(dst),passed=rec['passed'],error=rec.get('error'),GPU_executed=False)))
        return 0 if rec['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
