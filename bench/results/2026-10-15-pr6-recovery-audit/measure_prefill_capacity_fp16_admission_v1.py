#!/usr/bin/env python3
"""Close the partial-FP16-store capacity gap using the actual host planner.

No GPU work. Keep the earlier capacity receipts byte-identical: their partial
hidden counts are byte-only ceilings, before staging and chunk rounding.
"""
import fcntl
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = A/'prefill-capacity-alias-delta-v1.json'
        base = json.loads(previous.read_text())
        require(base['passed'] and base['complete'] and not base['active'], 'closed capacity arithmetic')
        header = P/'include/strata/prefill/layer_major_plan.hpp'
        require(sha(header) == base['production_source_pins'][str(header.relative_to(P))], 'same production planner')
        fixture = A/'prefill-capacity-fp16-admission-v1/fixture.cpp'
        compiler = Path('/usr/bin/g++')
        root = B/'main-prefill-capacity-fp16-admission-root-v1'
        receipt = A/'prefill-capacity-fp16-admission-v1.json'
        require(not root.exists() and not receipt.exists(), 'fresh host capacity gate')
        root.mkdir(mode=0o700)
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        record = dict(schema='prefill-capacity-FP16-partial-admission-v1', passed=False,
            complete=False, active=True, GPU_executed=False, commands=[],
            recipe_sha256=sha(Path(__file__)), fixture_sha256=sha(fixture),
            references={previous.name:sha(previous)}, source_pins={str(header.relative_to(P)):sha(header)},
            compiler=str(compiler), compiler_sha256=sha(compiler), cases=[],
            quality_qualified=False, performance_qualified=False, physical_model_262144_qualified=False,
            scope='Hypothetical active-layer-only layout with current FP32 workspace and FP16 saved hidden. Actual production host planner only; no model/GPU maximum-context admission or numeric qualification.')
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        persist()
        env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, args):
            c, out, err = owner.run(label, args, env, P, wall=30, cpu=30,
                rss_cap=2<<30, text_cap=1<<20, total_cap=8<<20, file_cap=4<<20)
            require(module.closed(c) and c['exit_code'] == 0, label+': '+err.read_text())
            return out.read_text()
        try:
            record['compiler_version'] = run('compiler-version', [str(compiler), '--version'])
            exe = root/'capacity-planner'
            run('compile-production-planner', [str(compiler), '-std=c++17', '-O2', '-Wall', '-Wextra',
                '-Werror', '-I', str(P/'include'), str(fixture), '-o', str(exe)])
            record['binary_sha256'] = sha(exe)
            jobs = []
            for old in base['cases']:
                if old['layout'] != 'one_layer_bank':
                    continue
                for runtime in (512<<20, 1<<30):
                    budget = old['available_for_KV_hidden_bytes'] - (runtime - (512<<20))
                    kv = old['full_262144_QSA_bytes']
                    chunk = old['chunk']
                    row = 20480
                    ceiling = min(262144, (budget-kv)//row)
                    staging = chunk*row if ceiling < 262144 else 0
                    prefix = 262144 if ceiling == 262144 else ((budget-kv-staging)//row//chunk)*chunk
                    rate = 34956 if old['format'] == 'K8V8' else 32076
                    max_context = min(262144, ((budget-43200)//rate)//4*4)
                    require(kv+prefix*row+staging <= budget and
                        (prefix == 262144 or kv+(prefix+chunk)*row+staging > budget), 'maximum complete GPU prefix')
                    require(rate*max_context+43200 <= budget and
                        (max_context == 262144 or rate*(max_context+4)+43200 > budget), 'maximum full-context page')
                    case = dict(chunk=chunk, format=old['format'], scratch_alias=old['scratch_alias'],
                        runtime_headroom_bytes=runtime, available_for_KV_hidden_bytes=budget,
                        full_262144_QSA_bytes=kv, byte_only_hidden_ceiling=ceiling,
                        partial_hidden_conversion_staging_bytes=staging,
                        actual_planner_GPU_hidden_rows=prefix,
                        actual_planner_GPU_hidden_bytes=prefix*row,
                        actual_planner_RAM_hidden_rows=262144-prefix,
                        remaining_bytes=budget-kv-prefix*row-staging,
                        max_context_with_all_hidden_rows=max_context)
                    # auto prefix, exact maximum full store, and next page rejection.
                    specs = [(262144, chunk, budget-kv, -1),
                        (max_context, chunk, budget-(rate-20480)*max_context-43200, max_context),
                        (max_context+4, chunk, budget-(rate-20480)*(max_context+4)-43200, max_context+4)]
                    jobs.append((case, specs))
            input_file = root/'planner-input.txt'
            input_file.write_text(''.join(' '.join(map(str,s))+'\n' for _,ss in jobs for s in ss))
            # A fixed argv/stdin adapter avoids shell interpolation and exposes no GPU runtime.
            runner = 'import subprocess,sys; r=subprocess.run([sys.argv[1]],stdin=open(sys.argv[2],"rb"));sys.exit(r.returncode)'
            lines = run('actual-production-planner', ['/usr/bin/python3', '-B', '-c', runner,
                str(exe), str(input_file)]).splitlines()
            require(len(lines) == 3*len(jobs), 'all planner cases returned')
            for i,(case,_) in enumerate(jobs):
                auto, full, reject = lines[i*3:i*3+3]
                prefix = case['actual_planner_GPU_hidden_rows']
                stage = case['partial_hidden_conversion_staging_bytes']
                require(auto == f'accepted {prefix} {prefix*20480} {(262144-prefix)*20480} {stage}', 'actual partial admission')
                n = case['max_context_with_all_hidden_rows']
                require(full == f'accepted {n} {n*20480} 0 0', 'actual full-store admission')
                require(reject == 'rejected', 'next full-store page rejected')
                case['production_planner_outputs'] = [auto, full, reject]
                record['cases'].append(case)
            record.update(passed=True, complete=True, host_planner_cases=len(lines),
                original_partial_counts_are_byte_only=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with receipt.open('x') as out:
                json.dump(record, out, indent=2)
                out.write('\n')
        print(json.dumps(dict(passed=record['passed'], host_planner_cases=record['host_planner_cases'],
            main_cases=[c for c in record['cases'] if c['scratch_alias'] and c['runtime_headroom_bytes'] == 512<<20])))


if __name__ == '__main__':
    main()
