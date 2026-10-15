#!/usr/bin/env python3
"""Main's serial host checks, before changing the historical benchmark client."""
import ast
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def compare_client_paths(old_text, new_text):
    old, new = ast.parse(old_text), ast.parse(new_text)
    allowed = {'Suite', 'compare_table', 'main', 'validate_workload_id'}
    def dump(node):
        return ast.dump(node, include_attributes=False)
    require([dump(n) for n in old.body if getattr(n, 'name', None) not in allowed] ==
            [dump(n) for n in new.body if getattr(n, 'name', None) not in allowed],
            'client change outside reviewed constructor/provenance/CLI/comparison')
    suites = [next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Suite')
              for tree in (old, new)]
    methods = [{n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for tree in suites]
    require(set(methods[1]) == set(methods[0]) | {'workload_identity'}, 'unexpected Suite method delta')
    unchanged = sorted(set(methods[0]) - {'__init__'})
    for name in unchanged:
        require(dump(methods[0][name]) == dump(methods[1][name]), 'timed/request method changed: ' + name)
    return unchanged


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # No test is allowed to overlap the live full0, or alter its inputs.
        previous = json.loads((A / 'selective-iq-clean-bench-full-0-v1.json').read_text())
        require(not previous['active'] and all(c.get('direct_child_reaped') and c.get('session_empty')
                                              for c in previous['commands']), 'full0 owners not retired')
        for c in previous['commands']:
            for identity in c.get('owners', []):
                current = module.identity(identity['pid'])
                require(current is None or current['start_ticks'] != identity['start_ticks'],
                        'full0 owned identity still present')
        proposals = [A / 'bench-fixed-workload-v1', A / 'bench-request-identity-v1']
        proposal_inputs = {}
        for proposal in proposals:
            for name in ('pins.json', 'artifact-sha256.json'):
                manifest = proposal / name
                proposal_inputs[str(manifest.relative_to(P))] = sha(manifest)
            for rel, pin in json.loads((proposal / 'pins.json').read_text()).items():
                require(sha(P / rel) == pin, 'original client/helper changed before qualification: ' + rel)
            for rel, pin in json.loads((proposal / 'artifact-sha256.json').read_text()).items():
                require(sha(proposal / rel) == pin, 'proposed input tool changed: ' + rel)
                proposal_inputs[str((proposal / rel).relative_to(P))] = pin
        old = P / 'tools/strata_bench.py'
        proposed = proposals[0] / 'proposed/tools/strata_bench.py'
        unchanged = compare_client_paths(old.read_text(), proposed.read_text())
        root = B / 'main-bench-input-tools-host-root-v1'
        receipt = A / 'bench-input-tools-host-v1.json'
        require(not root.exists() and not receipt.exists(), 'fresh host qualification root required')
        root.mkdir(mode=0o700)
        record = dict(schema='main-bench-input-tools-host-v1', active=True, passed=False, complete=False,
                      GPU_executed=False, commands=[], original_client_sha256=sha(old),
                      proposed_client_sha256=sha(proposed), unchanged_suite_methods=unchanged,
                      proposal_inputs=proposal_inputs,
                      unchanged_module_paths=True, original_benchmark_sha256=sha(A / 'selective-iq-clean-bench-full-0-v1.json'),
                      recipe_sha256=sha(Path(__file__)), applied=False,
                      scope='Host tools and unchanged timed/request paths only; no input-pair, native, speed or quality approval.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8', LC_ALL='C.UTF-8',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        try:
            for label, script in [('fixed-workload', proposals[0] / 'test_fixed_workload.py'),
                                  ('request-identity', proposals[1] / 'test_compare_requests.py')]:
                command, out, err = owner.run(label, ['/usr/bin/python3', '-B', str(script)], env, P,
                                              wall=60, cpu=60, rss_cap=2 << 30, text_cap=4 << 20,
                                              total_cap=256 << 20, file_cap=8 << 20)
                require(module.closed(command) and command['exit_code'] == 0,
                        label + ' failed: ' + err.read_text()[-3000:])
                match = re.search(r'Ran (\d+) tests? in .*\n\nOK\s*$', err.read_text())
                require(match and int(match[1]) >= 6, label + ' did not execute the expected tests')
                record.setdefault('tests', []).append(dict(label=label, source_sha256=sha(script),
                    tests=int(match[1]), stdout_sha256=sha(out), stderr_sha256=sha(err)))
                persist()
            require(sha(old) == record['original_client_sha256'] and sha(proposed) == record['proposed_client_sha256'],
                    'client changed during host checks')
            require(all(sha(P / rel) == pin for rel, pin in proposal_inputs.items()), 'proposal changed during host checks')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with receipt.open('x') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], tests=record.get('tests'), GPU_executed=False)))


if __name__ == '__main__':
    main()
