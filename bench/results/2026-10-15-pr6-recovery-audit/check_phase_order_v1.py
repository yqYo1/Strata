#!/usr/bin/env python3
"""Independent same-chunk ordering case: later row / earlier operation first."""
import fcntl
import importlib.util
import json
import re
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed = json.loads((D / 'phase-audit-build-v1.json').read_text())
        assert seed['passed'] and seed['complete'] and not seed['active']
        spec = importlib.util.spec_from_file_location('owner', OWNER)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        folder = Path(re.search(r'artifacts=(.+)', seed['CPU_outputs']['phase']['stdout'])[1])
        source = folder / 'case-0'
        lines = source.read_text().splitlines()
        raw = Path(str(source) + '.raw').read_bytes()
        records = [dict(x.split('=', 1) for x in line.split()) for line in lines[3:-1]]
        names = ['gdn_input_mixed_h','gdn_out_bo','attn_write_R','ffn_mixed_h','ffn_mixed_bf',
                 'router_logits','router_ids','router_weights','host_grouping','moe_combine_bo','final_R']
        records.sort(key=lambda x: (names.index(x['tensor']), int(x['row'])))
        output = B / 'main-primary-phase-order-root-v1'; assert not output.exists(); output.mkdir(mode=0o700)
        texts, bodies = [lines[0].replace('chunk=4 ', 'chunk=8 '), *lines[1:3]], []
        for i, r in enumerate(records):
            offset, size = int(r['offset']), int(r['bytes'])
            r.update(record=str(i), chunk='0', start='0', count='8', offset=str(sum(map(len, bodies))),
                     source_offset=str(int(r['row']) * (40 if r['tensor'] == 'host_grouping' else size)))
            texts.append(' '.join(k + '=' + v for k, v in r.items()))
            bodies.append(raw[offset:offset+size])
        texts.append(lines[-1])
        a, b = output / 'reference', output / 'changed'
        a.write_text('\n'.join(texts) + '\n'); b.write_bytes(a.read_bytes())
        joined = b''.join(bodies)
        Path(str(a) + '.raw').write_bytes(joined)
        changed = bytearray(joined)
        for r in records:
            if (r['row'], r['tensor']) in [('1','final_R'), ('6','gdn_input_mixed_h')]:
                changed[int(r['offset'])] ^= 1
        Path(str(b) + '.raw').write_bytes(changed)
        result = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[], cases=[],
                      recipe_sha256=sha(Path(__file__)), comparator_sha256=sha(D/'compare_layer26_phase_v2.py'),
                      fixture_source_sha256=sha(source), fixture_raw_sha256=sha(Path(str(source)+'.raw')),
                      scope='Synthetic CPU comparator operation-order fixture only; no numerical or GPU claim.')
        def persist(): (output / 'record.json').write_text(json.dumps(result, indent=2) + '\n')
        owner = module.Owner(output / 'owned', persist); result['commands'] = owner.commands; persist()
        try:
            for label, right, expected in [('same', a, 0), ('two-differences', b, 1)]:
                command, stdout, stderr = owner.run(label, ['/usr/bin/python3',str(D/'compare_layer26_phase_v2.py'),str(a),str(right)],
                    seed['environment'], P, wall=30, rss_cap=1 << 30, text_cap=1 << 20, total_cap=16 << 20)
                assert module.closed(command) and command['exit_code'] == expected, stderr.read_text()
                parsed = json.loads(stdout.read_text()); assert parsed['coverage_complete']
                if expected:
                    assert parsed['first_selected_difference']['row'] == 6
                    assert parsed['first_selected_difference']['tensor'] == 'gdn_input_mixed_h'
                    assert parsed['different'][1]['row'] == 1 and parsed['different'][1]['tensor'] == 'final_R'
                result['cases'].append(dict(label=label, expected_exit=expected, result=parsed)); persist()
            result.update(passed=True, complete=True)
        except BaseException as error: result['error'] = repr(error); raise
        finally:
            result['active'] = False; persist()
            (D / 'phase-order-v1.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(passed=True, GPU_executed=False, comparisons=len(result['cases']))))


if __name__ == '__main__': main()
