"""Synthetic actual-input admission controls; no compiler, runtime or GPU use."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapped_row_copy_component as component


class InputCensus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(component, 'ROOT', self.root).start()
        self.paths = []

        def file(name):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'synthetic input\n')
            return p

        self.file = file
        sources = {}
        for name in component.UNITS:
            p = file('source/' + name)
            sources[name] = component.sha(p)
            self.paths.append(p)
        dependencies = {}
        for i in range(101):
            p = file(f'sdk/header{i}.hpp')
            dependencies[str(p)] = component.sha(p)
            self.paths.append(p)
        self.links = [file(f'sdk/lib{i}.so') for i in range(9)]
        external = file('external.json')
        external.write_text(json.dumps({str(p): component.sha(p) for p in self.links}))
        self.row = dict(sources=sources, dependency_pins=dependencies,
            external_inputs=dict(path=str(external)), tools={}, source_commit='0' * 40)

    def read(self, paths=None, links=None):
        deps = '\n'.join('    ' + str(p) for p in (self.paths if paths is None else paths))
        return component.census(self.row, deps,
            '\n'.join(map(str, self.links if links is None else links)))

    def test_complete_actual_inventory(self):
        r = self.read()
        self.assertEqual((r['compiler_files'], r['external_link_files'], r['uncovered']), (106, 9, 0))
        self.assertEqual(r['translation_units'], component.UNITS)
        self.assertEqual(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(), r['sha256'])

    def test_unknown_compiler_input_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'unfrozen actual compiler input'):
            self.read([*self.paths, self.file('sdk/unpinned.hpp')])

    def test_missing_translation_unit_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'exact five-unit compiler census'):
            self.read(self.paths[1:])

    def test_changed_header_refused(self):
        self.paths[-1].write_bytes(b'changed header\n')
        with self.assertRaisesRegex(RuntimeError, 'unfrozen actual compiler input'):
            self.read()

    def test_unknown_link_input_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'unfrozen actual linker input'):
            self.read(links=[*self.links, self.file('sdk/unpinned.so')])


class GraphTiming(unittest.TestCase):
    @staticmethod
    def transcript():
        lines = [f'CLOCK median_stamp_ns=100 floor_ns=10000000 batch={component.BATCH} process=0']
        for kind in ('host_usm', 'device'):
            for banks in (1, 64):
                for hits in (4, 20):
                    for variant in component.ORDERS[0]:
                        for sample in range(5):
                            values = dict(process=0, variant=variant, source=kind,
                                rows=40, width=2560, hits=hits, banks=banks, sample=sample,
                                launches=component.BATCH, route='compat_graph',
                                graph_launches=component.BATCH, nodes_per_graph=1,
                                wall_ns=20000000, cpu_ns=10000000,
                                submit_wall_ns=15000000, wait_wall_ns=5000000,
                                submit_cpu_ns=9000000, wait_cpu_ns=1000000)
                            lines.append('SAMPLE ' + ' '.join(f'{k}={v}' for k,v in values.items()))
        lines.append('TIMING passed=1 samples=200 model_speed=0 physical_traffic=0')
        return '\n'.join(lines)

    def test_complete_actual_graph_dispatch(self):
        value = component.collect_timing(self.transcript(), 0)
        self.assertEqual(len(value['cells']), 8)
        self.assertEqual(sum(len(rows) for cell in value['cells'].values()
                             for rows in cell.values()), 200)

    def test_direct_fallback_refused(self):
        text = self.transcript().replace('route=compat_graph', 'route=direct', 1)
        with self.assertRaisesRegex(RuntimeError, 'actual graph route differs'):
            component.collect_timing(text, 0)

    def test_missing_replay_refused(self):
        text = self.transcript().replace(f'graph_launches={component.BATCH}',
            f'graph_launches={component.BATCH-1}', 1)
        with self.assertRaisesRegex(RuntimeError, 'sample shape/order differs'):
            component.collect_timing(text, 0)


if __name__ == '__main__':
    unittest.main()
