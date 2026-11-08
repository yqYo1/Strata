"""Q6 DPAS actual-input and timing-reader controls; no compiler/GPU use."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import q6_dpas_component as component

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
        self.assertEqual((r['compiler_files'], r['external_link_files'], r['uncovered']), (len(component.UNITS)+101, 9, 0))
        self.assertEqual(r['translation_units'], component.UNITS)
        self.assertEqual(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(), r['sha256'])

    def test_unknown_compiler_input_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'unfrozen actual compiler input'):
            self.read([*self.paths, self.file('sdk/unpinned.hpp')])

    def test_missing_translation_unit_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'exact component-unit compiler census'):
            self.read(self.paths[1:])

    def test_changed_header_refused(self):
        self.paths[-1].write_bytes(b'changed header\n')
        with self.assertRaisesRegex(RuntimeError, 'unfrozen actual compiler input'):
            self.read()

    def test_unknown_link_input_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'unfrozen actual linker input'):
            self.read(links=[*self.links, self.file('sdk/unpinned.so')])

class Timing(unittest.TestCase):
    @staticmethod
    def transcript(index=0):
        out=[f'CLOCK median_stamp_ns=100 floor_ns=10000000 process={index}']
        for n in (640,248320):
            for arm in ((1,0) if index==1 else (0,1)):
                for sample in range(5):
                    batch=4096 if n==640 else 256
                    d=dict(process=index,arm=arm,k=2560,n=n,columns=4,sample=sample,
                        launches=batch,graph_launches=batch,nodes_per_graph=1,route='compat_graph',
                        wall_ns=20000000,cpu_ns=10000000,submit_wall_ns=15000000,
                        submit_cpu_ns=9000000,wait_wall_ns=5000000,wait_cpu_ns=1000000)
                    out.append('SAMPLE '+' '.join(f'{k}={v}' for k,v in d.items()))
        out.extend(['NUMERIC passed=1 synthetic=1 guards=1']*4)
        out.append('TIMING passed=1 samples=20 model_speed=0 physical_traffic=0')
        return '\n'.join(out)

    def test_all_complete_process_orders(self):
        for i in range(3):
            row=component.collect_timing(self.transcript(i),i)
            self.assertEqual(row['index'],i)
            self.assertEqual(sum(len(v) for c in row['cells'].values() for v in c.values()),20)

    def test_missing_actual_replay(self):
        with self.assertRaisesRegex(RuntimeError,'actual timing shape/order/replay count'):
            component.collect_timing(self.transcript().replace('graph_launches=4096','graph_launches=4095',1),0)

    def test_direct_fallback(self):
        with self.assertRaisesRegex(RuntimeError,'actual graph route differs'):
            component.collect_timing(self.transcript().replace('route=compat_graph','route=direct',1),0)

    def test_duration_floor(self):
        with self.assertRaisesRegex(RuntimeError,'timing floor/overhead'):
            component.collect_timing(self.transcript().replace('wall_ns=20000000','wall_ns=9000000',1),0)

    def test_clock_conservation(self):
        with self.assertRaisesRegex(RuntimeError,'timing conservation'):
            component.collect_timing(self.transcript().replace('wait_cpu_ns=1000000','wait_cpu_ns=1000001',1),0)

if __name__=='__main__':
    unittest.main()
