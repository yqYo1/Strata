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


if __name__ == '__main__':
    unittest.main()
