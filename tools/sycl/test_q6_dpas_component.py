"""Q6 DPAS actual-input and timing-reader controls; no compiler/GPU use."""
import hashlib
import json
from pathlib import Path
import sys
import struct
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

class LinkedSpirv(unittest.TestCase):
    @staticmethod
    def image(calls=2,unreachable=False):
        def instruction(op,*args):return [((len(args)+1)<<16)|op,*args]
        def text(value):
            b=value.encode()+b'\0';b+=b'\0'*((-len(b))%4)
            return struct.unpack('<'+str(len(b)//4)+'I',b)
        w=[0x07230203,0x10600,0,100,0]
        w+=instruction(5,9,*text('llvm.genx.dpas2.v64i32.v64i32.v128i32.v32i32'))
        w+=instruction(15,6,7,*text(component.KERNEL_ENTRY))
        w+=instruction(54,1,9,0,2)+instruction(56)
        w+=instruction(54,1,8 if unreachable else 7,0,2)
        for i in range(calls):w+=instruction(57,1,21+i,9)
        w+=instruction(56)
        if unreachable:w+=instruction(54,1,7,0,2)+instruction(56)
        return struct.pack('<'+str(len(w))+'I',*w)

    @staticmethod
    def elf(images):
        names=bytearray(b'\0');offsets={}
        for name in ('.shstrtab','__CLANG_OFFLOAD_BUNDLE__sycl-spir64','.strtab','.symtab'):
            offsets[name]=len(names);names+=name.encode()+b'\0'
        strings=bytearray(b'\0');symbols=bytearray(24);data=bytearray();position=0
        for i,image in enumerate(images):
            offset=len(strings);strings+=f'image{i}'.encode()+b'\0'
            symbols+=struct.pack('<IBBHQQ',offset,1,0,2,0x1000+position,len(image))
            data+=image;position+=len(image)
        blob=bytearray(64);sections=[(0,)*10]
        for name,kind,addr,payload,link,alignment,entry in (
            ('.shstrtab',3,0,names,0,1,0),
            ('__CLANG_OFFLOAD_BUNDLE__sycl-spir64',1,0x1000,data,0,4,0),
            ('.strtab',3,0,strings,0,1,0),('.symtab',2,0,symbols,3,8,24)):
            at=len(blob);blob+=payload
            sections.append((offsets[name],kind,2 if addr else 0,addr,at,len(payload),link,0,alignment,entry))
        blob+=b'\0'*((-len(blob))%8);section_at=len(blob)
        for section in sections:blob+=struct.pack('<IIQQQQIIQQ',*section)
        blob[:64]=struct.pack('<16sHHIQQQIHHHHHH',b'\x7fELF\x02\x01\x01'+b'\0'*9,
            2,62,1,0,0,section_at,0,64,0,0,64,len(sections),1)
        return bytes(blob)

    def test_exact_linked_calls(self):
        image=self.image();r=component.embedded_dpas(self.elf([image]))
        self.assertEqual((r['dpas_static_calls'],r['device_images']),(2,1))
        self.assertEqual(r['image_sha256'],hashlib.sha256(image).hexdigest())
        self.assertEqual(r['entrypoint'],component.KERNEL_ENTRY)

    def test_single_call_refused(self):
        with self.assertRaisesRegex(RuntimeError,'at least two reachable'):
            component.embedded_dpas(self.elf([self.image(calls=1)]))

    def test_unreachable_declaration_does_not_prove_calls(self):
        with self.assertRaisesRegex(RuntimeError,'at least two reachable'):
            component.embedded_dpas(self.elf([self.image(unreachable=True)]))

    def test_truncated_instruction_refused(self):
        image=bytearray(self.image());image[-4:]=struct.pack('<I',(2<<16)|56)
        with self.assertRaisesRegex(RuntimeError,'SPIR-V instruction extent'):
            component.embedded_dpas(self.elf([image]))

    def test_duplicate_kernel_image_refused(self):
        with self.assertRaisesRegex(RuntimeError,'one exact linked C4'):
            component.embedded_dpas(self.elf([self.image(),self.image()]))

    def test_bad_elf_span_refused(self):
        blob=bytearray(self.elf([self.image()]));struct.pack_into('<Q',blob,40,len(blob)+64)
        with self.assertRaisesRegex(RuntimeError,'ELF file span'):
            component.embedded_dpas(blob)

if __name__=='__main__':
    unittest.main()
