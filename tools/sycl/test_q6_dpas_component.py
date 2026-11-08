"""Six CPU-only linked-ELF/SPIR-V callgraph controls;repository LICENSE."""
import hashlib
from pathlib import Path
import struct
import sys
import unittest
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import q6_dpas_component as component

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


if __name__ == '__main__':
    unittest.main()
