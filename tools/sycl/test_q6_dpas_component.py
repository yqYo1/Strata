"""CPU-only linked-image and current NativeHead protocol controls;repository LICENSE."""
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import q6_dpas_component as component
import q6_head_owner as head_owner

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


class NativeHeadProtocol(unittest.TestCase):
    @staticmethod
    def output(packed=1):
        lines=[f'HEAD admitted=1 packed={packed} canonical_bytes=521472000 packed_device_bytes={521472000*packed} packed_host_bytes={521472000*packed} max_alloc_bytes=10132807680',
            'TRANSITION phase=load cycle=-1 wall_ns=100 thread_cpu_ns=1 scope=whole_wrapper']
        lines += [f'CAPTURE graphs=3 nodes_each=2 packed_decode_calls={packed} packed_prompt_calls=0 packed_C1_calls=0']*3
        lines += [f'NUMERIC cycle={cycle} seed={seed} outputs=993280 oracle_rows=543 max_abs_pair_error=0.001 C1=PASS prompt=reference guards=PASS'
            for cycle,seed in [(-1,7),(-1,19),(0,31),(1,32)]]
        for cycle in range(2):
            lines += ['BLOCKED phase='+phase+' canonical=PASS packed=PASS' for phase in ('detached','restored_not_published')]
            lines += [f'RECLAIM cycle={cycle} overwritten_device_bytes={521472000*(1+packed)} tail=PASS']
            lines += [f'RETIREMENT cycle={cycle} canonical_span=retired packed_span='+('retired' if packed else 'absent')]
            lines += [f'TRANSITION phase={phase} cycle={cycle} wall_ns=100 thread_cpu_ns=1 scope=whole_wrapper'
                for phase in ('prepare','retire_readers','detach','restore','resume')]
        lines += [f'OWNER_CHECK packed={packed} numeric_cases=4 blocked_cases=4 restore_cycles=2 cancel_cases=1 graph_replays=12 canonical_immutable=PASS packed_immutable='+('PASS' if packed else 'absent')+' model=0 Verifier_owner=0',
            'HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified']
        return '\n'.join(lines)+'\n'

    def test_both_admitted_arms(self):
        for arm in (0,1):
            value=head_owner.parse_native(self.output(arm),arm)
            self.assertEqual(value['graph_replays'],12)
            self.assertEqual(len(value['numeric']),4)

    def test_truncated_numerics(self):
        text=self.output();line=next(v for v in text.splitlines() if v.startswith('NUMERIC '))
        with self.assertRaises(RuntimeError):head_owner.parse_native(text.replace(line+'\n','',1),1)

    def test_prompt_route_and_graph_extent(self):
        for a,b in [('packed_prompt_calls=0','packed_prompt_calls=1'),('nodes_each=2','nodes_each=1')]:
            with self.assertRaises(RuntimeError):head_owner.parse_native(self.output().replace(a,b),1)

    def test_detach_and_reclaim_extents(self):
        for a,b in [('overwritten_device_bytes=1042944000','overwritten_device_bytes=521472000'),
                    ('BLOCKED phase=restored_not_published','BLOCKED phase=published')]:
            with self.assertRaises(RuntimeError):head_owner.parse_native(self.output().replace(a,b),1)

    def test_wrong_packed_arm_and_bytes(self):
        with self.assertRaises(RuntimeError):head_owner.parse_native(self.output(0),1)
        with self.assertRaises(RuntimeError):head_owner.parse_native(self.output().replace('packed_host_bytes=521472000','packed_host_bytes=0'),1)

    def test_transition_is_not_optional(self):
        text=self.output();line=next(v for v in text.splitlines() if v.startswith('TRANSITION phase=restore '))
        with self.assertRaises(RuntimeError):head_owner.parse_native(text.replace(line+'\n','',1),1)

    def test_duplicate_or_nonfinite_exit(self):
        for text in [self.output()+'HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified\n',
                     self.output().replace('max_abs_pair_error=0.001','max_abs_pair_error=nan'),
                     self.output().replace('model=0','model=1')]:
            with self.assertRaises(RuntimeError):head_owner.parse_native(text,1)

    def test_actual_fault_finalizer_acceptance_contract(self):
        # Real shared finalizer,only its OS reads mocked. Normal exit alone and
        # partial evidence must not qualify;the complete caller must set its
        # mandatory acceptance field before finalization (the V1 defect).
        scope=SimpleNamespace(active=None,commands=[])
        complete=dict(passed=True,assets_pre_post_match=True,native_exit_code=0,
            owner={'numeric_cases':4},validation={'preinit':37})
        with patch.object(head_owner.validation.subprocess,'check_output',return_value=''), \
             patch.object(head_owner.validation.guards,'driver_guard',return_value={'driver':'fixture'}), \
             patch.object(head_owner.validation,'Path',return_value=SimpleNamespace(read_text=lambda:'fixture-boot\n')):
            good=dict(complete);head_owner.finish_gpu_probe(good,scope,'fixture-cursor','fixture-boot')
            self.assertTrue(good['host_probe_accepted'] and good['passed'])
            for missing in complete:
                partial=dict(complete);partial.pop(missing)
                head_owner.finish_gpu_probe(partial,scope,'fixture-cursor','fixture-boot')
                self.assertFalse(partial['passed'])
            failed=dict(complete,error='original failure must survive')
            head_owner.finish_gpu_probe(failed,scope,'fixture-cursor','fixture-boot')
            self.assertFalse(failed['passed'])


if __name__ == '__main__':
    unittest.main()
