"""SOURCE ONLY: independent CPU parser adversarial fixtures; root executes."""
import struct
import unittest
from cache_route_pairs_parser_v1 import parse_request, static_profile, REQUEST_LIMIT

DONE='DONE 64 32768 1.0 1.0 length 1 2 0 0 480 0 0 0.0 32768 0'
FNV='0123456789abcdef'

def golden():
    lines=[
        'CACHE_ROUTE_PAIRS_V1 BEGIN request=1 layers=48 experts=512 enabled=1 supported=1 truncated=0 error=0 complete=1',
        f'CACHE_ROUTE_PAIRS_V1 RESIDENCY request=1 cells=24576 unchanged=1 start_fnv1a64={FNV} end_fnv1a64={FNV}',
    ]
    lines += [f'CACHE_ROUTE_PAIRS_V1 PAIR {layer} 20 10 0 10 0 1' for layer in range(48)]
    lines += ['CACHE_ROUTE_PAIRS_V1 END request=1 cells=48 callbacks=48 entries=480 hits=0 refused=480 offloaded=0 callback_pairs=48 done_hits=0 done_look=480 done_offload=0 reconciled=1 complete=1']
    return ('ordinary driver message\n'+'\n'.join(lines)+'\n').encode('ascii')

class ParserFixtures(unittest.TestCase):
    def parse(self,data=None,done=DONE,residents=frozenset(),ordinal=1):
        return parse_request(golden() if data is None else data,ordinal,done,residents,FNV)
    def rejected(self,data=None,done=DONE,residents=frozenset(),ordinal=1):
        with self.assertRaises(ValueError): self.parse(data,done,residents,ordinal)
    def test_golden_and_units(self):
        result=self.parse()
        self.assertEqual(result['end']['entries'],480)
        self.assertEqual(result['end']['callback_pairs'],48)
        self.assertEqual(result['layer_entries'],[10]*48)
        self.assertLess(result['frame_bytes'],REQUEST_LIMIT)
    def test_status_and_binding(self):
        for old,new in [('supported=1','supported=0'),('complete=1','complete=0'),('error=0','error=1'),
                        ('truncated=0','truncated=1'),('unchanged=1','unchanged=0'),('layers=48','layers=47'),
                        ('experts=512','experts=511'),('reconciled=1','reconciled=0'),('request=1','request=2'),
                        ('end_fnv1a64='+FNV,'end_fnv1a64=0000000000000000')]:
            with self.subTest(new=new): self.rejected(golden().replace(old.encode(),new.encode(),1))
        self.rejected(ordinal=2)
        self.rejected(done=DONE.replace('length','cancel'))
        for old,new in [('0 0 480','0 1 480'),('32768 0','32767 0'),('DONE 64','DONE 63')]:
            self.rejected(done=DONE.replace(old,new))
    def test_framing_fields_and_numeric_limits(self):
        self.rejected(golden()[:-1])
        self.rejected(golden().replace(b'\n',b'\r\n'))
        self.rejected(golden()+b'CACHE_ROUTE_PAIRS_V1 MODE enabled=1 supported=0\n')
        self.rejected(golden()+golden())
        self.rejected(golden().replace(b'entries=480',b'entries=0480'))
        self.rejected(golden().replace(b'entries=480',b'entries=18446744073709551616'))
        self.rejected(golden().replace(b'entries=480',b'entries=-1'))
        self.rejected(golden().replace(b'hits=0 refused=',b'unknown=0 refused='))
        self.rejected(golden().replace(b'CACHE_ROUTE_PAIRS_V1 PAIR 0',b'junk CACHE_ROUTE_PAIRS_V1 PAIR 0'))
        self.rejected(golden().replace(b'PAIR 0 20',b'PAIR \xff 20'))
        self.rejected(b'x'*65536+b'\n'+golden())
        self.rejected(golden()+b'CACHE_ROUTE_PAIRS_V1 unknown\n'*(REQUEST_LIMIT//20))
    def test_sparse_order_classification_and_sums(self):
        for old,new in [(b'PAIR 1 20',b'PAIR 0 20'),(b'PAIR 47 20',b'PAIR 48 20'),
                        (b'PAIR 0 20',b'PAIR 0 512'),(b'20 10 0 10 0 1',b'20 10 0 9 0 1'),
                        (b'20 10 0 10 0 1',b'20 10 0 9 1 1'),
                        (b'20 10 0 10 0 1',b'20 10 0 10 0 2'),(b'cells=48 callbacks',b'cells=47 callbacks'),
                        (b'callbacks=48',b'callbacks=47'),(b'refused=480',b'refused=479')]:
            self.rejected(golden().replace(old,new,1))
        self.rejected(residents={(0,20)})
        self.rejected(golden().replace(b'PAIR 0 20 10 0 10 0 1\n',b''))
    def test_static_profile_independent_reference(self):
        ranked=[(l,e) for l in range(48) for e in range(512)]
        data=b'STRP'+struct.pack('<5I',1,48,512,24576,24576)
        data+=b''.join(struct.pack('<HH',l,e) for l,e in ranked)
        data+=struct.pack('<24576i',*range(24576))
        resident,fp=static_profile(data)
        expected={(l,e) for l in range(47) for e in (0,1)}|{(47,e) for e in range(34)}
        self.assertEqual(resident,expected)
        # Independently implement byte-ordered FNV over hand-derived slot identities.
        value=14695981039346656037
        for l in range(48):
            for e in range(512):
                slot=l*2+e if (l,e) in expected else -1
                for byte in int(slot & 0xffffffff).to_bytes(4,'little'):
                    value=((value ^ byte)*1099511628211) % (1<<64)
        self.assertEqual(fp,format(value,'016x'))
        for malformed in [data[:-1],b'NOPE'+data[4:],data[:24]+b'\xff\xff\xff\xff'+data[28:],
                          data[:28]+data[24:28]+data[32:],data[:-4]+struct.pack('<i',-1)]:
            with self.assertRaises(ValueError): static_profile(malformed)

if __name__=='__main__': unittest.main()
