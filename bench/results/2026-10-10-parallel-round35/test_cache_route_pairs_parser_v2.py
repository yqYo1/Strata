"""SOURCE ONLY: independent CPU fixtures for immutable cache_route_pairs_parser_v1.

No producer serializer or patched parser helpers. Root owns serial execution.
Multiple distinct expert pairs per callback are legal: a top10 NT1 callback
has ten callback-pair occurrences, not one. Do not impose the false R41 sum cap.
"""
import hashlib
from pathlib import Path
import re
import struct
import unittest

PARSER_SHA = '2ff44099f7385054dc5663f04a8e226d364d4d349a84b2493bc2bb1d6f3585cd'
parser_file = Path(__file__).with_name('cache_route_pairs_parser_v1.py')
if hashlib.sha256(parser_file.read_bytes()).hexdigest() != PARSER_SHA:
    raise RuntimeError('immutable v1 parser pin differs')
from cache_route_pairs_parser_v1 import parse_request, static_profile

MARKER = 'CACHE_ROUTE_PAIRS_V1'
U64 = (1 << 64)-1


def independent_profile():
    ranked = [(layer,expert) for layer in range(48) for expert in range(512)]
    header = b'STRP' + struct.pack('<5I',1,48,512,24576,24576)
    pairs = b''.join(struct.pack('<HH',layer,expert) for layer,expert in ranked)
    tail = b''.join(struct.pack('<i',rank) for rank in range(24576))
    residents = {(layer,expert) for layer in range(47) for expert in (0,1)}
    residents |= {(47,expert) for expert in range(34)}
    # Independently hand-derive slot placement for this layer-major reference.
    fingerprint = 14695981039346656037
    for layer,expert in ranked:
        slot = layer*2+expert if (layer,expert) in residents else -1
        for byte in (slot & 0xffffffff).to_bytes(4,'little'):
            fingerprint = ((fingerprint ^ byte)*1099511628211) % (1 << 64)
    return header+pairs+tail,residents,format(fingerprint,'016x')


PROFILE, RESIDENTS, FNV = independent_profile()


class Frame:
    """Test-owned text constructor based on explicit callback route rows."""
    def __init__(self, ordinal=1, rows=((0,1,2,3,4,5,6,7,8,9),)):
        self.ordinal = ordinal
        self.records = []
        for layer in range(48):
            count = {}; callbacks = {}
            for row in rows:
                for expert in row: count[expert] = count.get(expert,0)+1
                for expert in set(row): callbacks[expert] = callbacks.get(expert,0)+1
            for expert in sorted(count):
                entries=count[expert]
                hits=entries if (layer,expert) in RESIDENTS else 0
                self.records.append([layer,expert,entries,hits,entries-hits,0,callbacks[expert]])
        sums = [sum(record[index] for record in self.records) for index in range(2,7)]
        entries,hits,refused,offloaded,callback_pairs=sums
        self.begin = dict(request=ordinal,layers=48,experts=512,enabled=1,supported=1,truncated=0,error=0,complete=1)
        self.residency = dict(request=ordinal,cells=24576,unchanged=1,start_fnv1a64=FNV,end_fnv1a64=FNV)
        self.end = dict(request=ordinal,cells=len(self.records),callbacks=48*len(rows),entries=entries,hits=hits,
                        refused=refused,offloaded=offloaded,callback_pairs=callback_pairs,
                        done_hits=hits,done_look=entries,done_offload=offloaded,reconciled=1,complete=1)
        self.done = ['DONE','64','32768','1.0','1.0','length','1','2','0',str(hits),str(entries),'0','0','0.0','32768','0']
    def lines(self):
        def keyed(kind,values):
            return MARKER+' '+kind+' '+' '.join(str(k)+'='+str(v) for k,v in values.items())
        return [keyed('BEGIN',self.begin),keyed('RESIDENCY',self.residency)] + [
            MARKER+' PAIR '+' '.join(map(str,record)) for record in self.records] + [keyed('END',self.end)]
    def data(self): return ('\n'.join(self.lines())+'\n').encode('ascii')
    def done_text(self): return ' '.join(self.done)


class ExpandedParserFixtures(unittest.TestCase):
    def parse(self,frame,data=None,done=None,ordinal=None):
        return parse_request(frame.data() if data is None else data,
                             frame.ordinal if ordinal is None else ordinal,
                             frame.done_text() if done is None else done,RESIDENTS,FNV)
    def reject(self,branch,frame=None,data=None,done=None,ordinal=None):
        """Require the named parser branch, so an earlier rejection cannot pass."""
        frame = Frame() if frame is None else frame
        with self.assertRaisesRegex(ValueError,'^'+re.escape(branch)+'$'):
            self.parse(frame,data,done,ordinal)
    def from_lines(self,lines): return ('\n'.join(lines)+'\n').encode('ascii')

    def test_accept_mixed_static_hits_nt1_top10_all_ordinals(self):
        for ordinal in range(1,5):
            with self.subTest(ordinal=ordinal):
                result=self.parse(Frame(ordinal))
                self.assertEqual(result['end']['hits'],104) # 47*2 resident routes + 10 in final layer
                self.assertEqual(result['end']['refused'],376)
                self.assertEqual(result['end']['entries'],480)
                self.assertEqual(result['layer_entries'],[10]*48)
                self.assertEqual(result['end']['callbacks'],48)
                for layer in range(48):
                    occurrences=sum(p['callback_pairs'] for p in result['pairs'] if p['layer']==layer)
                    self.assertEqual(occurrences,10)
                    # Guard against R41's explicitly corrected false aggregate invariant.
                    self.assertGreater(occurrences,result['end']['callbacks']//48)
    def test_accept_repeated_ids_within_and_across_callbacks(self):
        row=(0,2,2,3,4,5,6,7,8,9)
        result=self.parse(Frame(rows=(row,row)))
        self.assertEqual(result['layer_entries'],[20]*48)
        pair=next(p for p in result['pairs'] if (p['layer'],p['expert'])==(0,2))
        self.assertEqual((pair['entries'],pair['refused'],pair['callback_pairs']),(4,4,2))
        self.assertEqual(result['end']['callbacks'],96)
        self.assertEqual(result['end']['callback_pairs'],48*18)
        # These validate parser route units, not a claim of native NT histogram qualification.
    def test_accept_ordinary_stderr_interleaving(self):
        frame=Frame();lines=frame.lines();lines.insert(3,'ordinary bounded driver diagnostic')
        self.assertTrue(self.parse(frame,self.from_lines(lines))['passed'])

    def test_missing_markers_reach_frame_position_branch(self):
        for index,name in [(0,'BEGIN'),(1,'RESIDENCY'),(-1,'END')]:
            frame=Frame();lines=frame.lines();del lines[index]
            with self.subTest(missing=name):
                self.reject('wrong frame type/field count',frame,self.from_lines(lines))
        frame=Frame()
        self.reject('missing/empty/oversized frame',frame,self.from_lines(frame.lines()[:2]+frame.lines()[-1:]))
    def test_duplicate_markers_and_nested_frame(self):
        frame=Frame();lines=frame.lines();lines.insert(1,lines[0])
        self.reject('wrong frame type/field count',frame,self.from_lines(lines)) # duplicate BEGIN occupies RESIDENCY
        for marker_index in (1,-1):
            frame=Frame();lines=frame.lines();lines.insert(2,lines[marker_index])
            self.reject('unexpected/duplicate frame marker or PAIR',frame,self.from_lines(lines))
        frame=Frame();lines=frame.lines();inner=Frame(2).lines();lines[5:5]=inner
        self.reject('unexpected/duplicate frame marker or PAIR',frame,self.from_lines(lines))
    def test_out_of_order_markers_and_pairs(self):
        frame=Frame();lines=frame.lines();lines[0],lines[1]=lines[1],lines[0]
        self.reject('wrong frame type/field count',frame,self.from_lines(lines))
        frame=Frame();lines=frame.lines();lines[2],lines[3]=lines[3],lines[2]
        self.reject('unsorted/duplicate/out-of-range pair',frame,self.from_lines(lines))
        frame=Frame();frame.records[1]=frame.records[0][:]
        self.reject('unsorted/duplicate/out-of-range pair',frame)
        for layer,expert in [(48,0),(0,512)]:
            frame=Frame();frame.records[0][0:2]=[layer,expert]
            self.reject('unsorted/duplicate/out-of-range pair',frame)

    def test_bad_fnv_encoding_and_identity(self):
        for field in ('start_fnv1a64','end_fnv1a64'):
            for value in ('A'+FNV[1:],FNV[:-1],'g'+FNV[1:],'0x'+FNV,'0'*17):
                frame=Frame();frame.residency[field]=value
                self.reject('noncanonical FNV',frame)
            frame=Frame();frame.residency[field]='0'*16
            self.reject('resident identity changed/not pinned static map',frame)
        for field,value in [('unchanged',0),('cells',24575),('request',2)]:
            frame=Frame();frame.residency[field]=value
            self.reject('resident identity changed/not pinned static map',frame)
    def test_begin_status_dimensions_ordinal(self):
        for field,value in [('request',2),('layers',47),('experts',511),('enabled',0),('supported',0),
                            ('truncated',1),('error',1),('complete',0)]:
            frame=Frame();frame.begin[field]=value
            self.reject('unsupported/error/incomplete/incorrect BEGIN',frame)
        for ordinal in (0,5): self.reject('ordinal outside four-request job',ordinal=ordinal)
        self.reject('unsupported/error/incomplete/incorrect BEGIN',ordinal=2)
        for field,value in [('request',2),('complete',0),('reconciled',0)]:
            frame=Frame();frame.end[field]=value
            self.reject('END incomplete/mismatched ordinal',frame)
    def test_key_grammar_and_numeric_encoding(self):
        frame=Frame();lines=frame.lines();lines[0]=lines[0].replace('layers=48','wrong=48')
        self.reject('unknown/duplicate/out-of-order field',frame,self.from_lines(lines))
        frame=Frame();lines=frame.lines();lines[0]=lines[0].replace('layers=48','request=48')
        self.reject('unknown/duplicate/out-of-order field',frame,self.from_lines(lines))
        for value in ('01','-1','+1','1.0','184467440737095516160'):
            frame=Frame();frame.records[0][2]=value
            self.reject('noncanonical uint64',frame)
        frame=Frame();frame.records[0][2]=U64+1
        self.reject('uint64 overflow',frame)
    def test_pair_classification_addition_and_callback_caps(self):
        frame=Frame();frame.records[0][2:]=[U64,U64,1,0,1]
        self.reject('invalid pair classification/callback units',frame) # sum exceeds entries, all fields individually legal
        for entries,hits,refused,offloaded,callbacks in [(0,0,0,0,0),(1,1,0,0,0),(1,1,0,0,2),(2,2,0,0,2)]:
            frame=Frame();frame.records[0][2:]=[entries,hits,refused,offloaded,callbacks]
            self.reject('invalid pair classification/callback units',frame)
        # Last case has callbacks<=entries but exceeds callbacks/48=1.
        frame=Frame();frame.records[0][2:]=[1,0,0,1,1]
        self.reject('fixed no-offload profile changed',frame)
        frame=Frame();frame.records[0][2:]=[1,0,1,0,1]
        self.reject('classification inconsistent with fixed pinned resident set',frame)
        frame=Frame();frame.records[2][2:]=[1,1,0,0,1]
        self.reject('classification inconsistent with fixed pinned resident set',frame)
    def test_sparse_total_overflow_reaches_checked_addition(self):
        frame=Frame()
        # First two records are resident pairs: each satisfies classification,
        # owner callback bound and uint64 grammar. Second addition overflows.
        frame.records[0][2:]=[U64,U64,0,0,1]
        frame.records[1][2:]=[U64,U64,0,0,1]
        self.reject('sparse total overflow',frame)
    def test_sparse_end_totals_and_done_counters(self):
        frame=Frame();frame.end['cells']-=1
        self.reject('sparse count mismatch',frame)
        for field in ('entries','hits','refused','offloaded','callback_pairs'):
            frame=Frame();frame.end[field]+=1
            self.reject('sparse sums differ from END',frame)
        for field in ('done_hits','done_look','done_offload'):
            frame=Frame();frame.end[field]+=1
            self.reject('DONE reconciliation/admission/offload mismatch',frame)
        for index in (9,10,15):
            frame=Frame();frame.done[index]=str(int(frame.done[index])+1)
            self.reject('DONE reconciliation/admission/offload mismatch',frame)
    def test_callback_count_and_layer_coverage(self):
        for value in (0,47,49):
            frame=Frame();frame.end['callbacks']=value
            self.reject('invalid complete-layer callback count',frame)
        frame=Frame();frame.records[-1][2:]=[2,2,0,0,1]
        # Keep END and DONE coherent so only unequal layer-entry coverage rejects.
        frame.end['entries']+=1;frame.end['hits']+=1;frame.end['done_hits']+=1;frame.end['done_look']+=1
        frame.done[9]=str(frame.end['hits']);frame.done[10]=str(frame.end['entries'])
        self.reject('missing/unequal layer route coverage or wrong top-k10',frame)
    def test_done_completion_and_shape_branches(self):
        frame=Frame();frame.done[5]='cancel'
        self.reject('incomplete/cancelled DONE',frame)
        self.reject('incomplete/cancelled DONE',done=Frame().done_text()+' 0')
        for index,value in [(1,'63'),(2,'32767'),(8,'1'),(14,'32767')]:
            frame=Frame();frame.done[index]=value
            self.reject('not a complete fresh64/32K request',frame)
    def test_byte_line_and_frame_budgets_without_large_allocations(self):
        frame=Frame()
        self.reject('interval text budget/type',frame,data='not bytes')
        self.reject('incomplete request stderr interval',frame,data=frame.data()[:-1])
        self.reject('oversized/incomplete stderr line',frame,data=b'x'*65536+b'\n'+frame.data())
        self.reject('noncanonical frame newline',frame,data=frame.data().replace(b'\n',b'\r\n',1))
        self.reject('non-ASCII frame',frame,data=frame.data().replace(b'PAIR 0',b'PAIR \xff',1))
        self.reject('misplaced/noncanonical frame prefix',frame,data=b'junk CACHE_ROUTE_PAIRS_V1 BEGIN\n'+frame.data())
        # Exactly 65 lines of 65536 marker bytes: 4,259,840 bytes. Budget
        # branch triggers during scanning, before the deliberately bad grammar.
        line=b'CACHE_ROUTE_PAIRS_V1 '+b'x'*(65536-22)+b'\n'
        self.assertEqual(len(line),65536)
        self.reject('request pair output exceeds 4 MiB',frame,data=line*65)
        # Record-count bound triggers after bounded scanning, before parsing.
        short=b'CACHE_ROUTE_PAIRS_V1 X\n'
        self.reject('missing/empty/oversized frame',frame,data=short*24580)
        # No >64MiB fixture allocation: interval type branch is separate and
        # whole-job/owned interval enforcement belongs to the controller.

    def test_profile_reference_and_independent_resident_slot_map(self):
        residents,fp=static_profile(PROFILE)
        self.assertEqual(residents,RESIDENTS);self.assertEqual(fp,FNV)
        self.assertEqual(len(residents),128)
        self.assertEqual([sum(l==layer for l,e in residents) for layer in range(48)],[2]*47+[34])
    def test_profile_size_magic_and_geometry(self):
        for data in (PROFILE[:-1],b'NOPE'+PROFILE[4:]):
            with self.assertRaisesRegex(ValueError,'^profile size/magic$'): static_profile(data)
        data=bytearray(PROFILE);struct.pack_into('<I',data,8,47)
        with self.assertRaisesRegex(ValueError,'^profile geometry$'): static_profile(bytes(data))
    def test_profile_pair_bounds_duplicates_and_inverse(self):
        for layer,expert in ((48,0),(0,512)):
            data=bytearray(PROFILE);struct.pack_into('<HH',data,24,layer,expert)
            with self.assertRaisesRegex(ValueError,'^profile bounds/duplicates$'): static_profile(bytes(data))
        data=bytearray(PROFILE);data[28:32]=data[24:28]
        with self.assertRaisesRegex(ValueError,'^profile bounds/duplicates$'): static_profile(bytes(data))
        for value in (-1,1,24576):
            data=bytearray(PROFILE);struct.pack_into('<i',data,24+24576*4,value)
            with self.assertRaisesRegex(ValueError,'^profile inverse rank table$'): static_profile(bytes(data))


if __name__=='__main__': unittest.main()
