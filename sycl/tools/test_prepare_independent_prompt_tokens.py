"""Independent fake host fixtures; never actual pack-tokenization proof."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
import prepare_independent_prompt_tokens as prep


class FakeTokenizer:
    def encode(self,text): return [ord(c) for c in text]


def render(body): return 'U:'+body+':A'


def manifest():
    a=b'alpha complete document';b=b'other unrelated document'
    def entry(path,source,data):
        return dict(path=path,source_id=source,sha256=prep.digest(data),version='v1')
    return dict(schema='independent-documents-v1',selection_rule='manifest-order-first-complete-prefix',model_identity='literal-fake-model',corpora=dict(train=[entry('a','train-source',a)],validation=[entry('b','validation-source',b)])),{'a':a,'b':b}


class PromptFixtures(unittest.TestCase):
    def test_complete_document_first_prefix(self):
        docs=[({'source_id':'first'},'abcd'),({'source_id':'second'},'efgh'),({'source_id':'unused'},'ijkl')]
        got=prep.select_prompt(docs,render,FakeTokenizer().encode,256,minimum=10,maximum=20)
        self.assertEqual(got['rendered'],'U:abcd\n\nefgh:A')
        self.assertEqual(got['ids'],[85,58,97,98,99,100,10,10,101,102,103,104,58,65])
        self.assertEqual(got['selected_documents'],[{'source_id':'first'},{'source_id':'second'}])

    def test_overage_insufficient_and_id_bounds(self):
        with self.assertRaisesRegex(ValueError,'overage'):
            prep.select_prompt([({},'abcdefghijk')],render,FakeTokenizer().encode,256,minimum=10,maximum=12)
        with self.assertRaisesRegex(ValueError,'insufficient'):
            prep.select_prompt([({},'a')],render,FakeTokenizer().encode,256,minimum=10,maximum=20)
        with self.assertRaisesRegex(ValueError,'token ID bounds'):
            prep.select_prompt([({},'a')],render,lambda _: [True]*10,256,minimum=10,maximum=20)
        with self.assertRaisesRegex(ValueError,'token array budget'):
            prep.checked_ids([0]*(prep.MAX_TOKENS+1),256)

    def test_declared_disjoint_documents(self):
        m,files=manifest()
        docs=prep.documents(m,lambda path,sha,cap: files[path])
        self.assertEqual(docs['train'][0][1],'alpha complete document')
        self.assertEqual(docs['validation'][0][1],'other unrelated document')
        for field,branch in (('source_id','source ID'),('sha256','document hash')):
            bad=copy.deepcopy(m);bad['corpora']['validation'][0][field]=bad['corpora']['train'][0][field]
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,branch):
                prep.documents(bad,lambda path,sha,cap: files[path])
        bad=copy.deepcopy(m);bad['corpora']['train'].append(copy.deepcopy(bad['corpora']['train'][0]))
        with self.assertRaisesRegex(ValueError,'source ID'): prep.documents(bad,lambda path,sha,cap: files[path])

    def test_missing_hash_invalid_utf8_and_budget(self):
        m,files=manifest();m['corpora']['train'][0]['sha256']=''
        with self.assertRaisesRegex(ValueError,'SHA256'): prep.documents(m,lambda path,sha,cap: files[path])
        m,files=manifest();files['a']=b'\xff';m['corpora']['train'][0]['sha256']=prep.digest(files['a'])
        with self.assertRaises(UnicodeDecodeError): prep.documents(m,lambda path,sha,cap: files[path])
        m,files=manifest();files['a']=b'x'*(prep.DOC_CAP+1);m['corpora']['train'][0]['sha256']=prep.digest(files['a'])
        with self.assertRaisesRegex(ValueError,'document bounds'): prep.documents(m,lambda path,sha,cap: files[path])
        # Total corpus budget branch using five individually bounded distinct docs.
        m,files=manifest();m['corpora']['train']=[]
        for i in range(5):
            data=bytes([97+i])*prep.DOC_CAP;path=str(i);files[path]=data
            m['corpora']['train'].append(dict(path=path,source_id=path,sha256=prep.digest(data),version='v1'))
        with self.assertRaisesRegex(ValueError,'total corpus budget'): prep.documents(m,lambda path,sha,cap: files[path])

    def test_actual_token_near_overlap_and_common_prefix(self):
        a=list(range(100));b=[0,1]+list(range(1000,1098))
        result=prep.overlap_gate(a,b)
        self.assertEqual((result['common_prefix_ids'],result['shared_shingles'],result['matching_coordinates']),(2,0,2))
        with self.assertRaisesRegex(ValueError,'near-overlap'): prep.overlap_gate(a,a[:-1]+[500])
        with self.assertRaisesRegex(ValueError,'near-overlap'): prep.overlap_gate(a,[999]+a)
        with self.assertRaisesRegex(ValueError,'ID type'): prep.overlap_gate([True]*100,b)

    def test_output_private_exclusive_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'ok';prep.write_bundle(root,{'train.tokens.txt':b'1 2\n','record.json':b'{}\n'})
            self.assertEqual((root/'train.tokens.txt').read_bytes(),b'1 2\n')
            self.assertEqual(root.stat().st_mode & 0o777,0o700)
            self.assertEqual((root/'record.json').stat().st_mode & 0o777,0o600)
            with self.assertRaises(FileExistsError): prep.write_bundle(root,{'record.json':b'{}\n'})
            failed=Path(tmp)/'failed'
            def break_write(fd,data):
                os.write(fd,data[:1]);raise OSError('literal partial write failure')
            with self.assertRaisesRegex(OSError,'partial write'):
                prep.write_bundle(failed,{'train.tokens.txt':b'1 2\n','record.json':b'{}\n'},break_write)
            self.assertFalse((failed/'record.json').exists())
            self.assertEqual((failed/'train.tokens.txt').read_bytes(),b'1')
            with self.assertRaisesRegex(ValueError,'total output budget'):
                prep.write_bundle(Path(tmp)/'oversized',{'record.json':b'x'*(prep.OUTPUT_CAP+1)})
            self.assertFalse((Path(tmp)/'oversized').exists())

    def test_verified_reader_hash_utf8_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'document';path.write_bytes(b'literal UTF8 document')
            self.assertEqual(prep.read_verified(path,prep.digest(path.read_bytes()),100),b'literal UTF8 document')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'): prep.read_verified(path,'0'*64,100)
            with self.assertRaisesRegex(ValueError,'bounded file'): prep.read_verified(path,prep.digest(path.read_bytes()),1)
            link=Path(tmp)/'link';link.symlink_to(path)
            with self.assertRaises(OSError): prep.read_verified(link,prep.digest(path.read_bytes()),100)

    def test_duplicate_json_keys(self):
        with self.assertRaisesRegex(ValueError,'duplicate JSON key'): prep.parse_json(b'{"corpora":{},"corpora":{}}')


if __name__=='__main__': unittest.main()
