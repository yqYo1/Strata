"""Independent fake host fixtures; never actual pack-tokenization proof."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
import prepare_independent_prompt_tokens_v2 as prep


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
            with self.assertRaises((OSError,ValueError)): prep.read_verified(link,prep.digest(path.read_bytes()),100)

    def test_duplicate_json_keys(self):
        with self.assertRaisesRegex(ValueError,'duplicate JSON key'): prep.parse_json(b'{"corpora":{},"corpora":{}}')

    def test_fifo_precheck_never_opens(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            fifo=Path(tmp)/'fifo';os.mkfifo(fifo)
            with patch.object(prep.os,'open',side_effect=AssertionError('FIFO open forbidden')):
                with self.assertRaisesRegex(ValueError,'regular file precheck'):
                    prep.read_verified(fifo,'0'*64,100)

    def test_unused_tail_is_still_hash_checked(self):
        m,files=manifest()
        tail=dict(path='tail',source_id='unused-source',sha256='1'*64,version='v1')
        m['corpora']['train'].append(tail);files['tail']=b'unused malformed hash'
        with self.assertRaisesRegex(ValueError,'document bounds/hash'):
            prep.documents(m,lambda path,sha,cap: files[path])

    def test_multibyte_metadata_utf8_boundary(self):
        m,files=manifest();m['corpora']['train'][0]['version']='é'*2048
        prep.documents(m,lambda path,sha,cap: files[path])
        m['corpora']['train'][0]['version']+='é'
        with self.assertRaisesRegex(ValueError,'document metadata'):
            prep.documents(m,lambda path,sha,cap: files[path])

    def test_r63_long_prefix_nearcopy(self):
        a=list(range(32768));b=a[:32719]+list(range(100000,100040))+a[32759:]
        self.assertEqual(len(b),32768)
        self.assertEqual(sum(x!=y for x,y in zip(a,b)),40)
        with self.assertRaisesRegex(ValueError,'near-overlap'): prep.overlap_gate(a,b)

    def test_exact_shingle_threshold_and_below(self):
        # Length115 has100 unique16-shingles. Prefix95 shares80 windows;
        # prefix94 shares79. Distinct tails prevent incidental extra matches.
        a=list(range(115))
        exact=a[:95]+list(range(1000,1020))
        below=a[:94]+list(range(2000,2021))
        with self.assertRaisesRegex(ValueError,'near-overlap'): prep.overlap_gate(a,exact)
        result=prep.overlap_gate(a,below)
        self.assertEqual((result['shared_shingles'],result['containment_denominator']),(79,100))
        self.assertEqual(result['common_prefix_ids'],94)

    def test_degenerate_low_unique_windows(self):
        with self.assertRaisesRegex(ValueError,'near-overlap'): prep.overlap_gate([7]*100,[7]*16+[8]*100)
        result=prep.overlap_gate([7]*100,[8]*100)
        self.assertEqual((result['train_unique_shingles'],result['validation_unique_shingles'],result['shared_shingles']),(1,1,0))
        with self.assertRaisesRegex(ValueError,'overlap length'): prep.overlap_gate([7]*15,[8]*16)
        self.assertEqual(prep.overlap_gate([7]*16,[8]*16)['containment_denominator'],1)

    def test_rendered_ceiling_before_encoder(self):
        def forbidden(_): raise AssertionError('encoder must not run')
        with self.assertRaisesRegex(ValueError,'rendered text budget'):
            prep.select_prompt([({},'x')],lambda _: 'é'*(prep.RENDERED_CAP//2+1),forbidden,256,minimum=1,maximum=2)

    def test_final_candidate_write_fsync_close_failures(self):
        from unittest.mock import patch
        # Literal candidate cannot self-certify success, even if parseable after
        # fsync/close failure. Root must separately inspect child exit and hashes.
        candidate=b'{"completed":false,"ready_for_owner_validation":true}\n'
        actual_open=os.open;actual_write=os.write;actual_fsync=os.fsync;actual_close=os.close
        for failure in ('write','fsync','close'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'candidate';fds={}
                def open_file(path,*args,**kwargs):
                    fd=actual_open(path,*args,**kwargs);fds[fd]=Path(path).name;return fd
                def write_file(fd,data):
                    if fds.get(fd)=='record.json' and failure=='write':
                        actual_write(fd,data[:4]);raise OSError('final write failure')
                    return actual_write(fd,data)
                def sync_file(fd):
                    if fds.get(fd)=='record.json' and failure=='fsync': raise OSError('final fsync failure')
                    return actual_fsync(fd)
                def close_file(fd):
                    name=fds.pop(fd,None);actual_close(fd)
                    if name=='record.json' and failure=='close': raise OSError('final close failure')
                with patch.object(prep.os,'open',open_file),patch.object(prep.os,'write',write_file),patch.object(prep.os,'fsync',sync_file),patch.object(prep.os,'close',close_file):
                    with self.assertRaisesRegex(OSError,'final '+failure+' failure'):
                        prep.write_bundle(out,{'train.tokens.txt':b'1 2\n','record.json':candidate})
                raw=(out/'record.json').read_bytes()
                if failure=='write': self.assertEqual(raw,b'{"co')
                else:
                    parsed=prep.parse_json(raw)
                    self.assertIs(parsed['completed'],False)
                    self.assertIs(parsed['ready_for_owner_validation'],True)

    def test_named_special_config_literal(self):
        tokens=['ordinary']*248320;types_=[1]*248320
        for name,index,kind in (('<|endoftext|>',248044,3),('<|im_start|>',248045,3),('<|im_end|>',248046,3),('<tool_call>',248058,4),('<tool_response>',248066,4),('<think>',248068,4)):
            tokens[index]=name;types_[index]=kind
        for index in range(248077,248320): tokens[index]='[PAD%d]'%index;types_[index]=5
        cfg={'special_ids':{'tokenizer.ggml.bos_token_id':248044,'tokenizer.ggml.eos_token_id':248046,'tokenizer.ggml.padding_token_id':248044}}
        prep.special_config_gate(cfg,tokens,types_)
        wrong=copy.deepcopy(cfg);wrong['special_ids']['tokenizer.ggml.eos_token_id']=248044
        with self.assertRaisesRegex(ValueError,'BOS/EOS/padding'): prep.special_config_gate(wrong,tokens,types_)
        types_[248068]=3
        with self.assertRaisesRegex(ValueError,'representative special'): prep.special_config_gate(cfg,tokens,types_)


if __name__=='__main__': unittest.main()
