"""Independent literal rules fixtures; no tokenizer, model or controller import."""
import copy
import unittest
import long_prompt_reference_rules_v1 as rules


def literal_request(n=36004,generated=1,finish='stop'):
    endpoints={36004:[8192,16384,24576,32768,36003],37462:[8192,16384,24576,32768,37461],32768:[8192,16384,24576,32767],65536:[8192,16384,24576,32768,40960,49152,57344,65535]}[n]
    lines=['RESUME 0']+['PP %d %d 100.0 1.0'%(x,n) for x in endpoints]+['REUSED 0']
    lp='LP -0.5 40:-0.5 41:-1.0 42:-2.0 43:-3.0 44:-4.0'
    for _ in range(generated): lines.extend(['T 40',lp])
    lines.append('DONE %d %d 100.0 20.0 %s 0 1 0 3 10 0 0 0.0 %d 0'%(generated,n,finish,n))
    return dict(protocol=lines,ids=[40]*generated,logprobs=[lp]*generated)


class ReferenceRules(unittest.TestCase):
    def test_variable_whole_inputs_progress(self):
        a=rules.validate_request(literal_request(),36004)
        b=rules.validate_request(literal_request(37462),37462)
        self.assertEqual(a['progress'],[8192,16384,24576,32768,36003])
        self.assertEqual(b['progress'],[8192,16384,24576,32768,37461])
        self.assertEqual((a['generated'],b['prompt_tokens']),(1,37462))

    def test_min_max_and_early_stop(self):
        for n in (32768,65536):
            self.assertTrue(rules.validate_request(literal_request(n,1,'stop'),n)['passed'])
            self.assertEqual(rules.validate_request(literal_request(n,64,'length'),n)['generated'],64)
        with self.assertRaisesRegex(ValueError,'length finish'): rules.validate_request(literal_request(36004,1,'length'),36004)
        for n in (32767,65537,True):
            with self.assertRaisesRegex(ValueError,'length bounds'): rules.expected_progress(n)

    def test_malformed_done_and_nonfinite(self):
        for replacement,branch in (('DONE 1 36004','exact16'),
                                  ('DONE 0 36004 1.0 1.0 stop 0 1 0 3 10 0 0 0.0 36004 0','count/prompt'),
                                  ('DONE 1 36004 nan 1.0 stop 0 1 0 3 10 0 0 0.0 36004 0','finite DONE'),
                                  ('DONE 1 36004 1.0 1.0 stop 2 1 0 3 10 0 0 0.0 36004 0','MTP/fresh')):
            request=literal_request();request['protocol'][-1]=replacement
            with self.subTest(branch=branch),self.assertRaisesRegex(ValueError,branch): rules.validate_request(request,36004)

    def test_progress_and_freshness_negative(self):
        for mutate,branch in ((lambda r:r['protocol'].__setitem__(0,'RESUME 1'),'fresh RESUME'),
                              (lambda r:r['protocol'].__setitem__(5,'PP 36004 36004 1.0 1.0'),'prefill progress'),
                              (lambda r:r['protocol'].__setitem__(6,'REUSED 1'),'fresh REUSED'),
                              (lambda r:r['protocol'].insert(6,'PP 36003 36004 1.0 1.0'),'fresh REUSED')):
            request=literal_request();mutate(request)
            with self.subTest(branch=branch),self.assertRaisesRegex(ValueError,branch): rules.validate_request(request,36004)

    def test_ids_all_lp_and_nonfinite(self):
        request=literal_request();request['ids']=[41]
        with self.assertRaisesRegex(ValueError,'stored IDs'): rules.validate_request(request,36004)
        request=literal_request();request['protocol'][8]='LP -0.5 40:-0.5 41:-1.0'
        with self.assertRaisesRegex(ValueError,'all5 LP'): rules.validate_request(request,36004)
        request=literal_request();request['protocol'][8]='LP -0.5 40:-0.5 41:nan 42:-2.0 43:-3.0 44:-4.0'
        with self.assertRaisesRegex(ValueError,'candidate syntax'): rules.validate_request(request,36004)
        request=literal_request();request['protocol'][7]='T 248320'
        with self.assertRaisesRegex(ValueError,'ID bound'): rules.validate_request(request,36004)

    def test_exact_repeat_distinctions(self):
        a=dict(ids=[40],logprobs=['literal LP'],mtp_counts=[0,1],finish_reason='stop',first_head=dict(sha256='a'*64))
        live=dict(part_evidence=[{} for _ in range(66)],different_live_parts=[])
        self.assertTrue(all(rules.repeat_checks(a,copy.deepcopy(a),live).values()))
        for field,value,key in (('ids',[41],'ids_equal'),('logprobs',['changed'],'all_LP_equal'),('mtp_counts',[1,1],'mtp_equal'),('finish_reason','length','finish_equal'),('first_head',dict(sha256='b'*64),'first_head_equal')):
            b=copy.deepcopy(a);b[field]=value
            checks=rules.repeat_checks(a,b,live)
            self.assertFalse(checks[key]);self.assertEqual(sum(not v for v in checks.values()),1)
        self.assertFalse(rules.repeat_checks(a,a,dict(live,different_live_parts=[65]))['all66_live_state_parts_equal'])
        self.assertFalse(rules.repeat_checks(a,a,dict(live,part_evidence=[{}]*65))['all66_live_state_parts_equal'])

    def test_literal_input_keys_and_hashes(self):
        rules.input_identity('train',36004,'ddeded38b2f8e2fd10e735a9f6d382434049566bd8d891534fa7f6aa84c24796')
        rules.input_identity('validation',37462,'1c7eb5878f521890c5f7e511bea7d1f73146b61a784d6c86169c16713bd36d3c')
        for key,n,sha in (('A',36004,'ddeded38b2f8e2fd10e735a9f6d382434049566bd8d891534fa7f6aa84c24796'),('train',32768,'ddeded38b2f8e2fd10e735a9f6d382434049566bd8d891534fa7f6aa84c24796'),('train',36004,'0'*64)):
            with self.assertRaisesRegex(ValueError,'frozen input'): rules.input_identity(key,n,sha)

    def test_state_geometry_literal(self):
        sizes=rules.state_sizes(36003)
        self.assertEqual(sizes[:6],[8,117669888,368640,18432,6144,48])
        self.assertEqual(sizes[6:11],[18434048,18434048,576064,576064,4608512])
        self.assertEqual(len(sizes),66)
        self.assertEqual(sizes[6:11],sizes[-5:])

    def test_finite_head_exact_geometry(self):
        import struct
        valid=b'\0'*(248320*4)
        self.assertEqual(rules.finite_head(valid),dict(floats=248320,finite=True))
        with self.assertRaisesRegex(ValueError,'exact bytes'): rules.finite_head(valid[:-4])
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaisesRegex(ValueError,'finite floats'):
                rules.finite_head(struct.pack('=f',value)+valid[4:])


if __name__=='__main__': unittest.main()
