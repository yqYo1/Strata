"""Reuse the maintained independent oracle for all three QSA barrier arms.

Main-authored reversible test overlay; repository LICENSE. No engine edits.
"""
from qsa_barrier_sources import once, PREFIX, EXPORT

ORACLE = 'tests/sycl/qsa_bf16_prefix.cpp'


def generated(project):
    original = (project / ORACLE).read_text()
    prefix = (project / PREFIX).read_text()
    wrapper = prefix[prefix.index('bool ' + EXPORT + '('):]
    signature = wrapper[:wrapper.index(' {')].replace(EXPORT, 'barrier_entry')
    arguments = ('query, pools, ids, steps, cap, shape, output, queries, bounds, '
                 'live_cells, scratch, stream')
    adapter = ('int tested_arm = 0;\n' + signature + ' {\n'
               '  if (!tested_arm) return ' + EXPORT + '(' + arguments + ');\n'
               '  const auto fn = tested_arm == 1\n'
               '      ? barrier_control::' + EXPORT + '\n'
               '      : barrier_merged::' + EXPORT + ';\n'
               '  return fn(' + arguments + ', nullptr);\n}\n')
    before = '''    require(attention(dq.data(),pools,di.data(),ds.data(),cap,shape,
        out.data()+16,queries,{pages,pages},&rt->compute()),"BF16 matrix unavailable");
    rt->wait();'''
    compare = '''
    const auto expected_all_arms=out.get();const int saved_arm=tested_arm;
    for (int arm : {0,1,2}) if (arm!=saved_arm) {
      tested_arm=arm;
      require(attention(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},&rt->compute()),"barrier arm refused");
      rt->wait();require(out.get()==expected_all_arms,"barrier arm output bits differ");
    }
    tested_arm=saved_arm;'''
    alternating = '''
    { // Distinct K/V witnesses for three consecutive tiles;restore all bytes.
      const auto saved_k=k,saved_v=v;const auto saved_ks=ks,saved_vs=vs;
      for (int cell=0;cell<5;++cell) for (int head=0;head<KV;++head) {
        const size_t row=(size_t(table[0])*KV+head)*Page+cell;
        for (int d=0;d<Dim;++d) {
          k[row*Dim+d]=int8_t(1+cell*7+head*3+d%5);
          v[row*Dim+d]=int8_t((cell%2?-1:1)*(2+cell*11+head*2+d%7));
        }
        for (int g=0;g<4;++g) {
          ks[row*4+g]=half(std::ldexp(1.f,-6-g));
          vs[row*4+g]=half(std::ldexp(1.f,-5-g));
        }
      }
      for (int b=0;b<queries;++b) for (int c=0;c<cap;++c) {
        const int tile=(c/32)%3;
        ids[size_t(b)*cap+c]=tile==2?4:tile*2+((c+b)&1);
      }
      dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);di.put(ids);
      invoke();(void)guarded();immutable();
      k=saved_k;v=saved_v;ks=saved_ks;vs=saved_vs;ids=original_ids;
      dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);di.put(ids);
      invoke();require(guarded()==y,"restore alternating witness");immutable();
      std::cout<<"case alternating_rows=PASS unique_KV_sentinels=PASS cap="<<cap
               <<" queries="<<queries<<" status=PASS\\n";
    }
'''
    mixed = '''
    if (queries>=3) {
      for (int b=0;b<queries;++b)
        steps[b*kStepCount+kStepWidth]=b%3==0?0:b%3==1?1:cap;
      ds.put(steps);invoke();auto mixed=guarded();immutable();
      for (int b=0;b<queries;b+=3) for (int h=0;h<Heads;++h) for (int d=0;d<Dim;++d)
        require(mixed[16+(b*Heads+h)*Dim+d]==0,"mixed zero width");
      std::cout<<"case mixed_widths=PASS cap="<<cap<<" queries="<<queries<<" status=PASS\\n";
    }
    for (int width : {std::min(64,cap),cap-1}) {
      for (int b=0;b<queries;++b) steps[b*kStepCount+kStepWidth]=width;
      ds.put(steps);invoke();(void)guarded();immutable();
    }
    steps=saved_steps;ds.put(steps);
'''
    edits = [
        ('#include "strata/kernels/qsa_bf16_prefix.hpp"',
         '#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include "qsa-barrier.hpp"'),
        ('namespace {\nstd::shared_ptr', 'namespace {\n' + adapter + 'std::shared_ptr'),
        (before, before + compare),
        ('#if defined(STRATA_QSA_COMPARE_PREFIX33)\n    // Explicit physical-page alias:',
         '#if 1 // Always exercise row-publication and sticky diagnostic cases.\n'
         '    // Explicit physical-page alias:'),
        ('table[1]=table[0];dt.put(table);',
         'table[1]=table[0];table[2]=-1;dt.put(table);'),
        ('require(pages>1 && live_cells>1 && cap>32,"row-publication diagnostic geometry");',
         'require(pages>2 && live_cells>32 && cap>32,"row-publication diagnostic geometry");'),
        ('    std::cout<<"case explicit_duplicate_ids=PASS',
         alternating + '    std::cout<<"case explicit_duplicate_ids=PASS'),
        ('    const auto saved_steps=steps;\n',
         '    const auto saved_steps=steps;\n' + mixed),
        ('{1,17,31,32,63,65,2051}', '{1,17,31,32,63,64,65,2051}'),
        ('      run(262144,2051,1,false,false);',
         '      run(97,65,3,false,true,71);run(2113,2051,3,false,true);\n'
         '      run(262144,2051,1,false,false);'),
        ('require(argc==2 &&', 'require(argc==3 &&'),
        ('    const char *g=std::getenv("STRATA_TEST_MAIN_GATES");',
         '    tested_arm=std::stoi(argv[2]);require(tested_arm>=0 && tested_arm<3,"arm0..2");\n'
         '    const char *g=std::getenv("STRATA_TEST_MAIN_GATES");'),
        ('           <<" packed_guard_elements="',
         '           <<" exact_barrier_arms=PASS tested_arm="<<tested_arm\n'
         '           <<" packed_guard_elements="')]
    # Replace only calls in the original test, before adding the adapter itself.
    calls = original.count('qsa_bf16_prefix_batch(')
    if calls < 5:
        raise ValueError('oracle admission coverage changed')
    value = original.replace('qsa_bf16_prefix_batch(', 'barrier_entry(')
    for before, after in edits:
        value = once(value, before, after)
    restored = value
    for before, after in reversed(edits):
        restored = once(restored, after, before)
    restored = restored.replace('barrier_entry(', 'qsa_bf16_prefix_batch(')
    if restored != original:
        raise ValueError('oracle overlay changed another source byte')
    return {'barrier-checks.cpp': value.encode()}
