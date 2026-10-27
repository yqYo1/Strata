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
    edits = [
        ('#include "strata/kernels/qsa_bf16_prefix.hpp"',
         '#include "strata/kernels/qsa_bf16_prefix.hpp"\n#include "qsa-barrier.hpp"'),
        ('namespace {\nstd::shared_ptr', 'namespace {\n' + adapter + 'std::shared_ptr'),
        (before, before + compare),
        ('#if defined(STRATA_QSA_COMPARE_PREFIX33)\n    // Explicit physical-page alias:',
         '#if 1 // Always exercise row-publication and sticky diagnostic cases.\n'
         '    // Explicit physical-page alias:'),
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
