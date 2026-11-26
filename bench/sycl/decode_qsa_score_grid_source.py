"""Generate the decode scorer candidate from the exact measured SYCL source."""

import hashlib


SOURCE = 'src/kernels/sycl/qsa_select.cpp'
SOURCE_SHA256 = '463a9ec02ed28125a5cd38b7b5d28784b849a4fbb199cb017042cd99078efee1'
GROUPS = 4096


def generate(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('expected the retained native QSA translation unit')
    source = raw.decode()
    validation = source[source.index('int64_t validate('):source.index('uint32_t order(')]
    start = source.index('void qsa_block_scores(')
    end = source.index('void qsa_block_scores_prefill(', start)
    score = source[start:end]
    score = score.replace('  auto event = queue_for(stream).parallel_for(',
                          f'  constexpr int64_t grid_limit = {GROUPS};\n'
                          '  const int64_t grid = launched < grid_limit ? launched : grid_limit;\n'
                          '  auto event = queue_for(stream).parallel_for(', 1)
    score = score.replace('size_t(nq * launched) * 32', 'size_t(nq * grid) * 32', 1)
    score = score.replace('it.get_group_linear_id() / launched,\n'
                          '                      b = it.get_group_linear_id() % launched;',
                          'it.get_group_linear_id() / grid,\n'
                          '                      first = it.get_group_linear_id() % grid;', 1)
    score = score.replace(' || nb >= launched || b > nb)', ' || nb >= launched)', 1)
    body_start = score.index('        const float *key = b == nb')
    body_end = score.index('      });', body_start)
    body = score[body_start:body_end]
    score = (score[:body_start] +
             '        // The device steps and group IDs are uniform across the subgroup.\n'
             '        for (int64_t b = first; b <= nb; b += grid) {\n' +
             ''.join('  ' + line for line in body.splitlines(keepends=True)) +
             '        }\n' + score[body_end:])
    if 'it.get_group_linear_id() % grid' not in score or 'b > nb' in score:
        raise ValueError('decode scorer grid transform did not apply')
    return ('#include "strata/kernels/qsa_select.hpp"\n'
            '#include "strata/sycl/launch.hpp"\n'
            '#include <climits>\n\n'
            'namespace strata::kernels::decode_score_grid {\n'
            'using namespace sycl_backend;\n'
            'namespace {\n' + validation + '}\n' + score +
            '} // namespace strata::kernels::decode_score_grid\n').encode()
