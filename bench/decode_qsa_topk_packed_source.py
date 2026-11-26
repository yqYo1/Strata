"""Project an exact two-bit selector onto the retained SYCL QSA source."""

import hashlib


SOURCE = 'src/kernels/sycl/qsa_select.cpp'
SOURCE_SHA256 = '463a9ec02ed28125a5cd38b7b5d28784b849a4fbb199cb017042cd99078efee1'
FIXTURE_SHA256 = '198ae6ae67db3fd59ff595e6362b1e62483d91d47b37fe3ecd2cc2cffd3a0327'
CAPACITY = 262144
FIELD_BITS = 19

THRESHOLD = r"""        uint32_t threshold = 0, prefix = 0;
        int rank = int(width);
        constexpr int field_bits = @FIELD_BITS@;
        constexpr uint64_t field_mask = (uint64_t(1) << field_bits) - 1;
        static_assert(@CAPACITY@ <= field_mask && 3 * field_bits <= 64);
        // Three disjoint weighted counts. Their whole-group sums cannot
        // carry into another field. Digit zero is the implicit remainder.
        for (int shift = 30; shift >= 0; shift -= 2) {
          uint64_t counts = 0;
          for (int64_t b = lo; b < hi; ++b) {
            const uint32_t key = order(sc[b]);
            if ((key & prefix) != threshold)
              continue;
            const uint32_t digit = (key >> shift) & 3u;
            if (digit)
              counts += uint64_t(weight(b)) << (field_bits * (digit - 1));
          }
          counts = sycl::reduce_over_group(it.get_group(), counts, sycl::plus<uint64_t>());
          uint32_t digit = 3;
          for (; digit > 0; --digit) {
            const int count = int((counts >> (field_bits * (digit - 1))) & field_mask);
            if (rank <= count)
              break;
            rank -= count;
          }
          threshold |= digit << shift;
          prefix |= 3u << shift;
        }
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('unexpected source preimage: ' + old.splitlines()[0])
    return text.replace(old, new)


def generate(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('expected the retained native QSA source')
    text = raw.decode()
    helpers = text[text.index('int64_t validate('):text.index('// Independently qualified register-only')]
    begin = text.index('sycl::event submit_topk(')
    finish = text.index('int prefill_selection_mapping()', begin)
    selector = text[begin:finish]
    first = selector.index('        uint32_t threshold = 0;')
    last = selector.index('        int gt = 0, eq = 0;', first)
    threshold = THRESHOLD.replace('@CAPACITY@', str(CAPACITY)).replace('@FIELD_BITS@', str(FIELD_BITS))
    selector = selector[:first] + threshold + selector[last:]
    selector = replace_once(selector, 'if (n < 0 || nb != n / 4',
                            f'if (n < 0 || n > {CAPACITY} || nb != n / 4')
    begin = text.index('void qsa_block_topk(')
    finish = text.index('void qsa_block_topk_prefill(', begin)
    entry = replace_once(text[begin:finish], 'submit_topk<0>', 'submit_topk')
    return ('#include "strata/kernels/qsa_select.hpp"\n'
            '#include "strata/sycl/launch.hpp"\n'
            '#include <climits>\n\n'
            'namespace strata::kernels::decode_topk_packed {\n'
            'using namespace sycl_backend;\n'
            'namespace {\n' + helpers + selector + '}\n' + entry +
            '} // namespace strata::kernels::decode_topk_packed\n').encode()


KEYS = r"""    auto compare_keys = [&](int context, int pivot) {
        set_steps(context);
        // Exercise the selector's ordering of negative and positive scores,
        // signed zero, infinities, NaNs and tied live or partial blocks.
        constexpr uint32_t pattern[] = {
            0x7fc00000u, 0xffc00001u, 0xff800000u, 0x7f800000u,
            0xbf800000u, 0x80000000u, 0u, 0x3f800000u,
        };
        auto input = unset_scores;
        for (int q = 0; q < width; ++q)
            for (int b = 0; b <= steps[q * kStepCount + kStepNBid]; ++b) {
                uint32_t key = b % 257 == 0 ? pattern[3] : pattern[0];
                if (b % 17 == 0)
                    key = pivot == 5 || pivot == 6 ? pattern[5 + b % 2] : pattern[pivot];
                if (pivot == 3)
                    key = pattern[3]; // All 262,144 cells can occupy the highest count field.
                input[guard + q * blocks + b] = std::bit_cast<float>(key);
            }
        scores.put(input); ids.put(unset_ids);
        graphs.run(0);
        const auto reference = ids.get();
        scores.put(input); ids.put(unset_ids);
        graphs.run(1);
        const auto actual = ids.get();
        const auto unchanged = scores.get();
        require(actual == reference && std::memcmp(input.data(), unchanged.data(), input.size() * sizeof(float)) == 0,
                "raw selector keys or selected IDs changed");
        check_outputs(width, cap, steps, unchanged, actual, false);
    };

"""


def generate_fixture(raw):
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError('expected the retained graph fixture')
    fixture = raw.decode().replace('#include <array>\n', '#include <array>\n#include <bit>\n', 1)
    begin = fixture.index('namespace strata::kernels::decode_score_grid {')
    finish = fixture.index('namespace {', begin)
    fixture = fixture[:begin] + '''namespace strata::kernels::decode_topk_packed {
void qsa_block_topk(const float*, const int32_t*, int64_t, int64_t, int64_t, const QsaShapes&,
                    int32_t*, void*, int64_t);
}

''' + fixture[finish:]
    fixture = replace_once(fixture, 'const std::vector<float>& scores, const std::vector<int32_t>& ids) {',
                           'const std::vector<float>& scores, const std::vector<int32_t>& ids, bool scored = true) {')
    fixture = replace_once(fixture, '        for (int b = 0; b < live; ++b) require(std::isfinite',
                           '        if (scored) for (int b = 0; b < live; ++b) require(std::isfinite')
    begin = fixture.index('        const bool grid = slot % 2')
    finish = fixture.index('        });', begin)
    fixture = fixture[:begin] + r"""        const bool candidate = slot % 2, with_scorer = slot >= 2;
        setups[slot] = graphs.capture(slot, with_scorer ? 2 : 1, profile, [&](void* stream) {
            if (with_scorer)
                qsa_block_scores(kp.data(), kd.data(), qq.data(), ss.data(), width, blocks, shape,
                                 scores.data() + guard, stream, -1);
            auto select = candidate ? decode_topk_packed::qsa_block_topk : qsa_block_topk;
            select(scores.data() + guard, ss.data(), width, blocks, cap, shape, ids.data() + guard, stream, -1);
""" + fixture[finish:]
    fixture = replace_once(fixture, '    if (mode == "smoke") {', KEYS + '    if (mode == "smoke") {')
    fixture = replace_once(fixture, '        compare(97); compare(98); compare(97); compare(-1);',
                           '        compare(97); compare(2055); compare_keys(2055, 4); compare(97); compare(-1);')
    fixture = replace_once(fixture, "        compare(96'203); compare(capacity); compare(96'203); compare(-1);",
                           "        compare(96'203); compare(capacity); compare(96'203); compare(-1);\n"
                           "        for (int pivot = 0; pivot < 8; ++pivot)\n"
                           "            compare_keys(pivot == 3 ? capacity : capacity - pivot % 4, pivot);")
    return fixture.encode()
