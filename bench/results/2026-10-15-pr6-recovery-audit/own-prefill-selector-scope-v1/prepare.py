#!/usr/bin/env python3
"""Recreate proposal files and patch only from the pinned package snapshot.

Does not apply changes, run qualification, build, or access the GPU.
"""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text())


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("snapshot anchor is not unique: " + old[:100])
    return text.replace(old, new, 1)


originals = {}
for item in manifest["base_files"]:
    raw = (HERE / "base" / item["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise RuntimeError("base hash mismatch: " + item["path"])
    originals[item["path"]] = raw.decode()
proposals = dict(originals)

path = "include/strata/kernels/qsa_select.hpp"
proposals[path] = replace_once(proposals[path],
    "/// The same ids on a thread-block cluster of 8 CTAs per query (sm_90+, CUDA; S19).",
    """#if defined(STRATA_ENABLE_SYCL)
/// Dedicated layer-major long-prefill selector. Geometry, active-block bound,
/// output and validation contracts match qsa_block_topk. This is the only SYCL
/// entrypoint that reads STRATA_SYCL_SELECT_RADIX4 / STRATA_SYCL_SELECT_CYCLIC:
/// exact \"1\" enables an arm; both enabled is invalid; otherwise Mapping0.
/// Set flags before exec and keep the environment immutable after runtime init.
/// Short prefill, decode, verification and reference callers use qsa_block_topk
/// or qsa_block_topk_ref, which always retain Mapping0 and read neither flag.
void qsa_block_topk_prefill(const float* scores, const int32_t* steps, int64_t nq,
                           int64_t max_blocks, int64_t cap, const QsaShapes& s,
                           int32_t* ids, void* stream, int64_t active_blocks = -1);
#endif
/// The same ids on a thread-block cluster of 8 CTAs per query (sm_90+, CUDA; S19).""")

path = "src/kernels/sycl/qsa_select.cpp"
text = proposals[path]
validation = """  const int64_t launched = validate(nq, blocks, active, s);
  if (cap <= 0 || cap > INT_MAX)
    throw std::invalid_argument("invalid SYCL block selection capacity");
  validate_spans({{ids, checked_count(nq, cap) * 4}},
                 {{scores, checked_count(nq, blocks) * 4},
                  {steps, checked_count(nq, kStepCount) * 4}});
"""
text = replace_once(text, "int selection_mapping() {", """int64_t validate_topk(const float *scores, const int32_t *steps, int64_t nq,
                      int64_t blocks, int64_t cap, const QsaShapes &s,
                      int32_t *ids, int64_t active) {
""" + validation + """  return launched;
}

int prefill_selection_mapping() {""")
text = replace_once(text, "  // Experimental, default off. Set before exec; process environment must",
    "  // Dedicated layer-major prefill only, default off. Set before exec; environment must")
start = text.index("void qsa_block_topk(const float *scores")
end = text.index("void qsa_block_topk_ref(", start)
old_dispatch = text[start:end]
dedicated = old_dispatch.replace("void qsa_block_topk(", "void qsa_block_topk_prefill(", 1)
dedicated = replace_once(dedicated, validation,
    "  const int64_t launched = validate_topk(scores, steps, nq, blocks, cap, s, ids, active);\n")
dedicated = replace_once(dedicated, "const int mapping = selection_mapping();",
                         "const int mapping = prefill_selection_mapping();")
original = """void qsa_block_topk(const float *scores, const int32_t *steps, int64_t nq,
                    int64_t blocks, int64_t cap, const QsaShapes &s,
                    int32_t *ids, void *stream, int64_t active) {
  const int64_t launched = validate_topk(scores, steps, nq, blocks, cap, s, ids, active);
  auto &queue = queue_for(stream);
  const auto event = submit_topk<0>(queue, scores, steps, nq, blocks, cap, ids, launched);
  finish(stream, event);
}
"""
proposals[path] = text[:start] + original + dedicated + text[end:]

path = "src/prefill/prefill.cpp"
text = proposals[path]
# Both the production selector and the optional FP16-key diagnostic stay in the
# same phase scope. Non-SYCL preprocessing leaves each original call unchanged.
for tail in ("m.sel_ids + t0 * m.cap, m.cs, active);", "ids16 + t0 * m.cap, m.cs);"):
    lines = text.splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if tail in line]
    if len(indices) != 1:
        raise RuntimeError("prefill call anchor not unique: " + tail)
    i = indices[0] - 1
    old = "".join(lines[i:i + 2])
    if "strata::kernels::qsa_block_topk(" not in old:
        raise RuntimeError("unexpected call layout")
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    new = "#if defined(STRATA_ENABLE_SYCL)\n" + indent + "if (m.layer_major_active)\n"
    new += "".join("    " + line.replace("qsa_block_topk(", "qsa_block_topk_prefill(", 1)
                   for line in lines[i:i + 2])
    new += indent + "else\n#endif\n" + old
    text = replace_once(text, old, new)
proposals[path] = text

path = "tests/sycl/qsa_select_radix.cpp"
text = proposals[path]
text = replace_once(text,
    "    qsa_block_topk(ds.data() + guard, dt.data(), nq, blocks, cap,",
    "    qsa_block_topk_prefill(ds.data() + guard, dt.data(), nq, blocks, cap,")
text = replace_once(text,
    "  auto validate = [&](const std::vector<int32_t> &actual) {",
    "  auto validate = [&](const std::vector<int32_t> &actual,\n"
    "                      const std::vector<std::vector<int32_t>> &references) {")
text = replace_once(text, "      const auto &ref = expected[size_t(q)];",
                         "      const auto &ref = references[size_t(q)];")
text = replace_once(text, "  validate(first);", "  validate(first, expected);")
text = replace_once(text, "    validate(repeated);", "    validate(repeated, expected);")
text = replace_once(text, "got[j] < rows[size_t(q)].n,",
                         "got[j] < steps[size_t(q) * kStepCount + kStepNKv],")
text = replace_once(text, "  const auto scores_after = ds.get();", """  // Public decode/verification/short-prefill API and the reference entrypoint
  // must keep the original result under each pre-exec candidate environment.
  // Reuse output storage and keep these full-oracle/guard checks outside timing.
  out.put(std::vector<int32_t>(output_size, sentinel));
  qsa_block_topk(ds.data() + guard, dt.data(), nq, blocks, cap,
                 qsa_real_shapes(), out.data() + guard,
                 &runtime->compute(), active);
  runtime->wait();
  const auto original = out.get();
  validate(original, expected);
  check(original == first, label + ": public original differs from prefill");
  // _ref has no active-block argument. An invalid-active-bound row can be
  // valid at its full capacity; compute its actual contract's oracle separately.
  auto reference_expected = expected;
  for (int64_t q = 0; q < nq; ++q) {
    if (rows[size_t(q)].invalid != 3) continue;
    const auto *st = steps.data() + q * kStepCount;
    reference_expected[size_t(q)] = oracle(scores.data() + guard + q * blocks,
                                          st[kStepNKv], st[kStepWidth]);
  }
  out.put(std::vector<int32_t>(output_size, sentinel));
  qsa_block_topk_ref(ds.data() + guard, dt.data(), nq, blocks, cap,
                     qsa_real_shapes(), out.data() + guard, &runtime->compute());
  runtime->wait();
  validate(out.get(), reference_expected);
  std::cout << "PASS public_original_and_ref label=" << label << " nq=" << nq << '\\n';
  const auto scores_after = ds.get();""")
text = replace_once(text, "void validation() {", "void validation(bool dedicated_prefill) {")
text = replace_once(text, "  auto shapes = qsa_real_shapes();",
    "  const auto select = dedicated_prefill ? qsa_block_topk_prefill : qsa_block_topk;\n"
    "  auto shapes = qsa_real_shapes();")
start = text.index("void validation(bool dedicated_prefill)")
end = text.index("int integer(", start)
validation_fixture = text[start:end].replace("qsa_block_topk(", "select(")
# Function-pointer calls do not inherit the declaration's default argument.
validation_fixture = validation_fixture.replace("&runtime->compute());",
                                                 "&runtime->compute(), -1);")
validation_fixture = replace_once(validation_fixture,
    '  std::cout << "PASS host_validation\\n";',
    '  std::cout << "PASS host_validation api="\n'
    '            << (dedicated_prefill ? "dedicated_prefill" : "public_original") << \'\\n\';')
text = text[:start] + validation_fixture + text[end:]
text = replace_once(text, "      validation();", "      validation(false);\n      validation(true);")
proposals[path] = text

patch = []
for path, before in originals.items():
    after = proposals[path]
    dest = HERE / "files" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(after)
    patch.extend(difflib.unified_diff(before.splitlines(keepends=True),
                                     after.splitlines(keepends=True),
                                     fromfile="a/" + path, tofile="b/" + path))
(HERE / "source.diff").write_text("".join(patch))
manifest["proposed_files"] = [
    {"path": path, "bytes": len(text.encode()),
     "sha256": hashlib.sha256(text.encode()).hexdigest()}
    for path, text in proposals.items()]
manifest["source_diff_sha256"] = hashlib.sha256((HERE / "source.diff").read_bytes()).hexdigest()
(HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Prepared source-only package; qualification remains UNRUN.")
