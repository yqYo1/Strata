// Optional model-level CPU reference. Link against the pinned llama.cpp build.
// Usage: llama_logits MODEL.gguf TOKENS.txt LOGITS.bin
// Output matches strata --dump-logits: int32 vocabulary/rows, then FP32 rows.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error("usage: llama_logits MODEL.gguf TOKENS.txt LOGITS.bin");
    std::ifstream input(argv[2]);
    if (!input)
      throw std::runtime_error("cannot read token file");
    std::string text((std::istreambuf_iterator<char>(input)), {});
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream ids(text);
    std::vector<llama_token> tokens;
    int64_t id;
    while (ids >> id) {
      if (id < 0 || id > INT32_MAX)
        throw std::runtime_error("invalid token id");
      tokens.push_back(static_cast<llama_token>(id));
    }
    if (!ids.eof() || tokens.empty() || tokens.size() > 4096)
      throw std::runtime_error("expected 1..4096 integer token ids");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    // Avoid CPU weight repacking: compare the original GGUF quantized tensors.
    mp.use_extra_bufts = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model)
      throw std::runtime_error("model load failed");
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    for (auto t : tokens)
      if (t >= vocab)
        throw std::runtime_error("token id outside vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max<size_t>(128, tokens.size());
    cp.n_batch = cp.n_ubatch = cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.offload_kqv = cp.op_offload = false;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx)
      throw std::runtime_error("context initialization failed");
    std::unique_ptr<std::FILE, decltype(&std::fclose)> output(
        std::fopen(argv[3], "wb"), std::fclose);
    if (!output)
      throw std::runtime_error("cannot open output");
    const int32_t header[] = {vocab, static_cast<int32_t>(tokens.size())};
    if (std::fwrite(header, sizeof header, 1, output.get()) != 1)
      throw std::runtime_error("cannot write header");
    auto batch = llama_batch_init(1, 0, 1);
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
      batch.n_tokens = 1;
      batch.token[0] = tokens[pos];
      batch.pos[0] = static_cast<llama_pos>(pos);
      batch.n_seq_id[0] = 1;
      batch.seq_id[0][0] = 0;
      batch.logits[0] = 1;
      const int rc = llama_decode(ctx.get(), batch);
      if (rc)
        throw std::runtime_error("CPU decode failed: " + std::to_string(rc));
      const float *row = llama_get_logits_ith(ctx.get(), 0);
      if (!row || !std::all_of(row, row + vocab, [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("invalid CPU logits");
      if (std::fwrite(row, sizeof(float), vocab, output.get()) != size_t(vocab))
        throw std::runtime_error("cannot write logits");
      std::fflush(output.get());
      std::fprintf(stderr, "reference position %zu: argmax %td\n", pos,
                   std::max_element(row, row + vocab) - row);
    }
    llama_batch_free(batch);
    ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
