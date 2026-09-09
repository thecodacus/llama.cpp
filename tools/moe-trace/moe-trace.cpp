// moe-trace: dump routed expert ids per layer per decode step.
//
// Captures the "ffn_moe_topk-<il>" id tensors through the scheduler eval
// callback (same mechanism as imatrix), so it works regardless of which
// backend computed the node (CPU-offloaded experts included).
//
// Output CSV, one row per (position, layer):   pos,layer,id0,id1,...
// Prompt rows are tagged with negative positions so the simulator can
// separate prefill routing from decode routing.
//
// Usage:
//   MOE_TRACE_OUT=trace.csv llama-moe-trace -m model.gguf -ngl 99 -ncmoe 26 -fa on \
//       -p "prompt text" -n 512

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct trace_ctx {
    FILE   * out       = nullptr;
    int      pos       = 0;     // current decode position (negative = prefill)
    bool     in_prompt = true;
    std::vector<int32_t> buf;

    // optional hidden-state capture, for offline router-prediction experiments.
    // records: pos(i32) layer(i32) tag(i32) n_embd(i32) then n_embd f32 values.
    // tag 0 = l_out (layer output = next layer's pre-attention input)
    //     1 = attn_residual (h + attn, pre-norm)
    //     2 = attn_post_norm (the actual router input)
    FILE * spec      = nullptr;   // speculative (pre-attention) router ids
    FILE * hid       = nullptr;
    int    hid_limit = 0;       // capture at most this many decode positions
    std::vector<float> fbuf;
};

// capture one f32 hidden-state row per token in the batch
static void capture_hidden(trace_ctx * tc, struct ggml_tensor * t, int tag, int layer) {
    if (tc->hid == nullptr || t->type != GGML_TYPE_F32) {
        return;
    }
    // only decode steps, and only the first hid_limit of them
    if (tc->in_prompt || tc->pos >= tc->hid_limit) {
        return;
    }
    const int n_embd   = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];
    const size_t nbytes = ggml_nbytes(t);
    tc->fbuf.resize((nbytes + sizeof(float) - 1) / sizeof(float));
    ggml_backend_tensor_get(t, tc->fbuf.data(), 0, nbytes);
    const char * base = (const char *) tc->fbuf.data();
    for (int j = 0; j < n_tokens; j++) {
        const int32_t hdr[4] = { tc->pos, layer, tag, n_embd };
        fwrite(hdr, sizeof(int32_t), 4, tc->hid);
        fwrite(base + j*t->nb[1], sizeof(float), n_embd, tc->hid);
    }
}

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    trace_ctx * tc = (trace_ctx *) user_data;

    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_spec = tc->spec != nullptr && strncmp(t->name, "ffn_moe_spec_topk-", 18) == 0;

    // hidden-state taps for the router-prediction experiment
    int hid_tag = -1;
    if (tc->hid != nullptr) {
        if      (strncmp(t->name, "l_out-",           6) == 0) hid_tag = 0;
        else if (strncmp(t->name, "attn_residual-",  14) == 0) hid_tag = 1;
        else if (strncmp(t->name, "attn_post_norm-", 15) == 0) hid_tag = 2;
    }

    if (ask) {
        return is_topk || is_spec || hid_tag >= 0;
    }
    if (hid_tag >= 0) {
        const char * dash = strrchr(t->name, '-');
        capture_hidden(tc, t, hid_tag, dash ? atoi(dash + 1) : -1);
    }
    if (!(is_topk || is_spec) || t->type != GGML_TYPE_I32) {
        return true;
    }

    FILE * const dst   = is_spec ? tc->spec : tc->out;
    const int layer    = atoi(strrchr(t->name, '-') + 1);
    const int n_used   = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];

    // topk is a non-contiguous view over the argsort rows: copy the full
    // strided byte range, then index by nb[] — sizing by n_used*n_tokens
    // would under-allocate and tensor_get would smash the heap.
    const size_t nbytes = ggml_nbytes(t);
    tc->buf.resize((nbytes + sizeof(int32_t) - 1) / sizeof(int32_t));
    ggml_backend_tensor_get(t, tc->buf.data(), 0, nbytes);
    const char * base = (const char *) tc->buf.data();

    for (int j = 0; j < n_tokens; j++) {
        // prefill batches carry n_tokens > 1; decode steps carry 1
        const int pos = tc->in_prompt ? -(tc->pos + n_tokens - j) : tc->pos;
        fprintf(dst, "%d,%d", pos, layer);
        for (int i = 0; i < n_used; i++) {
            const int32_t id = *(const int32_t *)(base + j*t->nb[1] + i*t->nb[0]);
            fprintf(dst, ",%d", id);
        }
        fputc('\n', dst);
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 256;

    // reuse the standard arg parser; -o (out_file) holds the trace path
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    // output path via env — the arg registry has no free slot for this example
    const char * out_path = getenv("MOE_TRACE_OUT");
    if (!out_path) {
        out_path = "moe-trace.csv";
    }

    common_init();

    trace_ctx tc;
    tc.out = fopen(out_path, "w");
    if (!tc.out) {
        LOG_ERR("failed to open %s for writing\n", out_path);
        return 1;
    }

    if (const char * spec_path = getenv("MOE_TRACE_SPEC")) {
        tc.spec = fopen(spec_path, "w");
        if (!tc.spec) {
            LOG_ERR("failed to open %s for writing\n", spec_path);
            return 1;
        }
    }

    // optional hidden-state dump for the router-prediction experiment
    if (const char * hid_path = getenv("MOE_TRACE_HIDDEN")) {
        tc.hid = fopen(hid_path, "wb");
        if (!tc.hid) {
            LOG_ERR("failed to open %s for writing\n", hid_path);
            return 1;
        }
        const char * lim = getenv("MOE_TRACE_HIDDEN_N");
        tc.hid_limit = lim ? atoi(lim) : 64;
        LOG_INF("capturing hidden states for the first %d decode steps -> %s\n", tc.hid_limit, hid_path);
    }

    params.cb_eval           = trace_cb;
    params.cb_eval_user_data = &tc;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init ? llama_init->model()   : nullptr;
    llama_context * lctx  = llama_init ? llama_init->context() : nullptr;
    if (model == nullptr || lctx == nullptr) {
        LOG_ERR("failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(lctx, params.prompt, true);
    if (tokens.empty()) {
        LOG_ERR("empty prompt\n");
        return 1;
    }
    LOG_INF("prompt: %zu tokens, decoding %d\n", tokens.size(), params.n_predict);

    // prefill
    tc.in_prompt = true;
    tc.pos       = 0;
    for (size_t i = 0; i < tokens.size(); i += params.n_batch) {
        const int n_eval = std::min((int) (tokens.size() - i), params.n_batch);
        if (llama_decode(lctx, llama_batch_get_one(tokens.data() + i, n_eval))) {
            LOG_ERR("prefill failed at %zu\n", i);
            return 1;
        }
        tc.pos += n_eval;
    }

    // greedy decode
    tc.in_prompt = false;
    llama_sampler * smpl = llama_sampler_init_greedy();
    llama_token tok = 0;
    for (int i = 0; i < params.n_predict; i++) {
        tok = llama_sampler_sample(smpl, lctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        tc.pos = i;
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("decode failed at %d\n", i);
            return 1;
        }
        if (i % 64 == 0) {
            LOG_INF("decoded %d/%d\n", i, params.n_predict);
        }
    }
    llama_sampler_free(smpl);

    fclose(tc.out);
    if (tc.hid)  { fclose(tc.hid); }
    if (tc.spec) { fclose(tc.spec); }
    LOG_INF("trace written to %s\n", out_path);

    llama_backend_free();
    return 0;
}
