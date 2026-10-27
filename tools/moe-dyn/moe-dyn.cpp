// moe-dyn: speed + quality benchmark for a dynamic MoE expert cache (fork).
//
// Runs a task file (jsonl {"id","prompt","max_tokens"}) with greedy decoding and reports prefill and
// decode tokens/s. The expert cache (--moe-cache-slots S + a routing profile for the initial fill) holds
// S experts per CPU-offloaded layer on the GPU.
//
//   DYN_MODE=static   the pack never changes (the fork's existing behaviour)
//   DYN_MODE=dyn      every DYN_WIN tokens each layer's pack is re-elected: top-S experts by a decayed
//                     count of what the UNMASKED router wanted (raw router scores sampled every
//                     DYN_SAMPLE decode steps via the eval callback), applied with
//                     llama_model_moe_cache_set_hot (resident experts stay, new ones are uploaded)
//   DYN_MODE=static_cb  like static, but with the eval callback installed (measures the hook's own overhead)
// The library's own re-election (in-graph counts, uploads on a worker thread) runs with --moe-cache-dyn N in any mode,
// best with static; --moe-cache-mask N restricts routing to the pack except the top-N raw picks.
//
// The cache persists across tasks, as in a server. Output: jsonl {"id","output"} per task + a RESULT line.
//
// Env: DYN_TASKS, DYN_OUT, DYN_MODE, DYN_WIN (32), DYN_SAMPLE (2), DYN_DECAY (0.9), DYN_K (experts/token, 8),
//      DYN_MAX_TASKS (all)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

struct dyn_ctx {
    bool  active = false;   // sample router scores on this step
    int   k      = 8;
    std::map<int, std::vector<float>> pending;   // layer -> demand since last re-election
    std::vector<float> buf;
    std::vector<int>   order;
};

static int env_int(const char * n, int d)       { const char * v = getenv(n); return v ? atoi(v) : d; }
static float env_float(const char * n, float d) { const char * v = getenv(n); return v ? (float) atof(v) : d; }

static bool is_probs_name(const char * name) {
    if (strncmp(name, "ffn_moe_probs-", 14) != 0 || name[14] == '\0') return false;
    for (const char * c = name + 14; *c; c++) if (*c < '0' || *c > '9') return false;
    return true;
}

static bool dyn_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    dyn_ctx * dc = (dyn_ctx *) user_data;
    if (ask) {
        return dc->active && is_probs_name(t->name);
    }
    if (!is_probs_name(t->name) || t->type != GGML_TYPE_F32) {
        return true;
    }
    const int il = atoi(t->name + 14);
    const int n_expert = (int) t->ne[0];
    const int n_tokens = (int) t->ne[1];
    const size_t nbytes = ggml_nbytes(t);
    dc->buf.resize(nbytes / sizeof(float) + 1);
    ggml_backend_tensor_get(t, dc->buf.data(), 0, nbytes);
    auto & dem = dc->pending[il];
    if (dem.empty()) dem.assign(n_expert, 0.0f);
    dc->order.resize(n_expert);
    const int k = std::min(dc->k, n_expert);
    for (int j = 0; j < n_tokens; j++) {
        const float * row = (const float *) ((const char *) dc->buf.data() + j * t->nb[1]);
        for (int e = 0; e < n_expert; e++) dc->order[e] = e;
        std::partial_sort(dc->order.begin(), dc->order.begin() + k, dc->order.end(),
                          [row](int a, int b) { return row[a] > row[b]; });
        for (int r = 0; r < k; r++) dem[dc->order[r]] += 1.0f;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    const char * tasks_path = getenv("DYN_TASKS");
    const char * out_path   = getenv("DYN_OUT");
    if (!tasks_path || !out_path) { LOG_ERR("set DYN_TASKS and DYN_OUT\n"); return 1; }
    const std::string mode = getenv("DYN_MODE") ? getenv("DYN_MODE") : "static";
    const int   win    = env_int("DYN_WIN", 32);
    const int   sample = std::max(1, env_int("DYN_SAMPLE", 2));
    const float decay  = env_float("DYN_DECAY", 0.9f);
    const int   max_tasks = env_int("DYN_MAX_TASKS", 1 << 30);

    dyn_ctx dc;
    dc.k = env_int("DYN_K", 8);
    common_init();
    if (mode == "dyn" || mode == "static_cb") {   // static_cb: callback installed but never sampling (overhead control)
        params.cb_eval           = dyn_cb;
        params.cb_eval_user_data = &dc;
    }
    params.warmup = false;
    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init ? init->model()   : nullptr;
    llama_context * ctx   = init ? init->context() : nullptr;
    if (!model || !ctx) { LOG_ERR("failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int n_layer = llama_model_n_layer(model);
    std::map<int, std::vector<float>> score;   // layer -> decayed demand
    int n_cached = 0, slots = 0;
    for (int il = 0; il < n_layer; il++) {
        const int s = llama_model_moe_cache_slots(model, il);
        if (s > 0) { n_cached++; slots = s; }
    }
    LOG_INF("mode=%s cached layers=%d slots=%d mask=%s win=%d sample=%d\n", mode.c_str(), n_cached, slots,
            params.moe_cache_mask >= 0 ? std::to_string(params.moe_cache_mask).c_str() : "off", win, sample);

    long long uploads = 0, reelections = 0;
    double t_reelect = 0, t_prefill = 0, t_decode = 0;
    long long n_prefill = 0, n_decode = 0, since = 0;
    using clk = std::chrono::steady_clock;

    auto reelect = [&]() {
        const auto t0 = clk::now();
        std::vector<int> order;
        for (auto & [il, dem] : dc.pending) {
            if (llama_model_moe_cache_slots(model, il) <= 0) continue;
            auto & sc = score[il];
            if (sc.empty()) sc.assign(dem.size(), 0.0f);
            for (size_t e = 0; e < dem.size(); e++) { sc[e] = sc[e] * decay + dem[e]; dem[e] = 0.0f; }
            order.resize(sc.size());
            for (size_t e = 0; e < sc.size(); e++) order[e] = (int) e;
            const int S = std::min<int>(llama_model_moe_cache_slots(model, il), (int) sc.size());   // fitted packs are built on first use
            std::partial_sort(order.begin(), order.begin() + S, order.end(), [&sc](int a, int b) { return sc[a] > sc[b]; });
            std::vector<int32_t> hot(order.begin(), order.begin() + S);
            const int32_t up = llama_model_moe_cache_set_hot(model, il, hot.data(), (int32_t) hot.size());
            if (up > 0) uploads += up;
        }
        reelections++;
        t_reelect += std::chrono::duration<double>(clk::now() - t0).count();
    };

    std::ifstream tf(tasks_path);
    FILE * out = fopen(out_path, "w");
    std::string line;
    int n_done = 0;
    const int W = std::max(1, params.n_batch);
    llama_batch batch = llama_batch_init(W, 0, 1);
    while (n_done < max_tasks && std::getline(tf, line)) {
        if (line.empty()) continue;
        nlohmann::json task = nlohmann::json::parse(line);
        const std::string id = task["id"];
        const int max_new = task.value("max_tokens", 400);
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> ptoks = common_tokenize(ctx, task["prompt"].get<std::string>(), false, true);
        int pos = 0;
        auto t0 = clk::now();
        for (size_t i = 0; i < ptoks.size(); i += W) {
            const int m = std::min((int) (ptoks.size() - i), W);
            batch.n_tokens = m;
            for (int j = 0; j < m; j++) {
                batch.token[j] = ptoks[i + j]; batch.pos[j] = pos + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                batch.logits[j] = (i + j + 1 == ptoks.size());
            }
            dc.active = (mode == "dyn");
            if (llama_decode(ctx, batch)) { LOG_ERR("prefill failed on %s\n", id.c_str()); return 1; }
            pos += m;
        }
        const double prefill_s = std::chrono::duration<double>(clk::now() - t0).count();
        t_prefill += prefill_s;
        n_prefill += (long long) ptoks.size();
        if (mode == "dyn") { reelect(); since = 0; }   // the prompt sets the topic before decoding starts

        std::string text;
        t0 = clk::now();
        for (int t = 0; t < max_new; t++) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            int am = 0;
            for (int v = 1; v < n_vocab; v++) if (lg[v] > lg[am]) am = v;
            if (llama_vocab_is_eog(vocab, am)) break;
            text += common_token_to_piece(ctx, am);
            batch.n_tokens = 1;
            batch.token[0] = am; batch.pos[0] = pos++; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
            dc.active = (mode == "dyn") && (t % sample == 0);
            if (llama_decode(ctx, batch)) { LOG_ERR("decode failed on %s\n", id.c_str()); return 1; }
            n_decode++;
            if (mode == "dyn" && ++since >= win) { reelect(); since = 0; }
        }
        t_decode += std::chrono::duration<double>(clk::now() - t0).count();
        nlohmann::json o = {{"id", id}, {"output", text}, {"prefill_tokens", ptoks.size()}, {"prefill_s", prefill_s}};
        fprintf(out, "%s\n", o.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace).c_str());
        fflush(out);
        if (++n_done % 10 == 0) {
            LOG_INF("%d tasks, decode %.1f tok/s, prefill %.1f tok/s\n", n_done, n_decode / t_decode, n_prefill / t_prefill);
        }
    }
    fclose(out);
    llama_batch_free(batch);
    n_cached = 0;
    for (int il = 0; il < n_layer; il++) {
        const int s = llama_model_moe_cache_slots(model, il);
        if (s > 0) { n_cached++; slots = std::max(slots, s); }
    }
    LOG_INF("RESULT mode=%s mask=%s slots=%d tasks=%d prefill=%.1f tok/s decode=%.2f tok/s decode_tokens=%lld "
            "reelections=%lld uploads/reelection/layer=%.2f reelect_time=%.1fs (%.1f%% of decode)\n",
            mode.c_str(), params.moe_cache_mask >= 0 ? std::to_string(params.moe_cache_mask).c_str() : "off", slots, n_done,
            n_prefill / t_prefill, n_decode / t_decode, n_decode, reelections,
            reelections && n_cached ? (double) uploads / reelections / n_cached : 0.0, t_reelect,
            100.0 * t_reelect / std::max(1e-9, t_decode));
    llama_backend_free();
    return 0;
}
