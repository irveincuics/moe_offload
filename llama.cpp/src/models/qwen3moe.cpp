#include "models.h"
#include "ggml-alloc.h"
#include <cstdlib>
#include <fstream>
#include <vector>

llama_model_qwen3moe::~llama_model_qwen3moe() {
    if (prefetch_ctx) { ggml_backend_buffer_free(prefetch_buffer); ggml_free(prefetch_ctx); }
}

void llama_model_qwen3moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const char * adapter = getenv("GGML_MOE_B_ADAPTER");
    if (adapter && adapter[0]) {
        std::ifstream in(adapter, std::ios::binary);
        uint32_t shape[4];
        in.read((char *) shape, sizeof(shape));
        GGML_ASSERT(in && shape[0] == 0x42505231 && shape[1] == uint32_t(n_embd) && shape[2] == uint32_t(n_layer));
        const int rank = shape[3];
        prefetch_ctx = ggml_init({size_t(2*n_layer + 1)*ggml_tensor_overhead() + 1024, nullptr, true});
        prefetch_down = ggml_new_tensor_2d(prefetch_ctx, GGML_TYPE_F32, n_embd, rank);
        ggml_set_name(prefetch_down, "prefetch_down");
        std::vector<ggml_tensor *> tensors = {prefetch_down};
        for (int l = 0; l < n_layer; ++l) {
            auto * up = ggml_new_tensor_2d(prefetch_ctx, GGML_TYPE_F32, rank, n_expert);
            auto * bias = ggml_new_tensor_1d(prefetch_ctx, GGML_TYPE_F32, n_expert);
            ggml_format_name(up, "prefetch_up_%d", l);
            ggml_format_name(bias, "prefetch_bias_%d", l);
            prefetch_up.push_back(up); prefetch_bias.push_back(bias);
            tensors.push_back(up); tensors.push_back(bias);
        }
        prefetch_buffer = ggml_backend_alloc_ctx_tensors_from_buft(prefetch_ctx,
                ggml_backend_dev_buffer_type(dev_layer(0)));
        ggml_backend_buffer_set_usage(prefetch_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * tensor : tensors) {
            std::vector<float> values(ggml_nelements(tensor));
            in.read((char *) values.data(), ggml_nbytes(tensor));
            GGML_ASSERT(in);
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
    }


    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
        }

        // MoE branch
        const int64_t n_ff_exp = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int expert_flags = params.ssd_moe_offload ? TENSOR_READ_LAZY : 0;

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, expert_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, expert_flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, expert_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const char * prefetch = getenv("GGML_MOE_PREFETCH");
        if (prefetch && prefetch[0] == 'b' && cur->ne[1] == 1 &&
                !(std::string(prefetch) == "bgu" && il + atoi(getenv("GGML_MOE_PREFETCH_DISTANCE")) >= n_layer)) {
            const int distance = atoi(getenv("GGML_MOE_PREFETCH_DISTANCE"));
            const int target = (il + distance) % n_layer;
            auto * prediction = ggml_mul_mat(ctx0, model.layers[target].ffn_gate_inp, cur);
            const auto & qm = static_cast<const llama_model_qwen3moe &>(model);
            if (qm.prefetch_down) {
                auto * low = ggml_mul_mat(ctx0, qm.prefetch_down, cur);
                auto * correction = ggml_mul_mat(ctx0, qm.prefetch_up[il], low);
                correction = ggml_add(ctx0, correction, qm.prefetch_bias[il]);
                prediction = ggml_add(ctx0, prediction, correction);
            }
            cb(prediction, "moe_prefetch_score", il);
            ggml_build_forward_expand(gf, prediction);
        }


        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);
        cur = moe_out;

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
