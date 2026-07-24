#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

void ggml_cuda_launch_mm_ids_zero_skipped_rows(
        const int32_t * ids, float * dst, int64_t ne0, int n_tokens, int n_expert_used,
        int si1, int64_t s_slot, int64_t s_token, cudaStream_t stream);
