#include "common.cuh"

// flash attention over the cells listed per query (src[5]), for head size 256 and up to 16 query heads per K/V head
bool ggml_cuda_flash_attn_ext_listed_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_listed(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
