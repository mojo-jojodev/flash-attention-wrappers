import torch
def execute_flash_attention(q, k, v):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
