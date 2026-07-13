## 2024-05-24 - PyTorch Scaled Dot Product Attention
**Learning:** PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention`, which is highly optimized and automatically uses FlashAttention, Memory-Efficient Attention (xFormers), or an efficient math fallback depending on the hardware and inputs. Manual `torch.softmax(q @ k * scale) @ v` is much slower and uses significantly more memory since it instantiates the full `(b, heads, seq_len, seq_len)` attention matrix.
**Action:** Always prefer `F.scaled_dot_product_attention` for attention computations instead of manually calculating the scaled dot product. It speeds up operations and reduces VRAM usage without additional dependencies.

## 2024-05-24 - Precomputing Static Tensors in Modules
**Learning:** In PyTorch, computing static tensors (like frequencies for Sinusoidal Positional Embeddings) inside the `forward` method causes unnecessary redundant calculations and tensor allocations every pass. Using `self.register_buffer(name, tensor, persistent=False)` in `__init__` precomputes it once and keeps it on the correct device automatically without saving it to the `state_dict`, avoiding backward compatibility issues with existing checkpoints.
**Action:** When working with positional embeddings or other modules with deterministic, input-independent static tensors, precompute them in `__init__` and register them as non-persistent buffers instead of re-evaluating them in `forward`.

## 2026-07-07 - PyTorch Tensor Batching Before Splitting
**Learning:** In PyTorch attention module implementations, applying tensor operations like `reshape` and `transpose` to the combined `qkv` tensor before using `unbind` or `chunk` significantly reduces PyTorch dispatcher overhead and avoids expensive memory allocations caused by reshaping non-contiguous tensors (which occurs when chunking along the channel dimension first).
**Action:** When working with combined `qkv` tensors, batch shape manipulations on the single tensor before splitting it into `q`, `k`, and `v`.

## 2024-05-24 - PyTorch Training Loop Optimization Defaults
**Learning:** Default PyTorch training loop configurations leave performance on the table. Setting `optimizer.zero_grad(set_to_none=True)` prevents unnecessary memory writes, using `non_blocking=True` for data transfers with pinned memory allows asynchronous copies overlapping CPU/GPU, and enabling `torch.backends.cudnn.benchmark = True` for fixed-size inputs allows optimal convolution algorithm selection.
**Action:** Always apply `set_to_none=True` in `zero_grad`, use `non_blocking=True` on tensor transfers to devices when `pin_memory=True` is active, and set `torch.backends.cudnn.benchmark = True` when input sizes are static.
