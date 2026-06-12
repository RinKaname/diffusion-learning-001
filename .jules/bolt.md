## 2024-05-24 - PyTorch Scaled Dot Product Attention
**Learning:** PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention`, which is highly optimized and automatically uses FlashAttention, Memory-Efficient Attention (xFormers), or an efficient math fallback depending on the hardware and inputs. Manual `torch.softmax(q @ k * scale) @ v` is much slower and uses significantly more memory since it instantiates the full `(b, heads, seq_len, seq_len)` attention matrix.
**Action:** Always prefer `F.scaled_dot_product_attention` for attention computations instead of manually calculating the scaled dot product. It speeds up operations and reduces VRAM usage without additional dependencies.

## 2024-05-24 - Precomputing Static Tensors in Modules
**Learning:** In PyTorch, computing static tensors (like frequencies for Sinusoidal Positional Embeddings) inside the `forward` method causes unnecessary redundant calculations and tensor allocations every pass. Using `self.register_buffer(name, tensor, persistent=False)` in `__init__` precomputes it once and keeps it on the correct device automatically without saving it to the `state_dict`, avoiding backward compatibility issues with existing checkpoints.
**Action:** When working with positional embeddings or other modules with deterministic, input-independent static tensors, precompute them in `__init__` and register them as non-persistent buffers instead of re-evaluating them in `forward`.

## 2024-05-24 - Batching Tensor Operations Before Unbind
**Learning:** Performing tensor operations like `reshape`, `transpose`, and `unsqueeze` individually on split tensors (`q`, `k`, `v`) incurs unnecessary PyTorch dispatcher overhead. Batching these operations onto the combined `qkv` tensor before splitting it with `unbind(dim=1)` significantly reduces this overhead, resulting in faster execution.
**Action:** For PyTorch attention module implementations, always apply tensor operations to the combined `qkv` tensor before splitting it into `q`, `k`, and `v`. Use `unbind(dim=1)` to extract the components after the batched operations.
