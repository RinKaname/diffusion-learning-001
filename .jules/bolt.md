## 2024-05-24 - PyTorch Scaled Dot Product Attention
**Learning:** PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention`, which is highly optimized and automatically uses FlashAttention, Memory-Efficient Attention (xFormers), or an efficient math fallback depending on the hardware and inputs. Manual `torch.softmax(q @ k * scale) @ v` is much slower and uses significantly more memory since it instantiates the full `(b, heads, seq_len, seq_len)` attention matrix.
**Action:** Always prefer `F.scaled_dot_product_attention` for attention computations instead of manually calculating the scaled dot product. It speeds up operations and reduces VRAM usage without additional dependencies.

## 2024-05-25 - PyTorch Persistent Buffers
**Learning:** Using `register_buffer` in PyTorch to store precomputed tensors for performance optimizations defaults to `persistent=True`, which saves the buffer to the `state_dict`. This breaks backward compatibility when loading older checkpoints.
**Action:** When adding precomputed tensors (e.g., frequencies for embeddings) via `register_buffer`, always pass `persistent=False` to ensure they are not saved to the `state_dict` and checkpoint compatibility is maintained.
