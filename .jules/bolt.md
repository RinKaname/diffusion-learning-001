## 2024-05-24 - PyTorch Scaled Dot Product Attention
**Learning:** PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention`, which is highly optimized and automatically uses FlashAttention, Memory-Efficient Attention (xFormers), or an efficient math fallback depending on the hardware and inputs. Manual `torch.softmax(q @ k * scale) @ v` is much slower and uses significantly more memory since it instantiates the full `(b, heads, seq_len, seq_len)` attention matrix.
**Action:** Always prefer `F.scaled_dot_product_attention` for attention computations instead of manually calculating the scaled dot product. It speeds up operations and reduces VRAM usage without additional dependencies.

## 2026-05-31 - Persistent Buffers for Precomputed Tensors
**Learning:** When storing precomputed tensors (e.g., frequencies for position embeddings) as buffers using `self.register_buffer`, by default they are saved into the `state_dict`. This breaks backward compatibility with existing checkpoints that don't have this buffer. Setting `persistent=False` avoids this issue while still allowing the buffer to be moved to the correct device automatically.
**Action:** Always set `persistent=False` when using `register_buffer` in PyTorch for tensors that are derived directly from other parameters or statically defined, to prevent backwards compatibility breakage with existing checkpoints.
