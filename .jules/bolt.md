## 2024-05-24 - PyTorch Scaled Dot Product Attention
**Learning:** PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention`, which is highly optimized and automatically uses FlashAttention, Memory-Efficient Attention (xFormers), or an efficient math fallback depending on the hardware and inputs. Manual `torch.softmax(q @ k * scale) @ v` is much slower and uses significantly more memory since it instantiates the full `(b, heads, seq_len, seq_len)` attention matrix.
**Action:** Always prefer `F.scaled_dot_product_attention` for attention computations instead of manually calculating the scaled dot product. It speeds up operations and reduces VRAM usage without additional dependencies.

## 2024-05-24 - Precomputing Sinusoidal Time Embeddings
**Learning:** Computing frequencies with `math.log(10000)`, `torch.exp`, and `torch.arange` on every `forward` pass of a time embedding layer adds unnecessary overhead.
**Action:** Always precompute frequency components in `__init__` and store them using `self.register_buffer('freqs', emb, persistent=False)` so they automatically move with the model without polluting checkpoints.
