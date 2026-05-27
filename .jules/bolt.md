## 2024-05-24 - PyTorch Scaled Dot Product Attention
**Learning:** PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention`, which is highly optimized and automatically uses FlashAttention, Memory-Efficient Attention (xFormers), or an efficient math fallback depending on the hardware and inputs. Manual `torch.softmax(q @ k * scale) @ v` is much slower and uses significantly more memory since it instantiates the full `(b, heads, seq_len, seq_len)` attention matrix.
**Action:** Always prefer `F.scaled_dot_product_attention` for attention computations instead of manually calculating the scaled dot product. It speeds up operations and reduces VRAM usage without additional dependencies.

## 2026-05-27 - F.linear is faster than 1x1 Conv2d for SelfAttention Sequence Flattening
**Learning:** When preparing inputs for `F.scaled_dot_product_attention`, using 1x1 `nn.Conv2d` followed by reshaping/chunking is significantly slower due to memory fragmentation and contiguous array allocations.
**Action:** Instead, flatten the sequence first (e.g., `x.view(b, c, h*w).transpose(1, 2)`), then use `torch.nn.functional.linear` with the `nn.Conv2d`'s reshaped weights (`weight.view(-1, c)`). This preserves checkpoint compatibility with existing 1x1 Convs while yielding an ~3x speedup.
