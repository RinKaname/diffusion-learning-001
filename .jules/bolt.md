## 2024-05-24 - PyTorch Scaled Dot Product Attention
**Learning:** PyTorch 2.0+ includes `torch.nn.functional.scaled_dot_product_attention`, which is highly optimized and automatically uses FlashAttention, Memory-Efficient Attention (xFormers), or an efficient math fallback depending on the hardware and inputs. Manual `torch.softmax(q @ k * scale) @ v` is much slower and uses significantly more memory since it instantiates the full `(b, heads, seq_len, seq_len)` attention matrix.
**Action:** Always prefer `F.scaled_dot_product_attention` for attention computations instead of manually calculating the scaled dot product. It speeds up operations and reduces VRAM usage without additional dependencies.

## 2024-05-18 - Avoid chunking and re-transposing separately before SDPA
**Learning:** In `SelfAttention`, using `qkv.chunk(3, dim=1)` and then individually `.reshape().transpose().unsqueeze()` for `q`, `k`, and `v` causes unnecessary intermediary allocations and slows down attention blocks significantly.
**Action:** Use a fused reshape and split: `qkv.view(b, 3, c, h * w)` followed by `unbind(dim=1)` before the final transpose on each head. This avoids copying memory inefficiently and measurably speeds up self-attention in the U-Net.

## 2024-05-18 - Faster elementwise ops in ResidualBlock with `torch.addcmul`
**Learning:** PyTorch handles elementwise math like `h * (scale + 1) + shift` separately, creating temporary intermediate tensors at each stage.
**Action:** Consolidate these operations into a single kernel call with `torch.addcmul(shift, h, scale + 1)` which noticeably speeds up heavily utilized code paths like our U-Net's `ResidualBlock` forward pass.
