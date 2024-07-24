assert semantic_frame_rate in ["25hz", "50hz"]
self.semantic_frame_rate = semantic_frame_rate
if semantic_frame_rate == "25hz":
    self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
else:
    self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

with autocast(enabled=False):
    maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
    with maybe_no_grad:
        if self.freeze_quantizer:
            self.ssl_proj.eval()
            self.quantizer.eval()
    ssl = self.ssl_proj(ssl)
    quantized, codes, commit_loss, quantized_list = self.quantizer(
        ssl, layers=[0]
    )

if self.semantic_frame_rate == "25hz":
    quantized = F.interpolate(
        quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
    )