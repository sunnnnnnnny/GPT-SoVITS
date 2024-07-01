# ref : https://zhuanlan.zhihu.com/p/679839992
# ssl -> wav map
import torch


def forward(ssl, y, y_lengths, text, text_lengths):
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
        y.dtype
    )
    ge = self.ref_enc(y * y_mask, y_mask)

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

    x, m_p, logs_p, y_mask = self.enc_p(
        quantized, y_lengths, text, text_lengths, ge
    )
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
    z_p = self.flow(z, y_mask, g=ge)

    z_slice, ids_slice = commons.rand_slice_segments(
        z, y_lengths, self.segment_size
    )
    o = self.dec(z_slice, g=ge)
    return (
        o,
        commit_loss,
        ids_slice,
        y_mask,
        y_mask,
        (z, z_p, m_p, logs_p, m_q, logs_q),
        quantized,
    )


def infer(self, ssl, y, y_lengths, text, text_lengths, test=None, noise_scale=0.5):
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
        y.dtype
    )
    ge = self.ref_enc(y * y_mask, y_mask)

    ssl = self.ssl_proj(ssl)
    quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
    if self.semantic_frame_rate == "25hz":
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )

    x, m_p, logs_p, y_mask = self.enc_p(
        quantized, y_lengths, text, text_lengths, ge, test=test
    )
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

    z = self.flow(z_p, y_mask, g=ge, reverse=True)

    o = self.dec((z * y_mask)[:, :, :], g=ge)
    return o, y_mask, (z, z_p, m_p, logs_p)
