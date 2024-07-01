#
import torch

def forward(self, x, x_lens, y, y_lens, bert_feature):
    """
    x: phoneme_ids
    y: semantic_ids
    """
    x = self.ar_text_embedding(x)
    x = x + self.bert_proj(bert_feature.transpose(1, 2))
    x = self.ar_text_position(x)
    x_mask = make_pad_mask(x_lens)

    y_mask = make_pad_mask(y_lens)
    y_mask_int = y_mask.type(torch.int64)
    codes = y.type(torch.int64) * (1 - y_mask_int)

    # Training
    # AR Decoder
    y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
    x_len = x_lens.max()
    y_len = y_lens.max()
    y_emb = self.ar_audio_embedding(y)
    y_pos = self.ar_audio_position(y_emb)

    xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
    ar_xy_padding_mask = xy_padding_mask

    x_attn_mask = F.pad(
        torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
        (0, y_len),
        value=True,
    )
    y_attn_mask = F.pad(
        torch.triu(
            torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        ),
        (x_len, 0),
        value=False,
    )
    xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)
    bsz, src_len = x.shape[0], x_len + y_len
    _xy_padding_mask = (
        ar_xy_padding_mask.view(bsz, 1, 1, src_len)
        .expand(-1, self.num_head, -1, -1)
        .reshape(bsz * self.num_head, 1, src_len)
    )
    xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
    new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
    new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
    xy_attn_mask = new_attn_mask
    # x 和完整的 y 一次性输入模型
    xy_pos = torch.concat([x, y_pos], dim=1)
    xy_dec, _ = self.h(
        (xy_pos, None),
        mask=xy_attn_mask,
    )
    logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
    # loss
    # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
    loss = F.cross_entropy(logits, targets, reduction="sum")
    acc = self.ar_accuracy_metric(logits.detach(), targets).item()
    return loss, acc


# 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
def infer(
        self,
        x,
        x_lens,
        prompts,
        bert_feature,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
):
    x = self.ar_text_embedding(x)
    x = x + self.bert_proj(bert_feature.transpose(1, 2))
    x = self.ar_text_position(x)

    # AR Decoder
    y = prompts
    prefix_len = y.shape[1]
    x_len = x.shape[1]
    x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
    stop = False
    for _ in tqdm(range(1500)):
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)
        # x 和逐渐增长的 y 一起输入给模型
        xy_pos = torch.concat([x, y_pos], dim=1)
        y_len = y.shape[1]
        x_attn_mask_pad = F.pad(
            x_attn_mask,
            (0, y_len),
            value=True,
        )
        y_attn_mask = F.pad(
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0).to(
            y.device
        )

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, -1])
        samples = topk_sampling(
            logits, top_k=top_k, top_p=1.0, temperature=temperature
        )

        if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
            print("use early stop num:", early_stop_num)
            stop = True

        if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
            # print(torch.argmax(logits, dim=-1)[0] == self.EOS, samples[0, 0] == self.EOS)
            stop = True
        if stop:
            if prompts.shape[1] == y.shape[1]:
                y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                print("bad zero prediction")
            print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
            break
        # 本次生成的 semantic_ids 和之前的 y 构成新的 y
        # print(samples.shape)#[1,1]#第一个1是bs
        # import os
        # os._exit(2333)
        y = torch.concat([y, samples], dim=1)
    return y