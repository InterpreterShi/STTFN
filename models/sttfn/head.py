import torch
import torch.nn as nn

class TransformerDecoderHead(nn.Module):
    def __init__(self, in_len, out_len, d_model, n_heads, num_layers, dropout):
        super(TransformerDecoderHead, self).__init__()
        self.out_len = out_len

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # decoder 输入的 learnable query
        self.query_embed = nn.Parameter(torch.randn(out_len, d_model))

        # 最后的预测层
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):  # x: [B, in_len, N, d_model]
        B, in_len, N, d_model = x.shape

        # TransformerDecoder Memory 来自 Encoder(st_out)
        memory = x.permute(2, 0, 1, 3).contiguous().view(N, B * in_len, d_model)  # [N, B*in_len, d_model]

        # decoder 输入的 learnable query
        tgt = self.query_embed.unsqueeze(1).repeat(1, B * N, 1)  # [out_len, B*N, d_model]

        # Decoder 输出
        output = self.decoder(tgt, memory)  # [out_len, B*N, d_model]

        # reshape 回原 shape
        output = output.permute(1, 0, 2).contiguous().view(B, N, self.out_len, d_model).permute(0, 2, 1, 3)

        pred = self.output_layer(output).squeeze(-1)  # [B, out_len, N]

        return pred