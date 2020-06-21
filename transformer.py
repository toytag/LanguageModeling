# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# multi-head scaled dot-product attention
class MultiHeadedScaledDotProductAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.temperature = np.sqrt(d_k)
        # --- maybe it can be learnable ---
        # self.temperature = torch.tensor(np.sqrt(d_k), dtype=torch.float, requires_grad=True)
        self.wq = nn.Linear(d_model, d_k * n_head)
        self.wk = nn.Linear(d_model, d_k * n_head)
        self.wv = nn.Linear(d_model, d_v * n_head)
        self.output = nn.Linear(d_v * n_head, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, residual=None, mask=None):
        # default residual connection
        if residual is None:
            residual = q
        # [batch_size, n, (seq_length, d)]
        q = self.wq(q).view(q.size(0), q.size(1), self.n, self.d_k).transpose(1, 2)
        k = self.wk(k).view(k.size(0), k.size(1), self.n, self.d_k).transpose(1, 2)
        v = self.wv(v).view(v.size(0), v.size(1), self.n, self.d_v).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            mask.unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(mask==0, -1e9)
        # actual attention
        attn = F.softmax(attn / self.temperature, dim=-1)
        attn_dot_v = torch.matmul(attn, v).transpose(1, 2).reshape(q.size(0), -1, self.d_v * self.n)
        output = self.output(attn_dot_v)
        output = self.dropout(output)
        output = self.norm(output + residual)
        return output, attn


# positionwise feed-forward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.L1 = nn.Linear(d_model, d_hidden)
        self.L2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.L2(F.relu(self.L1(x)))
        output = self.dropout(output)
        output = self.norm(output + residual)
        return output


# general transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadedScaledDotProductAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffnn = PositionwiseFeedForward(d_model, d_hidden, dropout=dropout)

    def forward(self, input_seq, input_mask=None):
        slf_attn_output, slf_attn = self.slf_attn(input_seq, input_seq, input_seq, mask=input_mask)
        pos_ffnn_output = self.pos_ffnn(slf_attn_output)
        return pos_ffnn_output, slf_attn


# encoder layer
class EncoderLayer(TransformerBlock):
    def __init__(self, d_model, d_hidden, n_head, d_k, d_v, dropout=0.1):
        super().__init__(d_model, d_hidden, n_head, d_k, d_v, dropout)


# decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadedScaledDotProductAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_dec_attn = MultiHeadedScaledDotProductAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffnn = PositionwiseFeedForward(d_model, d_hidden, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, enc_dec_attn_mask=None):
        slf_attn_output, slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        enc_dec_attn_output, enc_dec_attn = self.enc_dec_attn(
            slf_attn_output, enc_output, enc_output, mask=enc_dec_attn_mask)
        dec_output = self.pos_ffnn(enc_dec_attn_output)
        return dec_output, slf_attn, enc_dec_attn


# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, n_position=100):
        super().__init__()
        self.pos_table = self._sinusoid_encoding_table(n_position, d_model)

    def _sinusoid_encoding_table(self, n_position, d_model):
        sinusoid_table = np.array([pos / np.power(10000, np.arange(d_model) // 2 * 2 / d_model)
                                   for pos in range(n_position)], dtype=np.float)
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        # (batch_size, n_position, d_model)
        return torch.tensor(sinusoid_table, dtype=torch.float).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)]


# encoder
class Encoder(nn.Module):
    def __init__(self, n_src_vocab, d_model, d_hidden, n_head, d_k, d_v, n_position=100, n_layer=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_src_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.enc_layer_stack = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_head, d_k, d_v, dropout=dropout)
                                              for _ in range(n_layer)])

    def forward(self, src_seq, src_mask=None, return_attns=False):
        enc_slf_attn_list = []

        enc_output = self.dropout(self.pos_enc(self.embedding(src_seq)))
        for enc_layer in self.enc_layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, input_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        return enc_output, enc_slf_attn_list


# decoder
class Decoder(nn.Module):
    def __init__(self, n_trg_vocab, d_model, d_hidden, n_head, d_k, d_v, n_position=100, n_layer=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_trg_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.dec_layer_stack = nn.ModuleList([DecoderLayer(d_model, d_hidden, n_head, d_k, d_v, dropout=dropout)
                                              for _ in range(n_layer)])

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, enc_dec_attn_list = [], []

        dec_output = self.dropout(self.pos_enc(self.embedding(trg_seq)))
        for dec_layer in self.dec_layer_stack:
            dec_output, dec_slf_attn, enc_dec_attn = dec_layer(dec_output, enc_output,
                                                               slf_attn_mask=trg_mask, enc_dec_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            enc_dec_attn_list += [enc_dec_attn] if return_attns else []

        return dec_output, dec_slf_attn_list, enc_dec_attn_list


# transformer
class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_trg_vocab, d_model=256, d_hidden=1024,
                 n_head=8, d_k=64, d_v=64, n_position=100, n_layer=2, dropout=0.1,
                 emb_src_trg_weight_sharing=True, trg_emb_prj_weight_sharing=True):
        super().__init__()
        self.encoder = Encoder(n_src_vocab, d_model, d_hidden, n_head, d_k, d_v,
                               n_position=n_position, n_layer=n_layer, dropout=dropout)
        self.decoder = Decoder(n_trg_vocab, d_model, d_hidden, n_head, d_k, d_v,
                               n_position=n_position, n_layer=n_layer, dropout=dropout)
        self.trg_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        if emb_src_trg_weight_sharing:
            self.encoder.embedding.weight = self.decoder.embedding.weight
        if trg_emb_prj_weight_sharing:
            self.trg_prj.weight = self.decoder.embedding.weight
            self.x_logit_scale = d_model ** -0.5

    def forward(self, src_seq, src_mask: torch.Tensor, trg_seq, trg_mask: torch.Tensor):
        """
        mask should be the same size as seq
        """
        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        transformer_output = self.trg_prj(dec_output) * self.x_logit_scale

        return transformer_output


if __name__ == "__main__":
    # test
    model = Transformer(10, 10)
    with torch.no_grad():
        x1, x2 = torch.tensor([[3, 8, 9, 2, 4, 2, 3]]), torch.tensor([[1, 5, 2, 7, 9, 7]])
        print(torch.argmax(F.softmax(model(x1, None, x2, None), dim=-1), dim=-1))