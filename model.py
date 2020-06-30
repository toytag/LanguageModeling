import random
random.seed(1)


import torch
from torch import nn
from LearnTransformer import PositionalEncoding, TransformerBlock



class LanguageModel(nn.Module):
    mask_idx = 0

    def __init__(self, n_vocab, d_model=256, d_hidden=1024, n_layer=8,
                 n_head=8, d_k=32, d_v=32, n_position=64, dropout=0.1,
                 embed_weight_sharing=True):
        super(LanguageModel, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_model, padding_idx=LanguageModel.mask_idx)
        self.pos_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.transformers = nn.ModuleList([
            TransformerBlock(d_model, d_hidden, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(d_model, n_vocab, bias=False)

        # maybe this is better?
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        self.logit_scale = 1.
        if embed_weight_sharing:
            self.embed.weight = self.lm_head.weight
            self.logit_scale = d_model ** -0.5

    def forward(self, src_seq, src_mask=None):
        output = self.dropout(self.pos_enc(self.embed(src_seq)))
        for transformer_block in self.transformers:
            output, _ = transformer_block(output, src_mask)
        output = self.lm_head(output) * self.logit_scale
        return output, 

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # test
    model = LanguageModel(n_vocab=10000)
    print(model.num_params())
    with torch.no_grad():
        model.eval()
        input_, mask = torch.LongTensor([[1, 2, 4, 8]]), torch.LongTensor([[1, 1, 1, 0]])
        output, *_ = model(input_.masked_fill(mask==0, LanguageModel.mask_idx), src_mask=mask)
        print(output)
