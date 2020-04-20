import torch
import torch.nn as nn

from onmt.modules.util_class import Cast

class CosGenerator(torch.nn.Module):
    def __init__(self, dec_out_size, vocab_size, cast, gen_func, pretrained_embeddings=None, fix_word_vecs_dec=False):
        super(CosGenerator, self).__init__()
        self.linear = nn.Linear(dec_out_size, vocab_size, bias=False)
        if pretrained_embeddings is not None:
            self.linear.weight = nn.Parameter(pretrained_embeddings)

        if fix_word_vecs_dec:
            self.linear.weight.requires_grad = False
            self.linear_norm = self.linear.weight.norm(p=2, dim=1)

        self.cast = cast
        self.gen_func = gen_func

    def forward(self, input):
        norm_input = input.norm(p=2, dim=1)
        if self.linear.weight.requires_grad:
            self.linear_norm = self.linear.weight.norm(p=2, dim=1)
        norms = norm_input.unsqueeze(-1) @ self.linear_norm.unsqueeze(0)
        norms = torch.max(norms, 1e-8 * torch.ones_like(norms))
        cos_projection = self.linear(input) / norms
        cast = self.cast(cos_projection)
        return self.gen_func(cast)
