import torch
import torch.nn as nn

from onmt.utils.logging import logger

class CosGenerator(torch.nn.Module):
    def __init__(self, dec_out_size, vocab_size, pretrained_embeddings=None):
        super(CosGenerator, self).__init__()
        logger.info("Generator: Cos projection.")
        self.linear = nn.Linear(dec_out_size, vocab_size, bias=False)
        if pretrained_embeddings is not None:
            logger.info("Generator: Weight tie.")
            self.linear.weight = nn.Parameter(pretrained_embeddings)

        if pretrained_embeddings is not None and not pretrained_embeddings.requires_grad:
            self.linear.weight.requires_grad = False
            self.linear_norm = None
            logger.info("Generator: Fixed weights.")

    def forward(self, model_output):
        norm_model_output = model_output.norm(p=2, dim=1)
        if self.linear.weight.requires_grad or self.linear_norm is None:
            self.linear_norm = self.linear.weight.norm(p=2, dim=1)
        norms = norm_model_output.unsqueeze(-1) @ self.linear_norm.unsqueeze(0)
        norms = torch.max(norms, 1e-8 * torch.ones_like(norms, device=model_output.device))
        cos_projection = self.linear(model_output) / norms
        return cos_projection
