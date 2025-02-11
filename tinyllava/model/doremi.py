from dataclasses import dataclass
from typing import Optional

import torch

from tinyllava.utils.arguments import DoReMiArguments
from tinyllava.model.modeling_tinyllava import CausalLMOutputWithDomainIDs, TinyLlavaForConditionalGeneration

class TinyLlavaForDoReMi(TinyLlavaForConditionalGeneration):
    def __init__(self, reference_model: TinyLlavaForConditionalGeneration,
                 doremi_args: DoReMiArguments, 
                 *args, **kwargs):
        TinyLlavaForConditionalGeneration.__init__(self, *args, **kwargs)
        
        self.reference_model: TinyLlavaForConditionalGeneration = reference_model

        self.num_domains = doremi_args.num_domains

        self.update_counter = 0
        self.train_domain_weights = torch.ones(self.num_domains) / self.num_domains
        self.avg_domain_weights = torch.ones(self.num_domains) / self.num_domains
        self.perdomain_scores = torch.ones(self.num_domains) / self.num_domains

    def forward(self, *args, **kwargs):
        proxy_output = super().forward(*args, return_pertoken_losses=True, **kwargs)
        reference_output = self.reference_model(*args, return_pertoken_losses=True, **kwargs)

        output = CausalLMOutputWithDomainIDs(
            **proxy_output,
            reference_pertoken_loss=reference_output.pertoken_loss,
        )

        return output

