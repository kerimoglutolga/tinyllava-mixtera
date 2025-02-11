import os
import torch
from torch import nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from typing import List, Optional

import wandb

from ..utils.train_utils import *


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                connector_parameters = [name for name, _ in opt_model.named_parameters() if "connector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                ]

            if getattr(self.args, "moe_enable", False):
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    

class DoReMiTrainer(LLaVATrainer):
    def __init__(self, doremi_args, *args, **kwargs):
        LLaVATrainer.__init__(self, *args, **kwargs)

        self.num_domains = doremi_args.num_domains
        self.reweight_eta = doremi_args.reweight_eta
        self.reweight_eps = doremi_args.reweight_eps

        self.train_domain_weights_dict = {i: 1.0 / self.num_domains for i in range(self.num_domains)}
        self.domain_list = list(sorted(self.train_domain_weights_dict.keys()))
        self.sampling_weights = torch.tensor([self.train_domain_weights_dict[domain] for domain in self.domain_list])

        self.pertoken_scores = []
        self.token_masks = []
        self.domain_ids = []

    def write_weights(self, weights):
        self.model.update_counter += 1
        self.model.train_domain_weights[:] = weights.float()
        self.model.avg_domain_weights[:] = (self.model.avg_domain_weights * (self.model.update_counter - 1) + weights) / self.model.update_counter

    def read_weights(self):
        return self.model.train_domain_weights.clone()

    def set_attributes(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update_domain_weights(self, scores, scores_mask, domain_ids):
        wandb_log_dict = {}
        train_domain_weights = self.read_weights()

        scores = scores.detach()
        domain_ids = domain_ids.detach()

        perdomain_scores = []
        for domain_id in range(len(train_domain_weights)):
            domain_mask = (domain_ids == domain_id)
            perdomain_scores_mask = scores_mask[domain_mask]

            if domain_mask.sum() > 0:
                curr_domain_scores = torch.clip(scores[domain_mask], min=0).mean()
            else:
                curr_domain_scores = self.model.perdomain_scores[domain_id]
            perdomain_scores.append(curr_domain_scores)

        self.model.perdomain_scores[:] = torch.tensor(perdomain_scores).float()
        log_new_train_domain_weights = torch.log(train_domain_weights) + self.reweight_eta * self.model.perdomain_scores
        log_new_train_domain_weights = log_new_train_domain_weights - torch.logsumexp(log_new_train_domain_weights, dim=0)
        train_domain_weights = (1-self.reweight_eps) * torch.exp(log_new_train_domain_weights) + self.reweight_eps / len(log_new_train_domain_weights)
        self.write_weights(train_domain_weights)

        for domain_idx in range(len(train_domain_weights)):
            domain_name = self.domain_list[domain_idx]
            wandb_log_dict[f'avg_domain_weights/{domain_name}'] = self.model.avg_domain_weights[domain_idx].item()
            wandb_log_dict[f'train_domain_weights/{domain_name}'] = self.model.train_domain_weights[domain_idx].item()
            wandb_log_dict[f'perdomain_scores/{domain_name}'] = self.model.perdomain_scores[domain_idx].item()
        wandb_log_dict['max_domain_id'] = domain_ids.max().item()
        wandb.log(wandb_log_dict, commit=False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        # domain_ids is expected to be [batch_size]
        domain_ids = inputs["domain_ids"]
        batch_size = domain_ids.size(0)  # e.g., 8

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            # Assume outputs.pertoken_loss is a flat tensor over all tokens.
            # Compute sequence length per sample.
            seq_length = outputs.pertoken_loss.numel() // batch_size  # e.g. 7056 // 8 = 882 tokens per sample
            pertoken_loss = outputs.pertoken_loss
            reference_pertoken_loss = outputs.reference_pertoken_loss
            token_mask = outputs.token_mask
            excess_loss = pertoken_loss - reference_pertoken_loss

        # Expand domain_ids from [batch_size] to [batch_size * seq_length]:
        domain_ids = domain_ids.unsqueeze(1).repeat(1, seq_length).view(-1)

        # -----------------------
        # Distributed gathering with padding for variable lengths
        # -----------------------
        # Determine local lengths (they may be different on each process)
        local_excess_len = torch.tensor([excess_loss.numel()], device=excess_loss.device)
        local_token_mask = torch.tensor([token_mask.numel()], device=token_mask.device)
        local_domain_len = torch.tensor([domain_ids.numel()], device=domain_ids.device)

        # Create lists to hold lengths from all processes.
        # Make sure the dtype matches local_excess_len and local_domain_len.
        excess_lens_list = [
            torch.zeros(1, device=excess_loss.device, dtype=local_excess_len.dtype)
            for _ in range(self.args.world_size)
        ]
        domain_lens_list = [
            torch.zeros(1, device=domain_ids.device, dtype=local_domain_len.dtype)
            for _ in range(self.args.world_size)
        ]
        token_mask_lens_list = [
            torch.zeros(1, device=token_mask.device, dtype=local_token_mask.dtype)
            for _ in range(self.args.world_size)
        ]

        # Gather lengths from all processes.
        dist.all_gather(excess_lens_list, local_excess_len)
        dist.all_gather(domain_lens_list, local_domain_len)
        dist.all_gather(token_mask_lens_list, local_token_mask)

        # Convert gathered lengths to Python ints.
        excess_lens = [int(t.item()) for t in excess_lens_list]
        domain_lens = [int(t.item()) for t in domain_lens_list]
        token_mask_lens = [int(t.item()) for t in token_mask_lens_list]

        # Determine maximum lengths across all processes.
        max_excess_len = max(excess_lens)
        max_domain_len = max(domain_lens)
        max_token_mask_len = max(token_mask_lens)

        # Pad excess_loss and domain_ids to the maximum length if needed.
        if excess_loss.numel() < max_excess_len:
            pad_size = max_excess_len - excess_loss.numel()
            excess_loss_padded = torch.nn.functional.pad(excess_loss, (0, pad_size))
        else:
            excess_loss_padded = excess_loss

        if domain_ids.numel() < max_domain_len:
            pad_size = max_domain_len - domain_ids.numel()
            domain_ids_padded = torch.nn.functional.pad(domain_ids, (0, pad_size))
        else:
            domain_ids_padded = domain_ids
        
        if token_mask.numel() < max_token_mask_len:
            pad_size = max_token_mask_len - token_mask.numel()
            token_mask_padded = torch.nn.functional.pad(token_mask, (0, pad_size))
        else:
            token_mask_padded = token_mask

        # Gather the padded tensors from all processes.
        if self.is_local_process_zero():
            gathered_excess_list = [
                torch.zeros(max_excess_len, device=excess_loss.device, dtype=excess_loss.dtype)
                for _ in range(self.args.world_size)
            ]
            gathered_domain_list = [
                torch.zeros(max_domain_len, device=domain_ids.device, dtype=domain_ids.dtype)
                for _ in range(self.args.world_size)
            ]
            gathered_token_mask_list = [
                torch.zeros(max_token_mask_len, device=token_mask.device, dtype=token_mask.dtype)
                for _ in range(self.args.world_size)
            ]

            dist.all_gather(gathered_excess_list, excess_loss_padded)
            dist.all_gather(gathered_domain_list, domain_ids_padded)
            dist.all_gather(gathered_token_mask_list, token_mask_padded)

            # Trim each gathered tensor to its valid length.
            gathered_excess_losses = []
            gathered_domain_ids = []
            gathered_token_masks = []

            for i in range(self.args.world_size):
                valid_excess = gathered_excess_list[i][:excess_lens[i]]
                valid_domain = gathered_domain_list[i][:domain_lens[i]]
                valid_token_mask = gathered_token_mask_list[i][:token_mask_lens[i]]
                gathered_excess_losses.append(valid_excess)
                gathered_domain_ids.append(valid_domain)
                gathered_token_masks.append(valid_token_mask)

            gathered_excess_losses = torch.cat(gathered_excess_losses, dim=0)
            gathered_domain_ids = torch.cat(gathered_domain_ids, dim=0)
            gathered_token_masks = torch.cat(gathered_token_masks, dim=0)

            self.pertoken_scores.append(gathered_excess_losses.detach())
            self.domain_ids.append(gathered_domain_ids.detach())
            self.token_masks.append(gathered_token_masks.detach())

                
            if len(self.pertoken_scores) == self.args.gradient_accumulation_steps:
                pertoken_scores = torch.cat(self.pertoken_scores, dim=0)
                domain_ids_all = torch.cat(self.domain_ids, dim=0)
                token_masks_all = torch.cat(self.token_masks, dim=0)

                # Update domain weights using the gathered per-token scores and domain IDs.
                self.update_domain_weights(pertoken_scores, token_masks_all, domain_ids_all)

                # Reset accumulators.
                self.pertoken_scores = []
                self.domain_ids = []
                self.token_masks = []
        else:
            # For non-zero ranks, still participate in the all_gather.
            dummy_excess = torch.zeros(max_excess_len, device=excess_loss.device, dtype=excess_loss.dtype)
            dummy_domain = torch.zeros(max_domain_len, device=domain_ids.device, dtype=domain_ids.dtype)
            dummy_token_mask = torch.zeros(max_token_mask_len, device=token_mask.device, dtype=token_mask.dtype)
            dummy_excess_list = [dummy_excess.clone() for _ in range(self.args.world_size)]
            dummy_domain_list = [dummy_domain.clone() for _ in range(self.args.world_size)]
            dummy_token_mask_list = [dummy_token_mask.clone() for _ in range(self.args.world_size)]
            dist.all_gather(dummy_excess_list, excess_loss_padded)
            dist.all_gather(dummy_domain_list, domain_ids_padded)
            dist.all_gather(dummy_token_mask_list, token_mask_padded)

        # -----------------------
        # Reweighted loss computation (local branch)
        # -----------------------
        # Read current domain weights (assumed to be a tensor of shape [num_domains]).
        train_domain_weights = self.read_weights().to(pertoken_loss.device).float()

        # Adjust domain weights if doing non-uniform sampling.
        train_domain_weights = train_domain_weights / self.sampling_weights.to(train_domain_weights.device)
        train_domain_weights = train_domain_weights / train_domain_weights.sum()

        # Use the local domain_ids (expanded to [batch_size * seq_length]) to index into domain weights.
        curr_domain_weights = train_domain_weights[domain_ids].expand_as(pertoken_loss).detach()

        # Renormalize: compute a normalizer across tokens.
        normalizer = curr_domain_weights.detach().sum()
        # Gather normalizer across GPUs.
        dist.all_reduce(normalizer, op=torch.distributed.ReduceOp.SUM)
        normalizer = torch.clip(normalizer, min=1e-10) / self.args.world_size

        # Compute the final weighted loss.
        loss = (pertoken_loss * curr_domain_weights.detach()).sum() / normalizer
        loss = loss.mean()  # Average the loss for multi-GPU training.

        loss = self.deepspeed.backward(loss)
        return loss

