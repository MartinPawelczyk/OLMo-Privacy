from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric

from ..config import EvaluatorType, PrivacyEvaluatorConfig
from .downstream import ICLMetric
from .evaluator import Evaluator
from ..model import OLMo
from ..tokenizer import Tokenizer
from ..torch_util import get_global_rank, get_world_size
from tqdm import tqdm
from pathlib import Path
import logging

__all__ = ["PrivacyEvaluator"]


class PrivacyEvaluator():
    """
    Evaluator for privacy risk analysis.
    Computes and input gradients of loss for individual training examples.
    Computes dot products of loss gradients (with respect to the model's input embeddings) and Gaussian noise added to the clean input embeddings.
    This is essentially computes the correlation between the noise and the model and analyzes the privacy risk of the model with respect to the Gaussian noise added to the input embeddings.
    This is the Gaussian Privacy Score from "Pawelczyk et al. (2025); Machine unlearning Fails to Remove Data Poisoning Attacks; ICLR 2025".
    """
    def __init__(
        self,
        cfg: PrivacyEvaluatorConfig,
        data_loader: DataLoader,
    ):
        self.cfg = cfg
        self.data_loader = data_loader

    def evaluate(self, model: OLMo) -> Dict[str, float]:
        log.info(f"Starting privacy evaluation at step {step}...")
        model.eval()  # Set model to evaluation mode        
        return # self.gaussian_privacy_score(model self.cfg)

    '''
    def compute_causal_loss(self, pred_scores:torch.tensor, labels:torch.tensor) -> torch.tensor:
        """ Computes the causal language modeling loss for a given set of prediction scores and labels.
        Args:
            pred_scores (torch.tensor): The prediction scores from the model, shape: (batch_size, sequence_length, vocab_size).
            labels (torch.tensor): The ground truth labels, shape: (batch_size, sequence_length).
        Returns:
            torch.tensor: The computed loss value.
        """
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L1050
        # we are doing next-token prediction; shift prediction scores and input ids by one
        
        shift_logits = pred_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return lm_loss

    def _compute_dot(self, G: torch.tensor, N: torch.tensor, n_std: float) -> torch.tensor:
        """
        Computes the dot product between gradients and Gaussian noise, normalized by the standard deviation.
        
        Args:
            gradient: input gradients of loss with respect to input embeddings, shape: (batch_size, embedding_dim)
            noise: Gaussian noise added to the input embeddings, shape: (batch_size, embedding_dim)
            n_std: standard deviation of the Gaussian noise, a scalar
        Returns:
            torch.tensor: dot products of the gradients and noise, shape: (batch_size,)
        """
        # MP: must be double precision, otherwise overflow for large nets
        gradient = gradient.to(torch.float64)
        noise = noise.to(torch.float64)

        dot_prods = (gradient*noise) / n_std
        dot_prods = torch.sum(dot_prods, dim=1)
        dot_prods = dot_prods / torch.norm(gradient, 2, dim=1)
        dot_prods = dot_prods.detach()
        return dot_prods

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def gaussian_privacy_score(self, model) -> torch.tensor:

        """ Computes the privacy score based on the dot product of input gradients and Gaussian noise.
        Args:
            model (OLMo): The OLMo model to evaluate.
            loader (DataLoader): (Training) DataLoader for the evaluation of privacy risk.
            poison_delta (torch.tensor): Gaussian noise added to the input embeddings, shape: (batch_size, embedding_dim).
        Returns:
            torch.tensor: The computed privacy score, which is the dot product of gradients and noise.
        """

        gradients = []
        for batch_id, batch in enumerate(self.data_loader):

            micro_batches = self.split_batch(batch)


            inputs_embeds = batch['input_embeds']
            attention_mask = batch['attention_mask']
            
            pred_scores = model(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask
            ).logits
            labels = batch['input_ids']
            
            lm_loss = self.compute_causal_loss(pred_scores, labels)
            
            # zero out any old gradients before the backward pass
            model.zero_grad()
            
            # compute gradient of this instance's loss wrt its input embeddings
            inputs_embeds.requires_grad = True  # IMPORTANT: track gradients for this tensor
            lm_loss.backward(retain_graph=True)  # retain_graph for the next instance in the batch
            # Store the gradient. Detach to prevent further graph tracking.
            if inputs_embeds.grad is None:
                raise ValueError(f"Gradient for inputs_embeds is None at index {idx}. Check your model and data loader.")
            grads = torch.autograd.grad(lm_loss, inputs_embeds)[0].detach().float()
            gradient = grads.flatten(1)
            gradients.append(gradient)
        
        # concatenate all gradients
        if len(gradients) == 0:
            raise ValueError("No gradients computed. Check your data loader and model.")
        if len(gradients) == 1:
            gradients = gradients[0]
        else:
            # concatenate along the batch dimension
            gradients = torch.cat(gradients)

        # get noise
        noise = poison_delta.flatten(1)
        
        # compute dot products & return
        dot = self._compute_dot(gradients, noise, n_std)
        return dot 

    '''