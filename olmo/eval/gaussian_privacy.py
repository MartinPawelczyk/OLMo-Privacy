### BEGIN GAUSSIAN POISONING 
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

# MP: here to be consistent with the other evaluators.
# This is called within eval step to flag start of privacy evaluation.
class PrivacyEvaluator():
    def __init__(
        self,
        cfg: PrivacyEvaluatorConfig
    ):
        self.cfg = cfg

### END GAUSSIAN POISONING 
