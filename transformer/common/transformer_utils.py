# pylint: disable=arguments-differ,invalid-name
from typing import Tuple, Callable
import math
import copy

import numpy as np
import torch
import torch.nn.functional as F


def subsequent_mask(size: int, device: str = 'cpu', size2: int = None) -> torch.Tensor:
    """Mask out subsequent positions."""
    if size2 is None:
        size2 = size
    attn_shape = (1, size, size2)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = (torch.from_numpy(mask) == 0)
    return mask.to(device)


def to_subsequent_mask(mask: torch.Tensor, timesteps_value=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns 1 masks of shape (batch_size, timesteps, timesteps) representing
    1) non-padded elements, and
    2) elements of the sequence which are permitted to be involved in attention at a given timestep.
    """
    device = mask.device
    timesteps_key = mask.size(1)
    if timesteps_value is None:
        timesteps_value = timesteps_key
    # Shape (1, timesteps_key, timesteps_value)
    subsequent = subsequent_mask(timesteps_key, device, size2=timesteps_value).int()
    # Broadcasted logical and - we want zero
    # elements where either we have padding from the mask,
    # or we aren't allowed to use the timesteps.
    # Shape (batch_size, timesteps_key, timesteps_value)
    final_mask = mask.unsqueeze(-1) & subsequent

    return final_mask


def get_target_to_soruce_mask(source_mask: torch.Tensor, timesteps_target) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns 1 masks of shape (batch_size, timesteps, timesteps) representing
    1) non-padded elements, and
    2) elements of the sequence which are permitted to be involved in attention at a given timestep.
    """
    device = source_mask.device
    timesteps_key = timesteps_target
    timesteps_value = source_mask.size(1)

    # Shape (1, timesteps_key, timesteps_value)
    subsequent = subsequent_mask(timesteps_key, device, size2=timesteps_value).int()
    # Broadcasted logical and - we want zero
    # elements where either we have padding from the mask,
    # or we aren't allowed to use the timesteps.
    # Shape (batch_size, timesteps_key, timesteps_value)
    final_mask = source_mask.unsqueeze(1) & subsequent

    return final_mask


def reshape_padding_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns 1 mask of shape (batch_size, timesteps, timesteps) representing
    1) non-padded elements
    """
    device = mask.device
    # Forward case:
    timesteps = mask.size(1)
    # Shape (1, timesteps, timesteps)
    subsequent = subsequent_mask(timesteps, device).int()
    subsequent = torch.ones_like(subsequent)
    # Broadcasted logical and - we want zero
    # elements where either we have padding from the mask,
    # or we aren't allowed to use the timesteps.
    # Shape (batch_size, timesteps, timesteps)
    reshaped_mask = mask.unsqueeze(-1) & subsequent

    return reshaped_mask


def clones(module: torch.nn.Module, num_copies: int) -> torch.nn.ModuleList:
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def attention(query: torch.Tensor,
              key: torch.Tensor,
              value: torch.Tensor,
              mask: torch.Tensor = None,
              dropout: Callable = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        try:
            scores = scores.masked_fill(mask == 0, -1e9)
        except RuntimeError:
            print(scores.size(), mask.size())
            raise RuntimeError
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

