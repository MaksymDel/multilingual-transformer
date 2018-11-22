# pylint: disable=arguments-differ,invalid-name
from typing import Tuple, Callable
import math

import torch
import torch.nn.functional as F

from allennlp.modules.layer_norm import LayerNorm

from transformer.common.transformer_utils import clones, attention
class PositionalEncoding(torch.nn.Module):
    "Implement the Positional Encoding function."
    def __init__(self, input_dim: int, max_len: int = 5000) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, input_dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return x + self.positional_encoding[:, :x.size(1)]


class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."
    def __init__(self, input_dim: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = torch.nn.Linear(input_dim, ff_dim)
        self.w_2 = torch.nn.Linear(ff_dim, input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoder(torch.nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer: torch.nn.Module, num_layers: int, return_all_layers: bool = False) -> None:
        super().__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.size)
        self.return_all_layers = return_all_layers

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        all_layers = []
        for layer in self.layers:
            x = layer(x, mask)
            if self.return_all_layers:
                all_layers.append(x)

        if self.return_all_layers:
            all_layers[-1] = self.norm(all_layers[-1])
            return all_layers
        return self.norm(x)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(torch.nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: torch.nn.Module,
                 dropout: float) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, input_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be a multiple of num_heads"
        # We assume d_v always equals d_k
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        self.linears = clones(torch.nn.Linear(input_dim, input_dim), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            # Same mask applied to all h heads.
            # Shape (batch_size, num_heads, timesteps, timesteps)
            mask = mask.unsqueeze(1).expand([-1, self.num_heads, -1, -1])

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [layer(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for layer, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


def make_model(num_layers: int = 6,
               input_size: int = 512, # Attention size
               hidden_size: int = 2048, # FF layer size
               heads: int = 8,
               dropout: float = 0.1,
               return_all_layers: bool = False) -> TransformerEncoder:
    "Helper: Construct a model from hyperparameters."
    attn = MultiHeadedAttention(heads, input_size, dropout)
    ff = PositionwiseFeedForward(input_size, hidden_size, dropout)
    model = TransformerEncoder(EncoderLayer(input_size, attn, ff, dropout),
                               num_layers,
                               return_all_layers=return_all_layers)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform(p)
    return model


############### DECODER LAYERS ##################
class TransformerDecoder(torch.nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(torch.nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

