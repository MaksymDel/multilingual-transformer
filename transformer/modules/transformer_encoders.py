# pylint: disable=arguments-differ,invalid-name
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from transformer.common.transformer_utils import subsequent_mask, reshape_padding_mask
from transformer.modules.transformer_layers import *


@Seq2SeqEncoder.register("base_transformer_encoder")
class BaseTransformerEncoder(Seq2SeqEncoder):

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 1024,
                 num_heads: int = 4,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 input_dropout: float = None,
                 return_all_layers=False) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._return_all_layers = return_all_layers

        attn = MultiHeadedAttention(num_heads, input_dim, dropout)
        ff = PositionwiseFeedForward(input_dim, hidden_dim, dropout)
        self._transformer_encoder = TransformerEncoder(EncoderLayer(input_dim, attn, ff, dropout), num_layers,
                                                       return_all_layers=return_all_layers)

        self._position = PositionalEncoding(input_dim)
        # Initialize parameters with Glorot / fan_avg.
        for p in self._transformer_encoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        if input_dropout:
            self._dropout = torch.nn.Dropout(input_dropout)
        else:
            self._dropout = lambda x: x

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor):
        mask = reshape_padding_mask(mask.int())
        token_embeddings = self._position(token_embeddings)
        token_embeddings = self._dropout(token_embeddings)
        encoder_outputs = self._transformer_encoder(token_embeddings, mask)
        return encoder_outputs

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self) -> bool:
        return False


class BidirectionalTransformerEncoder(Seq2SeqEncoder):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 input_dropout: float = None,
                 return_all_layers: bool = False) -> None:
        super().__init__()

        self._return_all_layers = return_all_layers
        self.transformer_layers = num_layers
        self.num_layers = num_layers

        self._forward_transformer = make_model(input_size=input_dim,
                                               hidden_size=hidden_dim,
                                               num_layers=num_layers,
                                               dropout=dropout,
                                               return_all_layers=return_all_layers)
        self._backward_transformer = make_model(input_size=input_dim,
                                                hidden_size=hidden_dim,
                                                num_layers=num_layers,
                                                dropout=dropout,
                                                return_all_layers=return_all_layers)
        self._position = PositionalEncoding(input_dim)

        self.input_dim = input_dim
        self.output_dim = 2 * input_dim

        if input_dropout:
            self._dropout = torch.nn.Dropout(input_dropout)
        else:
            self._dropout = lambda x: x

        self.should_log_activations = False

    def get_attention_masks(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns 2 masks of shape (batch_size, timesteps, timesteps) representing
        1) non-padded elements, and
        2) elements of the sequence which are permitted to be involved in attention at a given timestep.
        """
        device = mask.device
        # Forward case:
        timesteps = mask.size(1)
        # Shape (1, timesteps, timesteps)
        subsequent = subsequent_mask(timesteps, device).int()
        # Broadcasted logical and - we want zero
        # elements where either we have padding from the mask,
        # or we aren't allowed to use the timesteps.
        # Shape (batch_size, timesteps, timesteps)
        forward_mask = mask.unsqueeze(-1) & subsequent
        # Backward case - exactly the same, but transposed.
        backward_mask = forward_mask.transpose(1, 2)

        return forward_mask, backward_mask

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        forward_mask, backward_mask = self.get_attention_masks(mask.int())
        token_embeddings = self._position(token_embeddings)
        token_embeddings = self._dropout(token_embeddings)
        forward_output = self._forward_transformer(token_embeddings, forward_mask)
        backward_output = self._backward_transformer(token_embeddings, backward_mask)

        if self._return_all_layers:
            to_return = []
            for forward, backward in zip(forward_output, backward_output):
                to_return.append(torch.cat([forward, backward], -1))
            return to_return

        return torch.cat([forward_output, backward_output], -1)

    def get_regularization_penalty(self):
        return 0.

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def is_bidirectional(self) -> bool:
        return True
