# pylint: disable=arguments-differ,invalid-name
import copy

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from transformer.modules.transformer_layers import *
from transformer.common.transformer_utils import reshape_padding_mask

@Seq2SeqEncoder.register("base_transformer_decoder")
class BaseTransformerDecoder(Seq2SeqEncoder):

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 1024,
                 num_heads: int = 4,
                 num_layers: int = 6,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._position = PositionalEncoding(input_dim)

        attn = MultiHeadedAttention(num_heads, input_dim, dropout)
        ff = PositionwiseFeedForward(input_dim, hidden_dim, dropout)
        self._transformer_decoder = TransformerDecoder(
            DecoderLayer(input_dim, copy.deepcopy(attn), copy.deepcopy(attn), ff, dropout), num_layers)

        # Initialize parameters with Glorot / fan_avg.
        for p in self._transformer_decoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, embedded_targets: torch.Tensor, encoder_outputs: torch.Tensor,
                target_to_source_padding_mask: torch.Tensor, only_target_subsequent_mask: torch.Tensor):
        """

        :param embedded_targets: embedded targets
        :param encoder_outputs: from seq2seq encoder
        :param source_padding_mask: raw padding mask (B, MAX_T)
        :param target_attention_mask:  prepared target mask depending on teacher forcing
        :return:
        """
        embedded_targets = self._position(embedded_targets)

        logits = self._transformer_decoder(embedded_targets, encoder_outputs, target_to_source_padding_mask,
                                           only_target_subsequent_mask)
        return logits

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self) -> bool:
        return False


