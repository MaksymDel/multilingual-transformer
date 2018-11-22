from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn.modules.linear import Linear

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BLEU
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.beam_search import BeamSearch
from allennlp.modules.time_distributed import TimeDistributed

from transformer.modules.transformer_decoders import BaseTransformerDecoder
from transformer.common.transformer_utils import to_subsequent_mask, get_target_to_soruce_mask

@Model.register("transformer")
class Transformer(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
     : ``str``, Predefined transformer architecture name
        Options: "transformer_iwslt_de_en" (default) / "transformer_wmt_en_de"

    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 target_namespace: str = "target_words",
                 max_decoding_steps: int = 128,
                 beam_size: int = None,
                 label_smoothing_ratio: int = None,
                 use_bleu: bool = True
                 ) -> None:
        super(Transformer, self).__init__(vocab)
        assert source_field_embedder.get_output_dim() == encoder.get_input_dim()
        # Dense embedding of source vocab tokens.
        self._source_field_embedded = source_field_embedder

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        self._decoder = BaseTransformerDecoder()

        self._target_namespace = target_namespace
        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        self._target_embedding = Embedding(num_classes, self._decoder.get_input_dim())

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = TimeDistributed(Linear(self._decoder.get_output_dim(), num_classes))
        self._max_decoding_steps = max_decoding_steps

        self._target_namespace = target_namespace
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, namespace=self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, namespace=self._target_namespace)

        self._label_smoothing_ratio = label_smoothing_ratio

        # At prediction time, we can use a beam search to find the most likely sequence of target tokens.
        # If the beam_size parameter is not given, we'll just use a greedy search (equivalent to beam_size = 1).
        self._max_decoding_steps = max_decoding_steps
        if beam_size is not None:
            raise NotImplementedError
            self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)
        else:
            self._beam_search = None

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                   self._target_namespace)  # pylint: disable=protected-access
            self.bleu = BLEU(exclude_indices={pad_index, self._end_index})
        else:
            self.bleu = None

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        # Encode source
        source_padding_mask = util.get_text_field_mask(source_tokens)
        embedded_source = self._source_field_embedded(source_tokens)
        encoded_source = self._encoder(embedded_source, source_padding_mask)

        if target_tokens is not None:
            return self._forward_with_targets(encoded_source, source_padding_mask, target_tokens)
        else:
            return self._greedy_decode(encoded_source, source_padding_mask)

    def _greedy_decode(self, encoded_source, source_padding_mask):
        batch_size = source_padding_mask.size(0)

        # init predictions pull
        predictions = torch.ones(batch_size, 1).fill_(self._start_index).long().to(encoded_source.device)

        for step in range(self._max_decoding_steps):
            # shape: bs x step x vocab_size
            logits = self._transform(encoded_source, source_padding_mask, {"tokens": predictions})
            # take token index from logits for the new predicted word
            # shape: bs x 1
            new_predictions = logits[:, -1].argmax(-1, True)
            # add new predictions to predictions pull
            predictions = torch.cat([predictions, new_predictions], dim=1)

        return {"predictions": predictions}

    def _forward_with_targets(self, encoded_source, source_padding_mask, target_tokens):
        target_paddding_mask = util.get_text_field_mask(target_tokens)
        logits = self._transform(encoded_source, source_padding_mask, target_tokens)

        # Compute loss
        loss = self._get_loss(logits, target_tokens["tokens"], target_paddding_mask, self._label_smoothing_ratio)

        # Get predicted indexes
        predictions = logits.argmax(-1)

        # Update metrics
        self.bleu(predictions, target_tokens["tokens"])

        # Return result dict
        output_dict = {"loss": loss,
                       "predictions": predictions}

        return output_dict

    def _transform(self, encoded_source, source_padding_mask, target_tokens):
        # Decode conditioning on source
        # embed targets
        embedded_target = self._target_embedding(target_tokens["tokens"])

        # get padding mask
        target_paddding_mask = util.get_text_field_mask(target_tokens)

        # get attention mask to hide padding and future timesteps
        only_target_subsequent_mask = to_subsequent_mask(target_paddding_mask.int())

        # get attention mask to hide source padding elements
        target_to_source_padding_mask = get_target_to_soruce_mask(source_padding_mask.int(),
                                                                  target_paddding_mask.size(
                                                                      1))  # torch.Size([1, 4, 5, 5]) torch.Size([1, 4, 3, 5])

        # print(embedded_target.size(), encoded_source.size(), source_padding_mask.size(), target_subsequent_mask.size())
        decoder_outputs = self._decoder(embedded_target,
                                        encoded_source,
                                        target_to_source_padding_mask,
                                        only_target_subsequent_mask)

        # Project decoder outputs back onto vocabulary
        logits = self._output_projection_layer(decoder_outputs)

        return logits

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)

            # Collect indices till the first end_symbol inclusive
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)+1]
            # Insert start index, if we fed it as an input
            if indices[0] != self._start_index:
                indices.insert(0, self._start_index)

            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor,
                  label_smoothing_ratio: int = None) -> torch.Tensor:
        """
        Compute loss.
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.
        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, max_len)
        # relevant_targets = targets[:, 1:].contiguous()
        relevant_targets = targets

        # shape: (batch_size, max_len)
        #relevant_mask = target_mask[:, 1:].contiguous()
        relevant_mask = target_mask

        return util.sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask,
                                                       label_smoothing=label_smoothing_ratio)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self.bleu:
            all_metrics.update(self.bleu.get_metric(reset=reset))
        return all_metrics
