from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL
from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BLEU

from fairseq.sequence_generator import SequenceGenerator
from fairseq.models.transformer import (TransformerEncoder,
                                        TransformerDecoder,
                                        TransformerModel,
                                        Embedding,
                                        base_architecture, transformer_iwslt_de_en, transformer_wmt_en_de
                                        )

from transformer.common.fairseq_wrapping_util import Args, Dictionary


@Model.register("multilingual_transformer_fseq")
class MultilingualTransformerFseq(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    architecture: ``str``, Predefined transformer architecture name
        Options: "transformer_iwslt_de_en" (default) / "transformer_wmt_en_de"

    """

    def __init__(self,
                 vocab: Vocabulary,
                 architecture: str = "transformer_iwslt_de_en",
                 use_bleu: bool = True) -> None:
        super(MultilingualTransformerFseq, self).__init__(vocab)
        self._source_namespace = "vocab_A"
        self._target_namespace = "vocab_B"

        # We need the end symbol to provide as the input at the first timestep of decoding, and
        # the same end symbol as a way to indicate the end of the decoded sequence.
        self._end_index = self.vocab.get_token_index(END_SYMBOL, namespace=self._target_namespace)

        self._src_dict = Dictionary(vocab, self._source_namespace,
                                    eos=END_SYMBOL,
                                    pad=vocab._padding_token,
                                    unk=vocab._oov_token)
        self._tgt_dict = Dictionary(vocab, self._target_namespace,
                                    eos=END_SYMBOL,
                                    pad=vocab._padding_token,
                                    unk=vocab._oov_token)

        if architecture == "transformer_iwslt_de_en":
            apply_architecture = transformer_iwslt_de_en
        elif architecture == "transformer_wmt_en_de":
            apply_architecture = transformer_wmt_en_de
        else:
            raise ConfigurationError("Typo in architecture name")

        args = Args()
        apply_architecture(args)

        self._model_A2B = self._build_transformer(args)

        # we should have a translator per each language direction
        self._translator_A2B = SequenceGenerator([self._model_A2B], self._tgt_dict, beam_size=7,
                                                 stop_early=True, maxlen=200)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                   self._target_namespace)  # pylint: disable=protected-access
            self.bleu = BLEU(exclude_indices={pad_index, self._end_index})
        else:
            self.bleu = None

    @overrides
    def forward(self,  # type: ignore
                tokens_A: Dict[str, torch.LongTensor] = None,
                tokens_B: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        tokens_A : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        tokens_B:  ````Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`. It is sequence that we want to teach our model to
           produce.
           The format is following: W1 W2 ... Wn EOS

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        encoder_input = self._prepare_encoder_input(source_tokens=tokens_A)
        encoder_output = self._model_A2B.encoder(**encoder_input)

        if tokens_B is not None:
            # Compute sequence logits
            decoder_input = self._prepare_decoder_input(target_tokens=tokens_B, encoder_output=encoder_output)
            logits, _ = self._model_A2B.decoder(**decoder_input)

            # Targets: <EOS, w1, w2, w3, PAD, PAD>
            # Desired: <w1, w2, w3, EOS, PAD, PAD>
            target_mask = util.get_text_field_mask(tokens_B)
            relevant_targets = move_eos_to_the_end(tokens_B["tokens"], target_mask).contiguous()

            # Compute loss
            loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, target_mask)

            # Update metrics
            predictions = logits.argmax(2)
            self.bleu(predictions, relevant_targets)

            return {"loss": loss, "predictions": predictions}

        else:
            list_of_dicts = self._translator_A2B.generate(encoder_input)
            best_predictions = [d[0]["tokens"].detach().cpu().numpy() for d in list_of_dicts]

            return {"predictions": best_predictions}

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
        if not isinstance(predicted_indices, numpy.ndarray) and not isinstance(predicted_indices, list):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]

            if not isinstance(indices, list):
                indices = list(indices)

            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    def _build_transformer(self, args) -> TransformerModel:
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = self._src_dict, self._tgt_dict

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)

            if path:
                raise ConfigurationError("Pretrained embeddings are not implemented yet")
            # if provided, load from preloaded dictionaries
            # if path:
            #    embed_dict = utils.parse_embedding(path)
            #    utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=False)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens, left_pad=False)
        return TransformerModel(encoder, decoder)

    @staticmethod
    def _prepare_encoder_input(source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        padding_mask = util.get_text_field_mask(source_tokens)
        source_tokens = source_tokens["tokens"]
        source_tokens, padding_mask = remove_eos_from_the_beginning(source_tokens, padding_mask)
        lengths = util.get_lengths_from_binary_sequence_mask(padding_mask)
        return {"src_tokens": source_tokens, "src_lengths": lengths}

    @staticmethod
    def _prepare_decoder_input(target_tokens: Dict[str, torch.Tensor], encoder_output: Dict[str, torch.Tensor]):
        target_tokens = target_tokens["tokens"]
        return {"prev_output_tokens": target_tokens, "encoder_out": encoder_output}

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self.bleu:
            all_metrics.update(self.bleu.get_metric(reset=reset))
        return all_metrics


def remove_eos_from_the_beginning(tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Our source sentences does not need EOS at the beginning while dataset reader appends it.
    """

    return tensor.clone()[:, 1:], mask.clone()[:, 1:]


def move_eos_to_the_end(tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assumes EOS is in the beginning. Useful to turn sequence that is an input for teacher forcing (EOS, w1, w2)
    to sequence that is suitable to compute loss (w1, w2, EOS). Takes padding into account.
    """
    batch_size = tensor.size(0)
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    eos_id = tensor[0][0]  # eos is the first symbol in all sequences
    tensor_without_eos, _ = remove_eos_from_the_beginning(tensor, mask)
    tensor_with_eos_at_the_end = torch.cat([tensor_without_eos, torch.zeros(batch_size, 1).long()], dim=1)
    for i, j in zip(range(batch_size), sequence_lengths):
        tensor_with_eos_at_the_end[i, j - 1] = eos_id

    return tensor_with_eos_at_the_end


def move_eos_to_the_beginning(tensor: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
    """
    Mask stays the same
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()

    tensor_with_eos_at_the_beginning = tensor.zeros_like(tensor)
    for i, j in enumerate(sequence_lengths):
        if j > 1:
            tensor_with_eos_at_the_beginning[i, 0] = tensor[i, (j - 1)]
            tensor_with_eos_at_the_beginning[i, 1:j] = tensor[i, :(j - 1)]

    return tensor_with_eos_at_the_beginning


def add_eos_to_the_beginning(tensor: torch.Tensor, eos_index: int):
    eos_column = tensor.new_full((tensor.size(0), 1), eos_index)
    return torch.cat([eos_column, torch.tensor], dim=1)
