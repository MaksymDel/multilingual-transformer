from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL
from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BLEU

# from fairseq.modules import (
#     AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
#     SinusoidalPositionalEmbedding
# )
#
# from fairseq.models import (
#     FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
#     register_model_architecture
# )

from fairseq.models.transformer import (TransformerEncoder, TransformerEncoderLayer,
                                        TransformerDecoder, TransformerDecoderLayer,
                                        TransformerModel,
                                        Embedding, LayerNorm, Linear, PositionalEmbedding,
                                        base_architecture, transformer_iwslt_de_en, transformer_wmt_en_de
                                        )

from fairseq.sequence_generator import SequenceGenerator

from transformer.common.fairseq_wrapping_util import Args, Dictionary


# TODO: work out BOS and EOS in taraining and generation (probably no eos)
# TODO: correct device for sequence generator
@Model.register("fairseq_transformer")
class FairseqTransformer(Model):
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
                 source_namespace: str,
                 target_namespace: str,
                 architecture: str = "transformer_iwslt_de_en",
                 use_bleu = True
                 ) -> None:
        super(FairseqTransformer, self).__init__(vocab)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace


        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._end_index = self.vocab.get_token_index(END_SYMBOL, namespace=self._target_namespace)

        self._src_dict = Dictionary(vocab, source_namespace,
                                    eos=END_SYMBOL,
                                    pad=vocab._padding_token,
                                    unk=vocab._oov_token)
        self._tgt_dict = Dictionary(vocab, target_namespace,
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
        args.left_pad = False

        apply_architecture(args)

        self._fairseq_transformer_model = self._build_fairseq_transformer_model(args)

        self._translator = SequenceGenerator([self._fairseq_transformer_model], self._tgt_dict, beam_size=7,
                                             stop_early=False, maxlen=100)

        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                   self._target_namespace)  # pylint: disable=protected-access
            self.bleu = BLEU(exclude_indices={pad_index, self._end_index})
        else:
            self.bleu = None

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                targets_for_teacher_forcing: Dict[str, torch.LongTensor] = None,
                targets_for_loss_computation: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        targets_for_teacher_forcing : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`. It is sequence that will be used as an input for
           teacher forcing.
           The format is following: EOS W1 W2 ... Wn
        targets_for_loss_computation :  ````Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`. It is sequence that we want to teach our model to
           produce.
           The format is following: W1 W2 ... Wn EOS

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        # fetch source padding mask
        source_padding_mask = util.get_text_field_mask(source_tokens)
        source_lengths = util.get_lengths_from_binary_sequence_mask(source_padding_mask)

        # unpack inputs
        source_tokens = source_tokens["tokens"]
        if targets_for_loss_computation is not None:
            targets_for_loss_computation = targets_for_loss_computation["token_ids"]
            targets_for_teacher_forcing = targets_for_teacher_forcing["tokens"]

        if targets_for_loss_computation is not None:
            model_input = (source_tokens, source_lengths, targets_for_teacher_forcing)
            net_output = self._fairseq_transformer_model(*model_input)
            lprobs = self._fairseq_transformer_model.get_normalized_probs(net_output, log_probs=True)
            lprobs_reshaped = lprobs.view(-1, lprobs.size(-1))
            loss = F.nll_loss(lprobs_reshaped, targets_for_loss_computation.view(-1),
                              size_average=False,
                              ignore_index=self._tgt_dict.pad(),
                              reduce=True)
            predictions = lprobs.argmax(2)
            output_dict = {"loss": loss, "predictions": predictions}
            self.bleu(predictions, targets_for_loss_computation)

        else:
            encoder_input = {"src_tokens": source_tokens, "src_lengths": source_lengths}
            list_of_dicts = self._translator.generate(encoder_input)
            best_predictions = [d[0]["tokens"].detach().cpu().numpy() for d in list_of_dicts]
            output_dict = {"predictions": best_predictions}

        return output_dict

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

    def _build_fairseq_transformer_model(self, args) -> TransformerModel:
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

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens, left_pad=args.left_pad)
        return TransformerModel(encoder, decoder)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self.bleu:
            all_metrics.update(self.bleu.get_metric(reset=reset))
        return all_metrics
