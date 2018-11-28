from typing import Dict, List

import numpy
from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL
from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BLEU, Average
from allennlp.common import Params

from fairseq.models.transformer import transformer_iwslt_de_en, transformer_wmt_en_de

from transformer.nn.bridging_helpers import (Args,
                                             Dictionary,
                                             BeamSearchSequenceGenerator,
                                             run_encoder,
                                             run_decoder,
                                             move_eos_to_the_end)

from transformer.nn.modules_builders import build_transformer_encoder_and_decoder


@Model.register("separate_encdec")
class SeparateEncoderDecoder(Model):
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
                 overrides: Params = None,
                 label_smoothing: float = None,
                 architecture: str = "transformer_iwslt_de_en",
                 identity_loss: bool = False,
                 bleu_metric: bool = True,
                 loss_metric: bool = False) -> None:
        super(SeparateEncoderDecoder, self).__init__(vocab)

        self._tags_namespace = "language_tags"
        self._target_namespace = None  # has to be set for each instance in forward
        self._identity_loss = identity_loss
        self._label_smoothing = label_smoothing

        language_tags = vocab.get_token_to_index_vocabulary(namespace="language_tags").keys()

        # init dictionaries that fairseq components expect
        self._fseq_dictionaries = {}

        for tag in language_tags:
            self._fseq_dictionaries[tag] = Dictionary(vocab,
                                                      namespace="vocab_" + tag,
                                                      eos=END_SYMBOL,
                                                      pad=vocab._padding_token,
                                                      unk=vocab._oov_token)

        if architecture is not None:
            if architecture == "transformer_iwslt_de_en":
                apply_architecture = transformer_iwslt_de_en
            elif architecture == "transformer_wmt_en_de":
                apply_architecture = transformer_wmt_en_de
            else:
                raise ConfigurationError("Typo in architecture name")
        else:
            apply_architecture = lambda x: x

        # create args in format fairseq expects
        args = Args()
        apply_architecture(args)  # set some args based on predefined architecture
        if overrides is not None:  # override some args
            for arg_name, arg in overrides.as_dict().items():
                setattr(args, arg_name, arg)

        # build encoders and decoders
        self._encoders = {}
        self._decoders = {}

        for tag in language_tags:
            tag_dict = self._fseq_dictionaries[tag]
            encoder, decoder = build_transformer_encoder_and_decoder(args, tag_dict, tag_dict)
            self._encoders[tag] = encoder
            self._decoders[tag] = decoder

        self._encoders = torch.nn.ModuleDict(self._encoders)
        self._decoders = torch.nn.ModuleDict(self._decoders)

        if identity_loss:
            all_lang_directions = [(l1, l2) for l1 in language_tags for l2 in language_tags]
        else:
            all_lang_directions = [(l1, l2) for l1 in language_tags for l2 in language_tags if l1 != l2]

        # build translators for all language directions since they are just cheap wrappers with no additional parameters
        self._translators = {}
        for l_src, l_tgt in all_lang_directions:
            self._translators[l_src + "->" + l_tgt] = BeamSearchSequenceGenerator(self._encoders[l_src],
                                                                                  self._decoders[l_tgt],
                                                                                  self._fseq_dictionaries[l_tgt])

        # init bleu metrics for all directions
        if bleu_metric:
            self._all_bleu_metrics = {}
            for l_src, l_tgt in all_lang_directions:
                pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                       "vocab_" + l_tgt)  # pylint: disable=protected-access
                end_index = self.vocab.get_token_index(END_SYMBOL,
                                                       "vocab_" + l_tgt)  # pylint: disable=protected-access
                self._all_bleu_metrics[l_src + "->" + l_tgt] = BLEU(exclude_indices={pad_index, end_index})
        else:
            self._all_bleu_metrics = None

        if loss_metric:
            self._all_loss_metrics = {}
            for l_src, l_tgt in all_lang_directions:
                self._all_loss_metrics[l_src + "->" + l_tgt] = Average()
        else:
            self._all_loss_metrics = None

    @overrides
    def forward(self,  # type: ignore
                language_a: List[str],
                language_a_indexed: torch.LongTensor,
                sentence_a: Dict[str, torch.LongTensor],
                language_b: List[str],
                language_b_indexed: torch.LongTensor,
                sentence_b: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        The forward method process each sentence pair in both ways: "EN-FR" becomes ("EN->FR", "FR->EN").
        And if identity loss attribute is true, also ("EN->EN", "FR->FR")
        ```
        TODO: Add Discriminator / PCL losses and metrics for them.

        Params:
        sentence_pairs: Size: batch_size x num_languages x 2 (num_parallel_sentences) x num_tokens
        """
        language_a, language_b = language_a[0], language_b[0]  # we always have the same language inside one batch
        inference = sentence_b is None
        validation = sentence_b is not None and not self.training

        if self.training or validation:
            sentence_pair_loss = self._forward_on_sentence_pair(sentence_a, language_a, sentence_b, language_b)
            return {"loss": sentence_pair_loss}
        else:
            assert inference
            language_direction = language_a + "->" + language_b
            return self._forward_seq2seq(language_direction, sentence_a)

    def _forward_on_sentence_pair(self, sentence_a, language_a, sentence_b, language_b):
        a2a_tag = language_a + "->" + language_b
        b2a_tag = language_b + "->" + language_a

        loss_a2b = self._forward_seq2seq(a2a_tag, sentence_a, sentence_b)["loss"]
        loss_b2a = self._forward_seq2seq(b2a_tag, sentence_b, sentence_a)["loss"]
        loss = loss_a2b + loss_b2a

        if self._identity_loss:
            a2a_tag = language_a + "->" + language_a
            b2b_tag = language_b + "->" + language_b
            loss_a2a = self._forward_seq2seq(a2a_tag, sentence_a, sentence_a)["loss"]
            loss_b2b = self._forward_seq2seq(b2b_tag, sentence_b, sentence_b)["loss"]
            loss = loss + loss_a2a + loss_b2b

        return loss

    def _forward_seq2seq(self, language_direction, source_tokens, target_tokens=None):
        if target_tokens is None:  # if called for inference
            return {"predictions": self._translators[language_direction](source_tokens)}

        source_lang_tag, target_lang_tag = language_direction.split("->")
        self._target_namespace = "vocab_" + target_lang_tag  # target namespace is different for each instance

        # Shape: src_len x batch_size x num_features
        encoder_output = run_encoder(encoder=self._encoders[source_lang_tag], source_tokens=source_tokens)

        # Shape: batch_size x tgt_len x num_features
        logits, _ = run_decoder(decoder=self._decoders[target_lang_tag],
                                target_tokens=target_tokens,
                                encoder_output=encoder_output)

        # Targets: <EOS, w1, w2, w3, PAD, PAD>
        # Desired: <w1, w2, w3, EOS, PAD, PAD>
        target_mask = util.get_text_field_mask(target_tokens)
        relevant_targets = move_eos_to_the_end(target_tokens["tokens"], target_mask).contiguous()

        # Compute loss
        loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, target_mask,
                                                       label_smoothing=self._label_smoothing)

        # Update metrics
        predictions = logits.argmax(2)

        if self._all_bleu_metrics is not None:
            self._all_bleu_metrics[language_direction](predictions, relevant_targets)

        if self._all_loss_metrics is not None:
            self._all_loss_metrics[language_direction](loss.item())

        return {"loss": loss, "predictions": predictions}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method does not do "decoding" in a sense of encoder-decoder architecture.

        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``run_decoder`` function.

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
            end_index = self.vocab.get_token_index(END_SYMBOL, namespace=self._target_namespace)
            if end_index in indices:
                indices = indices[:indices.index(end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}

        if self._all_bleu_metrics is not None:
            for lang_dir, bleu in self._all_bleu_metrics.items():
                all_metrics["BLEU_" + "-".join(lang_dir.split("->"))] = bleu.get_metric(reset=reset)["BLEU"]

        if self._all_loss_metrics is not None:
            for lang_dir, loss in self._all_loss_metrics.items():
                all_metrics["loss_" + "-".join(lang_dir.split("->"))] = loss.get_metric(reset=reset)

        return all_metrics
