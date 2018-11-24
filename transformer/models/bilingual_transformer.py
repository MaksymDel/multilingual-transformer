from typing import Dict, Any

import numpy
from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import END_SYMBOL
from allennlp.nn import util
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import BLEU

from fairseq.models.transformer import transformer_iwslt_de_en, transformer_wmt_en_de


from transformer.nn.bridging_helpers import (Args,
                                             Dictionary,
                                             BeamSearchSequenceGenerator,
                                             run_encoder,
                                             run_decoder,
                                             move_eos_to_the_end)

from transformer.nn.modules_builders import build_transformer_encoder_and_decoder


@Model.register("bilingual_transformer")
class BilingualTransformer(Model):
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
                 label_smoothing: float = None,
                 overrides: Dict[str, Any] = None,
                 architecture: str = None,
                 use_bleu: bool = True) -> None:
        super(BilingualTransformer, self).__init__(vocab)
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
        # build encoder and decoder
        self._encoder, self._decoder = build_transformer_encoder_and_decoder(args, self._src_dict, self._tgt_dict)

        # build translator that takes source tokens directly
        self._translator_A2B = BeamSearchSequenceGenerator(self._encoder, self._decoder, self._tgt_dict)

        self._label_smoothing = label_smoothing
        if use_bleu:
            pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                   self._target_namespace)  # pylint: disable=protected-access
            self.bleu = BLEU(exclude_indices={pad_index, self._end_index})
        else:
            self.bleu = None

    @overrides
    def forward(self,  # type: ignore
                tokens_A: Dict[str, torch.LongTensor],
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
        validation = not self.training and tokens_B is not None

        if self.training or validation:
            # Shape: src_len x batch_size x num_features
            encoder_output = run_encoder(encoder=self._encoder, source_tokens=tokens_A)

            # Shape: batch_size x tgt_len x num_features
            logits, _ = run_decoder(decoder=self._decoder, target_tokens=tokens_B, encoder_output=encoder_output)

            # Targets: <EOS, w1, w2, w3, PAD, PAD>
            # Desired: <w1, w2, w3, EOS, PAD, PAD>
            target_mask = util.get_text_field_mask(tokens_B)
            relevant_targets = move_eos_to_the_end(tokens_B["tokens"], target_mask).contiguous()

            # Compute loss
            loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, target_mask,
                                                           label_smoothing=self._label_smoothing)

            # Update metrics
            predictions = logits.argmax(2)
            self.bleu(predictions, relevant_targets)

            return {"loss": loss, "predictions": predictions}

        else:
            best_predictions = self._translator_A2B.generate(tokens_A)
            return {"predictions": best_predictions}

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
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self.bleu:
            all_metrics.update(self.bleu.get_metric(reset=reset))
        return all_metrics
