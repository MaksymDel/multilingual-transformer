from typing import Dict

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


@Model.register("multilingual_transformer")
class MultilingualTransformer(Model):
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
                 identity_loss: bool = False,
                 use_bleu: bool = True) -> None:
        super(MultilingualTransformer, self).__init__(vocab)

        self._tags_namespace = "language_tags"
        self._target_namespace = None  # has to be set for each instance in forward
        self._identity_loss = identity_loss

        language_tags = vocab.get_token_to_index_vocabulary(namespace="language_tags").keys()

        # init dictionaries that fairseq components expect
        self._fseq_dictionaries = {}

        for tag in language_tags:
            self._fseq_dictionaries[tag] = Dictionary(vocab,
                                                      namespace="vocab_" + tag,
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
        if use_bleu:
            self._all_bleu_metrics = {}
            for l_src, l_tgt in all_lang_directions:
                pad_index = self.vocab.get_token_index(self.vocab._padding_token,
                                                       "vocab_" + l_tgt)  # pylint: disable=protected-access
                end_index = self.vocab.get_token_index(END_SYMBOL,
                                                       "vocab_" + l_tgt)  # pylint: disable=protected-access
                self._all_bleu_metrics[l_src + "->" + l_tgt] = BLEU(exclude_indices={pad_index, end_index})
        else:
            self._all_bleu_metrics = None


    @overrides
    def forward(self,  # type: ignore
                source_lang_id: torch.LongTensor,
                source_tokens: Dict[str, torch.LongTensor],
                target_lang_id: torch.LongTensor,
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        # TODO: see below description
        By language pair/direction I mean just tags.
        By sentence pair/direction I mean concrete pair of sentences

        ---
        Dataset reader should return a list of sentence pairs (where each sentence pair is 2 text fields in sublist)

        Also it should return a list of language tags for sentence pairs (list of lists of label fields).
        I will use them (after they are indexed with LabelField's as_tensor(...) method) for discriminator training.

        The third thing it should return is metadata field with language pairs tags.
        E.g.: dict["language_pairs"] = ["EN-ET", "FR-DE", "ET-FI"] etc.
        I can derive them from label fields, but I think this way it is cleaner since I suppose
        that MetadataFields are cheap.

        At test time dataset reader returns a single Text Field called "test_time_source" and MetaData field
        containing language direction like "EN->ET".
        ---

        ---
        The forward method then process each sentence pair in both ways: "EN-FR" becomes ("EN->FR", "FR->EN").
        And if identity loss atribute is true, also ("EN->EN", "FR->FR")

        This class's code should have following methods: "._forward_on_setnece_direction", "_forward_seq2seq",
        Then the logic of the main forward methods becomes roughly:
        ```
        if training or validation:
            loss = 0
            for lang_pair_tag, sentence_pair in zip(language_pairs, instance_sentence_pairs):
                sp_loss = self._forward_on_sentence_pair(lang_pair_tag, sentence_pair)
                loss += sp_loss

            return loss

        if test_time:
            self._forward_seq2seq(lang_dir_tag, source_tokens)
        ```
        Where:
        ```
        def _forward_on_sentence_pair(self, lang_pair_tag, sentence_pair):
            sentence_X, sentence_Y = sentence_pair
            X_tag, Y_tag = lang_pair_tag.split("-")
            X2Y_tag = X_tag + "->" + Y_tag
            Y2X_tag = Y_tag + "->" + X_tag

            loss_X2Y = self._forward_seq2seq(X2Y_tag, sentence_X, sentence_Y)["loss"]
            loss_Y2X = self._forward_seq2seq(Y2X_tag, sentence_Y, sentence_X)["loss"]
            return loss_X2Y + loss_Y2X
        ```

        Metrics (language direction BLEU and/or losses) are updated inside _forward_seq2seq(...) function)
        Discriminator / PCL losses and metrics are also computed inside "_forward_seq2seq" function.
        ---

        TODO 2: More on DatasetReader
        1) Add a note to dataset that it currently works with synchronized epochs without duplicating sentences
        in shorter datasets to match the ones of longer ones.

        2) MB make a option with duplication at some point

        3) Rename dataset and model to something like "multilingual_separate" as opposite to "multilingual_shared"

        4) Dataset reader should now accept several parallel files, e.g.: "tarin.en-et, train.de-en, train.fr-et"
        We then do path.split(" ,") and parse them with itertools.izip or something that returns None in place of
        empty iterator output.

        5) Dataset reader should now expect a number of corresponding language directions instead of just language pairs
        e.g.: "EN-ET, DE-EN, FR-ET". These should be in the same order as files. I will then use them in two ways:
        1) parse them to obrtain a set of languages to create vocab namespaces (by creating tokend indexers)
        2) I will use them to attach language pair information to the LabelField and MetadataField for each instance
        That is, if the iterator of some file return me `None`, I will just know that at this place the lang_pair (and
        sentence pair) should be skipped.

        6) This way appending tag information to files is not needed any more and user can keep their files clean
        without any additional cumbersome stuff with appending language tags.

        TODO 3: More on test time
        So at test time we provide a single test file. But we need to tell our dataset reader that it should
        work completely differently from how it does at training time.

        For this one will have to override "test_language_direction" parameter of the dataset_reader (which will be None
        by default) specifying correct language direction.
        Dataset reader will check if test_language_direction is none to decide weather it is a training or test time.

        """
        validation = not self.training and target_tokens is not None
        test_time = not self.training and target_tokens is None

        source_tag = self.vocab.get_token_from_index(source_lang_id.item(), namespace=self._tags_namespace)
        target_tag = self.vocab.get_token_from_index(target_lang_id.item(), namespace=self._tags_namespace)
        curr_lang_dir = source_tag + "->" + target_tag
        self._target_namespace = "vocab_" + target_tag  # target namespace is different for each instance

        if self.training or validation:
            # Shape: src_len x batch_size x num_features
            encoder_output = run_encoder(encoder=self._encoders[source_tag], source_tokens=source_tokens)

            # Shape: batch_size x tgt_len x num_features
            logits, _ = run_decoder(decoder=self._decoders[target_tag],
                                    target_tokens=target_tokens,
                                    encoder_output=encoder_output)

            # Targets: <EOS, w1, w2, w3, PAD, PAD>
            # Desired: <w1, w2, w3, EOS, PAD, PAD>
            target_mask = util.get_text_field_mask(target_tokens)
            relevant_targets = move_eos_to_the_end(target_tokens["tokens"], target_mask).contiguous()

            # Compute loss
            loss = util.sequence_cross_entropy_with_logits(logits, relevant_targets, target_mask)

            # Update metrics
            predictions = logits.argmax(2)

            if self._all_bleu_metrics is not None:
                self._all_bleu_metrics[curr_lang_dir](predictions, relevant_targets)

            return {"loss": loss, "predictions": predictions}

        else:
            best_predictions = self._translators[curr_lang_dir](source_tokens)
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
                all_metrics[lang_dir] = bleu.get_metric(reset=reset)["BLEU"]
        return all_metrics
