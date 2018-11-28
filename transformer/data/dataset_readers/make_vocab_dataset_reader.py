from typing import Dict
import logging
from itertools import zip_longest

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("make_vocab")
class MakeVocabDatasetReader(DatasetReader):
    """

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

    `END_SYMBOL` tokens are added to beginning of all sentences.

    It currently works with synchronized epochs without duplicating sentences
    in shorter datasets to match the ones of longer ones.

    Dataset reader accepts several parallel files, e.g.: "tarin.en-et, train.de-en, train.fr-et"
    We then do path.split(" ,") and parses them with itertools.zip that returns None in place of
    empty iterator output.

    Dataset reader expects a number of corresponding language pairs, e.g.: "EN-ET, DE-EN, FR-ET".
    These should be in the same order as files. It will then use them in two ways:
    1) parse them to obrtain a set of languages to create vocab namespaces (by creating tokend indexers)
    2) attach language pair information to the LabelField and MetadataField for each instance
    That is, if the iterator of some file returns `None`, It will just know that at this place the lang_pair (and
    sentence pair) should be skipped.

    At test time we provide a single test file. But we need to tell our dataset reader that it should
    work completely differently from how it does at training time.

    For this one will have to override "test_time_language_direction" parameter of the dataset_reader
    (which is None by default) specifying correct language direction.
    Dataset reader will check if test_language_direction is none to decide weather it is a training or test time.

    Parameters
    ----------

    """

    def __init__(self,
                 language_pairs: str,
                 inference_language_direction: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

        # This one is only supplied at test time via "--overrides" flag of the predict command
        self._inference_language_direction = inference_language_direction

        # Language pairs in the same order as files supplied
        self._language_pairs = language_pairs.split(", ")

        # Get a set of all languages
        language_tags = set()
        for pair in self._language_pairs:
            l1, l2 = pair.split("-")
            language_tags.add(l1)
            language_tags.add(l2)

        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

        self._tags_namespace = "language_tags"
        self._token_indexers = {}
        for tag in language_tags:
            self._token_indexers[tag] = {"tokens": SingleIdTokenIndexer(namespace="vocab_" + tag)}

    @overrides
    def _read(self, file_path):
        if self._inference_language_direction is not None:
            logger.info("Supplied language direction: %s, so dataset reader is in inference mode now",
                        self._inference_language_direction)
            inference = True
        else:
            logger.info("No language direction is supplied, so dataset reader is in training mode now")
            inference = False

        if inference:
            logger.info("Reading source sentences from following file: %s", file_path)

            with open(cached_path(file_path)) as test_file:
                for line_num, source_sentence in enumerate(test_file):
                    yield self.text_to_instance(source_sentence)
        else:
            file_names = file_path.split(", ")

            logger.info("Reading instances from following files: %s", ", ".join(file_names))
            logger.info("These files correspond to the following language pairs: %s", ", ".join(self._language_pairs))

            files = [open(cached_path(f)) for f in file_names]

            for line_num, sentence_pairs in enumerate(zip_longest(*files)):
                sentence_pairs_list = []
                sentence_pairs_tags_list = []
                language_pairs_metadata = []
                for pair_num, sentence_pair in enumerate(sentence_pairs):
                    curr_lang_pair = self._language_pairs[pair_num]
                    lang_tag_1, lang_tag_2 = curr_lang_pair.split("-")

                    if sentence_pair is None:  # means that we are out of instances in this dataset
                        empty_tf = TextField([Token(END_SYMBOL)], self._token_indexers[lang_tag_1]).empty_field()
                        empty_lf = ListField([empty_tf, empty_tf]).empty_field()
                        sentence_pairs_list.append(empty_lf)
                        continue

                    sentence_pair = sentence_pair.strip("\n")
                    sentence_pair_parts = sentence_pair.split('\t')

                    if len(sentence_pair_parts) != 2:
                        raise ConfigurationError(
                            "Invalid line format: %s (line number %d)" % (sentence_pair, line_num + 1))

                    sentence_1, sentence_2 = sentence_pair_parts

                    sentence_field_1 = self._string_to_field(sentence_1, self._token_indexers[lang_tag_1])
                    sentence_field_2 = self._string_to_field(sentence_2, self._token_indexers[lang_tag_2])
                    lang_tag_field_1 = LabelField(lang_tag_1, label_namespace=self._tags_namespace)
                    lang_tag_field_2 = LabelField(lang_tag_2, label_namespace=self._tags_namespace)

                    sentence_pair_field = ListField([sentence_field_1, sentence_field_2])
                    sentence_pair_tags_field = ListField([lang_tag_field_1, lang_tag_field_2])

                    sentence_pairs_list.append(sentence_pair_field)
                    sentence_pairs_tags_list.append(sentence_pair_tags_field)
                    language_pairs_metadata.append(curr_lang_pair)

                sentence_pairs_field = ListField(sentence_pairs_list)
                sentence_pairs_tags_field = ListField(sentence_pairs_tags_list)
                language_pairs_metadata = MetadataField(language_pairs_metadata)

                instance = Instance({"sentence_pairs": sentence_pairs_field,
                                     "sentence_pairs_tags": sentence_pairs_tags_field,
                                     "language_tags_meta": language_pairs_metadata})

                yield instance

            for f in files:
                f.close()

    @overrides
    def text_to_instance(self, source_sentence: str) -> Instance:
        source_tag, _ = self._inference_language_direction.split("->")
        source_sentence_field = self._string_to_field(source_sentence, self._token_indexers(source_tag))
        return Instance({"source_sentence": source_sentence_field,
                         "language_tags_meta": self._inference_language_direction})

    def _string_to_field(self, string: str, token_indexers: Dict[str, TokenIndexer]):
        tokenized_string = self._tokenizer.tokenize(string)
        tokenized_string.insert(0, Token(END_SYMBOL))
        return TextField(tokenized_string, token_indexers)
