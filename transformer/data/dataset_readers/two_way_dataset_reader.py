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


@DatasetReader.register("two_way")
class TwoWayDatasetReader(DatasetReader):
    """

    By language pair/direction I mean just tags.
    By sentence pair/direction I mean concrete pair of sentences

    ---
    Parameters
    ----------

    """

    def __init__(self,
                 language_pair: str,
                 lazy: bool = False) -> None:
        super().__init__(lazy)

        self._language_pair = language_pair

        self._language_a, self._language_b = self._language_pair.split("-")

        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

        self._tags_namespace = "language_tags"
        self._token_indexers = {}
        for tag in [self._language_a, self._language_b]:
            self._token_indexers[tag] = {"tokens": SingleIdTokenIndexer(namespace="vocab_" + tag)}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path)) as data_file:
            logger.info("Reading %s sentence pairs from following file: %s", self._language_pair, file_path)

            for line_num, sentence_pair in enumerate(data_file):
                sentence_pair = sentence_pair.strip('\n').split('\t')
                if len(sentence_pair) == 2:
                    sentence_a, sentence_b = sentence_pair
                    yield self.text_to_instance(sentence_a, sentence_b)
                elif len(sentence_pair) == 1:
                    sentence_a = sentence_pair[0]
                    yield self.text_to_instance(sentence_a)

    @overrides
    def text_to_instance(self, sentence_a: str, sentence_b: str = None) -> Instance:
        tokenized_a = self._tokenizer.tokenize(sentence_a)
        tokenized_a.insert(0, Token(END_SYMBOL))

        sentence_a_field = TextField(tokenized_a, self._token_indexers[self._language_a])

        language_a_field = LabelField(self._language_a, label_namespace=self._tags_namespace)
        language_b_field = LabelField(self._language_b, label_namespace=self._tags_namespace)

        lanugage_a_metadata = MetadataField(self._language_a)
        lanugage_b_metadata = MetadataField(self._language_b)

        fields_dict = {"sentence_a": sentence_a_field,
                       "language_a_indexed": language_a_field,
                       "language_b_indexed": language_b_field,
                       "language_a": lanugage_a_metadata,
                       "language_b": lanugage_b_metadata,
                       }

        if sentence_b is not None:
            tokenized_b = self._tokenizer.tokenize(sentence_b)
            tokenized_b.insert(0, Token(END_SYMBOL))
            sentence_b_field = TextField(tokenized_b, self._token_indexers[self._language_b])
            fields_dict.update({"sentence_b": sentence_b_field})

        return Instance(fields_dict)

    @staticmethod
    def get_num_instances(file_path):
        # calculate total number of instances
        with open(file_path) as f:
            for i, _ in enumerate(f):
                pass
        return i + 1
