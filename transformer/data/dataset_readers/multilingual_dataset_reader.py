from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("multilingual")
class MultilingualDatasetReader(DatasetReader):
    """
    language_tags parameter accepts dash "-" separated set of languages

    Training time:
    expected format for each input line: <lang_tag_A>\t<langA_sequence_string>\t<lang_tag_B>\t<langB_sequence_string>\t

    Test time:
    expected format is <lang_tag_A>\t<langA_sequence_string>\t<lang_tag_B>

    Read a tsv file containing paired sequences, and create a dataset suitable for a
    FairSeq models.

    These models expect targets to begin with EOS so we just add EOS symbols to all
    sentences and rely on in-model utils to move EOS  to beginning for teacher forcing or to remove EOS in
    case of string being source sequence.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------

    """
    def __init__(self, language_tags: str, lazy: bool = False) -> None:
        super().__init__(lazy)

        language_tags = language_tags.split("-")
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

        self._tags_namespace = "language_tags"
        self._token_indexers = {}
        for tag in language_tags:
            self._token_indexers[tag] = {"tokens": SingleIdTokenIndexer(namespace="vocab_" + tag)}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')

                if len(line_parts) == 4:
                    src_tag, source_sequence, tgt_tag, target_sequence = line_parts
                    yield self.text_to_instance(src_tag, source_sequence, tgt_tag, target_sequence)
                elif len(line_parts) == 3:
                    src_tag, source_sequence, tgt_tag = line_parts
                    yield self.text_to_instance(src_tag, source_sequence, tgt_tag)
                else:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))

    @overrides
    def text_to_instance(self, source_lang_tag: str, source_string: str,
                         target_lang_tag, target_string: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields_dict = {}

        fields_dict["source_lang_id"] = LabelField(source_lang_tag, self._tags_namespace)
        fields_dict["source_tokens"] = self._string_to_field(source_string, source_lang_tag)
        fields_dict["target_lang_id"] = LabelField(target_lang_tag, self._tags_namespace)

        if target_string is not None:
            fields_dict["target_tokens"] = self._string_to_field(target_string, target_lang_tag)

        return Instance(fields_dict)

    def _string_to_field(self, string: str, lang_tag: str):
        tokenized_string = self._tokenizer.tokenize(string)
        tokenized_string.insert(0, Token(END_SYMBOL))
        return TextField(tokenized_string, self._token_indexers[lang_tag])

