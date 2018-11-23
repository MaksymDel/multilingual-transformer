from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("multilingual_transformer_fseq")
class MultilingualFseqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    FairSeq models.

    These models expect targets to begin with EOS so we just add EOS symbols to all
    sentences and rely on in-model utils to move EOS  to beginning for teacher forcing or to remove EOS in
    case of string being source sequence.

    Expected format for each input line: <langA_sequence_string>\t<langB_sequence_string>\t...

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------

    """
    def __init__(self, lazy: bool = False) -> None:
        super().__init__(lazy)
        tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

        self._tokenizer_A = tokenizer
        self._tokenizer_B = tokenizer
        self._token_indexers_A = {"tokens": SingleIdTokenIndexer(namespace="vocab_A")}
        self._token_indexers_B = {"tokens": SingleIdTokenIndexer(namespace="vocab_B")}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')

                if len(line_parts) == 2:
                    source_sequence, target_sequence = line_parts
                    yield self.text_to_instance(source_sequence, target_sequence)
                elif len(line_parts) == 1:
                    source_sequence = line_parts[0]
                    yield self.text_to_instance(source_sequence)
                else:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))

    @overrides
    def text_to_instance(self, string_A: str, string_B: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields_dict = {}
        if string_A is not None:
            field_dict = self.string_to_field(string_A, self._tokenizer_A, self._token_indexers_A, "A")
            fields_dict.update(field_dict)
        if string_B is not None:
            field_dict = self.string_to_field(string_B, self._tokenizer_B, self._token_indexers_B, "B")
            fields_dict.update(field_dict)
        return Instance(fields_dict)

    @staticmethod
    def string_to_field(string: str, tokenizer: Tokenizer, token_indexers: TokenIndexer, domain_tag: str):
        tokenized_string = tokenizer.tokenize(string)
        tokenized_string.insert(0, Token(END_SYMBOL))
        field = TextField(tokenized_string, token_indexers)
        field_name = "tokens_" + domain_tag
        return {field_name: field}
