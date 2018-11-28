# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

from transformer.data.dataset_readers import TwoWayDatasetReader


class TestTwoWayDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = TwoWayDatasetReader(lazy=lazy, language_pair="EN-ET")
        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'seq2seq_copy.tsv'))
        instances = ensure_list(instances)

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["sentence_a"].tokens] == ["@end@", "this", "is",
                                                                    "a", "sentence"]
        assert [t.text for t in fields["sentence_b"].tokens] == ["@end@", "this", "is",
                                                                    "a", "sentence"]
        fields = instances[1].fields
        assert [t.text for t in fields["sentence_a"].tokens] == ["@end@", "this", "is",
                                                                    "another"]
        assert [t.text for t in fields["sentence_b"].tokens] == ["@end@", "this", "is",
                                                                    "another"]
        fields = instances[2].fields
        assert [t.text for t in fields["sentence_a"].tokens] == ["@end@", "all", "these", "sentences",
                                                                 "should", "get", "copied"]
        assert [t.text for t in fields["sentence_b"].tokens] == ["@end@", "all", "these", "sentences",
                                                                 "should", "get", "copied"]

        assert fields["language_a_indexed"].label == fields["language_a"].metadata == "EN"
        assert fields["language_b_indexed"].label == fields["language_b"].metadata == "ET"
