# # pylint: disable=no-self-use,invalid-name
# import pytest
#
# from transformer.data.dataset_readers.bitext_dataset_reader import BitextDatasetReader
# from allennlp.common.util import ensure_list
# from allennlp.common.testing import AllenNlpTestCase
#
#
# class TestBitextDatasetReader:
#     @pytest.mark.parametrize("lazy", (True, False))
#     def test_default_format(self, lazy):
#         reader = BitextDatasetReader(lazy=lazy)
#         instances = reader.read("fixtures/data/seq2seq_diff_lens.tsv")
#         instances = ensure_list(instances)
#
#         assert len(instances) == 3
#         fields = instances[0].fields
#         assert [t.text for t in fields["source_tokens"].tokens] == ["@start@", "this", "is",
#                                                                     "a", "sentence", "@end@"]
#         assert [t.text for t in fields["target_tokens"].tokens] == ["@start@", "this", "is",
#                                                                     "a", "sentence", "@end@"]
#         fields = instances[1].fields
#         assert [t.text for t in fields["source_tokens"].tokens] == ["@start@", "this", "is",
#                                                                     "another", "@end@"]
#         assert [t.text for t in fields["target_tokens"].tokens] == ["@start@", "this", "is",
#                                                                     "another", "@end@"]
#         fields = instances[2].fields
#         assert [t.text for t in fields["source_tokens"].tokens] == ["@start@", "all", "these", "sentences",
#                                                                     "should", "get", "copied", "@end@"]
#         assert [t.text for t in fields["target_tokens"].tokens] == ["@start@", "all", "these", "sentences",
#                                                                     "should", "get", "copied", "@end@"]
