# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
import itertools, collections

from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.tests.data.iterators.basic_iterator_test import IteratorTest

from transformer.data.dataset_readers import TwoWayDatasetReader
from transformer.data.iterators import MultipleIterator


class TestMultiIterator(IteratorTest):
    def setUp(self):
        super().setUp()

    def test_multiple_generator(self):
        generator1 = self.build_multi_generator(2, 1, 6)
        generator2 = self.build_multi_generator(2, 2, 3)

        a = [i["language_a"] for i in generator1]
        b = [i["language_a"] for i in generator2]
        assert collections.Counter(itertools.chain(*a)) == collections.Counter(itertools.chain(*b))

        a = [i["language_b"] for i in generator1]
        b = [i["language_b"] for i in generator2]
        assert collections.Counter(itertools.chain(*a)) == collections.Counter(itertools.chain(*b))

    def build_multi_generator(self, batch_size, num_epoch, instances_per_epoch):
        base_file_path = "fixtures/data/en-et.tsv, fixtures/data/en-ru.tsv, fixtures/data/ua-ru.tsv"
        data_files_list = base_file_path.split(", ")
        self._language_pairs = "EN-ET, EN-RU, UA-RU".split(", ")
        self._dataset_readers = {}
        for language_pair in self._language_pairs:
            self._dataset_readers[language_pair] = TwoWayDatasetReader(lazy=False, language_pair=language_pair)

        self._data_files = {}
        for language_pair, data_file in zip(self._language_pairs, data_files_list):
            self._data_files[language_pair] = data_file
        # For some of the tests we need a vocab, we'll just use the base_reader for that.
        self.vocab = Vocabulary.from_files("fixtures/vocab/vocabulary")

        self._iterable_datasets = {}
        for language_pair in self._language_pairs:
            data_file = self._data_files[language_pair]
            reader = self._dataset_readers[language_pair]
            dataset_iterable = reader.read(data_file)

            self._iterable_datasets[language_pair] = dataset_iterable

        datasets_list = list(self._iterable_datasets.values())

        iterator_base = BasicIterator(batch_size=batch_size, instances_per_epoch=instances_per_epoch)
        multiple_iterator = MultipleIterator(iterator_base, 3)
        multiple_iterator.index_with(self.vocab)
        return multiple_iterator(datasets_list, num_epochs=num_epoch, shuffle=False)

