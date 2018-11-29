import logging
from typing import Dict, Union, Iterable, Iterator, List, Optional, Tuple
from itertools import cycle, islice
import copy

import torch
from allennlp.data.dataset import Batch

from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import DataIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name


@DataIterator.register("multi")
class MultiIterator(DataIterator):
    """
    Takes as an input an ordered list of iterators that will in turn iterate over an ordered list of datasets.
    I.e. iterator1 will iterate over dataset1, iterator2 over dataset2, etc.

    All if instances_per_epoch is provided, all the datasets will have a synchronized epochs, with smaller datasets
    repeating.

    Left thus hparam None if you want each dataset have its own epoch

    It has the same methods as :class:`allennlp.data.iterators.DataIterator`
    """

    ""

    def __init__(self, iterator_base: DataIterator, num_datasets: int) -> None:
        self._iterators = [copy.copy(iterator_base) for _ in range(num_datasets)]

        if self._iterators[0]._instances_per_epoch is None:
            self._synced = False
        else:
            self._synced = True

    def __call__(self, iterable_datasets: List[Iterable[Instance]],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:

        if len(self._iterators) != len(iterable_datasets):
            raise ValueError("Number of iterators has to be the same with the number of datasets. Got: %s vs %s",
                             len(self._iterators), len(iterable_datasets))

        generators = []
        for iterator, instances in zip(self._iterators, iterable_datasets):
            generators.append(iterator(instances, num_epochs, shuffle))

        # "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
        # Recipe credited to George Sakkis
        num_active = len(generators)
        nexts = cycle(it.__next__ for it in generators)
        while num_active:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                # Remove the iterator we just exhausted from the cycle.
                num_active -= 1
                nexts = cycle(islice(nexts, num_active))

    def index_with(self, vocab: Vocabulary):
        for iterator in self._iterators:
            iterator.index_with(vocab)

    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        """
        Returns the number of batches that ``dataset`` will be split into; if you want to track
        progress through the batch with the generator produced by ``__call__``, this could be
        useful.
        """
        if self._synced:
            return self._iterators[0].get_num_batches(instances)
        else:
            return max([it.get_num_batches(instances) for it in self._iterators])

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        pass

