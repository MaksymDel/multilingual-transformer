# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import ModelTestCase

from transformer import *

# pylint: disable=invalid-name

import json

import numpy
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import sequence_cross_entropy_with_logits


class FairseqTransformerModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model("fixtures/fairseq_transformer.json",
                          "fixtures/data/seq2seq_copy.tsv")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_model_works_without_labels(self):
        training_tensors = self.dataset.as_tensor_dict()
        source_tensors = training_tensors["source_tokens"]
        out = self.model(source_tensors)
        out1 = self.model(source_tensors)
        assert len(out["predictions"]) == 3

        self.set_up_model("fixtures/fairseq_transformer.json",
                          "fixtures/data/seq2seq_nolabels.txt")

        tensors = self.dataset.as_tensor_dict()
        source_tensors_test = tensors["source_tokens"]

        out2 = self.model(source_tensors_test)
