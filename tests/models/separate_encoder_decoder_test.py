# # pylint: disable=invalid-name,no-self-use,protected-access
# from allennlp.common.testing import ModelTestCase
#
# from transformer import *
#
# # pylint: disable=invalid-name
#
# import json
#
# import numpy
# import torch
#
# from allennlp.common.testing import ModelTestCase
# from allennlp.nn.beam_search import BeamSearch
# from allennlp.nn.util import sequence_cross_entropy_with_logits
#
#
# class MultilingualTransformerModelTest(ModelTestCase):
#     def setUp(self):
#         super().setUp()
#         self.set_up_model("fixtures/config/separate_encdec.json",
#                           "fixtures/data/en-et.tsv, fixtures/data/en-ru.tsv, fixtures/data/ua-ru.tsv")
#
#     def test_model_can_train_save_and_load(self):
#         self.model.train()
#         self.ensure_model_can_train_save_and_load(self.param_file)
#
#     def test_model_works_without_labels(self):
#         self.set_up_model("fixtures/config/separate_encdec_inference.json",
#                           "fixtures/data/seq2seq_nolabels.txt")
#         self.model.eval()
#
#         tensors = self.dataset.as_tensor_dict()
#
#         out = self.model(**tensors)
#         out = self.model.decode(out)
#         assert len(out["predicted_tokens"]) == 3
