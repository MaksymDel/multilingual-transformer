# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.pretrained import PretrainedModel

from transformer import *
from allennlp.data.dataset_readers import DatasetReader


class TestSeq2SeqPredictor(AllenNlpTestCase):
    def test_predict_no_labels(self):
        ar = load_archive("fixtures/serialization/separate_encdec/model.tar.gz")
        mdl = ar.model
        mdl.eval()
        reader_params = ar.config.pop("dataset_reader")
        reader_params.__setitem__("language_pair", "EN-ET")
        reader = DatasetReader.from_params(reader_params)
        instances = reader.read("fixtures/data/seq2seq_nolabels.txt")
        res = mdl.forward_on_instances(instances)
        assert len(res) == len(instances)
