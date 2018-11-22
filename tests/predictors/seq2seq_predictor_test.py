# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.pretrained import PretrainedModel

from transformer import *
from allennlp.data.dataset_readers import DatasetReader

class TestSeq2SeqPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        fixed = PretrainedModel("fixtures/fairseq_transformer/serialization/model.tar.gz", "seq2seq")
        predictor = fixed.predictor()
        input_tokens = "all these sentences should get copied"
        predicted_tokens = predictor.predict_json({"source": "all these sentences should get copied"})["predicted_tokens"]
        predicted_tokens = " ".join(predicted_tokens)
        # inputs = {
        #         "source": "What kind of test succeeded on its first attempt?",
        # }
        #
        # archive = load_archive(self.FIXTURES_ROOT / 'encoder_decoder' / 'simple_seq2seq' /
        #                        'serialization' / 'model.tar.gz')
        # predictor = Predictor.from_archive(archive, 'simple_seq2seq')
        #
        # result = predictor.predict_json(inputs)
        #
        # predicted_tokens = result.get("predicted_tokens")
        # assert predicted_tokens is not None
        # assert isinstance(predicted_tokens, list)
        # assert all(isinstance(x, str) for x in predicted_tokens)

    def test_on_single_instance(self):
        try:
            ar = load_archive("fixtures/fairseq_transformer/model.tar.gz")
            tr = ar.model
            dr = DatasetReader.from_params(ar.config.pop("dataset_reader"))

            i = dr.text_to_instance(
                "also, ,, ziehen, sie, los, und, fangen, sie, an, zu, erfinden, .",
                "so go ahead and start inv@@ enting .")

            i = dr.text_to_instance("vielen dank .",
                                    "thank you .")

            i = dr.text_to_instance("vielen dank .")

            # tr.forward_on_instance(i)
        except FileNotFoundError:
            pass

    def test_predict_no_labels(self):
        ar = load_archive("fixtures/fairseq_transformer/serialization/model.tar.gz")
        mdl = ar.model
        reader = DatasetReader.from_params(ar.config.pop("dataset_reader"))
        instances = reader.read("fixtures/data/seq2seq_nolabels.txt")
        res = mdl.forward_on_instances(instances)
        assert len(res) == len(instances)
