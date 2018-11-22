```python 

from allennlp import predictors
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.data import DatasetReader
from transformer import *

ar = load_archive("fixtures/fairseq_transformer/model.tar.gz")
tr = ar.model
dr = DatasetReader.from_params(ar.config.pop("dataset_reader"))

i = dr.text_to_instance(
"also, ,, ziehen, sie, los, und, fangen, sie, an, zu, erfinden, .", 
"so go ahead and start inv@@ enting .")

i = dr.text_to_instance("vielen dank .",
 "thank you .")
 
i = dr.text_to_instance("vielen dank .",
 "thank you .")

i = dr.text_to_instance("vielen dank .")

i = dr.text_to_instance("this is another")

tr.forward_on_instance(i)

```

