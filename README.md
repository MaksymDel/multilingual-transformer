```python 

python -m transformer.run train-multilingual fixtures/config/separate_encdec.json -s fixtures/serialization/separate_encdec -f
python -m transformer.run predict fixtures/serialization/separate_encdec/model.tar.gz fixtures/data/seq2seq_nolabels.txt --use-dataset-reader --predictor seq2seq_mulilingual -o '{"dataset_reader":{"language_pair": "EN-ET"}}'



python -m transformer.run train-multilingual training_config/separate_encdec.json -s ../output/separate_encdec-de-en -f
python -m transformer.run predict ../output/separate_encdec-de-en/model.tar.gz ../data/iwslt14.tokenized.de-en/test.de --use-dataset-reader --predictor seq2seq_mulilingual -o '{"dataset_reader":{"language_pair": "DE-EN"}}' --output-file hyps.de-en --cuda-device 0

```

