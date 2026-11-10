from pathlib import Path
from transformers import BertForSequenceClassification
import torch


# All subdirectories in the current directory, not recursive.
multiling_models = [f for f in Path('./models/monoling').iterdir() if f.is_dir()]
monoling_models = [f for f in Path('./models/multiling_solo').iterdir() if f.is_dir()]
models = multiling_models + monoling_models

for model in models:
    print(str(model)
    model = BertForSequenceClassification.from_pretrained(model)
    pos_test = open_read("source_persuade_test.txt").read().split("\n")
    neg_test = open_read("source_neg_test.txt").read().split("\n")

        