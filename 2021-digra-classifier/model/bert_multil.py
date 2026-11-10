from transformers import BertForSequenceClassification, BertTokenizer
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from mikatools import *


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def finetune(): 
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")


    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    train_batch = open_read("nmt/source_train.txt").read().split("\n")
    train_text = tokenizer(train_batch, return_tensors='pt', padding=True, truncation=True)
    valid_batch = open_read("nmt/source_valid.txt").read().split("\n")
    valid_text = tokenizer(valid_batch, return_tensors='pt', padding=True, truncation=True)
    
    train_labels = torch.tensor([int(x == "depression") for x in open_read("nmt/target_train.txt").read().split("\n")])
    valid_labels = torch.tensor([int(x == "depression") for x in open_read("nmt/target_valid.txt").read().split("\n")])
    #test_labels = torch.tensor([int(x == "depression") for x in open_read("nmt/target_test.txt").read().split("\n")])

    train_dataset = IMDbDataset(train_text, train_labels)
    val_dataset = IMDbDataset(valid_text, valid_labels)
    #test_dataset = IMDbDataset(test_text, test_labels)

    training_args = TrainingArguments(
        output_dir='./results_multil',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()

    trainer.save_model("finetuned-multi-bert")

def _resulter(two_turds):
	if two_turds[0] > two_turds[1]:
		return 0
	else:
		return 1

def eval(shit_pile):
    cor = 0.0
    tot = 0.0
    labs = {"depression":{"cor":0.0, "err":0.0}, "no_depression":{"cor":0.0, "err":0.0}}
    ls = ["no_depression", "depression"]
    for item in shit_pile:
        p = _resulter(item[0])
        c = item[1]

        if p == c:
            cor += 1
            labs[ls[c]]["cor"] +=1
        else:
            labs[ls[c]]["err"] +=1
        tot += 1
    print(cor, tot, cor/tot)
    for l, vs in labs.items():
        print(l, vs["cor"]/(vs["cor"]+vs["err"]))


def predict():
	tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
	test_batch = open_read("nmt/source_test.txt").read().split("\n")
	test_text = tokenizer(test_batch, return_tensors='pt', padding=True, truncation=True)
	test_labels = torch.tensor([int(x == "depression") for x in open_read("nmt/target_test.txt").read().split("\n")])
	test_dataset = IMDbDataset(test_text, test_labels)
	model = BertForSequenceClassification.from_pretrained("finetuned-multi-bert")
	trainer = Trainer(
		model=model,                         # the instantiated 🤗 Transformers model to be trained
		args=training_args,                  # training arguments, defined above
	)
	model.eval()

	ps = trainer.predict(test_dataset)
	softshit = torch.nn.Softmax()
	preds = softshit(torch.from_numpy(ps.predictions)).numpy()
	eval(list(zip(preds.tolist(), ps.label_ids.tolist())))

#finetune()
predict()

"""
3765.0 5016.0 0.7505980861244019
depression 0.7911959485781067
no_depression 0.708044099632503
"""

