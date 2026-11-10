# -*- coding: utf-8 -*-
"""Monolingual and multilingual BERT persuasion detection (2022 release version).

Originally exported from a Colab notebook (classifier_model.ipynb,
https://colab.research.google.com/drive/1hXq2h-MHR3HZIA49S3vrESyIu8JBmyav).

Revived in 2026 to run on Python 3.12 / current transformers and to read the
published Zenodo dataset directly (https://doi.org/10.5281/zenodo.6341173,
file ``digra.csv`` with columns sid, game, is_persuasion, en, de, fr, it, es).
The data preparation, model choices and training hyper-parameters are the
original ones; see the README in this folder for what was changed and for the
two known bugs that can be switched off with ``--fix-known-bugs``.

Examples (run from the repository root):
    python data/download.py
    # smoke test: tiny model, 200 sentences per group, 1 training step
    python 2022-dataset-release/models/multilingual_persuasion_detection.py \
        --langs en --runs mono --base-model prajjwal1/bert-tiny --max-sents 200 --max-steps 1
    # full original experiment (11 models, needs a GPU)
    python 2022-dataset-release/models/multilingual_persuasion_detection.py
"""

import argparse
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "digra.csv"
DEFAULT_WORK_DIR = REPO_ROOT / "runs" / "2022"

# Pre-trained models used in the paper (Hugging Face Hub ids; the original code
# used the legacy un-namespaced ids and a local ./camembert-base copy).
MONO_MODELS = {
    "en": "google-bert/bert-base-cased",
    "es": "dccuchile/bert-base-spanish-wwm-cased",
    "it": "dbmdz/bert-base-italian-cased",
    "de": "google-bert/bert-base-german-cased",
    "fr": "almanach/camembert-base",
}
MULTI_MODEL = "google-bert/bert-base-multilingual-cased"
LANGS = ["en", "es", "it", "de", "fr"]

seed = 42
torch.manual_seed(seed)

# Filled in by main() from the command line.
CFG = argparse.Namespace(
    data=DEFAULT_DATA,
    work_dir=DEFAULT_WORK_DIR,
    base_model=None,
    max_sents=None,
    max_steps=-1,
    epochs=5,
    fix_known_bugs=False,
)


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)


def _load_tokenizer(name):
    try:
        return AutoTokenizer.from_pretrained(name)
    except ValueError:  # e.g. prajjwal1/bert-tiny: config.json has no model_type
        return BertTokenizer.from_pretrained(name)


def _load_model(name, **kwargs):
    try:
        return AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
    except ValueError:
        return BertForSequenceClassification.from_pretrained(name, **kwargs)


def _ensure_punkt():
    for res in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


def sent_tokenize(source):
    source = [j for i in source for j in i]

    source = [str(x) for x in source if x]

    source = [nltk.sent_tokenize(sent) for sent in source]
    source = [j for i in source for j in i]
    return source


def _work(name):
    return Path(CFG.work_dir) / name


def _read_lines(name):
    with open(_work(name), encoding="utf-8") as f:
        return f.read().split("\n")


def get_traintest_data(langs):
    df = pd.read_csv(CFG.data, index_col=[0], encoding="utf-8")
    df_kotor1_neg = df.loc[(df.is_persuasion == False) & (df.game == "KotOR 1")]
    df_kotor2_neg = df.loc[(df.is_persuasion == False) & (df.game == "KotOR 2")]
    df_nwn_neg = df.loc[(df.is_persuasion == False) & (df.game == "NWN 1")]
    df_persuade_sents = df.loc[(df.is_persuasion == True)]
    df_kotor2_neg = df_kotor2_neg.dropna(subset=["de", "it", "es", "fr"])
    df_nwn_neg = df_nwn_neg.dropna(subset=["de", "it", "es", "fr"])
    df_persuade_sents = df_persuade_sents.dropna(subset=["de", "it", "es", "fr"])
    # (The 2021 version also dropped rows whose "translation" was detected as
    # English with langdetect; that step was commented out in this version.)
    print("kotor1:", len(df_kotor1_neg))
    print("kotor2:", len(df_kotor2_neg))
    print("nwn", len(df_nwn_neg))
    print("persuade", len(df_persuade_sents))
    kotor1_neg = []
    kotor2_neg = []
    nwn_neg = []
    pos_sents = []
    if type(langs) is list:
        for lang in langs:
            kotor1_neg.append(list(set(df_kotor1_neg[lang].tolist())))
            kotor2_neg.append(list(set(df_kotor2_neg[lang].tolist())))
            nwn_neg.append(list(set(df_nwn_neg[lang].tolist())))
            pos_sents.append(list(set(df_persuade_sents[lang].tolist())))
    else:
        kotor1_neg.append(list(set(df_kotor1_neg[langs].tolist())))
        kotor2_neg.append(list(set(df_kotor2_neg[langs].tolist())))
        nwn_neg.append(list(set(df_nwn_neg[langs].tolist())))
        pos_sents.append(list(set(df_persuade_sents[langs].tolist())))

    # NOTE: original bug -- KotOR 1 negatives are built from the KotOR 2 lines,
    # so KotOR 1 is never used and KotOR 2 negatives are counted twice.
    if CFG.fix_known_bugs:
        kotor1_neg = sent_tokenize(kotor1_neg)
    else:
        kotor1_neg = sent_tokenize(kotor2_neg)
    kotor2_neg = sent_tokenize(kotor2_neg)
    nwn_neg = sent_tokenize(nwn_neg)
    persuade_sents = sent_tokenize(pos_sents)

    if CFG.max_sents:  # smoke-test subset
        n = CFG.max_sents
        kotor1_neg, kotor2_neg, nwn_neg, persuade_sents = (
            kotor1_neg[:n],
            kotor2_neg[:n],
            nwn_neg[:n],
            persuade_sents[:n],
        )

    pos_num = int(len(persuade_sents) * 0.7)
    pos_test = int(len(persuade_sents) * 0.3)
    neg_sents = kotor2_neg[:pos_num] + nwn_neg[:pos_num] + kotor1_neg[:pos_num]

    pos_sents = persuade_sents[:pos_num]
    neg_sents_test = (
        kotor2_neg[-pos_test:] + nwn_neg[-pos_test:] + kotor1_neg[-pos_test:]
    )

    pos_sents_test = persuade_sents[-pos_test:]
    print("neg_sents", len(neg_sents))
    print("neg_sents_test", len(neg_sents_test))
    print("pos_sents_test", len(pos_sents_test))
    print("pos_sents", len(pos_sents))

    Path(CFG.work_dir).mkdir(parents=True, exist_ok=True)
    for name, sents in [
        ("source_neg.txt", neg_sents),
        ("source_persuade.txt", pos_sents),
        ("source_neg_test.txt", neg_sents_test),
        ("source_persuade_test.txt", pos_sents_test),
    ]:
        with open(_work(name), "w", encoding="utf-8") as f:
            for element in sents:
                f.write(str(element) + "\n")


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, zero_division=0)
    precision = precision_score(y_true=labels, y_pred=pred, zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=pred, zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _base_model_name(langs, multiling):
    if CFG.base_model:
        return CFG.base_model
    if multiling or type(langs) is list:
        return MULTI_MODEL
    return MONO_MODELS[langs]


def _output_name(langs, multiling):
    if multiling:
        if type(langs) is list:
            return "multilang-finetuned-multilingual-persuade-bert"
        return "{}-finetuned-multilingual-persuade-bert".format(langs)
    return "{}-finetuned-persuade-bert".format(langs)


def _training_args(epochs, eval_strategy="epoch"):
    return TrainingArguments(
        output_dir=str(_work("results_pers")),  # output directory
        eval_strategy=eval_strategy,  # was evaluation_strategy="epoch"
        num_train_epochs=epochs,  # total number of training epochs
        max_steps=CFG.max_steps,  # -1 = train for num_train_epochs (smoke tests use 1)
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        # (logging_dir="./logs" removed: no longer a TrainingArguments option)
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )


def finetune(langs, multiling):
    base = _base_model_name(langs, multiling)
    print("Fine-tuning", base)
    tokenizer = _load_tokenizer(base)
    model = _load_model(base, num_labels=2)

    source_neg = _read_lines("source_neg.txt")
    source_pos = _read_lines("source_persuade.txt")
    neg_labels = ["no_persuade"] * len(source_neg)
    pos_labels = ["persuade"] * len(source_pos)

    source = source_neg + source_pos
    labels = neg_labels + pos_labels

    train_batch, valid_batch, train_labels, valid_labels = train_test_split(
        source, labels, test_size=0.1
    )
    train_text = tokenizer(
        train_batch, return_tensors="pt", padding=True, truncation=True
    )
    valid_text = tokenizer(
        valid_batch, return_tensors="pt", padding=True, truncation=True
    )

    train_labels = torch.tensor([int(x == "persuade") for x in train_labels])
    valid_labels = torch.tensor([int(x == "persuade") for x in valid_labels])

    train_dataset = IMDbDataset(train_text, train_labels)
    val_dataset = IMDbDataset(valid_text, valid_labels)

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=_training_args(CFG.epochs),  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(_work(_output_name(langs, multiling))))


def _resulter(two_turds):
    if two_turds[0] > two_turds[1]:
        return 0
    else:
        return 1


def eval(shit_pile):
    cor = 0.0
    tot = 0.0
    labs = {
        "persuade": {"cor": 0.0, "err": 0.0},
        "no_persuade": {"cor": 0.0, "err": 0.0},
    }
    ls = ["no_persuade", "persuade"]
    for item in shit_pile:
        p = _resulter(item[0])
        c = item[1]

        if p == c:
            cor += 1
            labs[ls[c]]["cor"] += 1
        else:
            labs[ls[c]]["err"] += 1
        tot += 1
    print(cor, tot, cor / tot)
    for l, vs in labs.items():
        n = vs["cor"] + vs["err"]
        print(l, vs["cor"] / n if n else float("nan"))


def _predict_tokenizer_name(langs, multiling):
    """Tokenizer used at prediction time.

    NOTE: original bug -- for the single-language multilingual-BERT models the
    original code loaded the *monolingual* tokenizer of that language (e.g. the
    Italian BERT vocabulary for an mBERT model), so the mBERT models were
    evaluated on wrongly tokenised input.  Only the all-languages model used
    the mBERT tokenizer.  ``--fix-known-bugs`` uses the model's own tokenizer.
    """
    if CFG.base_model:
        return CFG.base_model
    if type(langs) is list:
        return MULTI_MODEL
    if multiling and CFG.fix_known_bugs:
        return MULTI_MODEL
    return MONO_MODELS[langs]


def predict(langs, multiling):
    tokenizer = _load_tokenizer(_predict_tokenizer_name(langs, multiling))
    model = str(_work(_output_name(langs, multiling)))

    pos_test = _read_lines("source_persuade_test.txt")
    neg_test = _read_lines("source_neg_test.txt")

    test_source = neg_test + pos_test
    test_labels = ["no_persuade"] * len(neg_test) + ["persuade"] * len(pos_test)

    test_text = tokenizer(
        test_source, return_tensors="pt", padding=True, truncation=True
    )
    test_labels = torch.tensor([int(x == "persuade") for x in test_labels])
    test_dataset = IMDbDataset(test_text, test_labels)
    model_path = str(model)
    model = _load_model(model)
    trainer = Trainer(
        model=model,  # the instantiated Transformers model
        args=_training_args(3, eval_strategy="no"),  # prediction only
    )
    model.eval()

    ps = trainer.predict(test_dataset)
    softshit = torch.nn.Softmax(dim=1)
    preds = softshit(torch.from_numpy(ps.predictions)).numpy()
    pred_labels = [int(np.argmax(pred)) for pred in preds]
    df = pd.DataFrame(
        {"sents": test_source, "gold": test_labels.tolist(), "preds": pred_labels}
    )
    df.to_csv(model_path + ".csv")
    eval(list(zip(preds.tolist(), ps.label_ids.tolist())))
    print(
        classification_report(
            df.gold,
            df.preds,
            labels=[0, 1],
            target_names=["no_persuade", "persuade"],
            zero_division=0,
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Zenodo digra.csv")
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                    help="where split files, models and prediction CSVs are written")
    ap.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--runs", nargs="+", default=["mono", "multi", "all"],
                    choices=["mono", "multi", "all"],
                    help="mono: language-specific BERT per language; multi: mBERT per "
                         "language; all: one mBERT on all --langs together")
    ap.add_argument("--base-model", default=None,
                    help="override every pre-trained model (e.g. prajjwal1/bert-tiny for smoke tests)")
    ap.add_argument("--max-sents", type=int, default=None,
                    help="cap each sentence group (per game / persuade) to N sentences")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap training steps (smoke tests)")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--fix-known-bugs", action="store_true",
                    help="use KotOR 1 negatives and the mBERT tokenizer for mBERT predictions")
    args = ap.parse_args(argv)
    vars(CFG).update(vars(args))
    if not Path(CFG.data).exists():
        ap.error(f"{CFG.data} not found - run `python data/download.py` first")
    _ensure_punkt()

    langs = args.langs
    for lang in langs:
        if "mono" in args.runs or "multi" in args.runs:
            get_traintest_data(lang)
        if "mono" in args.runs:
            print("Training", lang, "monolingual model")
            finetune(lang, multiling=False)
            predict(lang, multiling=False)
        if "multi" in args.runs:
            print("Training", lang, "monolingual multilingual BERT model")
            finetune(lang, multiling=True)
            predict(lang, multiling=True)

    if "all" in args.runs:
        get_traintest_data(langs)
        finetune(langs, multiling=True)
        predict(langs, multiling=True)


if __name__ == "__main__":
    main()
