# -*- coding: utf-8 -*-
"""Persuasion detection with monolingual and multilingual BERT (DiGRA 2022 paper version).

Originally exported from a Colab notebook (classifier_model.ipynb,
https://colab.research.google.com/drive/1hXq2h-MHR3HZIA49S3vrESyIu8JBmyav);
this is the Oct 2021 state that produced the results in
models/performance/ (Tables 2 and 3 of the paper).

Revived in 2026 for Python 3.12 / current transformers. The original read
three intermediate CSVs (data/en_removed/*.csv, never published); it now reads
the published Zenodo dataset (https://doi.org/10.5281/zenodo.6341173,
``digra.csv``) by default and selects the same subsets from it. Use
``--legacy-csv-dir`` to point at the original en_removed CSVs instead.
See ../README.md and this folder's README for details and known bugs.

Example (from the repository root):
    python data/download.py
    python 2021-digra-classifier/classifier_model.py --langs en --runs mono \
        --base-model prajjwal1/bert-tiny --max-lines 200 --max-steps 1
"""

import argparse
import random
import re
import string
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from langdetect import detect
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "digra.csv"
DEFAULT_WORK_DIR = REPO_ROOT / "runs" / "2021"

# Hub ids of the models used in the paper (original: legacy ids, ./camembert-base)
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
random.seed(seed)

CFG = argparse.Namespace(
    data=DEFAULT_DATA,
    legacy_csv_dir=None,
    work_dir=DEFAULT_WORK_DIR,
    base_model=None,
    max_lines=None,
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


def clean_source(source):
    source = [str(s).replace("\r", "") for s in source]
    source = [str(s).replace("&apos;", "'") for s in source]
    source = [str(s).replace("\n", "") for s in source]
    source = [str(s).replace("&apos;", "'") for s in source]
    source = [str(s).replace("&quot;", '"') for s in source]
    source = [str(s).replace("quot;", '"') for s in source]
    source = [str(s).replace("apos;", "'") for s in source]
    source = [str(s).replace("lt;StartAction", "") for s in source]
    source = [str(s).replace("PERSUASION : ", "") for s in source]
    source = [str(s).replace("&lt;", "") for s in source]
    source = [str(s).replace("&gt;", "") for s in source]
    source = [str(s).replace("</string", "") for s in source]
    source = [str(s).replace(">", "") for s in source]
    source = [str(s).replace("<", "") for s in source]
    source = [s.rstrip() for s in source]
    source = [s.lstrip(string.punctuation + string.whitespace) for s in source]
    source = [x for x in source if x]

    source = [nltk.sent_tokenize(sent) for sent in source]
    source = [j for i in source for j in i]
    return source


def del_eng_sents(source):
    source_copy = []
    for sent in source:
        try:
            detection = detect(sent)
        except Exception:
            detection = "en"
        if detection != "en":
            source_copy.append(sent)

    return source_copy


def remove_stuff_between_brackets(source):
    source = [j for i in source for j in i]
    source = [re.sub(r"\[[^>]+\]", "", str(s)) for s in source]
    source = [re.sub(r"\{[^>]+\}", "", s) for s in source]
    return source


def _load_frames():
    """Return (kotor2 negatives, NWN negatives, persuasion lines) data frames."""
    if CFG.legacy_csv_dir:
        d = Path(CFG.legacy_csv_dir)
        df_kotor2_neg = pd.read_csv(d / "kotor2_nonpersuade_en_removed.csv", index_col=[0], encoding="utf-8", engine="python")
        with open(d / "nwn_nonpersuade_en_removed.csv", encoding="utf-8", errors="backslashreplace") as input_fd:
            df_nwn_neg = pd.read_csv(input_fd, index_col=[0], engine="python")
        df_persuade_sents = pd.read_csv(d / "persuade_en_removed.csv", index_col=[0], encoding="utf-8", engine="python")
        return df_kotor2_neg, df_nwn_neg, df_persuade_sents
    # Zenodo digra.csv: sid, game, is_persuasion, en, de, fr, it, es
    df = pd.read_csv(CFG.data, index_col=[0], encoding="utf-8")
    df_kotor2_neg = df.loc[(df.is_persuasion == False) & (df.game == "KotOR 2")]
    df_nwn_neg = df.loc[(df.is_persuasion == False) & (df.game == "NWN 1")]
    df_persuade_sents = df.loc[df.is_persuasion == True]
    return df_kotor2_neg, df_nwn_neg, df_persuade_sents


def _work(name):
    return Path(CFG.work_dir) / name


def _read_lines(name):
    with open(_work(name), encoding="utf-8") as f:
        return f.read().split("\n")


def get_traintest_data(langs):
    # KotOR 1 negatives were not used in the paper version (commented out).
    df_kotor2_neg, df_nwn_neg, df_persuade_sents = _load_frames()

    df_kotor2_neg = df_kotor2_neg.dropna(subset=["en", "de", "it", "es", "fr"])
    df_nwn_neg = df_nwn_neg.dropna(subset=["en", "de", "it", "es", "fr"])
    df_persuade_sents = df_persuade_sents.dropna(subset=["en", "de", "it", "es", "fr"])
    # (original also wrote these frames back to ./*_en_removed.csv here)

    kotor2_neg = []
    nwn_neg = []
    pos_sents = []
    if type(langs) is list:
        for lang in langs:
            kotor2_neg.append(list(set(df_kotor2_neg[lang].tolist())))
            nwn_neg.append(list(set(df_nwn_neg[lang].tolist())))
            pos_sents.append(list(set(df_persuade_sents[lang].tolist())))
    else:
        kotor2_neg.append(list(set(df_kotor2_neg[langs].tolist())))
        nwn_neg.append(list(set(df_nwn_neg[langs].tolist())))
        pos_sents.append(list(set(df_persuade_sents[langs].tolist())))

    kotor2_neg = remove_stuff_between_brackets(kotor2_neg)
    nwn_neg = remove_stuff_between_brackets(nwn_neg)
    persuade_sents = remove_stuff_between_brackets(pos_sents)
    if CFG.max_lines:  # smoke-test subset
        n = CFG.max_lines
        kotor2_neg, nwn_neg, persuade_sents = kotor2_neg[:n], nwn_neg[:n], persuade_sents[:n]
    print("kotor2:", len(kotor2_neg))
    print("nwn", len(nwn_neg))
    print("persuade", len(persuade_sents))

    pos_num = int(len(persuade_sents) * 0.7)
    pos_test = int(len(persuade_sents) * 0.3)
    neg_sents = clean_source(kotor2_neg[: pos_num * 2]) + clean_source(nwn_neg[:pos_num])
    pos_sents = clean_source(persuade_sents[:pos_num])
    neg_sents_test = clean_source(kotor2_neg[-pos_test * 2:]) + clean_source(nwn_neg[-pos_test:])
    pos_sents_test = clean_source(persuade_sents[-pos_test:])
    print("neg_sents_test", len(neg_sents_test))
    print("neg_sents", len(neg_sents))
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
        max_steps=CFG.max_steps,  # -1 = use num_train_epochs (smoke tests: 1)
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_steps=10,  # (logging_dir='./logs' is no longer an option)
        save_strategy="no",
        report_to="none",
        seed=seed,
    )


def finetune(langs, multiling):
    if CFG.base_model:
        base = CFG.base_model
    elif multiling or type(langs) is list:
        base = MULTI_MODEL
    else:
        base = MONO_MODELS[langs]
    print("Fine-tuning", base)
    tokenizer = _load_tokenizer(base)
    model = _load_model(base, num_labels=2)

    source_neg = _read_lines("source_neg.txt")
    source_pos = _read_lines("source_persuade.txt")
    neg_labels = ["no_persuade"] * len(source_neg)
    pos_labels = ["persuade"] * len(source_pos)

    source = source_neg + source_pos
    labels = neg_labels + pos_labels

    train_batch, valid_batch, train_labels, valid_labels = train_test_split(source, labels, test_size=0.1)
    train_text = tokenizer(train_batch, return_tensors="pt", padding=True, truncation=True)
    valid_text = tokenizer(valid_batch, return_tensors="pt", padding=True, truncation=True)

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
    labs = {"persuade": {"cor": 0.0, "err": 0.0}, "no_persuade": {"cor": 0.0, "err": 0.0}}
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


def predict(langs, multiling):
    # NOTE: original bug -- the per-language mBERT models were evaluated with the
    # *monolingual* tokenizer of that language (only the all-languages model used
    # the mBERT tokenizer). This explains the poor Table 2 numbers; kept by
    # default for reproducibility, fixed with --fix-known-bugs.
    if CFG.base_model:
        tok_name = CFG.base_model
    elif type(langs) is list or (multiling and CFG.fix_known_bugs):
        tok_name = MULTI_MODEL
    else:
        tok_name = MONO_MODELS[langs]
    tokenizer = _load_tokenizer(tok_name)
    model = str(_work(_output_name(langs, multiling)))

    pos_test = _read_lines("source_persuade_test.txt")
    neg_test = _read_lines("source_neg_test.txt")

    test_source = neg_test + pos_test
    test_labels = ["no_persuade"] * len(neg_test) + ["persuade"] * len(pos_test)

    test_text = tokenizer(test_source, return_tensors="pt", padding=True, truncation=True)
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
    df = pd.DataFrame({"sents": test_source, "gold": test_labels.tolist(), "preds": pred_labels})
    df.to_csv(model_path + ".csv")
    eval(list(zip(preds.tolist(), ps.label_ids.tolist())))
    # same report as models/performance/precall.py
    print(classification_report(df.gold, df.preds, labels=[0, 1],
                                target_names=["no_persuade", "persuade"], zero_division=0))


def _ensure_punkt():
    for res in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Zenodo digra.csv")
    ap.add_argument("--legacy-csv-dir", type=Path, default=None,
                    help="read the original data/en_removed/*.csv files instead of digra.csv")
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR,
                    help="where split files, models and prediction CSVs are written")
    ap.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    ap.add_argument("--runs", nargs="+", default=["mono", "multi", "all"], choices=["mono", "multi", "all"])
    ap.add_argument("--base-model", default=None, help="override every pre-trained model (smoke tests)")
    ap.add_argument("--max-lines", type=int, default=None, help="cap each dialogue-line group to N lines")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap training steps (smoke tests)")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--fix-known-bugs", action="store_true",
                    help="evaluate per-language mBERT models with the mBERT tokenizer")
    args = ap.parse_args(argv)
    vars(CFG).update(vars(args))
    if not CFG.legacy_csv_dir and not Path(CFG.data).exists():
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
