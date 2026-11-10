# persuasion-detection

Multilingual persuasion detection with BERT, trained on dialogue from role-playing games
(2021–2022). Teemu Pöyhönen, with Mika Hämäläinen and Khalid Alnajjar (University of Helsinki).

This repository brings together the code behind

> Pöyhönen, T., Hämäläinen, M., Alnajjar, K. (2022). "Multilingual Persuasion Detection: Video Games as an
> Invaluable Data Source for NLP". *DiGRA '22 – Proceedings of the 2022 DiGRA International Conference.*

and the published dataset

> Pöyhönen, T., Hämäläinen, M., Alnajjar, K. (2022). *Multilingual Persuasion Dataset* (version of 2022-03-09).
> Zenodo. <https://doi.org/10.5281/zenodo.6341173> — CC BY 4.0.

## Summary

RPG dialogue is semi-annotated by the developers: lines where the player attempts a Persuade skill
check carry a tag such as `[Persuade]`. The dialogue lines of *Star Wars: Knights of the Old Republic* 1
and 2 and *Neverwinter Nights* were extracted from the games' `dialog.tlk` files in English, German,
French, Italian and Spanish, aligned by string id, labelled persuasion / non-persuasion from the tags,
cleaned (tags removed) and split into sentences. Eleven classifiers were fine-tuned: a language-specific
BERT for each language (BERT-base, German BERT, CamemBERT, Italian BERT, BETO), multilingual BERT for
each language, and one multilingual BERT on all languages. The language-specific models reached
0.87–0.90 accuracy and 0.78–0.83 macro-F1; the multilingual ones did much worse (see the tokenizer bug
below). The Part 2 research plan (Nov 2021) turned this into the MA thesis, which is in a separate repository.

## Structure

| Folder | What | Original repo |
|---|---|---|
| [`2021-digra-classifier/`](2021-digra-classifier) | Code and test-set predictions for the DiGRA paper (June–Oct 2021): classifier, 2021 extraction scripts, Pillars of Eternity exploration | `Teemursu/persuasion_detection` (private) + uncommitted work recovered from the old laptop |
| [`2022-dataset-release/`](2022-dataset-release) | Code released with the Zenodo dataset (Mar 2022): extraction pipeline and cleaned-up classifier | `Teemursu/multilingual_persuasion_dataset` (public) |
| [`data/`](data) | `download.py` for the Zenodo dataset; the data itself is git-ignored | — |

Each folder keeps its original commit history. Raw game files, their XML/TXT dumps and large
intermediate CSVs were removed from that history: they are copyrighted game assets and 5–40 MB each.

## Getting the data

```
python data/download.py      # -> data/digra.csv (61.8 MB, MD5-checked against the Zenodo API)
```

The file layout and columns are in [`data/README.md`](data/README.md). Both classifiers read
`data/digra.csv` directly.

## Running

Python 3.12:

```
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
.venv/Scripts/pip install -r requirements.txt
python data/download.py

# smoke tests (tiny model, small subset, 1 training step; a few minutes on CPU)
python 2022-dataset-release/models/multilingual_persuasion_detection.py --langs en fr --runs mono multi all \
    --base-model prajjwal1/bert-tiny --max-sents 200 --max-steps 1
python 2021-digra-classifier/classifier_model.py --langs en de --runs mono multi all \
    --base-model prajjwal1/bert-tiny --max-lines 200 --max-steps 1

# full experiments (11 BERT-base models x 5 epochs, GPU recommended)
python 2021-digra-classifier/classifier_model.py
python 2022-dataset-release/models/multilingual_persuasion_detection.py
```

Models, split files and prediction CSVs are written to `runs/` (git-ignored).

| Part | Status (Python 3.12, torch 2.15 CPU, transformers 5.22) |
|---|---|
| 2022 classifier | Runs: smoke-tested on the Zenodo file for every mode (mono / multi / all) and with `--fix-known-bugs`. |
| 2021 classifier | Runs: smoke-tested on the Zenodo file for all modes; `--legacy-csv-dir` checked against the original CSVs (data preparation). |
| 2022 extraction pipeline | Not run: needs the games' files (not distributed). Modules import; one KeyError and one Windows-only path fixed. |
| 2021 extraction / PoE scripts | Not run: need game files and intermediate CSVs with hard-coded names. |
| `2021-digra-classifier/test.py` | Unfinished scratch file, does not parse; kept as found. |

## How the code matches the paper

- Data extraction, tag-based labelling, cleaning and sentence tokenisation (label copied to every
  sentence of a line) match the paper's description. So do the 11 models and the 70/30 split.
- `2021-digra-classifier/models/performance/` holds the paper's predictions; recomputing the metrics
  reproduces Tables 2 and 3.
- **Tokenizer bug.** The per-language multilingual-BERT models were evaluated with the language-specific
  tokenizer (e.g. CamemBERT's SentencePiece for the French mBERT model). This most likely explains the
  poor Table 2 results (French mBERT: persuade recall 0.00), while the all-languages mBERT, evaluated
  with its own tokenizer, gets 0.96 persuade recall. The paper's conclusion that mBERT is worse than
  monolingual BERT is not supported by these numbers. `--fix-known-bugs` fixes it; by default the original
  behaviour is kept.
- Non-persuasion sentences came from KotOR 2 and NWN only (2021); the 2022 script means to add KotOR 1
  but a typo uses KotOR 2 twice.
- The paper's English-comment filter (`langdetect` over the translations) is not in the published data
  or the final scripts. That is why Zenodo has 1,110 persuasion lines and Table 1 reports 1,177 per language.
- Splits are not deterministic (`set()` order, unseeded `train_test_split`), so retraining will not
  give exactly the paper's numbers.

Details are in the two folder READMEs.

## Trained models

None are in git (each 400–700 MB). Surviving fine-tuned models from 2021 are on the archive drive:

- `R:\Memories\Old laptop (läppäri)\Users\temep\Code\Repositories\uni\persuasion_detection\persuasion_detection\`
  `en-finetuned-persuade-bert\`, `en-finetuned-multilingual-persuade-bert\`,
  `models\monoling\fr-finetuned-multiling-persuade-bert\`, `models\multiling_solo\es-finetuned-persuade-bert\`
- `R:\Documents\University\MA thesis\DiGRA 2022\Persuasion detection (code)\persuasion-BERT\fr-finetuned-multiling-persuade-bert\`

## How to cite

Please cite the DiGRA 2022 paper and the dataset (DOI [10.5281/zenodo.6341173](https://doi.org/10.5281/zenodo.6341173)), as above.
The dataset is CC BY 4.0. The dialogue text in it, and in the prediction CSVs under
`2021-digra-classifier/models/performance/`, comes from the games and remains © its rights holders.
