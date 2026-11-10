# 2022 — dataset release (`multilingual_persuasion_dataset`)

Code published with the Zenodo dataset in March 2022 (public GitHub repo
`Teemursu/multilingual_persuasion_dataset`): the extraction pipeline that produced the dataset and
a cleaned-up version of the classifier.

### If you use this repository or dataset, please cite

Pöyhönen, T., Hämäläinen, M., Alnajjar, K. (2022) "Multilingual Persuasion Detection: Video Games as an
Invaluable Data Source for NLP". *DiGRA '22 – Proceedings of the 2022 DiGRA International Conference.*

Dataset: <https://doi.org/10.5281/zenodo.6341173>

| Path | What it is |
|---|---|
| `data_extraction.py` | Runs the pipeline: string ids from the games' dialogue XML → aligned dialogue strings → cleaning. |
| `extracting_data/tools/{nwn,kotor1,kotor2}_xml.py` | Walk the `.dlg` XML conversions and collect string ids → `data/ids/*_dataset.csv`. |
| `extracting_data/tools/sid_to_str.py` | Look the ids up in each language's `dialog.tlk` XML and flag `[Persuade]`-type tags → `data/persuasion_dialogue_dataset.csv`. |
| `extracting_data/tools/data_processing.py` | Strip tags/entities and set `is_persuasion` to True/False. The Zenodo `digra.csv` is this output without the `filename`/`id` columns. |
| `models/multilingual_persuasion_detection.py` | The 11-model experiment (as in 2021), reading the released dataset. |

## Running

From the repository root, after `python data/download.py`:

```
# smoke test -- passes on CPU in well under a minute per model
python 2022-dataset-release/models/multilingual_persuasion_detection.py --langs en fr \
    --runs mono multi all --base-model prajjwal1/bert-tiny --max-sents 200 --max-steps 1

# full experiment (needs a GPU)
python 2022-dataset-release/models/multilingual_persuasion_detection.py [--fix-known-bugs]
```

Outputs go to `runs/2022/`. The extraction pipeline (`data_extraction.py`, run from this folder)
needs the XML conversions of the games' `dialog.tlk`/`.dlg` files under
`extracting_data/infinity_engine/{kotor1,kotor2,nwn}/{tlk,xml}/`. Those are copyrighted game files and
are not distributed (local copy: see `../data/README.md`). The pipeline was not re-run; its modules
import on Python 3.12 with the packages in `requirements.txt`.

### What was changed in 2026

- `models/multilingual_persuasion_detection.py`: reads `data/digra.csv` (`--data`), transformers-5 API
  (`eval_strategy`, no `logging_dir`, `Auto*` classes, Hub ids, `Softmax(dim=1)`), no `mikatools`,
  CLI flags for smoke tests, outputs in `--work-dir` instead of the working directory.
- `data_processing.py`: printed `df["persuasion"]` (KeyError, the column is `is_persuasion`) and used a
  Windows-only `data\...` path.

### Known issues in the classifier (kept by default, fixed with `--fix-known-bugs`)

- `kotor1_neg = sent_tokenize(kotor2_neg)`: KotOR 1 negatives are never used, and KotOR 2 negatives are used twice.
- Per-language mBERT models are evaluated with the monolingual tokenizer of their language (same bug as in
  the 2021 code; see `../2021-digra-classifier/README.md`).

Also: the `langdetect` filter for English text in translations is commented out, and splits depend on
`PYTHONHASHSEED` and an unseeded `train_test_split`.

### Not included

The local copy of this repo (`R:\Documents\University\MA thesis\DiGRA 2022\Persuasion detection (code)\multilingual_persuasion_dataset\`)
also holds unrelated April 2022 coursework (emotion and sarcasm classification: `week4.py`, `week5.py`,
`variation_preds.py`, `sarcasm_data.json`, `emotion_classification_*`, `sarcasm_classification_*`). Its
`results_pers/` model is a 6-label emotion classifier, not a persuasion model. None of it was added here.
