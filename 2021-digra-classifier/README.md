# 2021 — DiGRA paper classifier (`persuasion_detection`)

Code behind *Multilingual Persuasion Detection: Video Games as an Invaluable Data Source for NLP*
(DiGRA 2022), written June–October 2021. Originally the private GitHub repo
`Teemursu/persuasion_detection`; the Oct 2021 working copy (never pushed) was recovered from
the old laptop and is what this folder contains.

> Role playing games (RPGs) have a considerable amount of text in video game dialogues. Quite often
> this text is semi-annotated by the game developers. In this paper, we extract a multilingual dataset
> of persuasive dialogue from several RPGs. We show the viability of this data in building a persuasion
> detection system using a natural language processing (NLP) model called BERT.

| Path | What it is |
|---|---|
| `classifier_model.py` | Data split, fine-tuning and evaluation of the 11 models in the paper (5 language-specific BERTs, 5 per-language mBERTs, 1 all-languages mBERT). Revived to read the Zenodo `digra.csv`. |
| `models/performance/*.csv` | **The paper's test-set predictions** (`sents, gold, preds`) for all 11 models, Oct 2021. They reproduce Tables 2 and 3 exactly (e.g. English BERT acc 0.87 / macro-F1 0.79; all-languages mBERT acc 0.52 / persuade recall 0.96). |
| `models/performance/notsure/` | An earlier run of the same 11 models (kept as found; the folder name is the author's). |
| `models/performance/precall.py` | Prints `classification_report` for every prediction CSV (Windows paths; run from this folder). |
| `data_extraction/games/tlk_files/process_tlk.py`, `read_dlg.py` | 2021 extraction of `[Persuade]` lines from KotOR/NWN `dialog.tlk` XML dumps. |
| `data_extraction/align_nonpersuades.py`, `delete_identical_empty_rows.py` | Alignment / clean-up of the multilingual line CSVs. |
| `data_extraction/games/poe/` | Pillars of Eternity / Tyranny attribute-check extraction (explored for follow-up work, not used in the paper). |
| `model/bert_multil.py` | Early mBERT fine-tuning template (still has the "depression" labels of the template it was copied from). |
| `test.py` | Unfinished scratch script for re-evaluating saved models (does not parse; kept as found). |
| `pyproject.toml` | Original Poetry stub (Python ^3.7). Use the top-level `requirements.txt` instead. |

The extraction scripts need the raw game files and the intermediate CSVs, which are not
distributed (see `../data/README.md`); they were not re-run.

## Running

From the repository root, after `python data/download.py`:

```
# smoke test (tiny model, 200 dialogue lines per group, 1 training step) -- passes on CPU
python 2021-digra-classifier/classifier_model.py --langs en de --runs mono multi all \
    --base-model prajjwal1/bert-tiny --max-lines 200 --max-steps 1

# the paper's setup (11 models x 5 epochs; needs a GPU)
python 2021-digra-classifier/classifier_model.py

# use the original (unpublished) en_removed CSVs instead of Zenodo
python 2021-digra-classifier/classifier_model.py --legacy-csv-dir "<path>/data/en_removed"
```

Outputs (split files `source_*.txt`, fine-tuned models, prediction CSVs) go to `runs/2021/`
(`--work-dir`). Every prediction run prints the same `classification_report` as `precall.py`.

### What was changed in 2026

- Data: the original read `data/en_removed/{kotor2_nonpersuade,nwn_nonpersuade,persuade}_en_removed.csv`.
  It now takes the same subsets (KotOR 2 non-persuasion, NWN non-persuasion, all persuasion lines)
  from `data/digra.csv`; `--legacy-csv-dir` restores the old input. The side effect of writing those
  frames back to the working directory was dropped.
- transformers 5: `evaluation_strategy` → `eval_strategy`, `logging_dir` removed, the prediction-only
  `Trainer` uses `eval_strategy="no"`; models are loaded with `Auto*` classes (BERT fallback for
  `prajjwal1/bert-tiny`), Hub ids instead of legacy ids and the local `./camembert-base`.
- Mixed tab/space indentation fixed, `Softmax(dim=1)`, `mikatools.open_read` → `open`.
- CLI flags for smoke tests (`--base-model`, `--max-lines`, `--max-steps`, `--epochs`, `--langs`, `--runs`).
  The Oct 2021 file had the training calls commented out (only data preparation ran); `--runs`
  defaults to all three experiments, as in the September 2021 version.

### Differences from the paper and known issues

- **Tokenizer mismatch for the per-language mBERT models.** `predict()` loads the *monolingual*
  tokenizer of the language (Italian/German/Spanish BERT vocab, CamemBERT SentencePiece for French)
  for models fine-tuned from `bert-base-multilingual-cased`. Only the all-languages model is evaluated
  with the mBERT tokenizer. The poor Table 2 results (French mBERT: persuade recall 0.00) are very likely
  caused by this, not by mBERT itself. Kept by default for reproducibility; `--fix-known-bugs` uses the
  right tokenizer.
- The paper says data from all three games was used. Here the negatives come from KotOR 2 (twice as many)
  and NWN only; KotOR 1 negatives are commented out. The positives come from all three games.
- The paper removes English developer comments from non-English lines. That was done in an earlier
  (unrecorded) step that produced the `en_removed` CSVs: rows where `langdetect` found English in a
  translation were dropped (the commented-out loop in the 2022 script). The Zenodo file does not apply this
  filter, so a Zenodo run uses somewhat different, noisier lines (English: 935 de-duplicated persuasion
  lines from Zenodo vs 995 from `en_removed`; the paper's Table 1 reports 1,177 persuasion lines).
- Splits are not reproducible run to run: lines are de-duplicated with `list(set(...))` (order depends
  on `PYTHONHASHSEED`) and `train_test_split` has no `random_state`. Set `PYTHONHASHSEED=0` for a stable
  test split. Test = last 30 % of each list, train/validation = first 70 %.
- Each split file ends with a newline, so one empty "sentence" per class ends up in the train and test data.

## Trained models (not in git)

Weights stay on R: (each 400–700 MB):

- `R:\Memories\Old laptop (läppäri)\Users\temep\Code\Repositories\uni\persuasion_detection\persuasion_detection\en-finetuned-persuade-bert\` — English BERT (bert-base-cased), Sep 2021
- `...\persuasion_detection\en-finetuned-multilingual-persuade-bert\` — English mBERT, Sep 2021
- `...\persuasion_detection\models\monoling\fr-finetuned-multiling-persuade-bert\` — French mBERT, Aug 2021
- `...\persuasion_detection\models\multiling_solo\es-finetuned-persuade-bert\` — Spanish BETO, Aug 2021
- `R:\Documents\University\MA thesis\DiGRA 2022\Persuasion detection (code)\persuasion-BERT\fr-finetuned-multiling-persuade-bert\` — French mBERT, Sep 2021 (next to the Slurm `pers.err.*` training logs)

The other final models of the paper were not found locally.
