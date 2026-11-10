# Data

Everything in this folder except `download.py` and this file is git-ignored.

## Multilingual Persuasion Dataset (Zenodo)

> Pöyhönen, Teemu; Hämäläinen, Mika; Alnajjar, Khalid (2022). *Multilingual Persuasion Dataset*.
> Zenodo. <https://doi.org/10.5281/zenodo.6341173>. Licence: CC BY 4.0.

Download it with

```
python data/download.py
```

The script reads the file list and MD5 checksums from the Zenodo API
(`https://zenodo.org/api/records/6341173`), downloads anything missing into this
folder and checks the checksums.

### Expected layout

```
data/
  download.py
  README.md
  digra.csv          61.8 MB, md5 32a0eec9c34efae112f3d101bc36f612
```

Both training scripts read `data/digra.csv` by default (override with `--data PATH`).

### `digra.csv` format

UTF-8 CSV with a header row, 134,113 rows, no index column:

| column | content |
|---|---|
| `sid` | string id of the line in the game's `dialog.tlk` (float, e.g. `35659.0`; unique only within a game) |
| `game` | `KotOR 1`, `KotOR 2` or `NWN 1` |
| `is_persuasion` | `True` if the English line had a `[Persuade]`-style skill-check tag, else `False` |
| `en`, `de`, `fr`, `it`, `es` | the line in English, German, French, Italian and Spanish (aligned by `sid`) |

Tags such as `[Persuade]`, `{...}` and `(...)` and XML entities are already stripped
(see `2022-dataset-release/extracting_data/tools/data_processing.py`).
Lines may contain several sentences; the training scripts sentence-tokenise them with NLTK
and give every sentence the label of its line.

| game | `is_persuasion=False` | `is_persuasion=True` |
|---|---:|---:|
| KotOR 1 | 33,093 | 242 |
| KotOR 2 | 30,765 | 184 |
| NWN 1 | 69,145 | 684 |
| total | 133,003 | 1,110 |

`digra.csv` is `data/persuasion_dialogue_dataset.csv` from the 2022 extraction pipeline with
the `filename` and `id` columns dropped.

## Data that is not published

The raw game files (`dialog.tlk`, `.dlg`, their XML/TXT conversions, Pillars of Eternity
`.conversation`/`.stringtable` files, Planescape: Torment `.d` files) are copyrighted game
assets and are **not** in this repository or on Zenodo. To re-run the extraction you need the
games and must convert the files yourself. Local copies (read-only archive):

- `R:\Documents\University\MA thesis\DiGRA 2022\Persuasion detection (code)\multilingual_persuasion_dataset\extracting_data\infinity_engine\` — XML conversions used by the 2022 pipeline
- `R:\Documents\University\MA thesis\DiGRA 2022\Persuasion detection (code)\multilingual_persuasion_dataset\data\` — `persuasion_dialogue_dataset.csv` (72 MB, pipeline output) and `ids/`
- `R:\Memories\Old laptop (läppäri)\Users\temep\Code\Repositories\uni\persuasion_detection\persuasion_detection\data\` — 2021 intermediate CSVs, including `en_removed/` used for the DiGRA paper (`classifier_model.py --legacy-csv-dir` reads these)
- `R:\Documents\University\MA thesis\DiGRA 2022\digra.csv` — an earlier export of the Zenodo file (extra index columns, not byte-identical to the published one)
