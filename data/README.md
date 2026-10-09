# Data

The expected file is `raw/online_shoppers_intention.csv`, downloaded from the
[UCI dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
or its [DOI](https://doi.org/10.24432/C5F88Q). It is attributed to Sakar and
Kastro (2018) and is available under CC BY 4.0.

Run `python scripts/download_data.py`. The downloader extracts only the exact
expected CSV, refuses to overwrite it, and appends retrieval date, source,
SHA-256, and observed schema/count notes here. If the download is blocked,
download `online_shoppers_purchasing_intention_dataset.zip` manually from the
UCI page, extract only `online_shoppers_intention.csv` into `data/raw/`, and
record the retrieval date and checksum with the project team.

Generated cleaned data and split assignments belong in `processed/` and are
not source data.

## Retrieval record
- Date: 2026-10-09
- Source: https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip
- SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`
