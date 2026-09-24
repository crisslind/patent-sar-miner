# Patent SAR Miner

Patent SAR Miner is an offline RDKit workflow for validating and analysing
patent-derived structure–activity data supplied as a CSV file. It standardises
structures, audits measurements and replicates, identifies chemical series,
clusters compounds, extracts matched molecular pairs and R groups, and runs an
optional leakage-aware baseline potency model.

The program does not retrieve patent or ChEMBL data. The user supplies the
curated CSV and explicitly selects the structure and numerical analysis columns.
Every other source column is retained in `all_activities.csv` and in the
row-level standardised export.

## Installation on macOS

Install Miniforge for Apple Silicon or Intel, unzip the project, then run:

```bash
cd patent-sar-miner
conda env create -f environment.yml
conda activate patent-sar-miner
patent-sar-miner --help
```

## Command line

```bash
patent-sar-miner \
  --input activities.csv \
  --structure SMILES \
  --analysis_col pIC50 \
  --output results/
```

Column names are exact and case-sensitive. They may contain spaces when quoted:

```bash
patent-sar-miner \
  --input activities.csv \
  --structure "Registry SMILES" \
  --analysis_col "Measured pIC50" \
  --output results/
```

`--analysis-col` is accepted as an alias for `--analysis_col`. Use
`--skip-model` to run structural and measurement analysis without predictive
modelling.

## Measurement validation

The selected response is parsed as a numerical measurement. Relations embedded
in values, such as `<6.0`, and optional `standard_relation`, `relation`, or
`activity_relation` columns are retained and flagged. Only exact numerical
measurements enter replicate statistics and modelling.

Replicate measurements are grouped by standardised structure. The workflow
reports mean, median, standard deviation, median absolute deviation, range and
approximate 95% confidence intervals. Groups whose range exceeds the configured
threshold are flagged:

```bash
--replicate-disagreement-threshold 1.0
```

The units and meaning of the selected analysis column must already be
comparable. The software does not silently convert or combine different assay
types, targets, endpoints or units.

## Predictive validation

When enough data are available, the workflow trains a deterministic Random
Forest baseline using radius-2 Morgan fingerprints. Evaluation uses a Murcko
scaffold-group holdout rather than a random molecular split. Cross-validated
training residuals define 90% conformal prediction intervals. If `patent_id` is
present, leave-one-patent-out metrics are also calculated.

The model is skipped with an explicit reason when there are fewer than 12
usable compounds or fewer than three scaffolds. This baseline is for validation
and comparison, not a production potency claim.

## Outputs

| File | Contents |
|---|---|
| `all_activities.csv` | Original CSV, unchanged, with every source column |
| `standardized_records.csv` | Row-level source data plus standardised structure and QC fields |
| `measurement_qc.csv` | Numerical parsing, relation and usability flags |
| `replicate_summary.csv` | Per-structure replicate statistics and confidence intervals |
| `uncertainty_summary.json` | Dataset-level measurement-quality counts |
| `standardized_compounds.csv/.sdf` | Deduplicated compounds, descriptors, scaffolds and clusters |
| `matched_pairs.csv` | Shared cores, fragments and response differences |
| `r_group_table.csv` | R-group decomposition of the largest series |
| `model_predictions.csv` | Observed/predicted values and prediction intervals |
| `model_validation.json` | Scaffold-holdout and optional patent-holdout metrics |
| `run_manifest.json` | Input hash, selected columns, versions, seed and configuration |
| `report.html` | Self-contained summary report |

If `standard_type` is present, the unchanged source rows are additionally split
into `ic50_activities.csv` and `non_ic50_activities.csv`. These files do not
control the analysis; `--analysis_col` does.

## Notebook

Start JupyterLab from the project root:

```bash
jupyter lab
```

Open `notebooks/Analysis_of_Results.ipynb`. It loads `results/`, shows
measurement quality, replicate uncertainty, model validation, chemical series,
matched-pair structures and numerically ordered R groups. R-group columns that
are hydrogen for every row are omitted from the displayed dataframe.

## API (work in progress)...

Start the optional backend service:

```bash
uvicorn patent_sar_miner.api:app --host 127.0.0.1 --port 8000
```

`POST /analysis` accepts JSON records plus `structure` and `analysis_col` field
names. `GET /health` reports service status.

## Tests

```bash
pytest
ruff check src tests
```

This workflow supports scientific exploration. Similarity and SAR analysis do
not establish patent scope, validity, infringement or freedom to operate.
