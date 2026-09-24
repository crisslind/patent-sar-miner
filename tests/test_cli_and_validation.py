import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from patent_sar_miner.api import app
from patent_sar_miner.cli import main
from patent_sar_miner.modeling import validate_potency_model
from patent_sar_miner.quality import measurement_quality
from patent_sar_miner.sar import add_scaffolds


def input_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Registry SMILES": ["CCOc1ccccc1", "CCOc1ccccc1", "CCCOc1ccccc1"],
        "Measured pIC50": [7.0, 7.2, 6.4],
        "project_note": ["first", "replicate", "second"],
        "assignee": ["Company A", "Company A", "Company B"],
        "standard_relation": ["=", "=", "<"],
    })


def test_explicit_columns_and_all_metadata_are_retained(tmp_path: Path):
    source = tmp_path / "activities.csv"
    output = tmp_path / "results"
    input_frame().to_csv(source, index=False)
    assert main([
        "--input", str(source),
        "--structure", "Registry SMILES",
        "--analysis_col", "Measured pIC50",
        "--output", str(output),
        "--skip-model",
    ]) == 0

    original = pd.read_csv(source)
    exported = pd.read_csv(output / "all_activities.csv")
    pd.testing.assert_frame_equal(exported, original)
    standardized = pd.read_csv(output / "standardized_records.csv")
    assert set(original.columns).issubset(standardized.columns)
    assert standardized["project_note"].tolist() == original["project_note"].tolist()
    deduplicated = pd.read_csv(output / "standardized_compounds.csv")
    assert deduplicated.iloc[0]["source_rows"] == "1 | 2"
    qc = pd.read_csv(output / "measurement_qc.csv")
    assert qc["measurement_qc_flag"].tolist() == [
        "usable_exact_value", "usable_exact_value", "censored_or_approximate"
    ]
    summary = json.loads((output / "uncertainty_summary.json").read_text())
    assert summary["structures_with_replicates"] == 1
    assert summary["censored_or_approximate_records"] == 1


def test_missing_selected_column_is_reported(tmp_path: Path, capsys):
    source = tmp_path / "activities.csv"
    input_frame().to_csv(source, index=False)
    assert main([
        "--input", str(source), "--structure", "missing",
        "--analysis_col", "Measured pIC50",
    ]) == 1
    assert "Available columns" in capsys.readouterr().err


def test_non_csv_input_is_rejected(tmp_path: Path, capsys):
    source = tmp_path / "activities.tsv"
    source.write_text("smiles\tvalue\nCC\t1\n")
    assert main([
        "--input", str(source), "--structure", "smiles", "--analysis_col", "value"
    ]) == 1
    assert "Input must be a CSV" in capsys.readouterr().err


def test_inline_relations_are_detected():
    records = pd.DataFrame({
        "compound_id": ["A", "B"],
        "source_row": [1, 2],
        "standardized_smiles": ["CC", "CCC"],
        "response": [">6.0", "7.1"],
    })
    qc, _, summary = measurement_quality(records, "response")
    assert qc["analysis_relation"].tolist() == [">", ""]
    assert summary["usable_exact_records"] == 1


def test_model_skips_small_data_with_explicit_reason():
    compounds = add_scaffolds(pd.DataFrame({
        "compound_id": ["A", "B"],
        "standardized_smiles": ["CCOc1ccccc1", "CCOc1ccncc1"],
    }))
    replicates = pd.DataFrame({
        "standardized_smiles": compounds["standardized_smiles"],
        "median": [7.0, 6.0],
        "n_measurements": [1, 1],
    })
    predictions, summary = validate_potency_model(compounds, replicates, "pIC50")
    assert predictions.empty
    assert summary["status"] == "insufficient_data"


def test_model_runs_scaffold_holdout_and_prediction_intervals():
    smiles = [
        "Cc1ccccc1", "CCc1ccccc1", "COc1ccccc1", "Fc1ccccc1",
        "Cc1ccncc1", "CCc1ccncc1", "COc1ccncc1", "Fc1ccncc1",
        "CC1CCCCC1", "CCC1CCCCC1", "COC1CCCCC1", "FC1CCCCC1",
    ]
    compounds = add_scaffolds(pd.DataFrame({
        "compound_id": [f"C{i:02d}" for i in range(len(smiles))],
        "standardized_smiles": smiles,
    }))
    replicates = pd.DataFrame({
        "standardized_smiles": smiles,
        "median": [5.0 + index * 0.15 for index in range(len(smiles))],
        "n_measurements": [1] * len(smiles),
    })
    predictions, summary = validate_potency_model(compounds, replicates, "pIC50")
    assert summary["status"] == "completed"
    assert summary["holdout_metrics"]["n"] > 0
    assert {
        "prediction_interval_lower_90", "prediction_interval_upper_90", "split"
    }.issubset(predictions.columns)


def test_api_uses_records_and_explicit_columns():
    client = TestClient(app)
    response = client.post("/analysis", json={
        "records": input_frame().to_dict(orient="records"),
        "structure": "Registry SMILES",
        "analysis_col": "Measured pIC50",
        "run_model": False,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["manifest"]["counts"]["input_records"] == 3
    assert payload["uncertainty"]["structures_with_replicates"] == 1
