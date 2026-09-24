from pathlib import Path

import nbformat
import pandas as pd
import pytest

from patent_sar_miner.visualization import load_results, year_stack_counts


def test_year_stacks_deduplicate_activities_and_other_categories():
    associations = pd.DataFrame([
        {"standardized_smiles": "CC", "year": 2020, "patent_id": "P1"},
        {"standardized_smiles": "CC", "year": 2020, "patent_id": "P1"},
        {"standardized_smiles": "CC", "year": 2020, "patent_id": "P2"},
        {"standardized_smiles": "CC", "year": 2020, "patent_id": "P3"},
        {"standardized_smiles": "CCC", "year": 2020, "patent_id": "P3"},
    ])
    by_patent = year_stack_counts(associations, by="patent", max_stacks=3)
    assert by_patent.loc[2020, "P1"] == 1
    combined = year_stack_counts(associations, by="patent", max_stacks=1)
    assert combined.loc[2020, "Other"] == 1


def test_load_results_includes_validation_exports(tmp_path: Path):
    output = tmp_path / "results"
    output.mkdir()
    pd.DataFrame([{"smiles": "CC", "value": 1.0}]).to_csv(
        output / "all_activities.csv", index=False
    )
    pd.DataFrame([{"n_measurements": 1}]).to_csv(output / "replicate_summary.csv", index=False)
    (output / "uncertainty_summary.json").write_text('{"usable_exact_records": 1}')
    data = load_results(output)
    assert len(data["replicate_summary"]) == 1
    assert data["uncertainty_summary"]["usable_exact_records"] == 1
    with pytest.raises(FileNotFoundError, match="all_activities.csv"):
        load_results(tmp_path / "missing")


def test_notebook_schema_and_offline_execution(tmp_path: Path, monkeypatch):
    notebook_path = Path(__file__).parents[1] / "notebooks" / "Analysis_of_Results.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    output = tmp_path / "results"
    output.mkdir()
    pd.DataFrame([{"SMILES": "CCOc1ccccc1", "pIC50": 7.0}]).to_csv(
        output / "all_activities.csv", index=False
    )
    pd.DataFrame([{
        "standardized_smiles": "CCOc1ccccc1", "murcko_scaffold": "c1ccccc1",
        "cluster_id": "C001", "compound_id": "A", "molecular_weight": 122.0,
        "clogp": 1.1,
    }]).to_csv(output / "standardized_compounds.csv", index=False)
    pd.DataFrame([{
        "standardized_smiles": "CCOc1ccccc1", "measurement_qc_flag": "usable_exact_value"
    }]).to_csv(output / "measurement_qc.csv", index=False)
    pd.DataFrame([{
        "standardized_smiles": "CCOc1ccccc1", "n_measurements": 2, "range": 0.2,
        "replicate_disagreement": False,
    }]).to_csv(output / "replicate_summary.csv", index=False)
    pd.DataFrame([{
        "standardized_smiles": "CCOc1ccccc1", "compound_id": "A"
    }]).to_csv(output / "standardized_records.csv", index=False)
    pd.DataFrame([
        {"compound_id": "A", "Core": "c1ccccc1", "R1": "[H][*:1]",
         "R10": "C[*:10]", "R2": "F[*:2]", "assignees": "Company A"},
        {"compound_id": "B", "Core": "c1ccccc1", "R1": "[H][*:1]",
         "R10": "Cl[*:10]", "R2": "Br[*:2]", "assignees": "Company A"},
    ]).to_csv(output / "r_group_table.csv", index=False)
    (output / "uncertainty_summary.json").write_text(
        '{"usable_exact_records": 1, "structures_with_replicates": 1}'
    )
    (output / "model_validation.json").write_text('{"status": "insufficient_data"}')
    (output / "run_manifest.json").write_text(
        '{"input": {"structure_column": "SMILES", "analysis_column": "pIC50"}}'
    )
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"notebook/{cell.id}", "exec"), namespace)  # noqa: S102
    assert namespace["analysis_column"] == "pIC50"
    assert namespace["hydrogen_only"] == ["R1"]
    assert namespace["r_groups"].columns[-2:].tolist() == ["R2", "R10"]
    assert "assignees" in namespace["r_groups"].columns
