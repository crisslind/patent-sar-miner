from pathlib import Path
import pandas as pd
from patent_sar_miner.analysis import AnalysisConfig, run_analysis


def test_end_to_end(tmp_path: Path):
    source = Path(__file__).parents[1] / "examples" / "demo_patent_compounds.csv"
    output = tmp_path / "result"
    manifest = run_analysis(
        AnalysisConfig(
            input_path=str(source),
            structure_column="smiles",
            analysis_column="pIC50",
            output_dir=str(output),
            query_smiles="COc1ccc(Nc2ncc(C)cn2)cc1",
        )
    )
    expected_records = len(pd.read_csv(source))
    assert manifest["counts"]["input_records"] == expected_records
    
    assert (output / "report.html").exists()
    assert (output / "standardized_compounds.csv").exists()
    assert (output / "matched_pairs.csv").exists()
    assert (output / "r_group_table.csv").exists()
    assert (output / "measurement_qc.csv").exists()
    assert (output / "replicate_summary.csv").exists()
    assert (output / "model_validation.json").exists()
