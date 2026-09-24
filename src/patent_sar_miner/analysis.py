from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import rdkit

from patent_sar_miner.io import prepare_compounds, read_compounds, standardized_records, write_sdf
from patent_sar_miner.modeling import SEED, validate_potency_model
from patent_sar_miner.quality import measurement_quality
from patent_sar_miner.report import generate_report
from patent_sar_miner.sar import (
    add_scaffolds,
    butina_clusters,
    matched_pairs,
    nearest_neighbours,
    r_group_table,
)


@dataclass(frozen=True)
class AnalysisConfig:
    input_path: str
    structure_column: str
    analysis_column: str
    output_dir: str = "patent_sar_output"
    query_smiles: str | None = None
    similarity_threshold: float = 0.55
    neighbour_limit: int = 20
    replicate_disagreement_threshold: float = 1.0
    run_model: bool = True
    stack_by: str = "auto"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_analysis(config: AnalysisConfig) -> dict[str, object]:
    if not 0.0 < config.similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if config.replicate_disagreement_threshold <= 0:
        raise ValueError("replicate_disagreement_threshold must be positive")
    if config.stack_by not in {"auto", "patent", "assignee"}:
        raise ValueError("stack_by must be auto, patent or assignee")

    input_path = Path(config.input_path).expanduser().resolve()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source, raw = read_compounds(input_path, config.structure_column, config.analysis_column)
    source.to_csv(output_dir / "all_activities.csv", index=False)

    if "standard_type" in source.columns:
        ic50_mask = source["standard_type"].fillna("").astype(str).str.upper().eq("IC50")
        source.loc[ic50_mask].to_csv(output_dir / "ic50_activities.csv", index=False)
        source.loc[~ic50_mask].to_csv(output_dir / "non_ic50_activities.csv", index=False)
    else:
        source.iloc[0:0].to_csv(output_dir / "ic50_activities.csv", index=False)
        source.to_csv(output_dir / "non_ic50_activities.csv", index=False)

    accepted_records, rejected = standardized_records(raw)
    if accepted_records.empty:
        rejected.to_csv(output_dir / "rejected_records.csv", index=False)
        raise ValueError("No valid structures remained after standardisation")

    qc_records, replicates, qc_summary = measurement_quality(
        accepted_records,
        config.analysis_column,
        config.replicate_disagreement_threshold,
    )
    qc_records.to_csv(output_dir / "standardized_records.csv", index=False)
    qc_records.to_csv(output_dir / "measurement_qc.csv", index=False)
    replicates.to_csv(output_dir / "replicate_summary.csv", index=False)
    (output_dir / "uncertainty_summary.json").write_text(
        json.dumps(qc_summary, indent=2), encoding="utf-8"
    )
    if qc_summary["usable_exact_records"] == 0:
        rejected.to_csv(output_dir / "rejected_records.csv", index=False)
        raise ValueError(
            f"Selected analysis column {config.analysis_column!r} has no usable numerical values"
        )

    compounds, rejected = prepare_compounds(raw)
    compounds = compounds.merge(
        replicates[["standardized_smiles", "median", "n_measurements"]],
        on="standardized_smiles",
        how="left",
    ).rename(columns={"median": f"{config.analysis_column}_median"})
    compounds = add_scaffolds(compounds)
    compounds = butina_clusters(compounds, config.similarity_threshold)
    pairs = matched_pairs(compounds, f"{config.analysis_column}_median")
    r_groups = r_group_table(compounds, f"{config.analysis_column}_median")

    neighbours = None
    canonical_query = None
    if config.query_smiles:
        neighbours, canonical_query = nearest_neighbours(
            compounds, config.query_smiles, config.neighbour_limit
        )

    model_predictions = pd.DataFrame()
    model_summary: dict[str, object] = {
        "status": "skipped", "reason": "disabled with --skip-model"
    }
    if config.run_model:
        model_predictions, model_summary = validate_potency_model(
            compounds, replicates, config.analysis_column
        )

    compounds.to_csv(output_dir / "standardized_compounds.csv", index=False)
    rejected.to_csv(output_dir / "rejected_records.csv", index=False)
    pairs.to_csv(output_dir / "matched_pairs.csv", index=False)
    r_groups.to_csv(output_dir / "r_group_table.csv", index=False)
    model_predictions.to_csv(output_dir / "model_predictions.csv", index=False)
    (output_dir / "model_validation.json").write_text(
        json.dumps(model_summary, indent=2), encoding="utf-8"
    )
    write_sdf(compounds, output_dir / "standardized_compounds.sdf")
    if neighbours is not None:
        neighbours.to_csv(output_dir / "query_neighbours.csv", index=False)
    if "patent_id" in qc_records.columns:
        qc_records.to_csv(output_dir / "patent_compound_associations.csv", index=False)

    generate_report(
        compounds,
        pairs,
        r_groups,
        output_dir / "report.html",
        neighbours,
        canonical_query,
        qc_records,
        config.stack_by,
        qc_summary=qc_summary,
        model_summary=model_summary,
    )

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "input": {
            "filename": input_path.name,
            "sha256": _file_hash(input_path),
            "columns": list(source.columns),
            "retained_columns": list(source.columns),
            "structure_column": config.structure_column,
            "analysis_column": config.analysis_column,
        },
        "configuration": asdict(config),
        "random_seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "pandas": pd.__version__,
            "scikit_learn": version("scikit-learn"),
        },
        "counts": {
            "input_records": len(source),
            "usable_exact_measurements": qc_summary["usable_exact_records"],
            "unique_standardized_compounds": len(compounds),
            "rejected_records": len(rejected),
            "matched_pairs": len(pairs),
            "structures_with_replicates": qc_summary["structures_with_replicates"],
        },
        "outputs": {
            "row_level_source": "all_activities.csv",
            "row_level_standardized": "standardized_records.csv",
            "quality": "measurement_qc.csv",
            "replicates": "replicate_summary.csv",
            "model": "model_validation.json",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
