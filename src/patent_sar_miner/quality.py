from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

RELATION_ALIASES = ("standard_relation", "relation", "activity_relation")
VALUE_PATTERN = re.compile(r"^\s*(<=|>=|<|>|=|~)?\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def _relation_column(frame: pd.DataFrame) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    return next((lookup[name] for name in RELATION_ALIASES if name in lookup), None)


def measurement_quality(
    records: pd.DataFrame,
    analysis_column: str,
    disagreement_threshold: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Audit selected measurements and quantify exact-value replicate variability."""
    relation_column = _relation_column(records)
    parsed_relations: list[str] = []
    parsed_values: list[float] = []
    for value in records[analysis_column]:
        if pd.isna(value):
            parsed_relations.append("")
            parsed_values.append(float("nan"))
            continue
        match = VALUE_PATTERN.match(str(value))
        parsed_relations.append(match.group(1) or "" if match else "")
        parsed_values.append(float(match.group(2)) if match else float("nan"))

    relation = pd.Series(parsed_relations, index=records.index, dtype="string")
    if relation_column:
        explicit = records[relation_column].fillna("").astype(str).str.strip()
        relation = explicit.where(explicit.ne(""), relation)
    numeric = pd.Series(parsed_values, index=records.index, dtype="float64")
    exact = relation.isin(["", "="])

    qc = records.copy()
    qc["analysis_value"] = numeric
    qc["analysis_relation"] = relation
    qc["is_exact_measurement"] = exact
    qc["measurement_qc_flag"] = np.select(
        [numeric.isna(), ~exact],
        ["non_numeric_or_missing", "censored_or_approximate"],
        default="usable_exact_value",
    )

    usable = qc.loc[qc["is_exact_measurement"] & qc["analysis_value"].notna()].copy()
    rows: list[dict[str, object]] = []
    for smiles, group in usable.groupby("standardized_smiles", sort=True):
        values = group["analysis_value"].astype(float)
        count = len(values)
        std = float(values.std(ddof=1)) if count > 1 else float("nan")
        sem = std / math.sqrt(count) if count > 1 else float("nan")
        median = float(values.median())
        mad = float((values - median).abs().median())
        spread = float(values.max() - values.min())
        rows.append({
            "standardized_smiles": smiles,
            "compound_id": str(group.iloc[0]["compound_id"]),
            "n_measurements": count,
            "mean": float(values.mean()),
            "median": median,
            "std": std,
            "mad": mad,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "range": spread,
            "ci95_lower": float(values.mean() - 1.96 * sem) if count > 1 else float("nan"),
            "ci95_upper": float(values.mean() + 1.96 * sem) if count > 1 else float("nan"),
            "replicate_disagreement": bool(count > 1 and spread > disagreement_threshold),
            "source_rows": " | ".join(str(value) for value in group["source_row"]),
        })
    replicates = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "analysis_column": analysis_column,
        "total_records": len(qc),
        "usable_exact_records": int((exact & numeric.notna()).sum()),
        "censored_or_approximate_records": int((~exact).sum()),
        "non_numeric_or_missing_records": int(numeric.isna().sum()),
        "unique_standardized_structures": int(qc["standardized_smiles"].nunique()),
        "structures_with_replicates": int((replicates.get("n_measurements", pd.Series()).gt(1)).sum()),
        "replicate_disagreement_threshold": disagreement_threshold,
        "structures_with_replicate_disagreement": int(
            replicates.get("replicate_disagreement", pd.Series(dtype=bool)).sum()
        ),
    }
    return qc, replicates, summary
