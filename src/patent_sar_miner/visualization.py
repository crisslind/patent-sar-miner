"""Offline analysis helpers for notebooks/Analysis_of_Results.ipynb."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPORTS = (
    "all_activities", "ic50_activities", "non_ic50_activities",
    "standardized_compounds", "patent_compound_associations",
    "matched_pairs", "r_group_table", "query_neighbours", "rejected_records",
    "standardized_records", "measurement_qc", "replicate_summary", "model_predictions",
)


def load_results(path: str | Path) -> dict[str, object]:
    """Read existing exports without network access or mutation of the result directory."""
    directory = Path(path).expanduser().resolve()
    if not (directory / "all_activities.csv").is_file():
        raise FileNotFoundError(f"No all_activities.csv at {directory}; run the CLI first")
    result: dict[str, object] = {"directory": directory}
    for name in EXPORTS:
        filename = directory / f"{name}.csv"
        try:
            result[name] = pd.read_csv(filename) if filename.is_file() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            result[name] = pd.DataFrame()
    for name in ("run_manifest", "uncertainty_summary", "model_validation"):
        filename = directory / f"{name}.json"
        result[name] = json.loads(filename.read_text(encoding="utf-8")) if filename.is_file() else {}
    return result


def endpoint_counts(activities: pd.DataFrame) -> pd.Series:
    if activities.empty:
        return pd.Series(dtype="int64", name="activity_records")
    if "standard_type" not in activities.columns:
        endpoints = pd.Series("Untyped", index=activities.index)
    else:
        endpoints = activities["standard_type"].fillna("Untyped").astype(str).str.strip()
        endpoints = endpoints.replace("", "Untyped")
    return endpoints.value_counts().rename("activity_records")


def _publication_year(frame: pd.DataFrame) -> pd.Series:
    year = pd.Series(float("nan"), index=frame.index)
    if "publication_date" in frame.columns:
        text_dates = frame["publication_date"].astype("string")
        numeric = pd.to_numeric(text_dates, errors="coerce")
        numeric = numeric.where(numeric.between(1800, 2200))
        dates = pd.to_datetime(text_dates.where(numeric.isna()), errors="coerce").dt.year
        year = numeric.fillna(dates)
    if "year" in frame.columns:
        fallback = pd.to_numeric(frame["year"], errors="coerce").where(
            lambda values: values.between(1800, 2200)
        )
        year = year.fillna(fallback)
    return year


def year_stack_counts(
    associations: pd.DataFrame, by: str = "auto", max_stacks: int = 8
) -> pd.DataFrame:
    """Distinct structures per year/category; duplicates in Other stay deduplicated."""
    if by not in {"auto", "patent", "assignee"}:
        raise ValueError("by must be auto, patent or assignee")
    if associations.empty or "standardized_smiles" not in associations.columns:
        return pd.DataFrame()
    frame = associations.copy()
    frame["publication_year"] = _publication_year(frame)
    frame = frame.dropna(subset=["publication_year", "standardized_smiles"])
    if frame.empty:
        return pd.DataFrame()
    frame["publication_year"] = frame["publication_year"].astype(int)
    if by == "patent":
        category = "patent_id"
    elif by == "assignee":
        category = "assignee"
    else:
        category = "assignee" if "assignee" in frame.columns and frame["assignee"].notna().any() \
            else "patent_id" if "patent_id" in frame.columns else "dataset"
    if category == "dataset":
        frame["dataset"] = "All compounds"
    elif category not in frame.columns:
        return pd.DataFrame()
    frame[category] = frame[category].fillna("Unknown").astype(str)
    rankings = frame.groupby(category)["standardized_smiles"].nunique().sort_values(ascending=False)
    keep = set(rankings.head(max_stacks).index)
    frame["stack"] = frame[category].where(frame[category].isin(keep), "Other")
    counts = frame.groupby(["publication_year", "stack"])["standardized_smiles"].nunique()
    return counts.unstack(fill_value=0).sort_index().astype(int)


def scaffold_counts(compounds: pd.DataFrame, limit: int = 12) -> pd.Series:
    if compounds.empty or "murcko_scaffold" not in compounds.columns:
        return pd.Series(dtype="int64", name="unique_compounds")
    return compounds.groupby("murcko_scaffold")["standardized_smiles"].nunique().sort_values(
        ascending=False
    ).head(limit).rename("unique_compounds")


def cluster_counts(compounds: pd.DataFrame) -> pd.Series:
    if compounds.empty or "cluster_id" not in compounds.columns:
        return pd.Series(dtype="int64", name="unique_compounds")
    return compounds.groupby("cluster_id")["standardized_smiles"].nunique().sort_values(
        ascending=False
    ).rename("unique_compounds")


def ic50_pchembl(ic50: pd.DataFrame, target: str | None = None) -> pd.Series:
    """Keep assay rows separate; refuse to pool potency across annotated targets."""
    if ic50.empty or "pchembl_value" not in ic50.columns:
        return pd.Series(dtype="float64", name="pchembl_value")
    filtered = ic50.copy()
    if "target_pref_name" in filtered.columns:
        labels = filtered["target_pref_name"].fillna("Unannotated target").astype(str)
        unique = sorted(labels.unique())
        if target is None and len(unique) > 1:
            raise ValueError(f"Multiple targets present ({', '.join(unique[:6])}); select TARGET")
        if target is not None:
            filtered = filtered.loc[labels.eq(target)]
            if filtered.empty:
                raise ValueError(f"No IC50 records for target {target!r}")
    if "standard_relation" in filtered.columns:
        filtered = filtered.loc[filtered["standard_relation"].fillna("").eq("=")]
    return pd.to_numeric(filtered["pchembl_value"], errors="coerce").dropna().rename(
        "pchembl_value"
    )


def draw_bar(series: pd.Series, title: str, xlabel: str = "Number of records") -> None:
    if series.empty:
        print(f"{title}: no records to plot")
        return
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(series))))
    series.iloc[::-1].plot.barh(ax=ax, color="#28536b")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    plt.show()
