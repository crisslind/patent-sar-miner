from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

SEED = 42


def _fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    matrix = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles):
        fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(str(value)))
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    return {
        "n": len(observed),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)) if len(observed) > 1 else None,
        "spearman": (
            float(pd.Series(observed).corr(pd.Series(predicted), method="spearman"))
            if len(observed) > 1 else None
        ),
    }


def validate_potency_model(
    compounds: pd.DataFrame,
    replicates: pd.DataFrame,
    analysis_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit an interpretable baseline with scaffold holdout and conformal intervals."""
    summary: dict[str, object] = {
        "status": "not_run",
        "analysis_column": analysis_column,
        "algorithm": "RandomForestRegressor on radius-2 Morgan fingerprints",
        "split": "deterministic scaffold-group holdout nearest 25%",
        "random_seed": SEED,
    }
    if replicates.empty or "median" not in replicates.columns:
        summary.update(status="insufficient_data", reason="no usable numerical measurements")
        return pd.DataFrame(), summary

    data = compounds.merge(
        replicates[["standardized_smiles", "median", "n_measurements"]],
        on="standardized_smiles",
        how="inner",
    ).dropna(subset=["median", "murcko_scaffold"])
    groups = data["murcko_scaffold"].astype(str)
    if len(data) < 12 or groups.nunique() < 3:
        summary.update(
            status="insufficient_data",
            reason="requires at least 12 compounds across 3 Murcko scaffolds",
            n_compounds=len(data),
            n_scaffolds=int(groups.nunique()),
        )
        return pd.DataFrame(), summary

    x = _fingerprints(data["standardized_smiles"])
    y = data["median"].to_numpy(dtype=float)
    group_sizes = groups.value_counts()
    candidates = group_sizes[(group_sizes >= 2) & ((len(data) - group_sizes) >= 8)]
    if candidates.empty:
        summary.update(
            status="insufficient_data",
            reason="no scaffold group supports a holdout with at least 2 test and 8 train compounds",
            n_compounds=len(data),
            n_scaffolds=int(groups.nunique()),
        )
        return pd.DataFrame(), summary
    target_size = len(data) * 0.25
    test_group = min(candidates.index, key=lambda value: abs(candidates[value] - target_size))
    test_mask = groups.eq(test_group).to_numpy()
    test_index = np.flatnonzero(test_mask)
    train_index = np.flatnonzero(~test_mask)
    train_groups = groups.iloc[train_index]

    folds = min(5, train_groups.nunique())
    calibration_residuals: list[float] = []
    if folds >= 2:
        for fit_index, calibration_index in GroupKFold(n_splits=folds).split(
            x[train_index], y[train_index], train_groups
        ):
            fold_model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
            fold_model.fit(x[train_index][fit_index], y[train_index][fit_index])
            fold_prediction = fold_model.predict(x[train_index][calibration_index])
            calibration_residuals.extend(
                np.abs(y[train_index][calibration_index] - fold_prediction).tolist()
            )
    conformal_radius = (
        float(np.quantile(calibration_residuals, 0.9, method="higher"))
        if calibration_residuals else float("nan")
    )

    model = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    model.fit(x[train_index], y[train_index])
    test_prediction = model.predict(x[test_index])
    all_prediction = model.predict(x)
    predictions = data[["compound_id", "standardized_smiles", "murcko_scaffold"]].copy()
    predictions["observed"] = y
    predictions["predicted"] = all_prediction
    predictions["prediction_interval_lower_90"] = all_prediction - conformal_radius
    predictions["prediction_interval_upper_90"] = all_prediction + conformal_radius
    predictions["split"] = "train"
    predictions.loc[predictions.index[test_index], "split"] = "scaffold_holdout"

    summary.update(
        status="completed",
        n_compounds=len(data),
        n_scaffolds=int(groups.nunique()),
        conformal_coverage_target=0.9,
        conformal_radius=conformal_radius,
        holdout_metrics=_metrics(y[test_index], test_prediction),
        mean_baseline_mae=float(mean_absolute_error(
            y[test_index], np.repeat(y[train_index].mean(), len(test_index))
        )),
    )

    if "patent_id" in data.columns and data["patent_id"].dropna().nunique() >= 2:
        patent_rows = data["patent_id"].notna().to_numpy()
        patent_groups = data.loc[patent_rows, "patent_id"].astype(str).to_numpy()
        patent_x, patent_y = x[patent_rows], y[patent_rows]
        patent_predictions = np.full(len(patent_y), np.nan)
        for fit_index, held_out_index in LeaveOneGroupOut().split(
            patent_x, patent_y, patent_groups
        ):
            if len(fit_index) < 2:
                continue
            fold_model = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
            fold_model.fit(patent_x[fit_index], patent_y[fit_index])
            patent_predictions[held_out_index] = fold_model.predict(patent_x[held_out_index])
        usable = ~np.isnan(patent_predictions)
        if usable.any():
            summary["leave_one_patent_out_metrics"] = _metrics(
                patent_y[usable], patent_predictions[usable]
            )
    return predictions, summary
