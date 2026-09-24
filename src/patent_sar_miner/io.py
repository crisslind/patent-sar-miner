from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

from patent_sar_miner.chemistry import calculate_descriptors, standardize_smiles

ID_ALIASES = ("compound_id", "molecule_chembl_id", "id", "name")


def _first_column(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    return next((lookup[name] for name in aliases if name in lookup), None)


def read_compounds(
    path: str | Path, structure_column: str, analysis_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an untouched source table and an analysis copy with internal columns."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Input must be a CSV file")
    source = pd.read_csv(path)
    missing = [name for name in (structure_column, analysis_column) if name not in source.columns]
    if missing:
        raise ValueError(
            f"Missing selected column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, source.columns))}"
        )
    if structure_column == analysis_column:
        raise ValueError("--structure and --analysis_col must name different columns")

    frame = source.copy()
    frame["source_row"] = range(1, len(frame) + 1)
    frame["raw_smiles"] = frame[structure_column]
    detected_id = _first_column(frame.columns, ID_ALIASES)
    frame["compound_id"] = (
        frame[detected_id].astype(str)
        if detected_id is not None
        else [f"compound_{index + 1}" for index in range(len(frame))]
    )
    return source, frame


def standardized_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = [standardize_smiles(value) for value in frame["raw_smiles"]]
    prepared = frame.copy()
    prepared["standardized_smiles"] = [result.standardized_smiles for result in results]
    prepared["status"] = [result.status for result in results]
    rejected = prepared[prepared["status"] != "accepted"].copy()
    accepted = prepared[prepared["status"] == "accepted"].copy().reset_index(drop=True)
    return accepted, rejected


def prepare_compounds(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted, rejected = standardized_records(frame)
    accepted["duplicate_group_size"] = accepted.groupby("standardized_smiles")[
        "standardized_smiles"
    ].transform("size")
    accepted["source_rows"] = accepted.groupby("standardized_smiles")["source_row"].transform(
        lambda values: " | ".join(str(value) for value in values)
    )
    for column, destination in (("patent_id", "patent_ids"), ("assignee", "assignees")):
        if column in accepted.columns:
            associations = accepted.groupby("standardized_smiles")[column].transform(
                lambda values: " | ".join(sorted({str(value) for value in values.dropna()
                                                    if str(value).strip()}))
            )
            accepted[destination] = associations
    accepted = accepted.drop_duplicates("standardized_smiles", keep="first").reset_index(drop=True)
    mols = [Chem.MolFromSmiles(value) for value in accepted["standardized_smiles"]]
    descriptor_frame = pd.DataFrame([calculate_descriptors(mol) for mol in mols])
    accepted = pd.concat([accepted, descriptor_frame], axis=1)
    return accepted, rejected


def write_sdf(frame: pd.DataFrame, path: str | Path) -> None:
    writer = Chem.SDWriter(str(path))
    for record in frame.to_dict(orient="records"):
        mol = Chem.MolFromSmiles(str(record["standardized_smiles"]))
        mol.SetProp("_Name", str(record["compound_id"]))
        for key, value in record.items():
            if key not in {"raw_smiles", "standardized_smiles"} and pd.notna(value):
                mol.SetProp(str(key), str(value))
        writer.write(mol)
    writer.close()
