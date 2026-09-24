from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMMPA, rdRGroupDecomposition
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles: list[str]):
    return [FP_GENERATOR.GetFingerprint(Chem.MolFromSmiles(value)) for value in smiles]


def add_scaffolds(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    scaffolds: list[str] = []
    for smiles in output["standardized_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds.append(Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() else "ACYCLIC")
    output["murcko_scaffold"] = scaffolds
    counts = Counter(scaffolds)
    order = {scaffold: index + 1 for index, (scaffold, _) in enumerate(counts.most_common())}
    output["scaffold_id"] = [f"S{order[value]:03d}" for value in scaffolds]
    return output


def butina_clusters(frame: pd.DataFrame, similarity_threshold: float = 0.55) -> pd.DataFrame:
    output = frame.copy()
    fps = fingerprints(output["standardized_smiles"].tolist())
    distances: list[float] = []
    for index in range(1, len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[index], fps[:index])
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fps),
        1.0 - similarity_threshold,
        isDistData=True,
    )
    labels = np.zeros(len(output), dtype=int)
    for cluster_id, members in enumerate(clusters, start=1):
        for member in members:
            labels[member] = cluster_id
    output["cluster_id"] = [f"C{value:03d}" for value in labels]
    return output


def nearest_neighbours(
    frame: pd.DataFrame, query_smiles: str, limit: int = 20
) -> tuple[pd.DataFrame, str]:
    query = Chem.MolFromSmiles(query_smiles)
    if query is None:
        raise ValueError("Query SMILES is invalid")
    canonical = Chem.MolToSmiles(query, canonical=True, isomericSmiles=True)
    query_fp = FP_GENERATOR.GetFingerprint(query)
    fps = fingerprints(frame["standardized_smiles"].tolist())
    output = frame.copy()
    output["query_tanimoto"] = DataStructs.BulkTanimotoSimilarity(query_fp, fps)
    output = (
        output.sort_values("query_tanimoto", ascending=False).head(limit).reset_index(drop=True)
    )
    return output, canonical


def _single_cut_fragments(smiles: str) -> set[tuple[str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    fragments: set[tuple[str, str]] = set()
    for _, fragmented in rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = fragmented.split(".")
        if len(parts) != 2:
            continue
        ranked = sorted(parts, key=lambda value: Chem.MolFromSmiles(value).GetNumHeavyAtoms())
        variable, core = ranked[0], ranked[1]
        if Chem.MolFromSmiles(core).GetNumHeavyAtoms() < 5:
            continue
        fragments.add((core, variable))
    return fragments


def matched_pairs(
    frame: pd.DataFrame,
    activity_column: str | None = None,
    max_pairs: int = 500,
) -> pd.DataFrame:
    index: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_index, smiles in enumerate(frame["standardized_smiles"]):
        for core, variable in _single_cut_fragments(smiles):
            index[core].append((row_index, variable))

    records: list[dict[str, object]] = []
    seen: set[tuple[int, int, str]] = set()
    for core, members in index.items():
        for left_pos in range(len(members)):
            for right_pos in range(left_pos + 1, len(members)):
                left_index, left_fragment = members[left_pos]
                right_index, right_fragment = members[right_pos]
                if left_index == right_index or left_fragment == right_fragment:
                    continue
                key = (min(left_index, right_index), max(left_index, right_index), core)
                if key in seen:
                    continue
                seen.add(key)
                left = frame.iloc[left_index]
                right = frame.iloc[right_index]
                record: dict[str, object] = {
                    "compound_a": left["compound_id"],
                    "compound_b": right["compound_id"],
                    "shared_core": core,
                    "fragment_a": left_fragment,
                    "fragment_b": right_fragment,
                }
                if "assignees" in frame.columns:
                    record["assignees_a"] = left.get("assignees", "")
                    record["assignees_b"] = right.get("assignees", "")
                if activity_column and activity_column in frame.columns:
                    value_a = pd.to_numeric(left[activity_column], errors="coerce")
                    value_b = pd.to_numeric(right[activity_column], errors="coerce")
                    record[f"{activity_column}_a"] = value_a
                    record[f"{activity_column}_b"] = value_b
                    record["activity_delta_b_minus_a"] = value_b - value_a
                records.append(record)
                if len(records) >= max_pairs:
                    return pd.DataFrame(records)
    return pd.DataFrame(records)


def r_group_table(frame: pd.DataFrame, activity_column: str | None = None) -> pd.DataFrame:
    """Decompose the largest non-acyclic Murcko series into labelled R groups."""
    eligible = frame[frame["murcko_scaffold"] != "ACYCLIC"]
    if eligible.empty:
        return pd.DataFrame()
    top_scaffold = eligible["murcko_scaffold"].value_counts().index[0]
    series = eligible[eligible["murcko_scaffold"] == top_scaffold].reset_index(drop=True)
    if len(series) < 2:
        return pd.DataFrame()
    core = Chem.MolFromSmiles(top_scaffold)
    mols = [Chem.MolFromSmiles(value) for value in series["standardized_smiles"]]
    decomposed, unmatched = rdRGroupDecomposition.RGroupDecompose([core], mols, asSmiles=True)
    unmatched_set = set(unmatched)
    records: list[dict[str, object]] = []
    result_index = 0
    for row_index, row in series.iterrows():
        if row_index in unmatched_set:
            continue
        record = {
            "compound_id": row["compound_id"],
            "patent_id": row.get("patent_id", ""),
        }
        if "assignees" in series.columns:
            record["assignees"] = row.get("assignees", "")
        if activity_column and activity_column in series.columns:
            record[activity_column] = row.get(activity_column, np.nan)
        record.update(decomposed[result_index])
        records.append(record)
        result_index += 1
    return pd.DataFrame(records)
