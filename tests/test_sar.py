import pandas as pd

from patent_sar_miner.sar import (
    add_scaffolds,
    butina_clusters,
    matched_pairs,
    nearest_neighbours,
    r_group_table,
)


def example_frame():
    return pd.DataFrame(
        {
            "compound_id": ["A", "B", "C"],
            "standardized_smiles": [
                "CC(=O)Nc1ccc(OC)cc1",
                "CC(=O)Nc1ccc(OCC)cc1",
                "c1ccncc1",
            ],
            "pIC50": [5.1, 5.8, 4.0],
        }
    )


def test_scaffolds_and_clusters_are_assigned():
    frame = butina_clusters(add_scaffolds(example_frame()))
    assert frame["scaffold_id"].notna().all()
    assert frame["cluster_id"].notna().all()


def test_query_ranking_places_identical_structure_first():
    ranked, canonical = nearest_neighbours(example_frame(), "CC(=O)Nc1ccc(OC)cc1")
    assert canonical == "COc1ccc(NC(C)=O)cc1"
    assert ranked.iloc[0]["compound_id"] == "A"
    assert ranked.iloc[0]["query_tanimoto"] == 1.0


def test_matched_pairs_include_activity_delta():
    pairs = matched_pairs(example_frame().iloc[:2], "pIC50")
    assert not pairs.empty
    assert "activity_delta_b_minus_a" in pairs.columns


def test_r_group_table_decomposes_largest_series():
    frame = add_scaffolds(example_frame())
    frame["assignees"] = ["Company A | Company B", "Company B", "Company C"]
    table = r_group_table(frame, "pIC50")
    assert len(table) == 2
    assert "Core" in table.columns
    assert "pIC50" in table.columns
    assert table["assignees"].tolist() == ["Company A | Company B", "Company B"]
