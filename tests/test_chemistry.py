from patent_sar_miner.chemistry import standardize_smiles


def test_standardization_removes_counterion():
    result = standardize_smiles("CC(=O)[O-].[Na+]")
    assert result.status == "accepted"
    assert result.standardized_smiles == "CC(=O)O"


def test_invalid_smiles_is_audited():
    result = standardize_smiles("not_a_smiles")
    assert result.status == "invalid_smiles"
    assert result.mol is None
