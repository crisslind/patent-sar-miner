from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class StandardizedMolecule:
    raw_smiles: str
    standardized_smiles: str | None
    mol: Chem.Mol | None
    status: str


def standardize_smiles(smiles: str) -> StandardizedMolecule:
    """Return the neutral canonical parent while preserving a failure reason."""
    raw = "" if smiles is None else str(smiles).strip()
    if not raw:
        return StandardizedMolecule(raw, None, None, "missing_smiles")
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return StandardizedMolecule(raw, None, None, "invalid_smiles")
    try:
        clean = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean)
        neutral = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(neutral)
        canonical = Chem.MolToSmiles(neutral, canonical=True, isomericSmiles=True)
        return StandardizedMolecule(raw, canonical, neutral, "accepted")
    except (RuntimeError, ValueError):
        return StandardizedMolecule(raw, None, None, "standardization_failed")


def calculate_descriptors(mol: Chem.Mol) -> dict[str, float | int]:
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 3),
        "clogp": round(Descriptors.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
    }
