import base64

import pandas as pd

from patent_sar_miner.report import _stacked_year_image


def test_stacked_year_plot_is_a_png():
    frame = pd.DataFrame(
        {
            "publication_date": ["2022-01-01", "2022-06-01", "2023-02-01"],
            "assignee": ["Company A", "Company B", "Company A"],
            "standardized_smiles": ["CC", "CCC", "CCCC"],
        }
    )
    encoded = _stacked_year_image(frame)
    assert encoded is not None
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_stacked_year_plot_accepts_numeric_year_column():
    frame = pd.DataFrame({"year": [2020, 2021], "patent_id": ["P1", "P2"]})
    assert _stacked_year_image(frame) is not None


def test_stacked_year_plot_accepts_year_in_publication_date_column():
    frame = pd.DataFrame({"publication_date": [2020, 2021], "patent_id": ["P1", "P2"]})
    assert _stacked_year_image(frame) is not None


def test_stacked_year_plot_requires_year_information():
    assert _stacked_year_image(pd.DataFrame({"smiles": ["CC"]})) is None


def test_stacked_year_plot_falls_back_when_publication_date_is_missing():
    frame = pd.DataFrame({"year": [2018, 2018], "publication_date": [None, None],
                          "patent_id": ["P1", "P1"], "standardized_smiles": ["CC", "CC"]})
    assert _stacked_year_image(frame, stack_by="patent") is not None
