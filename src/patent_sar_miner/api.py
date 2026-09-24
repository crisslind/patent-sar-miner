from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from patent_sar_miner.analysis import AnalysisConfig, run_analysis

app = FastAPI(title="Patent SAR Miner", version="0.4.0")


class AnalysisRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=100_000)
    structure: str
    analysis_col: str
    similarity_threshold: float = 0.55
    replicate_disagreement_threshold: float = 1.0
    run_model: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analysis")
def analyse(request: AnalysisRequest) -> dict[str, object]:
    frame = pd.DataFrame(request.records)
    with TemporaryDirectory(prefix="patent-sar-miner-") as directory:
        input_path = f"{directory}/input.csv"
        output_path = f"{directory}/results"
        frame.to_csv(input_path, index=False)
        try:
            manifest = run_analysis(AnalysisConfig(
                input_path=input_path,
                structure_column=request.structure,
                analysis_column=request.analysis_col,
                output_dir=output_path,
                similarity_threshold=request.similarity_threshold,
                replicate_disagreement_threshold=request.replicate_disagreement_threshold,
                run_model=request.run_model,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        uncertainty = json.loads(
            (Path(output_path) / "uncertainty_summary.json").read_text(encoding="utf-8")
        )
        model = json.loads(
            (Path(output_path) / "model_validation.json").read_text(encoding="utf-8")
        )
    return {"manifest": manifest, "uncertainty": uncertainty, "model": model}
