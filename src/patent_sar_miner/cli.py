from __future__ import annotations

import argparse
import json
import sys

from patent_sar_miner.analysis import AnalysisConfig, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patent-sar-miner",
        description="Validate and analyse SAR from a user-supplied CSV file.",
    )
    parser.add_argument("--input", required=True, dest="input_path", help="Input CSV file")
    parser.add_argument(
        "--structure", required=True, dest="structure_column",
        help="Name of the column containing SMILES structures",
    )
    parser.add_argument(
        "--analysis_col", "--analysis-col", required=True, dest="analysis_column",
        help="Name of the numerical response column used for SAR and modelling",
    )
    parser.add_argument("--output", default="patent_sar_output", dest="output_dir")
    parser.add_argument("--query-smiles")
    parser.add_argument("--similarity-threshold", type=float, default=0.55)
    parser.add_argument("--neighbour-limit", type=int, default=20)
    parser.add_argument("--replicate-disagreement-threshold", type=float, default=1.0)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--stack-by", choices=("auto", "patent", "assignee"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = run_analysis(AnalysisConfig(
            input_path=args.input_path,
            output_dir=args.output_dir,
            structure_column=args.structure_column,
            analysis_column=args.analysis_column,
            query_smiles=args.query_smiles,
            similarity_threshold=args.similarity_threshold,
            neighbour_limit=args.neighbour_limit,
            replicate_disagreement_threshold=args.replicate_disagreement_threshold,
            run_model=not args.skip_model,
            stack_by=args.stack_by,
        ))
    except (ValueError, OSError) as exc:
        print(f"patent-sar-miner: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
