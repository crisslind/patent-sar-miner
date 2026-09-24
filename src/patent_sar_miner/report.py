from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
from matplotlib.ticker import MaxNLocator
from rdkit import Chem
from rdkit.Chem import Draw


def _mol_image(smiles: str, size: tuple[int, int] = (260, 180)) -> str:
    mol = Chem.MolFromSmiles(smiles)
    image = Draw.MolToImage(mol, size=size)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _plot_image(frame: pd.DataFrame, column: str, title: str) -> str | None:
    if column not in frame.columns or frame[column].dropna().empty:
        return None
    counts = (frame.assign(_category=frame[column].fillna("Unknown").astype(str))
              .groupby("_category")["standardized_smiles"].nunique()
              .sort_values().tail(12))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    counts.plot.barh(ax=axis, color="#28536b")
    axis.set_title(title)
    axis.set_xlabel("Unique standardised compounds")
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def _stacked_year_image(frame: pd.DataFrame, max_stacks: int = 8,
                        stack_by: str = "auto") -> str | None:
    """Plot unique compound counts per publication year, stacked by source owner."""
    if "publication_date" in frame.columns:
        raw_dates = frame["publication_date"]
        numeric_years = pd.to_numeric(raw_dates, errors="coerce")
        numeric_years = numeric_years.where(numeric_years.between(1800, 2200))
        parsed_years = pd.to_datetime(
            raw_dates.where(numeric_years.isna()), errors="coerce"
        ).dt.year
        years = numeric_years.fillna(parsed_years)
        if "year" in frame.columns:
            years = years.fillna(pd.to_numeric(frame["year"], errors="coerce"))
    elif "year" in frame.columns:
        years = pd.to_numeric(frame["year"], errors="coerce")
    else:
        return None
    if years.dropna().empty:
        return None

    if stack_by == "assignee" and "assignee" not in frame.columns:
        return None
    if stack_by == "patent" and "patent_id" not in frame.columns:
        return None
    if (stack_by == "assignee" or stack_by == "auto" and
            "assignee" in frame.columns and frame["assignee"].notna().any()):
        stack_column = "assignee"
        stack_label = "assignee"
    elif "patent_id" in frame.columns and frame["patent_id"].notna().any():
        stack_column = "patent_id"
        stack_label = "patent document"
    else:
        stack_column = "_series"
        stack_label = "dataset"

    plot_frame = frame.copy()
    if "standardized_smiles" not in plot_frame.columns:
        # Legacy callers may supply only bibliographic rows; each row then
        # represents an unknown distinct compound, not a structure dedup key.
        plot_frame["standardized_smiles"] = [f"row_{i}" for i in range(len(plot_frame))]
    plot_frame["publication_year"] = years
    plot_frame = plot_frame.dropna(subset=["publication_year"])
    plot_frame["publication_year"] = plot_frame["publication_year"].astype(int)
    plot_frame[stack_column] = (
        "All compounds"
        if stack_column == "_series"
        else plot_frame[stack_column].fillna("Unknown").astype(str)
    )

    totals = plot_frame.groupby(stack_column)["standardized_smiles"].nunique().sort_values(ascending=False)
    retained = set(totals.head(max_stacks).index)
    plot_frame["stack_group"] = plot_frame[stack_column].where(
        plot_frame[stack_column].isin(retained), "Other"
    )
    counts = (
        plot_frame.groupby(["publication_year", "stack_group"], observed=True)[
            "standardized_smiles"
        ].nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    ordered_columns = counts.sum().sort_values(ascending=False).index
    counts = counts[ordered_columns]

    figure_width = max(8.0, min(14.0, 0.75 * len(counts.index) + 4.0))
    figure, axis = plt.subplots(figsize=(figure_width, 4.8))
    counts.plot.bar(stacked=True, ax=axis, width=0.78, colormap="tab20c")
    axis.set_title(f"Compounds published per year, stacked by {stack_label}")
    axis.set_xlabel("Publication year")
    axis.set_ylabel("Number of unique compounds")
    axis.tick_params(axis="x", rotation=0)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title=stack_label.title(), bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


TEMPLATE = Template(
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Patent SAR Miner report</title>
<style>
body{font-family:Inter,Arial,sans-serif;margin:0;color:#17252d;background:#f4f7f8}main{max-width:1180px;margin:auto;padding:34px}
h1{margin-bottom:4px}.sub{color:#56646b;margin-top:0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}
.card,section{background:white;border:1px solid #dbe3e6;border-radius:9px;padding:18px}.value{font-size:28px;font-weight:700;color:#28536b}
section{margin:18px 0;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #e2e8ea}th{background:#eaf0f2}
.mols{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.mol{text-align:center;border:1px solid #e2e8ea;padding:8px}.mol img{max-width:100%}
.plot{max-width:850px;width:100%}.warning{background:#fff7df;border-left:5px solid #b7791f;padding:14px}code{font-size:12px}.small{font-size:12px;color:#56646b}
@media(max-width:800px){.cards,.mols{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<h1>Patent SAR Miner</h1><p class="sub">Reproducible chemical-series and similarity analysis</p>
<div class="cards"><div class="card"><div class="value">{{ n_compounds }}</div>unique compounds</div>
<div class="card"><div class="value">{{ n_scaffolds }}</div>Murcko scaffolds</div><div class="card"><div class="value">{{ n_clusters }}</div>similarity clusters</div>
<div class="card"><div class="value">{{ n_pairs }}</div>matched pairs</div></div>
<section><h2>Measurement validation</h2><p><b>{{ usable_measurements }}</b> usable exact measurements; <b>{{ replicate_series }}</b> structures with replicates; <b>{{ disagreements }}</b> replicate groups exceeding the configured disagreement threshold.</p>{{ qc_table }}</section>
<section><h2>Predictive validation</h2><p>Status: <b>{{ model_status }}</b></p>{{ model_table }}</section>
<section><h2>Dataset composition</h2>{% if timeline_plot %}<img class="plot" src="data:image/png;base64,{{ timeline_plot }}">{% endif %}
{% if patent_plot %}<img class="plot" src="data:image/png;base64,{{ patent_plot }}">{% endif %}
{% if assignee_plot %}<img class="plot" src="data:image/png;base64,{{ assignee_plot }}">{% endif %}</section>
<section><h2>Largest chemical series</h2><div class="mols">{% for item in scaffold_cards %}<div class="mol"><img src="data:image/png;base64,{{ item.image }}"><b>{{ item.scaffold_id }}</b><br>{{ item.count }} compounds</div>{% endfor %}</div>{{ scaffold_table }}</section>
<section><h2>R-group decomposition of largest series</h2>{{ r_group_table }}</section>
{% if neighbours %}<section><h2>Nearest neighbours to query</h2><p><code>{{ query_smiles }}</code></p><div class="mols">
{% for item in neighbours %}<div class="mol"><img src="data:image/png;base64,{{ item.image }}"><b>{{ item.compound_id }}</b><br>Tanimoto {{ item.similarity }}<br><span class="small">{{ item.patent }}</span></div>{% endfor %}
</div></section>{% endif %}
<section><h2>Matched molecular pairs</h2>{{ pair_table }}</section>
<section><h2>Method</h2><p>Structures were parsed with RDKit, cleaned, reduced to the largest fragment, neutralised, canonicalised and deduplicated. Chemical series use Bemis–Murcko scaffolds. Clusters use radius-2, 2048-bit Morgan fingerprints and Butina clustering. Matched pairs use single acyclic cuts and a shared-core index.</p>
<p class="small">Generated from the exact tabular exports written beside this report.</p></section>
<p class="warning"><b>Scientific and legal limitation.</b> Structural similarity, scaffold overlap and matched-pair relationships do not establish patent scope, validity, infringement, freedom to operate or legal status. Claims must be reviewed by qualified patent counsel.</p>
</main></body></html>"""
)


def generate_report(
    compounds: pd.DataFrame,
    pairs: pd.DataFrame,
    r_groups: pd.DataFrame,
    path: str | Path,
    neighbours: pd.DataFrame | None = None,
    query_smiles: str | None = None,
    timeline: pd.DataFrame | None = None,
    stack_by: str = "auto",
    qc_summary: dict[str, object] | None = None,
    model_summary: dict[str, object] | None = None,
) -> None:
    qc_summary = qc_summary or {}
    model_summary = model_summary or {}
    source_frame = timeline if timeline is not None else compounds
    scaffold_table = (
        compounds.groupby(["scaffold_id", "murcko_scaffold"])
        .size()
        .reset_index(name="compound_count")
        .sort_values("compound_count", ascending=False)
        .head(20)
    )
    scaffold_cards = [
        {
            "scaffold_id": row["scaffold_id"],
            "count": int(row["compound_count"]),
            "image": _mol_image(row["murcko_scaffold"]),
        }
        for row in scaffold_table.head(8).to_dict(orient="records")
        if row["murcko_scaffold"] != "ACYCLIC"
    ]
    pair_columns = [
        column
        for column in (
            "compound_a",
            "compound_b",
            "fragment_a",
            "fragment_b",
            "activity_delta_b_minus_a",
        )
        if column in pairs.columns
    ]
    neighbour_cards: list[dict[str, object]] = []
    if neighbours is not None:
        for row in neighbours.head(12).to_dict(orient="records"):
            neighbour_cards.append(
                {
                    "compound_id": row["compound_id"],
                    "similarity": f"{row['query_tanimoto']:.3f}",
                    "patent": row.get("patent_id", ""),
                    "image": _mol_image(row["standardized_smiles"]),
                }
            )
    html = TEMPLATE.render(
        n_compounds=len(compounds),
        n_scaffolds=compounds["murcko_scaffold"].nunique(),
        n_clusters=compounds["cluster_id"].nunique(),
        n_pairs=len(pairs),
        usable_measurements=qc_summary.get("usable_exact_records", 0),
        replicate_series=qc_summary.get("structures_with_replicates", 0),
        disagreements=qc_summary.get("structures_with_replicate_disagreement", 0),
        qc_table=pd.DataFrame([qc_summary]).to_html(index=False, escape=True),
        model_status=model_summary.get("status", "not run"),
        model_table=pd.json_normalize(model_summary).to_html(index=False, escape=True),
        timeline_plot=_stacked_year_image(source_frame, stack_by=stack_by),
        patent_plot=_plot_image(source_frame, "patent_id", "Compounds by patent document"),
        assignee_plot=_plot_image(source_frame, "assignee", "Compounds by assignee"),
        scaffold_table=scaffold_table.to_html(index=False, escape=True),
        scaffold_cards=scaffold_cards,
        r_group_table=(
            r_groups.head(50).to_html(index=False, escape=True)
            if not r_groups.empty
            else "<p>No multi-compound scaffold was available for decomposition.</p>"
        ),
        pair_table=(
            pairs[pair_columns].head(50).to_html(index=False, escape=True)
            if not pairs.empty
            else "<p>No matched pairs found.</p>"
        ),
        neighbours=neighbour_cards,
        query_smiles=query_smiles,
    )
    Path(path).write_text(html, encoding="utf-8")
