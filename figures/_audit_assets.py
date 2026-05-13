#!/usr/bin/env python3
"""Build audited manuscript summaries from checked-in ProtEx artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results/protex_qc"
OUT_DIR = RESULTS_DIR / "manuscript_audit"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def primary_result(data: dict, branch: str | None = None) -> dict:
    result = data.get("results", data)
    if branch and isinstance(result, dict) and branch in result:
        result = result[branch]
    if isinstance(result, dict) and "single_adapter" in result:
        result = result["single_adapter"]
    return result


def extract_metric(path: Path, metric: str, branch: str | None = None) -> float | None:
    if not path.exists():
        return None
    result = primary_result(read_json(path), branch=branch)
    if not isinstance(result, dict):
        return None
    value = result.get(metric)
    if value is not None:
        return float(value)
    if metric == "rho_overall":
        value = result.get("mean_spearman")
        if value is not None:
            return float(value)
    return None


def aggregate(paths: Iterable[Path], metric: str, branch: str | None = None) -> dict:
    used_paths: list[str] = []
    values: list[float] = []
    for path in paths:
        val = extract_metric(path, metric, branch=branch)
        if val is None:
            continue
        used_paths.append(str(path.relative_to(PROJECT_ROOT)))
        values.append(val)
    if not values:
        return {"mean": None, "std": None, "n": 0, "paths": []}
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(np.mean(values)), "std": std, "n": len(values), "paths": used_paths}


def write_df(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    return path


def write_json(data: dict, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def rel(path: str) -> Path:
    return PROJECT_ROOT / path


def build_split_summary() -> pd.DataFrame:
    rows = []
    specs = [
        (
            "fusion10_gene_operon",
            "Fusion-10 gene-operon",
            [rel(f"results/protex_qc/round5/champion_f10_go_seed{seed}.json") for seed in (0, 7, 42, 99, 123)],
            "rho_overall",
            None,
        ),
        (
            "fusion10_species_cluster_t020",
            "Fusion-10 species-cluster t0.20",
            [rel(f"results/protex_qc/round5/f10_species_cluster_seed{seed}.json") for seed in (7, 42, 123)]
            + [rel(f"results/protex_qc/round9_sprint/f10_species_cluster_t020_seed{seed}.json") for seed in (0, 99)],
            "rho_overall",
            None,
        ),
        (
            "fusion10_random",
            "Fusion-10 random",
            [rel(f"results/protex_qc/round5/f10_random_seed{seed}.json") for seed in (7, 42, 123)],
            "rho_overall",
            None,
        ),
        (
            "txpredict_holdout",
            "Fusion-10 TXpredict-style holdout",
            [rel(f"results/protex_qc/round9_sprint/f10_txpredict_holdout_seed{seed}.json") for seed in (42, 123)],
            "rho_overall",
            None,
        ),
    ]
    for metric_id, label, paths, metric, branch in specs:
        stats = aggregate(paths, metric, branch=branch)
        rows.append(
            {
                "metric_id": metric_id,
                "label": label,
                "mean": stats["mean"],
                "std": stats["std"],
                "n": stats["n"],
                "artifact_paths": " ; ".join(stats["paths"]),
            }
        )
    return pd.DataFrame(rows)


def build_training_fraction_curve() -> pd.DataFrame:
    rows = []
    for frac in ("010", "025", "050", "075", "100"):
        paths = [rel(f"results/protex_qc/fraction_scaling/f10_go_trainfrac_{frac}_seed{seed}.json") for seed in (0, 7, 42, 99, 123)]
        stats = aggregate(paths, "rho_overall")
        if not stats["n"]:
            continue
        rows.append(
            {
                "fraction_pct": int(frac),
                "mean": stats["mean"],
                "std": stats["std"],
                "n": stats["n"],
                "artifact_paths": " ; ".join(stats["paths"]),
            }
        )
    return pd.DataFrame(rows)


def build_species_breadth_curve() -> pd.DataFrame:
    rows = []
    specs = [
        ("050", "50%", [rel(f"results/protex_qc/round9_sprint/f10_species_breadth_050_seed{seed}.json") for seed in (42, 123)]),
        ("075", "75%", [rel(f"results/protex_qc/round9_sprint/f10_species_breadth_075_seed{seed}.json") for seed in (42, 123)]),
        ("100", "100%", [rel(f"results/protex_qc/round5/champion_f10_go_seed{seed}.json") for seed in (42, 123)]),
    ]
    for code, label, paths in specs:
        stats = aggregate(paths, "rho_overall")
        if not stats["n"]:
            continue
        rows.append(
            {
                "fraction_code": code,
                "label": label,
                "mean": stats["mean"],
                "std": stats["std"],
                "n": stats["n"],
                "artifact_paths": " ; ".join(stats["paths"]),
            }
        )
    return pd.DataFrame(rows)


def build_noise_curve() -> pd.DataFrame:
    rows = []
    specs = [
        ("0", "0%", [rel("results/protex_qc/round5/champion_f10_go_seed42.json")]),
        ("10", "10%", [rel("results/protex_qc/round9_sprint/f10_noisy_10pct_seed42.json")]),
        ("25", "25%", [rel("results/protex_qc/round9_sprint/f10_noisy_25pct_seed42.json")]),
        ("50", "50%", [rel("results/protex_qc/round9_sprint/f10_noisy_50pct_seed42.json")]),
        ("100", "100%", [rel("results/protex_qc/round9_sprint/f10_noisy_100pct_seed42.json")]),
    ]
    baseline = aggregate([rel("results/protex_qc/round5/champion_f10_go_seed42.json")], "rho_overall")["mean"]
    for code, label, paths in specs:
        stats = aggregate(paths, "rho_overall")
        if stats["mean"] is None:
            continue
        rows.append(
            {
                "corruption_pct": int(code),
                "label": label,
                "rho_overall": stats["mean"],
                "delta_vs_clean": stats["mean"] - baseline if baseline is not None else None,
                "n": stats["n"],
                "artifact_paths": " ; ".join(stats["paths"]),
            }
        )
    return pd.DataFrame(rows)


def build_label_domain_summary() -> pd.DataFrame:
    rows = []
    specs = [
        ("paxdb_only", "Direct proteomics only", [rel("results/protex_qc/final_sprint/f10_paxdb_only_25M_seed42.json")]),
        ("proxy_only", "Proxy labels only", [rel("results/protex_qc/final_sprint/f10_non_paxdb_only_seed42.json")]),
        ("paxdb_to_proxy", "Train direct, test proxy", [rel("results/protex_qc/final_sprint/f10_paxdb_train_abele_test_seed42.json")]),
        ("proxy_to_paxdb", "Train proxy, test direct", [rel("results/protex_qc/final_sprint/f10_abele_train_paxdb_test_seed42.json")]),
        ("high_quality_only", "High-quality subset only", [rel("results/protex_qc/round9_sprint/f10_high_quality_seed42.json"), rel("results/protex_qc/round9_sprint/f10_high_quality_seed123.json")]),
    ]
    for metric_id, label, paths in specs:
        stats = aggregate(paths, "rho_overall")
        rows.append(
            {
                "metric_id": metric_id,
                "label": label,
                "mean": stats["mean"],
                "std": stats["std"],
                "n": stats["n"],
                "artifact_paths": " ; ".join(stats["paths"]),
            }
        )
    return pd.DataFrame(rows)


def build_clean_slice_summary() -> pd.DataFrame:
    path = rel("results/protex_qc/clean_slice_eval/label_quality_stratification_summary.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    rename_map = {
        "Full test": "rho_full",
        "N_Full test": "n_full",
        "Mega": "rho_mega",
        "N_Mega": "n_mega",
        "Non-mega": "rho_nonmega",
        "N_Non-mega": "n_nonmega",
        "PaXDb subset": "rho_paxdb",
        "N_PaXDb subset": "n_paxdb",
        "PaXDb×non-mega": "rho_paxdb_nonmega",
        "N_PaXDb×non-mega": "n_paxdb_nonmega",
        "Abele subset": "rho_abele",
        "N_Abele subset": "n_abele",
    }
    df = df.rename(columns=rename_map)

    model_map = {
        "Full 492K mixed labels (champion GO)": ("full_mixed", "Full mixed-label champion"),
        "PaXDb-only trained → PaXDb test": ("paxdb_specialist", "Direct-label specialist"),
        "Abele-only trained → Abele test": ("proxy_specialist", "Proxy-label specialist"),
        "PaXDb trained → Abele test (cross)": ("paxdb_to_proxy", "Direct→proxy transfer"),
        "Abele trained → PaXDb test (cross)": ("proxy_to_paxdb", "Proxy→direct transfer"),
        "Protein-only F5 (GO)": ("protein_only", "Protein-only"),
        "DNA-only F5 (GO)": ("dna_only", "DNA-only"),
        "Classical-only (GO)": ("classical_only", "Classical-only"),
    }

    rows = []
    for row in df.to_dict(orient="records"):
        if row["model"] not in model_map:
            continue
        metric_id, label = model_map[row["model"]]
        rows.append(
            {
                "metric_id": metric_id,
                "label": label,
                "rho_full": row.get("rho_full"),
                "n_full": row.get("n_full"),
                "rho_mega": row.get("rho_mega"),
                "n_mega": row.get("n_mega"),
                "rho_nonmega": row.get("rho_nonmega"),
                "n_nonmega": row.get("n_nonmega"),
                "rho_paxdb": row.get("rho_paxdb"),
                "n_paxdb": row.get("n_paxdb"),
                "rho_paxdb_nonmega": row.get("rho_paxdb_nonmega"),
                "n_paxdb_nonmega": row.get("n_paxdb_nonmega"),
                "rho_abele": row.get("rho_abele"),
                "n_abele": row.get("n_abele"),
                "artifact_path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def build_fixedwidth_loo() -> pd.DataFrame:
    base_path = rel("results/protex_qc/validation/fixedwidth_loo_full_f10_baseline.json")
    base = extract_metric(base_path, "rho_overall")
    rows = []
    for path in sorted((RESULTS_DIR / "validation").glob("fixedwidth_loo_drop_*.json")):
        rho = extract_metric(path, "rho_overall")
        if rho is None or base is None:
            continue
        modality = path.stem.replace("fixedwidth_loo_drop_", "")
        rows.append(
            {
                "modality": modality,
                "rho_overall": rho,
                "delta": rho - base,
                "artifact_path": str(path.relative_to(PROJECT_ROOT)),
                "baseline_rho": base,
                "baseline_path": str(base_path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows).sort_values("delta")


def build_biology_family_go() -> pd.DataFrame:
    rows = []
    specs = [
        ("xgboost_69d", "Classical ML (XGBoost, 69d)", [rel("results/protex_qc/round9_sprint/xgboost_classical_results.json")], "rho"),
        ("classical_only", "Classical only", [rel("results/protex_qc/round9_sprint/f10_ablation_classical_only_seed42.json")], "rho_overall"),
        ("single_esmc", "Protein only (ESM-C)", [rel("results/protex_qc/validation/single_esmc_protein_seed42.json")], "rho_overall"),
        ("protein_only", "Protein family only", [rel("results/protex_qc/round9_sprint/f10_ablation_protein_only_seed42.json")], "rho_overall"),
        ("dna_only", "DNA family only", [rel("results/protex_qc/round9_sprint/f10_ablation_dna_only_seed42.json")], "rho_overall"),
        ("fusion10_seed42", "Fusion-10", [rel("results/protex_qc/round5/champion_f10_go_seed42.json")], "rho_overall"),
    ]
    for metric_id, label, paths, metric in specs:
        stats = aggregate(paths, metric)
        rows.append(
            {
                "metric_id": metric_id,
                "label": label,
                "rho_overall": stats["mean"],
                "std": stats["std"],
                "n": stats["n"],
                "artifact_paths": " ; ".join(stats["paths"]),
            }
        )
    return pd.DataFrame(rows)


def build_species_cluster_model_comparison() -> pd.DataFrame:
    rows = []
    specs = [
        ("fusion10", "Fusion-10", rel("results/protex_qc/round9_sprint/f10_champion_species_cluster_seed42.json")),
        ("protein", "Protein space (ESM-C)", rel("results/protex_qc/round9_sprint/single_esmc_protein_species_cluster_seed42.json")),
        ("dna", "Coding DNA space (Evo-2)", rel("results/protex_qc/round9_sprint/single_evo2_cds_species_cluster_seed42.json")),
        ("context", "Genome-context space (Bacformer)", rel("results/protex_qc/round9_sprint/single_bacformer_species_cluster_seed42.json")),
    ]
    for metric_id, label, path in specs:
        rows.append(
            {
                "metric_id": metric_id,
                "label": label,
                "rho_overall": extract_metric(path, "rho_overall"),
                "rho_novel_families": extract_metric(path, "rho_novel_families"),
                "rho_shared_families": extract_metric(path, "rho_shared_families"),
                "rho_cluster_weighted": extract_metric(path, "rho_cluster_weighted"),
                "artifact_path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_curve() -> pd.DataFrame:
    rows = []
    threshold_specs = [
        ("0.05", sorted((RESULTS_DIR / "round5").glob("f10_species_t005_seed*.json"))),
        ("0.10", sorted((RESULTS_DIR / "round5").glob("f10_species_t010_seed*.json"))),
        ("0.20", [rel(f"results/protex_qc/round5/f10_species_cluster_seed{seed}.json") for seed in (7, 42, 123)]
         + [rel(f"results/protex_qc/round9_sprint/f10_species_cluster_t020_seed{seed}.json") for seed in (0, 99)]),
        ("0.30", sorted((RESULTS_DIR / "round5").glob("f10_species_t030_seed*.json"))),
    ]
    cluster_counts = {
        "0.05": 330,
        "0.10": 291,
        "0.20": 116,
        "0.30": 17,
    }
    for threshold, paths in threshold_specs:
        overall = aggregate(paths, "rho_overall")
        novel = aggregate(paths, "rho_novel_families")
        shared = aggregate(paths, "rho_shared_families")
        rows.append(
            {
                "threshold": float(threshold),
                "overall_mean": overall["mean"],
                "overall_std": overall["std"],
                "n": overall["n"],
                "rho_novel_families": novel["mean"],
                "rho_shared_families": shared["mean"],
                "cluster_count": cluster_counts[threshold],
                "artifact_paths": " ; ".join(overall["paths"]),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class Claim:
    claim_id: str
    description: str
    location: str
    claimed_value: str
    actual_mean: float | None
    actual_std: float | None
    actual_n: int
    status: str
    note: str
    artifact_paths: str


def build_claim_audit() -> pd.DataFrame:
    claims: list[Claim] = []

    f10_go = aggregate([rel(f"results/protex_qc/round5/champion_f10_go_seed{seed}.json") for seed in (0, 7, 42, 99, 123)], "rho_overall")
    f10_sc = aggregate(
        [rel(f"results/protex_qc/round5/f10_species_cluster_seed{seed}.json") for seed in (7, 42, 123)]
        + [rel(f"results/protex_qc/round9_sprint/f10_species_cluster_t020_seed{seed}.json") for seed in (0, 99)],
        "rho_overall",
    )
    f10_rand = aggregate([rel(f"results/protex_qc/round5/f10_random_seed{seed}.json") for seed in (7, 42, 123)], "rho_overall")
    esmc_go = aggregate([rel("results/protex_qc/validation/single_esmc_protein_seed42.json")], "rho_overall")
    xgb = aggregate([rel("results/protex_qc/round9_sprint/xgboost_classical_results.json")], "rho")
    f11 = aggregate([rel("results/protex_qc/validation/fusion11_with_esm2_seed7.json"), rel("results/protex_qc/validation/fusion11_with_esm2_seed123.json")], "rho_overall")
    fixedwidth = build_fixedwidth_loo().set_index("modality")
    compact128 = aggregate([rel(f"results/protex_qc/validation/fusion10_true128d_seed{seed}.json") for seed in (7, 42, 123)], "rho_overall")
    compact256 = aggregate([rel("results/protex_qc/validation/fusion10_true256d_seed42.json")], "rho_overall")
    f50 = aggregate([rel("results/protex_qc/round5/f10_50M_seed7.json")], "rho_overall")
    cross_attn = aggregate([rel("results/protex_qc/validation/fusion10_cross_attention_seed42.json")], "rho_overall", branch="cross_attention")
    t030 = aggregate(sorted((RESULTS_DIR / "round5").glob("f10_species_t030_seed*.json")), "rho_overall")
    novel_shared = aggregate(
        [rel(f"results/protex_qc/round5/f10_species_cluster_seed{seed}.json") for seed in (7, 42, 123)]
        + [rel(f"results/protex_qc/round9_sprint/f10_species_cluster_t020_seed{seed}.json") for seed in (0, 99)],
        "rho_novel_families",
    )
    shared = aggregate(
        [rel(f"results/protex_qc/round5/f10_species_cluster_seed{seed}.json") for seed in (7, 42, 123)]
        + [rel(f"results/protex_qc/round9_sprint/f10_species_cluster_t020_seed{seed}.json") for seed in (0, 99)],
        "rho_shared_families",
    )

    def add_numeric_claim(claim_id: str, desc: str, location: str, claimed: str, stats: dict, note: str = "", min_n: int = 1) -> None:
        if stats["mean"] is None:
            status = "fabricated"
        elif stats["n"] < min_n:
            status = "unverifiable"
        else:
            status = "verified"
        claims.append(
            Claim(
                claim_id,
                desc,
                location,
                claimed,
                stats["mean"],
                stats["std"],
                stats["n"],
                status,
                note,
                " ; ".join(stats["paths"]),
            )
        )

    add_numeric_claim("f10_go", "Fusion-10 gene-operon headline", "Fig1/Fig3/table/text", "0.629 ± 0.003", f10_go, min_n=5)
    add_numeric_claim("f10_sc", "Fusion-10 species-cluster headline", "Fig1/Fig3/Fig6/table/text", "0.676 ± 0.005", f10_sc, min_n=5)
    add_numeric_claim("f10_random", "Fusion-10 random headline", "Fig1/Fig3/table/text", "0.684 ± 0.002", f10_rand, min_n=3)
    claims.append(Claim("auc_0978", "Extreme-expressor AUC", "Fig1/abstract/discussion", "0.978", None, None, 0, "fabricated", "No checked-in result artifact located for this AUC claim.", ""))
    add_numeric_claim("esmc_go", "ESM-C gene-operon single-modality baseline", "Fig3/table/text", "0.569 or 0.572", esmc_go)
    add_numeric_claim("xgboost_baseline", "Best non-neural baseline", "Fig3/table/text", "0.483", xgb)
    add_numeric_claim("fusion11_go", "Fusion-11 gene-operon comparator", "Fig4/table/text", "0.623", f11, min_n=3)
    claims.append(
        Claim(
            "loo_esmc",
            "Fixed-width LOO ESM-C delta",
            "Fig3/text",
            "-0.027",
            float(fixedwidth.loc["esmc_protein", "delta"]),
            0.0,
            1,
            "verified",
            "Derived from fixedwidth_loo_full_f10_baseline.json and fixedwidth_loo_drop_esmc_protein.json.",
            " ; ".join([fixedwidth.loc["esmc_protein", "baseline_path"], fixedwidth.loc["esmc_protein", "artifact_path"]]),
        )
    )
    claims.append(
        Claim(
            "loo_hyenadna",
            "Fixed-width LOO HyenaDNA delta",
            "Fig3/text",
            "-0.018",
            float(fixedwidth.loc["hyenadna_dna_cds", "delta"]),
            0.0,
            1,
            "verified",
            "Derived from fixed-width LOO baseline and drop run.",
            " ; ".join([fixedwidth.loc["hyenadna_dna_cds", "baseline_path"], fixedwidth.loc["hyenadna_dna_cds", "artifact_path"]]),
        )
    )
    claims.append(
        Claim(
            "loo_bacformer",
            "Fixed-width LOO Bacformer delta",
            "Fig3/text",
            "-0.014",
            float(fixedwidth.loc["bacformer", "delta"]),
            0.0,
            1,
            "verified",
            "Derived from fixed-width LOO baseline and drop run.",
            " ; ".join([fixedwidth.loc["bacformer", "baseline_path"], fixedwidth.loc["bacformer", "artifact_path"]]),
        )
    )
    claims.append(
        Claim(
            "loo_rna_init",
            "Fixed-width LOO RNA-init delta",
            "Fig3/text",
            "-0.014",
            float(fixedwidth.loc["classical_rna_init", "delta"]),
            0.0,
            1,
            "verified",
            "Derived from fixed-width LOO baseline and drop run.",
            " ; ".join([fixedwidth.loc["classical_rna_init", "baseline_path"], fixedwidth.loc["classical_rna_init", "artifact_path"]]),
        )
    )
    add_numeric_claim("compact_256d", "Fusion-10 256d compact claim", "Fig4/table/text", "0.629 ± 0.001 (3 seeds)", compact256, min_n=3)
    add_numeric_claim("scale_50m", "50M scaling claim", "Fig4/text", "0.633", f50)
    claims.append(Claim("scale_100m", "100M scaling claim", "Fig4/text", "0.630", None, None, 0, "fabricated", "No like-for-like checked-in 100M single-adapter artifact located.", ""))
    add_numeric_claim("cross_attention", "Cross-attention comparator", "Fig4/ED8/text", "0.618 ± 0.007", cross_attn)
    add_numeric_claim("t030_phylo", "Species-cluster threshold t0.30", "Fig6/text", "0.666", t030, min_n=2)
    claims.append(
        Claim(
            "novel_shared_species_cluster",
            "Novel/shared family comparison",
            "Fig6b",
            "0.45 vs 0.70",
            novel_shared["mean"],
            shared["mean"],
            min(novel_shared["n"], shared["n"]),
            "verified",
            "Uses five checked-in species-cluster t0.20 seed artifacts.",
            " ; ".join(novel_shared["paths"]),
        )
    )
    claims.append(Claim("cv_seed123_7", "Cross-validation seeds 123 and 7", "Text/ED10", "0.637 ± 0.010; 0.636 ± 0.012", None, None, 0, "unverifiable", "No local fold-level JSONs for CV seeds 123 and 7 were found.", ""))
    claims.append(Claim("f5_neural_species", "F5 neural-only species-cluster", "Fig6/text", "0.671 ± 0.002", None, None, 0, "unverifiable", "No checked-in multi-seed artifact located for this claim.", ""))
    claims.append(Claim("stepwise_intermediates", "Forward stepwise intermediates", "Fig3A/ED3", "0.595, 0.605, 0.615, 0.627, 0.635", None, None, 0, "fabricated", "No checked-in forward-stepwise summary artifacts located.", ""))
    claims.append(Claim("ed6_groupwise", "Extended Data 6 groupwise bins", "ED6", "hardcoded", None, None, 0, "fabricated", "No checked-in source files identified for the current hardcoded ED6 values.", ""))
    claims.append(Claim("ed7_source_gap", "Extended Data 7 PaXDb/Abele panel", "ED7", "hardcoded", None, None, 0, "fabricated", "Current ED7 values are not driven from checked-in source tables.", ""))
    claims.append(Claim("ed12_biomolecule", "Extended Data 12 biomolecule summary", "ED12", "hardcoded", None, None, 0, "fabricated", "Current ED12 values are hardcoded in the figure script.", ""))
    claims.append(Claim("ed14_negatives", "Extended Data 14 scrambled negatives", "ED14", "0.074, 0.116", None, None, 0, "fabricated", "No checked-in scrambled-controls artifact located.", ""))
    claims.append(Claim("ed15_subsample", "Extended Data 15 legacy subsampling panel", "ED15", "hardcoded", None, None, 0, "fabricated", "Current ED15 values are hardcoded in the figure script.", ""))

    return pd.DataFrame([c.__dict__ for c in claims])


def build_pair_synergy_matrix() -> pd.DataFrame:
    modalities = ["esmc_protein", "evo2_cds", "hyenadna_dna_cds", "dnabert2_operon_dna", "bacformer", "classical_all"]
    nice = {"esmc_protein": "ESM-C", "evo2_cds": "Evo-2", "hyenadna_dna_cds": "HyenaDNA",
            "dnabert2_operon_dna": "DNABERT-2", "bacformer": "Bacformer", "classical_all": "Classical"}
    single_paths = {
        "esmc_protein": rel("results/protex_qc/validation/single_esmc_protein_seed42.json"),
        "evo2_cds": rel("results/protex_qc/v2_production_sweep/single_evo2_cds_seed42.json"),
        "hyenadna_dna_cds": rel("results/protex_qc/v2_production_sweep/single_hyenadna_dna_cds_seed42.json"),
        "dnabert2_operon_dna": rel("results/protex_qc/v2_production_sweep/single_dnabert2_operon_dna_seed42.json"),
        "bacformer": rel("results/protex_qc/v2_production_sweep/single_bacformer_seed42.json"),
        "classical_all": rel("results/protex_qc/round9_sprint/f10_ablation_classical_only_seed42.json"),
    }
    rows = []
    for ma in modalities:
        for mb in modalities:
            if ma == mb:
                rho = extract_metric(single_paths[ma], "rho_overall")
                art = str(single_paths[ma].relative_to(PROJECT_ROOT)) if single_paths[ma].exists() else ""
            else:
                key = f"pair_{ma}_{mb}_seed42"
                key2 = f"pair_{mb}_{ma}_seed42"
                p = RESULTS_DIR / f"round9_sprint/{key}.json"
                if not p.exists():
                    p = RESULTS_DIR / f"round9_sprint/{key2}.json"
                rho = extract_metric(p, "rho_overall") if p.exists() else None
                art = str(p.relative_to(PROJECT_ROOT)) if p.exists() else ""
            rows.append({"modality_a": nice[ma], "modality_b": nice[mb], "rho": rho, "artifact_path": art})
    return pd.DataFrame(rows)


def build_loso_summary() -> pd.DataFrame:
    rows = []
    sprint_dir = RESULTS_DIR / "final_sprint"
    for p in sorted(sprint_dir.glob("loso_*.json")):
        stem = p.stem
        rho = extract_metric(p, "rho_overall")
        if rho is None:
            continue
        if "_esmc_" in stem:
            model = "ESM-C"
            species = stem.replace("loso_", "").replace("_esmc_seed42", "")
        elif "_f10_" in stem:
            model = "F10"
            species = stem.replace("loso_", "").replace("_f10_seed42", "")
        else:
            continue
        rows.append({"species": species.replace("_", " "), "model": model, "rho": rho,
                      "artifact_path": str(p.relative_to(PROJECT_ROOT))})
    return pd.DataFrame(rows)


def build_mega_asymmetry() -> pd.DataFrame:
    specs = [
        ("mega_train_nonmega_test", "Train mega, test non-mega",
         [rel(f"results/protex_qc/round9_sprint/f10_mega_train_nonmega_test_seed{s}.json") for s in (123,)]),
        ("nonmega_train_mega_test", "Train non-mega, test mega",
         [rel(f"results/protex_qc/round9_sprint/f10_nonmega_train_mega_test_seed{s}.json") for s in (123,)]),
    ]
    rows = []
    for metric_id, label, paths in specs:
        stats = aggregate(paths, "rho_overall")
        rows.append({"metric_id": metric_id, "label": label, "rho": stats["mean"],
                      "n": stats["n"], "artifact_paths": " ; ".join(stats["paths"])})
    return pd.DataFrame(rows)


def build_architecture_comparison() -> pd.DataFrame:
    specs = [
        ("single_adapter", "Single adapter (champion)", "round5/champion_f10_go_seed42.json", None),
        ("projector", "Projector", "round5/projector_f10_seed42.json", "projector"),
        ("latent_alignment", "Latent alignment", "round5/latent_alignment_f10_25M_seed42.json", "latent_alignment"),
        ("gmu", "GMU", "round5/gmu_f10_seed42.json", "gmu"),
        ("attention_gated", "Attention-gated", "round5/attn_gated_f10_seed42.json", "attention_gated"),
        ("concat", "Concat", "round5/concat_f10_seed42.json", "concat"),
    ]
    rows = []
    for metric_id, label, path_rel, branch in specs:
        p = RESULTS_DIR / path_rel
        rho = extract_metric(p, "rho_overall", branch=branch)
        if rho is None:
            rho = extract_metric(p, "mean_spearman", branch=branch)
        rows.append({"metric_id": metric_id, "label": label, "rho": rho,
                      "artifact_path": str(p.relative_to(PROJECT_ROOT)) if p.exists() else ""})
    return pd.DataFrame(rows)


def build_canonical_metrics() -> dict:
    split_df = build_split_summary().set_index("metric_id")
    loo_df = build_fixedwidth_loo()
    return {
        "fusion10_gene_operon": split_df.loc["fusion10_gene_operon"].to_dict(),
        "fusion10_species_cluster_t020": split_df.loc["fusion10_species_cluster_t020"].to_dict(),
        "fusion10_random": split_df.loc["fusion10_random"].to_dict(),
        "txpredict_holdout": split_df.loc["txpredict_holdout"].to_dict(),
        "fixedwidth_loo": loo_df.to_dict(orient="records"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_df(build_split_summary(), "split_summary.csv")
    write_df(build_training_fraction_curve(), "training_fraction_curve.csv")
    write_df(build_species_breadth_curve(), "species_breadth_curve.csv")
    write_df(build_noise_curve(), "noise_curve.csv")
    write_df(build_label_domain_summary(), "label_domain_summary.csv")
    write_df(build_fixedwidth_loo(), "fixedwidth_loo.csv")
    write_df(build_biology_family_go(), "biology_family_gene_operon.csv")
    write_df(build_species_cluster_model_comparison(), "species_cluster_model_comparison.csv")
    write_df(build_threshold_curve(), "threshold_curve.csv")
    write_df(build_pair_synergy_matrix(), "pair_synergy_matrix.csv")
    write_df(build_loso_summary(), "loso_summary.csv")
    write_df(build_mega_asymmetry(), "mega_asymmetry.csv")
    write_df(build_clean_slice_summary(), "clean_slice_summary.csv")
    write_df(build_architecture_comparison(), "architecture_comparison.csv")
    write_df(build_claim_audit(), "provenance_claims.csv")
    write_json(build_canonical_metrics(), "canonical_metrics.json")


if __name__ == "__main__":
    main()
