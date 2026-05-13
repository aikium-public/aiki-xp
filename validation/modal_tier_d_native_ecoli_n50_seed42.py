"""Sample N E. coli K12 test-set genes, fetch real CDS, score via Modal native,
compute Spearman rho vs truth and vs precomputed CV tier_d.

Results saved to /tmp/native_aggregate_results.json for local analysis.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("biopython", "pandas", "pyarrow", "scipy", "httpx")
)

genomes_vol = modal.Volume.from_name("aikixp-genomes", create_if_missing=False)
lookups_vol = modal.Volume.from_name("aikixp-lookups", create_if_missing=False)

app = modal.App("native-aggregate-test", image=image)


@app.function(volumes={"/genomes": genomes_vol, "/lookups": lookups_vol}, timeout=3600)
def run(n_genes: int = 50, seed: int = 42):
    import pickle, json, time
    import pandas as pd
    import numpy as np
    import httpx
    from scipy.stats import spearmanr, pearsonr

    lookup = pd.read_parquet("/lookups/tier_predictions_lookup.parquet")
    ecoli = lookup[lookup["gene_id"].str.startswith("Escherichia_coli_K12")].copy()
    print(f"E. coli K12 test genes in lookup: {len(ecoli)}")
    print(f"Overall lookup Spearman truth vs tier_d (E. coli K12 cohort): "
          f"{spearmanr(ecoli['true_expression'], ecoli['tier_d_prediction']).correlation:.4f}")

    ecoli["protein_id"] = ecoli["gene_id"].str.split("|").str[-1]

    with open("/genomes/NC_000913.3.pkl", "rb") as f:
        genome = pickle.load(f)
    genome_seq = str(genome.seq).upper()
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    def rc(s): return "".join(comp[c] for c in reversed(s))

    pid_to_cds = {}
    for feat in genome.features:
        if feat.type != "CDS" or "translation" not in feat.qualifiers or "pseudo" in feat.qualifiers:
            continue
        for pid in feat.qualifiers.get("protein_id", []):
            s, e = int(feat.location.start), int(feat.location.end)
            strand = feat.location.strand
            cds_raw = genome_seq[s:e]
            cds_dna = rc(cds_raw) if strand == -1 else cds_raw
            pid_to_cds[pid] = {
                "protein": feat.qualifiers["translation"][0],
                "cds": cds_dna,
            }

    have_cds = ecoli[ecoli["protein_id"].isin(pid_to_cds.keys())]
    print(f"Test genes with CDS in NC_000913.3: {len(have_cds)}")

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(have_cds), size=min(n_genes, len(have_cds)), replace=False)
    sample = have_cds.iloc[sample_idx].reset_index(drop=True)
    print(f"Scoring {len(sample)} genes via native mode...")

    endpoint = "https://aikium--aikixp-tier-d-predict-tier-d-endpoint.modal.run"
    results = []
    batch_size = 10
    with httpx.Client(timeout=600) as client:
        for i in range(0, len(sample), batch_size):
            batch = sample.iloc[i:i+batch_size]
            payload = {
                "proteins": [pid_to_cds[pid]["protein"] for pid in batch["protein_id"]],
                "cds":      [pid_to_cds[pid]["cds"]     for pid in batch["protein_id"]],
                "host": "NC_000913.3",
                "mode": "native",
                "anchor": "lacZ",
            }
            t0 = time.time()
            r = client.post(endpoint, json=payload, follow_redirects=True)
            dt = time.time() - t0
            if r.status_code != 200:
                print(f"  batch {i//batch_size}: HTTP {r.status_code} ({dt:.1f}s) — {r.text[:200]}")
                continue
            data = r.json()
            preds = data.get("predictions", [])
            for (_, row), pred_obj in zip(batch.iterrows(), preds):
                results.append({
                    "protein_id": row["protein_id"],
                    "truth": float(row["true_expression"]),
                    "modal_9mod": float(pred_obj["predicted_expression"]),
                    "precomputed_tier_d": float(row["tier_d_prediction"]),
                    "precomputed_tier_c": float(row["tier_c_prediction"]),
                    "operon_source": pred_obj.get("operon_source"),
                    "bacformer_available": pred_obj.get("bacformer_available"),
                })
            n_native = sum(1 for r in results[-len(preds):] if r["operon_source"] == "native")
            n_bf = sum(1 for r in results[-len(preds):] if r["bacformer_available"])
            print(f"  batch {i//batch_size}: {len(preds)} genes in {dt:.1f}s, "
                  f"native={n_native}, bacformer={n_bf}")

    if not results:
        print("No successful predictions — aborting.")
        return

    df = pd.DataFrame(results)
    print(f"\n=== Aggregate results over {len(df)} E. coli K12 test genes ===")
    print(f"  native-lookup success: {(df['operon_source'] == 'native').sum()}/{len(df)}")
    print(f"  bacformer available:   {df['bacformer_available'].sum()}/{len(df)}")

    rho_modal = spearmanr(df["truth"], df["modal_9mod"]).correlation
    rho_cv    = spearmanr(df["truth"], df["precomputed_tier_d"]).correlation
    rho_tc    = spearmanr(df["truth"], df["precomputed_tier_c"]).correlation
    rho_consistency = spearmanr(df["modal_9mod"], df["precomputed_tier_d"]).correlation

    print(f"\n  Spearman rho(truth, Modal 9-mod)         = {rho_modal:.4f}")
    print(f"  Spearman rho(truth, precomputed tier_d)  = {rho_cv:.4f}")
    print(f"  Spearman rho(truth, precomputed tier_c)  = {rho_tc:.4f}")
    print(f"  Spearman rho(Modal, precomputed tier_d)  = {rho_consistency:.4f}")

    r_modal = pearsonr(df["truth"], df["modal_9mod"])[0]
    r_cv    = pearsonr(df["truth"], df["precomputed_tier_d"])[0]
    print(f"\n  Pearson  r(truth, Modal 9-mod)           = {r_modal:.4f}")
    print(f"  Pearson  r(truth, precomputed tier_d)    = {r_cv:.4f}")

    mae_modal = (df["truth"] - df["modal_9mod"]).abs().mean()
    mae_cv    = (df["truth"] - df["precomputed_tier_d"]).abs().mean()
    print(f"\n  MAE(truth, Modal 9-mod)                  = {mae_modal:.3f}")
    print(f"  MAE(truth, precomputed tier_d)           = {mae_cv:.3f}")

    print("\n=== FULL TABLE ===")
    print(df.to_string(index=False))

    out = {
        "n_genes": len(df),
        "rho_truth_vs_modal": rho_modal,
        "rho_truth_vs_cv_tier_d": rho_cv,
        "rho_truth_vs_cv_tier_c": rho_tc,
        "rho_modal_vs_cv": rho_consistency,
        "pearson_truth_vs_modal": r_modal,
        "pearson_truth_vs_cv": r_cv,
        "mae_truth_vs_modal": float(mae_modal),
        "mae_truth_vs_cv": float(mae_cv),
        "native_rate": float((df["operon_source"] == "native").mean()),
        "bacformer_rate": float(df["bacformer_available"].mean()),
        "rows": df.to_dict("records"),
    }
    print("\n=== RESULT_JSON ===")
    print(json.dumps(out, indent=2))


@app.local_entrypoint()
def main(n: int = 50, seed: int = 42):
    run.remote(n, seed)
