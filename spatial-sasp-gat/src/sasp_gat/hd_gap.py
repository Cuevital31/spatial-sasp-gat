"""External-HD vs locked-TEST performance gap, descriptive (R2.6 / R1.5 / R3.4; Supplement S5).
Logic from Revision-1 cell 8-R-SAVE v1.1. Inputs: the per-slide scenario records (rand25), the slide -> cancer/group map and
the spatial nearest-neighbour distance audit. Nothing is selected or tuned; no p-values."""
import re

import numpy as np
import pandas as pd

SCN = "rand25"
MODELS = ["full", "expression+spatial"]
TRAINFREE = ["Baseline A", "Ring2Mean", "NeighborMean"]
METHODS = MODELS + TRAINFREE
LAB = {"test": "TEST", "hd": "HD"}


def slide_cancer_map(samples_test, samples_hd):
    rows = []
    for coh, smp in (("test", samples_test), ("hd", samples_hd)):
        for sid, d in smp.items():
            rows.append(dict(slide=sid, cohort=coh, cancer=str(d.cancer), group=d.leakage_group_id))
    return pd.DataFrame(rows)


def slide_and_pooled_tables(res, cmap):
    r = res[(res.scn == SCN) & res.cohort.isin(["test", "hd"]) & res.method.isin(METHODS)].copy()
    r["seed"] = r["seed"].astype(str)
    assert r[r.cohort == "test"].slide.nunique() == 13 and r[r.cohort == "hd"].slide.nunique() == 12
    cov = r.groupby(["cohort", "slide", "method"]).seed.nunique().unstack("method")
    assert len(cov) == 25 and (cov[MODELS] == 3).all().all() and (cov[TRAINFREE] == 1).all().all()
    assert (r.groupby(["cohort", "slide", "mask"]).n.nunique() == 1).all()
    g = r.groupby(["cohort", "slide", "method", "seed"]).agg(ss=("ss", "sum"), n=("n", "sum")).reset_index(); g["mse"] = g.ss / g.n
    S = g.groupby(["cohort", "slide", "method"]).agg(mse=("mse", "mean"), n=("n", "first")).reset_index()
    W = S.pivot_table(index=["cohort", "slide"], columns="method", values="mse").reset_index()
    W = W.merge(S[S.method == "full"][["slide", "n"]].rename(columns={"n": "n_targets_per_seed_3masks"}), on="slide", validate="one_to_one")
    W = W.merge(cmap[["slide", "cancer", "group"]], on="slide", how="left", validate="one_to_one")
    W["skill_vs_BaselineA"] = 1 - W["full"] / W["Baseline A"]; W["skill_vs_Ring2Mean"] = 1 - W["full"] / W["Ring2Mean"]
    W["cohort"] = W.cohort.map(LAB); W = W.sort_values(["cohort", "cancer", "slide"]).reset_index(drop=True)
    ct = W.groupby("cohort").cancer.apply(lambda s: sorted(set(s))).to_dict(); shared = sorted(set(ct["TEST"]) & set(ct["HD"]))
    assert len(shared) > 0
    r["cancer"] = r.slide.map(dict(zip(cmap.slide, cmap.cancer)))
    r["subset"] = np.where(r.cancer.isin(shared), "ortak", np.where(r.cohort == "test", "yalnız TEST", "yalnız HD"))

    def pooled(df, by):
        h = df.groupby(by + ["method", "seed"]).agg(ss=("ss", "sum"), n=("n", "sum"), ns=("slide", "nunique")).reset_index(); h["mse"] = h.ss / h.n
        rows = []
        for key, d in h.groupby(by + ["method"]):
            meth = key[-1]; k = dict(zip(by + ["method"], key)); vals = d.mse.to_numpy()
            k.update(mse=float(vals.mean()), sd_seed=float(np.std(vals, ddof=1)) if (meth in MODELS and len(vals) == 3) else np.nan,
                     n_seeds=len(vals), n_slides=int(d.ns.iloc[0]), n_targets=int(d.n.iloc[0]))
            rows.append(k)
        a = pd.DataFrame(rows); sk = []
        for key, d in h[h.method == "full"].groupby(by):
            key = key if isinstance(key, tuple) else (key,)
            msk = np.ones(len(h), dtype=bool)
            for c, v in zip(by, key): msk &= (h[c] == v).to_numpy()
            for base in ("Baseline A", "Ring2Mean"):
                bv = h[msk & (h.method == base)]; assert len(bv) == 1
                v = (1 - d.mse.to_numpy() / float(bv.mse.iloc[0])); kk = dict(zip(by, key))
                kk.update(method="full", reference=base, skill_mean=float(v.mean()), skill_sd_seed=float(np.std(v, ddof=1)) if len(v) == 3 else np.nan)
                sk.append(kk)
        return a, pd.DataFrame(sk)
    B1, K1 = pooled(r, ["cohort", "subset"]); B2, K2 = pooled(r, ["cohort", "cancer"])
    for t in (B1, B2, K1, K2): t["cohort"] = t.cohort.map(LAB)
    POOL = pd.concat([B1.assign(level="ortak/yalnız-kohort", group_label=B1.subset).drop(columns="subset"),
                      B2.assign(level="kanser türü", group_label=B2.cancer).drop(columns="cancer")], ignore_index=True)[
        ["level", "cohort", "group_label", "method", "mse", "sd_seed", "n_seeds", "n_slides", "n_targets"]]
    SKILL = pd.concat([K1.assign(level="ortak/yalnız-kohort", group_label=K1.subset).drop(columns="subset"),
                       K2.assign(level="kanser türü", group_label=K2.cancer).drop(columns="cancer")], ignore_index=True)[
        ["level", "cohort", "group_label", "method", "reference", "skill_mean", "skill_sd_seed"]]
    return W, POOL, SKILL, shared


def geometry_table(h5_dir, slides, cmap, nn, res):
    """Spot diameter / coordinate consistency per evaluation slide, read from the HEST h5ad metadata (header only)."""
    import anndata as ad
    from scipy.spatial import cKDTree
    rows = []
    for sid in slides:
        row = dict(slide=sid, n_obs=np.nan, n_lib=np.nan, spot_diameter_fullres=np.nan, nn1_h5ad_all=np.nan, scalefactor_keys="", micron_like_keys="")
        f = list(h5_dir.glob(f"{sid}.h5ad")); assert len(f) == 1, sid
        a = ad.read_h5ad(f[0], backed="r")
        try:
            row["n_obs"] = int(a.n_obs)
            libs = list(a.uns["spatial"].keys()); row["n_lib"] = len(libs); d = a.uns["spatial"][libs[0]]; sf = d.get("scalefactors", {})
            row["spot_diameter_fullres"] = float(sf.get("spot_diameter_fullres", np.nan)); row["scalefactor_keys"] = ",".join(sorted(sf.keys()))
            keys = list(a.uns.keys()) + list(d.keys()) + list(d.get("metadata", {}).keys() if isinstance(d.get("metadata"), dict) else [])
            row["micron_like_keys"] = ",".join([k for k in keys if re.search(r"micron|µm|_um\b|um_|\bmpp", str(k), re.I)])
            C = np.asarray(a.obsm["spatial"]).astype(float); assert C.ndim == 2 and C.shape[1] == 2 and np.isfinite(C).all()
            row["nn1_h5ad_all"] = float(np.median(cKDTree(C).query(C, k=2)[0][:, 1]))
        finally:
            a.file.close()
        rows.append(row)
    T = (pd.DataFrame(rows).merge(cmap[["slide", "cohort"]], on="slide", validate="one_to_one")
         .merge(nn[["sample_id", "platform", "n_spots", "nn1_median", "theo_spot_diameter", "theo_inter_spot_dist"]]
                .rename(columns={"sample_id": "slide", "n_spots": "n_spots_audit", "nn1_median": "nn1_audit"}), on="slide", validate="one_to_one"))
    q = res[(res.scn == SCN) & (res.method == "full") & (res.seed.astype(str) == "42") & res.slide.isin(slides)].groupby("slide").n.sum().rename("n_targets_3masks").reset_index()
    T = T.merge(q, on="slide", validate="one_to_one")
    T["target_ratio"] = T.n_targets_3masks / (T.n_spots_audit * 54 * 0.25 * 3)
    T["n_obs_equals_n_spots_audit"] = T.n_obs == T.n_spots_audit
    T["nn1_audit_over_h5ad"] = T.nn1_audit / T.nn1_h5ad_all
    T["nn1_over_spot_diameter_fullres"] = T.nn1_audit / T.spot_diameter_fullres
    assert T.n_obs_equals_n_spots_audit.all() and T.nn1_audit_over_h5ad.between(0.999, 1.001).all()
    assert (T.n_lib == 1).all() and T.spot_diameter_fullres.notna().all()
    T["cohort"] = T.cohort.map(LAB).fillna(T.cohort)
    return T.drop(columns=["n_targets_3masks"])


# ---------------------------------------------------------------- depth QC (S5.4) and TENX128 / TENX155 checks (cells QC-1 ... QC-3)
MODELS4 = ["expression-only", "expression+PPI", "expression+spatial", "full"]
ORDER8 = ["Baseline A", "Baseline B", "NeighborMean", "Ring2Mean"] + MODELS4


def slide_mse_table(res):
    """Per-slide pooled MSE of the full model (3-seed mean), Baseline A and NeighborMean (rand25)."""
    R = res[res.scn == SCN]

    def mse(m):
        z = R[R.method == m].groupby(["cohort", "slide"])[["ss", "n"]].sum(); return z.ss / z.n

    def mse_model(m):
        z = R[R.method == m].groupby(["cohort", "slide", "seed"])[["ss", "n"]].sum()
        assert (z.groupby(["cohort", "slide"]).n.nunique() == 1).all()
        return (z.ss / z.n).groupby(["cohort", "slide"]).mean()
    T = pd.DataFrame({"mse_full": mse_model("full"), "mse_baselineA": mse("Baseline A"), "mse_NM": mse("NeighborMean")}).reset_index()
    T["rel_red_vs_A_pct"] = 100 * (1 - T.mse_full / T.mse_baselineA)
    return T


def depth_qc(T, h5_dir, pyg):
    """Per-slide sequencing depth / detection from the raw counts of the h5ad, joined with the slide MSE; Spearman within cohort."""
    import anndata
    import scipy.sparse as sp
    from scipy.stats import spearmanr
    rows = []
    for sid in T.slide:
        a = anndata.read_h5ad(h5_dir / f"{sid}.h5ad"); d = pyg[sid]
        assert a.n_obs == d.x.shape[0], sid
        M = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        assert np.allclose(M.data, np.round(M.data)) and (M.min() >= 0 if M.nnz else True), f"{sid}: X is not non-negative integer counts"
        tc = np.asarray(M.sum(1)).ravel(); ng = np.asarray((M > 0).sum(1)).ravel()
        rows.append(dict(slide=sid, n_spots=M.shape[0], med_total_counts=float(np.median(tc)), med_genes_detected=float(np.median(ng))))
    T = T.merge(pd.DataFrame(rows), on="slide")
    cor = []
    for coh in ["test", "hd"]:
        z = T[T.cohort == coh]
        for c in ["med_total_counts", "med_genes_detected", "n_spots"]:
            r, p = spearmanr(z[c], z.mse_full); r2, p2 = spearmanr(z[c], z.rel_red_vs_A_pct)
            cor.append(dict(cohort=coh, x=c, rho_vs_mse_full=r, rho_vs_rel_red_vs_A=r2, n=len(z)))
    return T, pd.DataFrame(cor)


def _row_keys(X, min_nnz):
    """blake2b digest of the (gene index, count) vector of every spot with >= min_nnz nonzero genes (rows with few nonzeros can coincide by chance)."""
    import hashlib
    X = X.tocsr(); X.sort_indices(); keys = {}
    for i in range(X.shape[0]):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        if hi - lo >= min_nnz:
            keys[i] = hashlib.blake2b(X.indices[lo:hi].tobytes() + X.data[lo:hi].astype(np.float32).tobytes(), digest_size=12).digest()
    return keys


def tenx_overlap(h5_dir, min_nnz=50):
    """TENX128 vs TENX155 by count VECTOR (not by barcode): the two slides share most spot count vectors under different barcodes and coordinates."""
    import anndata
    import scipy.sparse as sp
    a = anndata.read_h5ad(h5_dir / "TENX128.h5ad"); b = anndata.read_h5ad(h5_dir / "TENX155.h5ad")
    ka, kb = _row_keys(sp.csr_matrix(a.X), min_nnz), _row_keys(sp.csr_matrix(b.X), min_nnz)
    sa, sb = set(ka.values()), set(kb.values()); oa, ob = list(a.obs_names.astype(str)), list(b.obs_names.astype(str))
    pos_b = {}
    for j_, k in kb.items(): pos_b.setdefault(k, []).append(ob[j_])
    same_bc = sum(1 for i, k in ka.items() if oa[i] in pos_b.get(k, []))
    ca, cb = np.asarray(a.obsm["spatial"], dtype=float), np.asarray(b.obsm["spatial"], dtype=float)
    common = [x for x in oa if x in set(ob)]; ia = {x: i for i, x in enumerate(oa)}; ib = {x: i for i, x in enumerate(ob)}
    return dict(n_spots_128=len(oa), n_spots_155=len(ob), n_common_barcodes=len(common), rows_used_128=len(ka), rows_used_155=len(kb),
                shared_count_vectors=len(sa & sb), frac_of_128=len(sa & sb) / max(len(sa), 1), frac_of_155=len(sa & sb) / max(len(sb), 1),
                same_barcode_identical_vector=same_bc,
                coordinates_identical_on_common_barcodes=bool(np.allclose(ca[[ia[x] for x in common]], cb[[ib[x] for x in common]])))


def row_vector_audit(files, slides, split, min_nnz=50, verbose=True):
    """Exact duplicate count vectors across ALL slides (cross-split pairs would be leakage). Returns the table of slide pairs sharing >= 1 vector."""
    import anndata
    import scipy.sparse as sp
    H = {}
    for k, s_ in enumerate(slides, 1):
        H[s_] = set(_row_keys(sp.csr_matrix(anndata.read_h5ad(files[s_]).X), min_nnz).values())
        if verbose and k % 10 == 0: print(f"  {k}/{len(slides)} slides")
    rows = []
    for i, a_ in enumerate(slides):
        for b_ in slides[i + 1:]:
            c = len(H[a_] & H[b_])
            if c: rows.append(dict(slide_a=a_, split_a=split[a_], slide_b=b_, split_b=split[b_], shared_row_vectors=c,
                                   frac_of_a=c / max(len(H[a_]), 1), frac_of_b=c / max(len(H[b_]), 1)))
    return pd.DataFrame(rows, columns=["slide_a", "split_a", "slide_b", "split_b", "shared_row_vectors", "frac_of_a", "frac_of_b"]).sort_values("shared_row_vectors", ascending=False).reset_index(drop=True)


def tenx_sensitivity(res_all):
    """HD pooled MSE of all methods with all 12 slides, without TENX155, without TENX128, without both (per scenario)."""
    def table(z):
        out = {}
        for m in ORDER8:
            y = z[z.method == m]
            if m in MODELS4:
                s = y.groupby("seed")[["ss", "n"]].sum(); out[m] = (s.ss / s.n).mean()
            else:
                out[m] = y.ss.sum() / y.n.sum()
        return pd.Series(out)
    outs = {}
    for scn in sorted(res_all.scn.unique()):
        z = res_all[(res_all.scn == scn) & (res_all.cohort == "hd")]
        if len(z) == 0: continue
        outs[scn] = pd.DataFrame({"12 slides": table(z), "- TENX155": table(z[z.slide != "TENX155"]),
                                  "- TENX128": table(z[z.slide != "TENX128"]), "- both": table(z[~z.slide.isin(["TENX128", "TENX155"])])})
    return outs
