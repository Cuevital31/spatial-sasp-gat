"""Structure preservation of reconstructed SASP profiles (manuscript Sec. 3.10, Table 12, Supplement S6).
Logic from Revision-1 cells 9-A (profiles) and 9-B (metrics).

Held-out-entry reconstruction: every measured spot x gene entry is assigned to one of 4 folds (u ~ torch.rand(n,54) with
the seed of md5('{slide}_recon_fold'), fold = min(3, floor(4u))). For fold k the entries of that fold (and all unmeasured
entries) are hidden, the model / Ring2Mean predicts them from the remaining entries, and each measured entry is filled
exactly once. Structure metrics compare the reconstructed profile with the observed one after standardisation with
TRAIN gene mean/SD: k-means ARI (k = 3, 5, 7) in the PCA space of the observed profile, PCA-space R2, and correlations of
the mean-z SASP score. Descriptive only."""
import numpy as np
import pandas as pd
import torch

from .availability import effective_mask
from .baselines import nb_predict
from .config import N_GENES
from .masking import seed_gen
from .twohop import ring2_pairs

PROFILES = ["observed", "full_s42", "full_s123", "full_s2026", "ring2mean"]
FULL = ["full_s42", "full_s123", "full_s2026"]
KS = [3, 5, 7]


def make_folds(sid, n):
    return torch.floor(4 * torch.rand(n, N_GENES, generator=seed_gen(f"{sid}_recon_fold"))).long().clamp_(max=3)


def reconstruct_slide(sid, d, av, models, ppi_ei, mu, device):
    """Returns {profile: float32 array (n, n_measured)} and bookkeeping (gene_indices, fold, ring2 fallback mask)."""
    n = d.x.shape[0]; meas = torch.nonzero(av).flatten()
    fold = make_folds(sid, n)
    ei_np = d.edge_index.cpu().numpy(); es, et = ei_np[0].astype(np.int64), ei_np[1].astype(np.int64)
    r2s, r2t = [torch.as_tensor(v) for v in ring2_pairs(es, et, n)]
    x64 = d.x.double()
    rec2 = torch.full((n, N_GENES), float("nan"), dtype=torch.float64); fbm = torch.zeros(n, N_GENES, dtype=torch.bool)
    filled2 = torch.zeros(n, N_GENES, dtype=torch.long)
    for k in range(4):
        hide = effective_mask(fold == k, av)
        pred, has = nb_predict(x64, ~hide, r2s, r2t, mu)
        sel = (fold == k) & av.unsqueeze(0)
        rec2[sel] = pred[sel]; fbm[sel] = ~has[sel]; filled2[sel] += 1
    assert (filled2[:, av] == 1).all() and (filled2[:, ~av] == 0).all()
    out = {"observed": d.x[:, meas].numpy().astype(np.float32), "ring2mean": rec2[:, meas].numpy().astype(np.float32)}
    x = d.x.to(device); ei = d.edge_index.to(device); avd = av.to(device); foldd = fold.to(device)
    for s, m in models.items():
        rec = torch.full((n, N_GENES), float("nan"), device=device); filled = torch.zeros(n, N_GENES, dtype=torch.long, device=device)
        with torch.no_grad():
            for k in range(4):
                hide = effective_mask(foldd == k, avd); o = m(x, ei, hide, ppi_ei); sel = (foldd == k) & avd.unsqueeze(0)
                rec[sel] = o[sel]; filled += sel.long()
        assert (filled[:, avd] == 1).all() and (filled[:, ~avd] == 0).all()
        v = rec[:, meas.to(device)].cpu().numpy().astype(np.float32); assert np.isfinite(v).all()
        out[f"full_s{s}"] = v
    info = dict(gene_indices=meas.numpy().astype(np.int16), fold=fold[:, meas].numpy().astype(np.int8), r2_fallback=fbm[:, meas].numpy())
    return out, info


def compute_metrics(slides, get_profile, mu_tr, sd_tr, unm, n_meas_lock):
    """slides: [(cohort, slide)]; get_profile(slide, profile) -> (gene_indices, X float, ring2 fallback mask or None).
    Returns (slide-level metric table D, cohort summary CS) exactly as in the revision (1000 and 44 rows for 25 slides)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from scipy.stats import pearsonr, spearmanr
    ROWS = []

    def add(coh, sid, prof, seed, metric, k, value, seed_sd=None, status=None):
        v = float(value) if value is not None else float("nan")
        ROWS.append(dict(cohort=coh, slide=sid, profile=prof, seed=seed, metric=metric, k=k, value=v, seed_sd=seed_sd,
                         status=status if status else ("ok" if np.isfinite(v) else "not_computable")))

    def corr(a, b, fn):
        if np.ptp(a) == 0 or np.ptp(b) == 0: return float("nan"), "undefined_constant"
        r = fn(a, b); v = float(r[0]) if isinstance(r, tuple) else float(getattr(r, "statistic", r))
        return v, ("ok" if np.isfinite(v) else "not_computable")

    for coh, sid in slides:
        Z, ref = {}, None
        for p in PROFILES:
            gi, X, fm_ = get_profile(sid, p); gi = np.asarray(gi).astype(int)
            if ref is None:
                exp_gi = np.array([j for j in range(54) if j not in unm.get(sid, set())])
                assert np.array_equal(gi, exp_gi) and len(gi) == n_meas_lock[sid], sid
                ref = gi
            assert np.array_equal(gi, ref)
            Z[p] = (np.asarray(X, dtype=np.float64) - mu_tr[gi]) / sd_tr[gi]; assert np.isfinite(Z[p]).all()
            if p == "ring2mean": add(coh, sid, "ring2mean", None, "ring2_fallback_fraction", None, float(fm_.mean()))
        n, ng = Z["observed"].shape
        pca = PCA(n_components=10, svd_solver="full", random_state=0).fit(Z["observed"]); T = {p: pca.transform(Z[p]) for p in Z}
        den = float(((T["observed"] - T["observed"].mean(0)) ** 2).sum()); assert np.isfinite(den) and den > 0
        S = {p: Z[p].mean(axis=1) for p in Z}
        for k in KS:
            lab_obs = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(T["observed"])
            lab_obs1 = KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(T["observed"])
            add(coh, sid, "observed_stability", None, "ARI", k, adjusted_rand_score(lab_obs, lab_obs1))
            for p in FULL + ["ring2mean"]:
                lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(T[p]); sd_ = int(p.split("_s")[1]) if p.startswith("full") else None
                add(coh, sid, p, sd_, "ARI", k, adjusted_rand_score(lab_obs, lab))
        for p in FULL + ["ring2mean"]:
            sd_ = int(p.split("_s")[1]) if p.startswith("full") else None
            add(coh, sid, p, sd_, "PCA_R2", None, 1.0 - float(((T[p] - T["observed"]) ** 2).sum()) / den)
            for nm, fn in (("score_pearson", pearsonr), ("score_spearman", spearmanr)):
                v, st = corr(S["observed"], S[p], fn); add(coh, sid, p, sd_, nm, None, v, status=st)
    D = pd.DataFrame(ROWS)
    extra = []
    for (coh, sid, metric, k), g in D[D.profile.isin(FULL)].groupby(["cohort", "slide", "metric", "k"], dropna=False, sort=False):
        v = g.value.to_numpy(float)
        st = "ok" if np.isfinite(v).all() and len(v) == 3 else ("undefined_constant" if (g.status == "undefined_constant").any() else "not_computable")
        extra.append(dict(cohort=coh, slide=sid, profile="full_seedmean", seed=None, metric=metric, k=(None if pd.isna(k) else int(k)),
                          value=float(v.mean()) if st == "ok" else float("nan"), seed_sd=float(v.std(ddof=1)) if st == "ok" else float("nan"), status=st))
    D = pd.concat([D, pd.DataFrame(extra)], ignore_index=True)
    kkey = lambda s_: np.where(pd.isna(s_), -1, s_).astype(int)
    fs = D[D.profile == "full_seedmean"][["cohort", "slide", "metric", "k", "value"]].copy(); fs["k_key"] = kkey(fs.k)
    rg = D[(D.profile == "ring2mean") & (D.metric != "ring2_fallback_fraction")][["cohort", "slide", "metric", "k", "value"]].copy(); rg["k_key"] = kkey(rg.k)
    mg = fs.merge(rg[["cohort", "slide", "metric", "k_key", "value"]], on=["cohort", "slide", "metric", "k_key"], how="inner", validate="one_to_one", suffixes=("_full", "_r2"))
    mg["value"] = np.where(np.isfinite(mg.value_full) & np.isfinite(mg.value_r2), mg.value_full - mg.value_r2, np.nan)
    diff = pd.DataFrame(dict(cohort=mg.cohort, slide=mg.slide, profile="full_minus_ring2", seed=None, metric=mg.metric,
                             k=mg.k.map(lambda v_: None if pd.isna(v_) else int(v_)), value=mg.value, seed_sd=None,
                             status=np.where(np.isfinite(mg.value), "ok", "not_computable")))
    D = pd.concat([D, diff], ignore_index=True)
    D["k"] = D.k.astype("Int64"); D["seed"] = D.seed.astype("Int64")
    C = []
    for (coh, prof, metric, k), g in D.groupby(["cohort", "profile", "metric", D.k.fillna(-1)], sort=False):
        if prof in FULL: continue
        v = g.value.to_numpy(float); ok_ = v[np.isfinite(v)]
        row = dict(cohort=coh, profile=prof, metric=metric, k=(None if k == -1 else int(k)), n_valid_slides=len(ok_),
                   median=float(np.median(ok_)) if len(ok_) else float("nan"), q25=float(np.percentile(ok_, 25)) if len(ok_) else float("nan"),
                   q75=float(np.percentile(ok_, 75)) if len(ok_) else float("nan"), n_comparable=None, n_full_better=None, n_equal=None, n_ring2_better=None)
        if prof == "full_minus_ring2":
            row.update(n_comparable=len(ok_), n_full_better=int((ok_ > 0).sum()), n_equal=int((ok_ == 0).sum()), n_ring2_better=int((ok_ < 0).sum()))
        C.append(row)
    CS = pd.DataFrame(C)
    for c_ in ("k", "n_comparable", "n_full_better", "n_equal", "n_ring2_better"): CS[c_] = CS[c_].astype("Int64")
    return D, CS
