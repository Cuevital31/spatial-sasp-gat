"""Per-gene MSE tables (Revision-1 cell 22C) and slide / cancer-type stratification (cell 22D, section 3).
Everything is descriptive; no tests, no model selection."""
import numpy as np
import pandas as pd

from .config import N_GENES
from .scenarios import get_masks

PG_MODELS = ["expression-only", "expression+PPI", "expression+spatial", "full"]
PG_BASELINES = ["Baseline A", "NeighborMean", "Ring2Mean"]      # Baseline B is not meaningful per gene


def per_gene_long(pred, test_s, hd_s, avail, GR, gene_order, device):
    """Per (cohort, method, seed, gene): sum of squares, targets, slide-mask count, slides. 25% random masks."""
    ACC = {}
    for coh, smp, role in [("test", test_s, "test"), ("hd", hd_s, "hd")]:
        for sid, d in smp.items():
            x = d.x.to(device); ei = d.edge_index.to(device); av = avail[sid].to(device); g = GR[sid]; n = g["n"]
            x64 = x.double(); S = {}
            for i in range(3):
                hide, score, _ = get_masks(("rand", 0.25), sid, role, i, n, av, g, device)
                P, _ = pred.predict_all(x, ei, hide, g)
                sc = score.double(); cnt = score.sum(0).double().cpu().numpy()
                for k, p in P.items():
                    if isinstance(k, tuple):
                        meth, seed = k
                        if meth not in PG_MODELS: continue
                    else:
                        meth, seed = k, -1
                        if meth not in PG_BASELINES: continue
                    ssg = (((p - x64) ** 2) * sc).sum(0).cpu().numpy()
                    s = S.setdefault((coh, meth, seed), [np.zeros(N_GENES) for _ in range(4)])
                    s[0] += ssg; s[1] += cnt; s[2] += (cnt > 0); s[3] = np.maximum(s[3], (cnt > 0).astype(float))
            for key, s in S.items():
                a = ACC.setdefault(key, dict(ss=np.zeros(N_GENES), n=np.zeros(N_GENES), nsm=np.zeros(N_GENES), nsl=np.zeros(N_GENES)))
                a["ss"] += s[0]; a["n"] += s[1]; a["nsm"] += s[2]; a["nsl"] += s[3]
    rows = []
    for (coh, meth, seed), a in ACC.items():
        for j, gname in enumerate(gene_order):
            rows.append(dict(cohort=coh, method=meth, seed=seed, gene=gname, ss=a["ss"][j], n=int(a["n"][j]),
                             n_slide_mask=int(a["nsm"][j]), n_slides=int(a["nsl"][j])))
    LONG = pd.DataFrame(rows)
    for c in ["n", "n_slide_mask", "n_slides"]:
        assert LONG.groupby(["cohort", "method", "gene"])[c].nunique().max() == 1
    for c in ["n", "n_slides"]:
        assert LONG.groupby(["cohort", "gene"])[c].nunique().max() == 1
    return LONG


def per_gene_gate(LONG, RES17):
    """Gene sums must reproduce the pooled MSE and target counts of the scenario run (rand25)."""
    for coh in ["test", "hd"]:
        for meth in PG_MODELS + PG_BASELINES:
            for seed in ([42, 123, 2026] if meth in PG_MODELS else [-1]):
                z = LONG[(LONG.cohort == coh) & (LONG.method == meth) & (LONG.seed == seed)]
                r = RES17[(RES17.scn == "rand25") & (RES17.cohort == coh) & (RES17.method == meth) & (RES17.seed == seed)]
                assert len(z) == N_GENES and len(r) > 0 and int(z.n.sum()) == int(r.n.sum()), (coh, meth, seed)
                assert abs(z.ss.sum() / z.n.sum() - r.ss.sum() / r.n.sum()) < 1e-6, (coh, meth, seed)


def per_gene_summary(LONG):
    L = LONG.copy()
    L["mse"] = np.where(L.n > 0, L.ss / L.n.clip(lower=1), np.nan)

    def agg(df):
        v = df.mse.to_numpy(); ok = np.isfinite(v).sum()
        return pd.Series(dict(mse_mean=np.nanmean(v) if ok else np.nan, mse_sd=(np.nanstd(v, ddof=1) if ok > 1 else np.nan),
                              n_seeds=int(ok), n_targets=int(df.n.iloc[0]), n_slide_mask=int(df.n_slide_mask.iloc[0]),
                              n_slides=int(df.n_slides.iloc[0])))
    SUM = L.groupby(["cohort", "method", "gene"]).apply(agg).reset_index()
    assert (SUM[SUM.method.isin(PG_MODELS)].n_seeds == 3).all()
    return L, SUM


def per_gene_wide(SUM, coh):
    s = SUM[SUM.cohort == coh].pivot(index="gene", columns="method", values="mse_mean")
    f = SUM[(SUM.cohort == coh) & (SUM.method == "full")].set_index("gene")
    out = pd.DataFrame(dict(MSE_full_mean=s["full"], MSE_full_sd_ddof1=f.mse_sd, MSE_baselineA=s["Baseline A"],
                            MSE_expr_spatial=s["expression+spatial"], MSE_Ring2Mean=s["Ring2Mean"], MSE_NeighborMean=s["NeighborMean"],
                            n_targets=f.n_targets, n_slides=f.n_slides))
    out["rel_improve_vs_A_pct"] = 100 * (out.MSE_baselineA - out.MSE_full_mean) / out.MSE_baselineA
    out["full_lt_Ring2Mean"] = out.MSE_full_mean < out.MSE_Ring2Mean
    out["full_lt_expr_spatial"] = out.MSE_full_mean < out.MSE_expr_spatial
    return out.sort_values("rel_improve_vs_A_pct", ascending=False)


def slide_level(RES17, cancer_map=None):
    """Full vs Ring2Mean / expression+spatial per slide (model seeds averaged) and per cancer type (rand25)."""
    SEEDS3 = {42, 123, 2026}
    grp = RES17.groupby(["scn", "cohort", "slide", "group", "method", "seed"], as_index=False)[["ss", "n"]].sum()
    grp["mse"] = grp.ss / grp.n

    def piv(meth):
        return grp[grp.method == meth].pivot_table(index=["scn", "cohort", "slide", "group"], columns="seed", values="mse")
    FULL, ES = piv("full"), piv("expression+spatial")
    R2 = grp[grp.method == "Ring2Mean"].set_index(["scn", "cohort", "slide", "group"]).mse
    assert set(FULL.columns) == SEEDS3 and set(ES.columns) == SEEDS3 and FULL.index.equals(ES.index)
    R2a = R2.reindex(FULL.index); assert R2a.notna().all()
    S = pd.DataFrame(dict(mse_full=FULL.mean(1), mse_expr_spatial=ES.mean(1), mse_Ring2Mean=R2a))
    S["d_full_minus_Ring2"] = S.mse_full - S.mse_Ring2Mean
    S["d_full_minus_ES"] = S.mse_full - S.mse_expr_spatial
    S["seeds_full_better_than_Ring2"] = (FULL.sub(S.mse_Ring2Mean, axis=0) < 0).sum(1)
    S["seeds_full_better_than_ES"] = ((FULL - ES) < 0).sum(1)
    n_sl = grp[(grp.method == "full") & (grp.seed == 42)].set_index(["scn", "cohort", "slide", "group"]).n
    S["n_targets"] = n_sl.reindex(S.index)
    S = S.reset_index()
    CANC = None
    if cancer_map is not None:
        S2 = S[S.cohort.isin(["test", "hd"])].copy()
        S2["cancer"] = S2.slide.astype(str).map(cancer_map); assert S2.cancer.notna().all()
        rows = []
        for scn_ in ["rand25", "nbhd05"]:
            z = S2[S2.scn == scn_]
            g = z.groupby(["cohort", "cancer"]).agg(
                n_slides=("slide", "nunique"), n_groups=("group", "nunique"),
                mean_d_full_minus_Ring2=("d_full_minus_Ring2", "mean"), mean_d_full_minus_ES=("d_full_minus_ES", "mean"),
                slides_full_lt_Ring2=("d_full_minus_Ring2", lambda s: int((s < 0).sum())),
                slides_full_lt_ES=("d_full_minus_ES", lambda s: int((s < 0).sum()))).reset_index()
            g.insert(0, "scn", scn_); rows.append(g)
        CANC = pd.concat(rows, ignore_index=True)
    return S, CANC


# ---------------------------------------------------------------- final comparison (cell 22F)
FIN_MODELS = ["expression-only", "expression+PPI", "expression+spatial", "full"]
PPI_CTRL = ["full + edge-free", "full + rewired"]
FIN_ORDER = ["Baseline B", "Baseline A", "NeighborMean", "Ring2Mean", "expression-only", "expression+PPI",
             "expression+spatial", "full", "full + edge-free", "full + rewired"]


def final_comparison(R, EV):
    """R: scenario rows (RES17); EV: PPI-control rows ('full (STRING)', 'full + edge-free', 'full + rewired').
    No new evaluation: merges raw records. Returns (FIN long table, MAIN rand25 table)."""
    cols = ["scn", "cohort", "slide", "group", "mask", "method", "seed", "ss", "n"]
    R, EV = R[cols].copy(), EV[cols].copy()
    a = EV[EV.method == "full (STRING)"].groupby(["scn", "cohort", "seed"])[["ss", "n"]].sum()
    b = R[R.method == "full"].groupby(["scn", "cohort", "seed"])[["ss", "n"]].sum()
    assert a.index.isin(b.index).all()
    b = b.loc[a.index]
    assert (a.n == b.n).all() and np.allclose(a.ss, b.ss, rtol=1e-9, atol=1e-9), "full (STRING) differs from scenario run"
    ppi_scn = sorted(set(a.index.get_level_values("scn")))
    for m in PPI_CTRL:
        idx = EV[EV.method == m].groupby(["scn", "cohort", "seed"]).size().index
        assert set(idx) == set(a.index)
    ctrl = EV[EV.method.isin(PPI_CTRL) & EV.scn.isin(ppi_scn)]
    ALL = pd.concat([R, ctrl], ignore_index=True)
    assert not ALL.duplicated(["scn", "cohort", "slide", "mask", "method", "seed"]).any()

    def pooled(df, keys):
        g = df.groupby(keys, as_index=False)[["ss", "n"]].sum(); assert (g.n > 0).all(); g["mse"] = g.ss / g.n; return g
    K = ["scn", "cohort", "method", "seed"]
    mic = pooled(ALL, K); gm = pooled(ALL, K + ["group"])
    mac = gm.groupby(K, as_index=False).agg(macro_mse=("mse", "mean"), n_groups=("group", "nunique"))
    PS = mic.merge(mac, on=K).merge(ALL.groupby(K).slide.nunique().rename("n_slides").reset_index(), on=K)

    def agg(d):
        assert d.n.nunique() == 1
        return pd.Series(dict(n_seeds=len(d), n_targets=int(d.n.iloc[0]), n_groups=int(d.n_groups.iloc[0]), n_slides=int(d.n_slides.iloc[0]),
                              micro_mean=d.mse.mean(), micro_sd=(d.mse.std(ddof=1) if len(d) > 1 else np.nan),
                              macro_mean=d.macro_mse.mean(), macro_sd=(d.macro_mse.std(ddof=1) if len(d) > 1 else np.nan)))
    FIN = PS.groupby(["scn", "cohort", "method"]).apply(agg, include_groups=False).reset_index()
    FIN["kind"] = np.where(FIN.method.isin(FIN_MODELS + PPI_CTRL), "trained model", "training-free")
    for _, r in FIN.iterrows():
        assert r.n_seeds == (3 if r.kind == "trained model" else 1)
    for _, d in FIN.groupby(["scn", "cohort"]):
        assert d.n_targets.nunique() == 1
    FIN["rank_micro"] = FIN.groupby(["scn", "cohort"]).micro_mean.rank(method="min").astype(int)
    FIN["rank_macro"] = FIN.groupby(["scn", "cohort"]).macro_mean.rank(method="min").astype(int)
    A = FIN[FIN.method == "Baseline A"].set_index(["scn", "cohort"]).micro_mean
    FIN["rel_improve_vs_A_pct_micro"] = FIN.apply(
        lambda r: 100 * (A[(r.scn, r.cohort)] - r.micro_mean) / A[(r.scn, r.cohort)] if (r.scn, r.cohort) in A.index else np.nan, axis=1)
    FIN["m_ord"] = FIN.method.map({m: i for i, m in enumerate(FIN_ORDER)}).fillna(99)
    FIN = FIN.sort_values(["scn", "cohort", "m_ord"]).drop(columns="m_ord").reset_index(drop=True)

    def fmt(m, s): return f"{m:.4f}" if not np.isfinite(s) else f"{m:.4f} ± {s:.4f}"
    rows = []
    for meth in [m for m in FIN_ORDER if m in set(FIN.method)]:
        row = dict(method=meth, kind=FIN[FIN.method == meth].kind.iloc[0])
        for coh in ["test", "hd"]:
            z = FIN[(FIN.scn == "rand25") & (FIN.cohort == coh) & (FIN.method == meth)]
            if len(z) == 0:
                row[f"{coh}_pooled_MSE"] = row[f"{coh}_group_macro_MSE"] = "—"; row[f"{coh}_rank_pooled"] = np.nan; row[f"{coh}_rank_macro"] = np.nan; continue
            z = z.iloc[0]
            row[f"{coh}_pooled_MSE"] = fmt(z.micro_mean, z.micro_sd); row[f"{coh}_group_macro_MSE"] = fmt(z.macro_mean, z.macro_sd)
            row[f"{coh}_rank_pooled"] = int(z.rank_micro); row[f"{coh}_rank_macro"] = int(z.rank_macro)
        rows.append(row)
    return FIN, pd.DataFrame(rows)
