"""Spatial autocorrelation (Moran's I) of SASP genes versus matched control genes, and its relation to the spatial gain
(Revision-1 cells 7-M1 v1.3 and 7-M2 v1.1; reviewer comment R3.4). Descriptive only; the panels are not selected or changed by
Moran's I. Moran's I uses the model's spot graph (binary weights, self-loops removed), all spots of a slide."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .panels import cp10k_log1p, csr32, EXPECT_UNMEASURED

CP = 1e4


def moran_cols(W, X):
    n = X.shape[0]; Z = X - X.mean(0, keepdims=True); den = (Z * Z).sum(0)
    num = (Z * (W @ Z)).sum(0); out = np.full(X.shape[1], np.nan)
    good = den > 1e-12; out[good] = (n / W.sum()) * num[good] / den[good]
    return out


def moran_naive(W, x):
    Wc = W.tocoo(); z = x - x.mean(); n = len(x)
    return (n / Wc.data.sum()) * float(sum(w * z[i] * z[j] for i, j, w in zip(Wc.row, Wc.col, Wc.data))) / float((z * z).sum())


def moran_bank(files, DS, SL, COH, P, GENES, verbose=True):
    """Per slide x gene Moran's I for the SASP54 genes and all control genes of the matched panels (md and mdv)."""
    import anndata as ad
    ctrl = sorted(set(P.control_gene) - set(GENES)); UG = GENES + ctrl
    rows, unmeasured, xerr, gstat = [], {}, {}, {}
    for k, s in enumerate(SL, 1):
        d = DS[s]; n = int(d.x.shape[0]); ei = d.edge_index.cpu().numpy()
        W = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)); W.sum_duplicates(); W.data[:] = 1.0
        sym = bool((W != W.T).nnz == 0); selfloops = int(W.diagonal().sum()); W.setdiag(0); W.eliminate_zeros(); assert W.nnz > 0
        a = ad.read_h5ad(files[s][0]); vn = pd.Index(a.var_names.astype(str)); on = pd.Index(a.obs_names.astype(str))
        idx = on.get_indexer(pd.Index(d.obs_names)); assert (idx >= 0).all() and len(np.unique(idx)) == n
        Xall = csr32(a.X); del a
        dv = Xall.data[:200000].astype(np.float64); assert len(dv) > 0 and np.isfinite(dv).all() and (dv >= 0).all() and np.allclose(dv, np.round(dv))
        X = cp10k_log1p(Xall, idx); del Xall
        if vn.is_unique: pos = vn.get_indexer(UG)
        else:
            dup = set(vn[vn.duplicated(keep=False)]); assert not (set(UG) & dup)
            first = {}
            for i, g in enumerate(vn): first.setdefault(g, i)
            pos = np.array([first.get(g, -1) for g in UG])
        miss = np.flatnonzero(pos < 0); assert all(UG[j] in GENES for j in miss)
        if len(miss): unmeasured[s] = [UG[j] for j in miss]
        sel = np.flatnonzero(pos >= 0); Xd = X[:, pos[sel]].toarray().astype(np.float64)
        ref = d.x.detach().cpu().numpy().astype(np.float64); jj = [i for i, j in enumerate(sel) if j < 54]
        xerr[s] = float(np.abs(Xd[:, jj] - ref[:, [sel[i] for i in jj]]).max()); assert xerr[s] < 1e-4
        I = moran_cols(W, Xd)
        fin = np.flatnonzero(np.isfinite(I)); assert len(fin) >= 3
        for t in np.random.default_rng(0).choice(fin, size=3, replace=False):
            assert abs(I[t] - moran_naive(W, Xd[:, t])) < 1e-9
        for i, j in enumerate(sel):
            rows.append(dict(slide=s, cohort=COH[s], gene=UG[j], role="sasp" if j < 54 else "control", moran_I=float(I[i]),
                             mean_logcp10k=float(Xd[:, i].mean()), frac_detected=float((Xd[:, i] > 0).mean())))
        gstat[s] = dict(n_spots=n, n_edges=int(W.nnz), mean_degree=float(W.nnz / n), symmetric=sym, self_loops_removed=selfloops,
                        isolated=int((np.asarray(W.sum(1)).ravel() == 0).sum()))
        del X, Xd
        if verbose and (k % 5 == 0 or k == len(SL)): print(f"  Moran bank {k}/{len(SL)} slides")
    assert unmeasured == {s: v for s, v in EXPECT_UNMEASURED.items() if s in SL}, unmeasured
    return pd.DataFrame(rows), pd.DataFrame(gstat).T.rename_axis("slide").reset_index()


def panel_summaries(G, P, GENES, SL, COH):
    Iv = G.set_index(["slide", "gene"]).moran_I
    prow = []
    for (rule, panel), g in P.groupby(["rule", "panel"], sort=True):
        sg, cg = g.sasp_gene.tolist(), g.control_gene.tolist()
        for s in SL:
            a_ = np.array([Iv.get((s, x), np.nan) for x in sg]); b_ = np.array([Iv.get((s, x), np.nan) for x in cg]); v = np.isfinite(a_) & np.isfinite(b_)
            prow.append(dict(rule=rule, panel=int(panel), slide=s, cohort=COH[s], n_pairs=int(v.sum()), sasp_median_I=float(np.median(a_[v])),
                             control_median_I=float(np.median(b_[v])), paired_diff_median=float(np.median(a_[v] - b_[v]))))
    PS = pd.DataFrame(prow)
    assert len(PS) == 2 * 200 * 25 and PS.n_pairs.between(1, 54).all()
    srow = []
    for (rule, s), g in PS.groupby(["rule", "slide"]):
        sas = float(Iv.reindex(pd.MultiIndex.from_product([[s], GENES])).dropna().median()); cm = g.control_median_I.to_numpy(); sm = g.sasp_median_I.to_numpy()
        srow.append(dict(rule=rule, slide=s, cohort=COH[s], sasp54_median_I=sas, panels_median_of_median_I=float(np.median(cm)),
                         panels_q05=float(np.quantile(cm, .05)), panels_q95=float(np.quantile(cm, .95)),
                         paired_panel_win_fraction=float((np.sum(cm < sm) + 0.5 * np.sum(cm == sm)) / len(cm)),
                         paired_diff_median_of_panels=float(g.paired_diff_median.median())))
    SD = pd.DataFrame(srow).sort_values(["rule", "cohort", "slide"]).reset_index(drop=True)
    SC = SD.groupby(["rule", "cohort"]).agg(n_slides=("slide", "nunique"), sasp_I_median=("sasp54_median_I", "median"),
                                           panels_I_median=("panels_median_of_median_I", "median"),
                                           paired_win_fraction_median=("paired_panel_win_fraction", "median"),
                                           paired_win_fraction_min=("paired_panel_win_fraction", "min"),
                                           paired_win_fraction_max=("paired_panel_win_fraction", "max"),
                                           paired_diff_median=("paired_diff_median_of_panels", "median")).reset_index()
    return PS, SD, SC


def moran_vs_spatial_gain(MO, sp_df, ex_df, ps_h3, PAN):
    """7-M2: association between the Moran difference of (SASP_i, control_i) and the spatial gain (7-H3). Returns PL, PSUM, GL, GC."""
    KEY = ["kind", "rule", "panel", "seed", "cohort", "col"]
    mg = sp_df.merge(ex_df, on=KEY, suffixes=("_sp", "_ex"), validate="one_to_one")
    mg["dN"] = mg.NMSE_ex - mg.NMSE_sp; mg["dR"] = mg.RI_sp - mg.RI_ex; mg = mg.sort_values(KEY).reset_index(drop=True)
    SEEDS = sorted(mg.seed.unique())
    PANELS = {r: sorted(g.panel.unique()) for r, g in mg[mg.kind == "ctrl"].groupby("rule")}
    arr = {k: (g.dN.to_numpy(), g.dR.to_numpy()) for k, g in mg.groupby(["kind", "rule", "panel", "seed", "cohort"])}
    MO = MO.copy(); MO["coh"] = MO.cohort.str.lower().map({"test": "test", "hd": "hd"})
    nsl = MO.groupby("coh").slide.nunique().to_dict()
    gm = MO.groupby(["coh", "gene"]).moran_I.agg(n_fin=lambda v: int(v.notna().sum()),
                                                 I=lambda v: float(np.nanmedian(v.to_numpy())) if v.notna().any() else np.nan).reset_index()
    gm["need"] = gm.coh.map(lambda c: int(np.ceil(0.5 * nsl[c]))); gm.loc[gm.n_fin < gm.need, "I"] = np.nan
    IG = gm.set_index(["coh", "gene"]).I
    mg["I"] = [IG.get((c, g), np.nan) for c, g in zip(mg.cohort, mg.gene_sp)]
    GL = mg.groupby(["kind", "rule", "panel", "cohort", "col"]).agg(gene=("gene_sp", "first"), moran_I=("I", "first"),
                                                                   delta_NMSE=("dN", "mean"), delta_RI=("dR", "mean")).reset_index()
    assert len(GL) == 41 * 2 * 54
    sas = GL[GL.kind == "sasp"].set_index(["cohort", "col"]); rows = []
    for rule, pl in PANELS.items():
        for p in pl:
            for coh in ("test", "hd"):
                c = GL[(GL.kind == "ctrl") & (GL.rule == rule) & (GL.panel == p) & (GL.cohort == coh)].set_index("col").sort_index()
                s = sas.loc[coh].sort_index(); v = np.isfinite(c.moran_I.to_numpy()) & np.isfinite(s.moran_I.to_numpy())
                dI = float(np.median(s.moran_I.to_numpy()[v] - c.moran_I.to_numpy()[v]))
                cols_ok = np.flatnonzero(v); assert np.array_equal(c.index.to_numpy()[cols_ok], cols_ok)
                dv = {}
                for mi, met in ((0, "NMSE"), (1, "RI")):
                    a_s = np.array([arr[("sasp", "sasp", -1, sd, coh)][mi][cols_ok].mean() for sd in SEEDS])
                    a_c = np.array([arr[("ctrl", rule, p, sd, coh)][mi][cols_ok].mean() for sd in SEEDS]); dv[met] = float((a_s - a_c).mean())
                dd = ps_h3[(ps_h3.rule == rule) & (ps_h3.panel == p) & (ps_h3.cohort == coh) & (ps_h3["subset"] == "all54")].set_index("metric").D_spatial
                rows.append(dict(rule=rule, panel=int(p), cohort=coh, n_valid_cols=int(v.sum()), sasp_median_I=float(np.median(s.moran_I.to_numpy()[v])),
                                 control_median_I=float(np.median(c.moran_I.to_numpy()[v])), dI=dI, abs_dI=abs(dI),
                                 D_spatial_NMSE=float(dd["NMSE"]), D_spatial_RI=float(dd["RI"]), D_valid_pairs_NMSE=dv["NMSE"], D_valid_pairs_RI=dv["RI"]))
    PL = pd.DataFrame(rows)
    PL["proximity"] = np.where(PL.abs_dI <= PL.groupby(["rule", "cohort"]).abs_dI.transform("median"), "Moran yakın", "Moran uzak")
    sc = lambda x, y: float(pd.Series(np.asarray(x, float)).corr(pd.Series(np.asarray(y, float)), method="spearman"))
    srow = []
    for (rule, coh), d in PL.groupby(["rule", "cohort"]):
        for met in ("D_spatial_NMSE", "D_spatial_RI", "D_valid_pairs_NMSE", "D_valid_pairs_RI"):
            near, far = d[d.proximity == "Moran yakın"][met], d[d.proximity == "Moran uzak"][met]
            srow.append(dict(rule=rule, cohort=coh, metric=met, n_panels=len(d), dI_median=float(d.dI.median()), dI_min=float(d.dI.min()),
                             dI_max=float(d.dI.max()), spearman_dI_vs_D=sc(d.dI, d[met]), spearman_is_nan=bool(np.isnan(sc(d.dI, d[met]))),
                             D_median_all=float(d[met].median()), D_median_near=float(near.median()), D_median_far=float(far.median()),
                             n_near=len(near), n_far=len(far), frac_D_positive_all=float((d[met] > 0).mean()), frac_D_positive_near=float((near > 0).mean())))
    PSUM = pd.DataFrame(srow)
    grow = []
    for met in ("delta_NMSE", "delta_RI"):
        for coh in ("test", "hd"):
            s = GL[(GL.kind == "sasp") & (GL.cohort == coh)]; v = s.moran_I.notna()
            grow.append(dict(group="SASP54", rule="-", cohort=coh, metric=met, n_units=int(v.sum()), n_rho_finite=int(np.isfinite(sc(s.moran_I[v], s[met][v]))),
                             n_rho_nan=int(np.isnan(sc(s.moran_I[v], s[met][v]))), rho_median=sc(s.moran_I[v], s[met][v]),
                             rho_q25=np.nan, rho_q75=np.nan, frac_rho_positive=np.nan))
            for rule in PANELS:
                rr = []
                for p in PANELS[rule]:
                    c = GL[(GL.kind == "ctrl") & (GL.rule == rule) & (GL.panel == p) & (GL.cohort == coh)]; vv = c.moran_I.notna()
                    rr.append(sc(c.moran_I[vv], c[met][vv]))
                rr = np.array(rr); fin = rr[np.isfinite(rr)]
                grow.append(dict(group="kontrol paneller", rule=rule, cohort=coh, metric=met, n_units=len(rr), n_rho_finite=len(fin),
                                 n_rho_nan=int(len(rr) - len(fin)), rho_median=float(np.median(fin)) if len(fin) else np.nan,
                                 rho_q25=float(np.quantile(fin, .25)) if len(fin) else np.nan, rho_q75=float(np.quantile(fin, .75)) if len(fin) else np.nan,
                                 frac_rho_positive=float((fin > 0).mean()) if len(fin) else np.nan))
    return PL, PSUM, GL, pd.DataFrame(grow)
