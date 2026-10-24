"""Matched control gene panels (manuscript Supplement S3; logic from Revision-1 cells 7-B ... 7-H3).

Pipeline
  1. raw HEST counts (h5ad X) -> CP10k -> log1p, verified against the dataset package on every TRAIN slide;
     TRAIN-only per-gene mean / variance / detection rate (availability-aware for SASP genes);
  2. 200 matched control panels per rule (md: mean+detection; mdv: mean+detection+variance) by noisy min-cost
     one-to-one assignment over each SASP gene's 60 nearest candidates; acceptance thresholds were fixed beforehand
     and are not amended;
  3. training-free evaluation of the paired control columns (Baseline A, NeighborMean, Ring2Mean);
  4. re-evaluation of the stored trained panel runs (expression-only and expression+spatial; checkpoints in Drive);
     the training routines (train_epoch / train_and_eval) are provided for transparency but are NOT executed by the
     verification notebook (no new training; GPU training is not bit-reproducible);
  5. spatial-contribution comparison (7-H3), a pure function of the stored gene-level tables.
Everything is descriptive: no p-values, no significance claims.
"""
import copy
import hashlib
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment

from .config import N_GENES
from .masking import make_mask

TOL_NORM = 1e-4
VARIANT = "CP1e4"                   # rule found by the normalisation check: X -> CP10k -> log1p
HARD2, HARD4 = ["MMP1", "SERPINE1"], ["MMP1", "SERPINE1", "MMP2", "IGFBP7"]
EXPECT_UNMEASURED = {"MISC70": ["CCL2"], "MISC73": ["CSF2"], "TENX39": ["CXCL6"], "TENX49": ["CXCL6"]}


# ------------------------------------------------------------------ normalisation / statistics
def csr32(X):
    return (X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))).astype(np.float32)


def cp10k_log1p(Xall, idx):
    """Counts of ALL spots define the library size; rows `idx` (spots kept in the package) are returned."""
    tot = np.asarray(Xall.sum(1)).ravel()[idx]
    X = Xall[idx]
    sc = np.where(tot > 0, 1e4 / np.maximum(tot, 1e-12), 0.0).astype(np.float32)
    X = (sp.diags(sc) @ X).tocsr(); X.data = np.log1p(X.data)
    return X


def sasp_map(vn, genes):
    vn = pd.Index(vn); dup = vn.duplicated(keep=False); first = {}
    for i, n in enumerate(vn):
        first.setdefault(n, i)
    j_ok, c_ok, j_miss, n_amb = [], [], [], 0
    for j, g in enumerate(genes):
        if g not in first: j_miss.append(j)
        elif dup[first[g]]: n_amb += 1
        else: j_ok.append(j); c_ok.append(first[g])
    return np.array(j_ok, dtype=int), np.array(c_ok, dtype=int), np.array(j_miss, dtype=int), n_amb


def sasp_err(Xn, vn, ref, genes):
    j_ok, c_ok, j_miss, _ = sasp_map(vn, genes)
    e = float(np.abs(Xn[:, c_ok].toarray().astype(np.float64) - ref[:, j_ok]).max())
    em = float(np.abs(ref[:, j_miss]).max()) if len(j_miss) else 0.0
    return e, em


def find_h5(h5_dir):
    files = {}
    for p in h5_dir.rglob("*.h5ad"):
        files.setdefault(p.stem, []).append(p)
    return files


def scan_var_names(files, slides):
    """VNU[s]: uniquely named genes of slide s; DUPS[s]: duplicated names. Header-only (backed) read."""
    import anndata as ad
    VNU, DUPS = {}, {}
    for s in slides:
        a = ad.read_h5ad(files[s][0], backed="r"); vn = pd.Index(a.var_names.astype(str)); a.file.close()
        d = set(vn[vn.duplicated(keep=False)]); DUPS[s] = d; VNU[s] = set(vn) - d
    return VNU, DUPS


def intersect(VNU, slides):
    u = None
    for s in slides:
        u = set(VNU[s]) if u is None else (u & VNU[s])
    return u


def train_gene_stats(files, DS, genes, train_slides, VNU, verbose=True):
    """TRAIN-only per-gene mean, variance, detection rate of the log1p(CP10k) expression, over all uniquely named genes.
    Every slide's SASP columns must reproduce the package within TOL_NORM (otherwise the normalisation rule is wrong)."""
    import anndata as ad
    NAMES = sorted(set().union(*[VNU[s] for s in train_slides])); uidx = {n: i for i, n in enumerate(NAMES)}; nU = len(NAMES)
    n_g = np.zeros(nU); nsl = np.zeros(nU, dtype=int); s1 = np.zeros(nU); s2 = np.zeros(nU); nz = np.zeros(nU)
    errs = {}; t0 = time.time()
    for k, s in enumerate(train_slides, 1):
        a = ad.read_h5ad(files[s][0]); vn = pd.Index(a.var_names.astype(str))
        idx = pd.Index(a.obs_names.astype(str)).get_indexer(DS[s].obs_names); assert (idx >= 0).all()
        X = cp10k_log1p(csr32(a.X), idx); X.eliminate_zeros()
        ref = DS[s].x.detach().cpu().numpy().astype(np.float64)
        e, em = sasp_err(X, vn, ref, genes); errs[s] = e
        assert e < TOL_NORM and em <= TOL_NORM, f"{s}: normalisation does not match package (max err {e:.2e}, unmeasured-column {em:.1e})"
        cols = np.flatnonzero(~np.asarray(vn.duplicated(keep=False))); ug = np.array([uidx[n] for n in vn[cols]])
        Xs = X[:, cols].tocsc()
        s1[ug] += np.asarray(Xs.sum(0)).ravel(); s2[ug] += np.asarray(Xs.multiply(Xs).sum(0)).ravel(); nz[ug] += np.diff(Xs.indptr)
        n_g[ug] += Xs.shape[0]; nsl[ug] += 1
        del a, X, Xs
        if verbose and (k % 10 == 0 or k == len(train_slides)):
            print(f"  {k}/{len(train_slides)} TRAIN slides, {time.time() - t0:.0f} s")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n_g; var = s2 / n_g - mean ** 2; det = nz / n_g
    G = pd.DataFrame({"mean": mean, "var": var, "det": det, "n_slides": nsl}, index=NAMES)
    assert all(g in G.index for g in genes)
    return G, errs


# ------------------------------------------------------------------ matched panels
NPAN, NN, KSEL, SEED0, BIG = 200, 60, 20, 20261020, 1e6
RULES = {"md": ["mean", "det"], "mdv": ["mean", "det", "var"]}
P95_PANEL, MAX_PANEL = 0.10, 0.20
MED_PAIR, P95_PAIR, MAX_PAIR = 0.10, 0.30, 0.75
FE = ["mean", "det", "var"]
ROLE = {"md": "primary", "mdv": "sensitivity"}


def build_panels(S, POOL_E, rule):
    FS, FP = S[FE].astype(float), POOL_E[FE].astype(float)
    scale = FS.std(ddof=1)
    feats = RULES[rule]
    D = np.zeros((len(FS), len(FP)))
    for f in feats:
        D += ((FS[f].values[:, None] - FP[f].values[None, :]) / scale[f]) ** 2
    D = np.sqrt(D); order = np.argsort(D, axis=1)[:, :NN]; Ds = np.take_along_axis(D, order, 1)
    tau = np.maximum(Ds[:, KSEL - 1] - Ds[:, 0], 1e-6)
    union = np.unique(order.ravel()); col = {c: j for j, c in enumerate(union)}
    rows = []
    for p in range(NPAN):
        rng = np.random.default_rng(SEED0 + 1000 * list(RULES).index(rule) + p)
        C = np.full((len(FS), len(union)), BIG)
        for i in range(len(FS)):
            C[i, [col[c] for c in order[i]]] = Ds[i] + tau[i] * rng.random(NN)
        ri, ci = linear_sum_assignment(C)
        assert len(ri) == len(FS) and (C[ri, ci] < BIG).all()
        for i, j in zip(ri, ci):
            c = union[j]
            rows.append(dict(rule=rule, panel=p, tier=("primary5" if p < 5 else "extra"), sasp_gene=FS.index[i],
                             control_gene=FP.index[c], rank_in_nearest=int(np.where(order[i] == c)[0][0]), dist_sd_units=float(D[i, c]),
                             c_mean=FP["mean"].values[c], c_det=FP["det"].values[c], c_var=FP["var"].values[c]))
    return pd.DataFrame(rows), scale


def panel_quality(df, S, scale, genes, rule):
    FS = S[FE].astype(float)
    assert (df.groupby("panel").control_gene.nunique() == len(genes)).all()
    assert not set(df.control_gene) & set(genes)
    dev = {f: np.array([abs(g["c_" + f].mean() - FS[f].mean()) / scale[f] for _, g in df.groupby("panel")]) for f in FE}
    a2 = all(dev[f].max() <= MAX_PANEL and np.percentile(dev[f], 95) <= P95_PANEL for f in RULES[rule])
    pq = df.groupby("panel").dist_sd_units.agg(median="median", p95=lambda x: x.quantile(.95), maximum="max")
    a4 = bool(~((pq["median"] > MED_PAIR) | (pq["p95"] > P95_PAIR) | (pq["maximum"] > MAX_PAIR)).any())
    flag = (pq["median"] <= MED_PAIR) & (pq["p95"] <= P95_PAIR) & (pq["maximum"] <= MAX_PAIR)
    return dict(a2=a2, a4=a4, pq=pq, flag=flag, dev=dev)


def matched_panels(S, POOL_E, genes):
    """Returns (panel table with role and panel_a4_pass, per-rule quality dicts). md is the primary rule, mdv the sensitivity rule."""
    out, rep = [], {}
    for rule in RULES:
        df, scale = build_panels(S, POOL_E, rule)
        rep[rule] = panel_quality(df, S, scale, genes, rule)
        x = df.copy(); x["role"] = ROLE[rule]; x["panel_a4_pass"] = x.panel.map(rep[rule]["flag"]).values
        out.append(x)
    return pd.concat(out, ignore_index=True), rep


# ------------------------------------------------------------------ bank of panel genes for all slides
def build_bank(files, DS, SPLIT, PLAT, genes, P, DUPS, verbose=True):
    import anndata as ad
    ctrl = sorted(set(P.control_gene) - set(genes)); UG = list(genes) + ctrl; nU = len(UG)
    BANK, t0 = {}, time.time()
    for k, s in enumerate(sorted(DS), 1):
        a = ad.read_h5ad(files[s][0]); vn = pd.Index(a.var_names.astype(str))
        idx = pd.Index(a.obs_names.astype(str)).get_indexer(DS[s].obs_names); assert (idx >= 0).all()
        X = cp10k_log1p(csr32(a.X), idx)
        if vn.is_unique: pos = vn.get_indexer(UG)
        else:
            assert not (set(UG) & DUPS[s])
            first = {}
            for i, n in enumerate(vn): first.setdefault(n, i)
            pos = np.array([first.get(g, -1) for g in UG])
        miss = np.flatnonzero(pos < 0)
        assert all(UG[j] in genes for j in miss)
        sel = np.flatnonzero(pos >= 0)
        Pm = sp.csr_matrix((np.ones(len(sel), dtype=np.float32), (np.arange(len(sel)), sel)), shape=(len(sel), nU))
        Xf = (X[:, pos[sel]] @ Pm).tocsr(); Xf.eliminate_zeros()
        meas = np.ones(nU, dtype=bool); meas[miss] = False
        ref = DS[s].x.detach().cpu().numpy().astype(np.float64); got = Xf[:, :len(genes)].toarray().astype(np.float64)
        ok_cols = np.flatnonzero(meas[:len(genes)])
        e = float(np.abs(got[:, ok_cols] - ref[:, ok_cols]).max()); em = float(np.abs(ref[:, ~meas[:len(genes)]]).max()) if (~meas[:len(genes)]).any() else 0.0
        assert e < TOL_NORM and em <= TOL_NORM, (s, e, em)
        BANK[s] = dict(X=Xf, meas=meas, split=SPLIT[s], platform=PLAT[s], group=str(DS[s].leakage_group_id), n=Xf.shape[0])
        del a, X
        if verbose and (k % 15 == 0 or k == len(DS)): print(f"  bank {k}/{len(DS)} slides, {time.time() - t0:.0f} s")
    unm = {s: [UG[j] for j in np.flatnonzero(~BANK[s]["meas"])] for s in BANK if (~BANK[s]["meas"]).any()}
    assert unm == EXPECT_UNMEASURED, unm
    return BANK, UG


# ------------------------------------------------------------------ training-free evaluation of paired controls (7-E)
def make_mask_local(n, tag, ratio=0.25):
    seed = int(hashlib.md5(tag.encode()).hexdigest(), 16) % (2 ** 31)
    return torch.rand(n, N_GENES, generator=torch.Generator().manual_seed(seed)) < ratio


def training_free_eval(BANK, UG, P, DS, SPLIT, G, genes, device, ref_res17=None):
    from .twohop import ring2_pairs
    from .baselines import nb_predict
    METH = ["Baseline A", "NeighborMean", "Ring2Mean"]
    nU = len(UG); uidx = {g: i for i, g in enumerate(UG)}
    Pc = P.copy(); Pc["col"] = Pc.groupby(["rule", "panel"]).cumcount()
    assert (Pc.sasp_gene.values == np.tile(list(genes), len(Pc) // N_GENES)).all()
    Pc["ug"] = Pc.control_gene.map(uidx).astype(int)
    combos = [(j, j) for j in range(N_GENES)] + sorted(set(zip(Pc.ug, Pc.col)) - set((j, j) for j in range(N_GENES)))
    cid = {c: k for k, c in enumerate(combos)}; ncomb = len(combos)
    comb_ug = np.array([c[0] for c in combos]); comb_col = np.array([c[1] for c in combos])
    by_col = {c: np.flatnonzero(comb_col == c) for c in range(N_GENES)}
    mu_all = torch.as_tensor(G.loc[UG, "mean"].values.astype(np.float64), device=device)
    var_all = G.loc[UG, "var"].values.astype(np.float64)
    COHS = {"test": [s for s in sorted(DS) if SPLIT[s] == "test"], "hd": [s for s in sorted(DS) if SPLIT[s] == "external_hd"]}
    SSALL, NALL = {}, {}
    for coh, slides in COHS.items():
        SS = {m: np.zeros((len(slides), 3, ncomb)) for m in METH}; NNm = np.zeros((len(slides), 3, N_GENES))
        for si, s in enumerate(slides):
            b = BANK[s]; n = b["n"]; ei = DS[s].edge_index.cpu().numpy().astype(np.int64)
            src, dst = torch.as_tensor(ei[0], device=device), torch.as_tensor(ei[1], device=device)
            r2a, r2b = ring2_pairs(ei[0], ei[1], n); r2s, r2t = torch.as_tensor(r2a, device=device), torch.as_tensor(r2b, device=device)
            Xd = torch.as_tensor(b["X"].toarray(), dtype=torch.float64, device=device); av = torch.as_tensor(b["meas"][:N_GENES], device=device)
            for i in range(3):
                M = make_mask_local(n, f"{s}_{coh}_{i}").to(device)
                for col in range(N_GENES):
                    mcol = M[:, col]; score = mcol & av[col]; ns = int(score.sum().item()); NNm[si, i, col] = ns
                    if ns == 0: continue
                    ids = by_col[col]; ugs = torch.as_tensor(comb_ug[ids], device=device); x = Xd[:, ugs]; mu_v = mu_all[ugs]
                    valid = ((~mcol) & av[col]).unsqueeze(1).expand(-1, x.shape[1])
                    pnb, _ = nb_predict(x, valid, src, dst, mu_v); pr2, _ = nb_predict(x, valid, r2s, r2t, mu_v); pA = mu_v.unsqueeze(0).expand_as(x)
                    for m, pred in (("Baseline A", pA), ("NeighborMean", pnb), ("Ring2Mean", pr2)):
                        SS[m][si, i, ids] = ((pred - x) ** 2)[score].sum(0).cpu().numpy()
            del Xd
        SSALL[coh], NALL[coh] = SS, NNm
    # gate 1 (optional): SASP columns reproduce the per-slide scenario records
    if ref_res17 is not None:
        ref = ref_res17[(ref_res17.scn == "rand25") & ref_res17.method.isin(METH)]
        bad = 0
        for coh, slides in COHS.items():
            for si, s in enumerate(slides):
                for i in range(3):
                    for m in METH:
                        r = ref[(ref.cohort == coh) & (ref.slide == s) & (ref["mask"] == i) & (ref.method == m)]
                        assert len(r) == 1
                        ss_mine = float(SSALL[coh][m][si, i, :N_GENES].sum()); n_mine = int(NALL[coh][si, i].sum())
                        if int(r.n.iloc[0]) != n_mine or abs(ss_mine - r.ss.iloc[0]) > 1e-5 * max(abs(r.ss.iloc[0]), 1.0): bad += 1
        assert bad == 0, f"{bad} slide x mask x method records do not reproduce the scenario results"
    SSUM = {coh: {m: SSALL[coh][m].sum((0, 1)) for m in METH} for coh in COHS}
    NSUM = {coh: NALL[coh].sum((0, 1)) for coh in COHS}
    MSE = {coh: {m: SSUM[coh][m] / NSUM[coh][comb_col] for m in METH} for coh in COHS}
    vfloor = np.maximum(var_all[comb_ug], 0.01)
    MET = {}
    for coh in COHS:
        for m in METH:
            MET[(coh, "NMSE_" + {"Baseline A": "A", "NeighborMean": "NB", "Ring2Mean": "R2"}[m])] = MSE[coh][m] / vfloor
        MET[(coh, "skill_R2_vs_A")] = 1 - MSE[coh]["Ring2Mean"] / MSE[coh]["Baseline A"]
        assert np.all(np.isfinite(MET[(coh, "skill_R2_vs_A")])) and np.all(MSE[coh]["Baseline A"] > 0)
    metrics = ["NMSE_A", "NMSE_NB", "NMSE_R2", "skill_R2_vs_A"]
    SUBS = {"all54": np.ones(N_GENES, bool), "excl_2hard": ~np.isin(list(genes), HARD2), "excl_4hard": ~np.isin(list(genes), HARD4)}
    rows = []
    for rule in ("md", "mdv"):
        Pr = Pc[Pc.rule == rule]; panels = sorted(Pr.panel.unique())
        C = np.array([[cid[(int(g.loc[g.col == c, "ug"].iloc[0]), c)] for c in range(N_GENES)] for _, g in Pr.groupby("panel", sort=True)])
        flag = Pr.groupby("panel").panel_a4_pass.first().reindex(panels).values
        for coh in COHS:
            for mt in metrics:
                Mv = MET[(coh, mt)]; D = Mv[:N_GENES][None, :] - Mv[C]
                for sub, keep in SUBS.items():
                    stat = D[:, keep].mean(1)
                    for p, v, f in zip(panels, stat, flag):
                        rows.append(dict(rule=rule, panel=p, a4_pass=bool(f), cohort=coh, subset=sub, metric=mt, D=float(v)))
    PS = pd.DataFrame(rows)
    GL = pd.DataFrame([dict(cohort=coh, method=m, ug_gene=UG[comb_ug[k]], col=int(comb_col[k]), is_sasp_combo=bool(k < N_GENES),
                            ss=float(SSUM[coh][m][k]), n=float(NSUM[coh][comb_col[k]]))
                       for coh in COHS for m in METH for k in range(ncomb)])
    return GL, PS


# ------------------------------------------------------------------ trained panel runs: re-evaluation of stored checkpoints
def make_packs(kind, rule, panel, P, genes, UG, BANK, DS, SPLIT, device, roles=("test", "hd")):
    """x tensors for one run's gene set (SASP columns or the matched control panel) for test and external-HD slides."""
    uidx = {g: i for i, g in enumerate(UG)}
    if kind == "sasp":
        getx = lambda s: DS[s].x.cpu().numpy()
    else:
        pr = P[(P.rule == rule) & (P.panel == panel)]
        assert pr.sasp_gene.tolist() == list(genes) and pr.control_gene.nunique() == N_GENES
        cols = [uidx[g] for g in pr.control_gene]
        getx = lambda s: BANK[s]["X"][:, cols].toarray()
    sets = {"test": [s for s in DS if SPLIT[s] == "test"], "hd": [s for s in DS if SPLIT[s] == "external_hd"]}
    out = {}
    for name in roles:
        out[name] = {}
        for s in sets[name]:
            out[name][s] = dict(x=torch.as_tensor(np.asarray(getx(s)), dtype=torch.float32, device=device), ei=DS[s].edge_index.to(device),
                                av=torch.as_tensor(np.asarray(BANK[s]["meas"][:N_GENES], dtype=bool), device=device), n=BANK[s]["n"])
    return out


@torch.no_grad()
def eval_cohort(model, S, coh, device):
    """Per-gene pooled sums of squares and scored-target counts (score = mask & availability of the SASP partner)."""
    model.eval(); ss, nn_ = np.zeros(N_GENES), np.zeros(N_GENES)
    for sid, d in S.items():
        for i in range(3):
            mask = make_mask(d["n"], mask_ratio=0.25, deterministic_seed_str=f"{sid}_{coh}_{i}", device=device)
            recon = model(d["x"], d["ei"], mask | (~d["av"]).unsqueeze(0), None)
            sc = mask & d["av"].unsqueeze(0)
            ss += (((recon - d["x"]) ** 2) * sc.float()).sum(0).double().cpu().numpy(); nn_ += sc.sum(0).cpu().numpy()
    return ss, nn_


def reeval_run(ckpt_path, use_spatial, packs, device):
    from .model import DualGraphSASPModel
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m = DualGraphSASPModel(use_spatial=use_spatial, use_ppi=False).to(device)
    m.load_state_dict(ck["state_dict"], strict=True); m.eval()
    st, nt = eval_cohort(m, packs["test"], "test", device); sh, nh = eval_cohort(m, packs["hd"], "hd", device)
    return dict(ss_test=st, n_test=nt, ss_hd=sh, n_hd=nh), ck


def gene_rows(kind, rule, panel, seed, res, genes_of_run, R2ref, train_var):
    """Gene-level records of one run (RI against the Ring2Mean reference of the same gene-mask column; NMSE with floor 0.01)."""
    rows = []
    for coh in ("test", "hd"):
        ss, nn_ = res[f"ss_{coh}"], res[f"n_{coh}"]; assert (nn_ > 0).all()
        mse = ss / nn_; r2 = np.array([R2ref[(coh, genes_of_run[c], c)] for c in range(N_GENES)])
        fl = np.maximum(np.array([train_var[g] for g in genes_of_run]), 0.01)
        ok = np.isfinite(r2) & (r2 > 0); ri = np.full(N_GENES, np.nan); ri[ok] = (r2[ok] - mse[ok]) / r2[ok]
        nm = mse / fl
        for c in range(N_GENES):
            rows.append(dict(kind=kind, rule=rule, panel=int(panel), seed=int(seed), cohort=coh, col=c, gene=genes_of_run[c], ss=float(ss[c]),
                             n=float(nn_[c]), mse=float(mse[c]), ringmean_mse=float(r2[c]), RI=float(ri[c]), NMSE=float(nm[c])))
    return rows


def ring2_reference(free_df):
    """(cohort, gene, column) -> Ring2Mean MSE from the training-free gene-level table."""
    fr2 = free_df[free_df.method == "Ring2Mean"].copy(); fr2["mse"] = fr2.ss / fr2.n
    gk = fr2.groupby(["cohort", "ug_gene", "col"]).mse.agg(["min", "max"])
    assert ((gk["max"] - gk["min"]) <= 1e-12 * np.maximum(1.0, gk["max"].abs())).all()
    return {(c, g, int(k)): v for (c, g, k), v in gk["min"].items()}


# --- training routines (documentation only; the verification notebooks do not call them) ---------------------------
def _loss(recon, x, mask, av):
    sel = ((recon - x) ** 2)[mask & av.unsqueeze(0)]
    return (sel.mean() if sel.numel() else torch.tensor(0.0, device=recon.device, requires_grad=True)), int(sel.numel())


def train_and_eval(seed, PK, use_spatial, device, lr=1e-3, wd=1e-5, max_epochs=800, patience=30):
    """Panel training protocol: Adam, early stopping on 3 deterministic validation masks, test/HD only after training.
    NOT executed by the verification notebooks."""
    from .model import DualGraphSASPModel
    torch.manual_seed(seed)
    model = DualGraphSASPModel(use_spatial=use_spatial, use_ppi=False).to(device); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_imp = float("inf"), None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for sid, d in PK["tr"].items():
            mask = make_mask(d["n"], mask_ratio=0.25, device=device)
            opt.zero_grad(); recon = model(d["x"], d["ei"], mask | (~d["av"]).unsqueeze(0), None)
            loss, ns = _loss(recon, d["x"], mask, d["av"]); loss.backward(); opt.step()
        model.eval(); tl, tn = 0.0, 0
        with torch.no_grad():
            for sid, d in PK["va"].items():
                for i in range(3):
                    mask = make_mask(d["n"], mask_ratio=0.25, deterministic_seed_str=f"{sid}_val_{i}", device=device)
                    recon = model(d["x"], d["ei"], mask | (~d["av"]).unsqueeze(0), None)
                    l, ns = _loss(recon, d["x"], mask, d["av"]); tl += l.item() * ns; tn += ns
        va = tl / max(tn, 1)
        if va < best_val: best_val, best_state, no_imp = va, copy.deepcopy(model.state_dict()), 0
        else: no_imp += 1
        if no_imp >= patience: break
    model.load_state_dict(best_state)
    return model, best_val


# ------------------------------------------------------------------ 7-H3: spatial contribution (expression+spatial vs expression-only)
def spatial_contribution(sp_df, ex_df, train_gene_stats_df):
    """Pure function of the stored gene-level tables. dN = NMSE_ex - NMSE_sp, dR = RI_sp - RI_ex (>0: spatial branch helps)."""
    KEY = ["kind", "rule", "panel", "seed", "cohort", "col"]
    assert len(sp_df) == len(ex_df) == 13284
    mg = sp_df.merge(ex_df, on=KEY, suffixes=("_sp", "_ex"), validate="one_to_one")
    assert (mg.gene_sp == mg.gene_ex).all() and np.array_equal(mg.n_sp.values, mg.n_ex.values)
    assert np.array_equal(mg.ringmean_mse_sp.values, mg.ringmean_mse_ex.values)
    gs = train_gene_stats_df.set_index("gene" if "gene" in train_gene_stats_df.columns else train_gene_stats_df.columns[0])
    fl = np.maximum(gs.loc[mg.gene_sp, "var"].values.astype(float), 0.01)
    for arm in ("sp", "ex"):
        assert np.allclose(mg[f"mse_{arm}"], mg[f"ss_{arm}"] / mg[f"n_{arm}"], rtol=1e-12, atol=0)
        assert np.allclose(mg[f"RI_{arm}"], (mg[f"ringmean_mse_{arm}"] - mg[f"mse_{arm}"]) / mg[f"ringmean_mse_{arm}"], rtol=1e-12, atol=1e-14)
        assert np.allclose(mg[f"NMSE_{arm}"], mg[f"mse_{arm}"] / fl, rtol=1e-12, atol=0)
    mg["dN"] = mg.NMSE_ex - mg.NMSE_sp
    mg["dR"] = mg.RI_sp - mg.RI_ex
    SEEDS_, GN = sorted(mg.seed.unique()), 54
    mg = mg.sort_values(KEY)
    AR = {k: (g.dN.values, g.dR.values) for k, g in mg.groupby(["kind", "rule", "panel", "seed", "cohort"])}
    names = mg[(mg.kind == "sasp") & (mg.seed == SEEDS_[0]) & (mg.cohort == "test")].sort_values("col").gene_sp.tolist()
    SUB = {"all54": np.ones(GN, bool), "excl_2hard": ~np.isin(names, HARD2), "excl_4hard": ~np.isin(names, HARD4)}
    PAN = {r: sorted(mg[(mg.kind == "ctrl") & (mg.rule == r)].panel.unique()) for r in ("md", "mdv")}
    rows = []
    for rule, pl in PAN.items():
        for p in pl:
            for coh in ("test", "hd"):
                for sub, keep in SUB.items():
                    for mi, mt in ((0, "NMSE"), (1, "RI")):
                        ds = np.array([AR[("sasp", "sasp", -1, s, coh)][mi][keep].mean() for s in SEEDS_])
                        dc = np.array([AR[("ctrl", rule, p, s, coh)][mi][keep].mean() for s in SEEDS_])
                        rows.append(dict(rule=rule, panel=int(p), cohort=coh, subset=sub, metric=mt, D_spatial=float((ds - dc).mean()),
                                         seed_sd=float(np.std(ds - dc, ddof=1)), mean_delta_sasp=float(ds.mean()), mean_delta_ctrl=float(dc.mean())))
    ps = pd.DataFrame(rows)
    grow = []
    for coh in ("test", "hd"):
        sN = np.mean([AR[("sasp", "sasp", -1, s, coh)][0] for s in SEEDS_], axis=0); sR = np.mean([AR[("sasp", "sasp", -1, s, coh)][1] for s in SEEDS_], axis=0)
        cN = np.median([np.mean([AR[("ctrl", "md", p, s, coh)][0] for s in SEEDS_], axis=0) for p in PAN["md"]], axis=0)
        for c in range(GN):
            grow.append(dict(cohort=coh, col=c, gene=names[c], delta_NMSE_sasp=float(sN[c]), delta_RI_sasp=float(sR[c]),
                             delta_NMSE_ctrl_md_median=float(cN[c])))
    return ps, pd.DataFrame(grow)
