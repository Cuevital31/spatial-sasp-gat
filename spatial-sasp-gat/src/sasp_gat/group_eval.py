"""Group-level (leakage_group_id) evaluation of full vs expression+spatial and the consolidated tables
(logic from Revision-1 cells 14 and 15). Slides are not treated as independent replicates; the unit is the
leakage group. Everything is descriptive."""
import numpy as np
import pandas as pd
import torch

from .availability import effective_mask, score_mask
from .config import MASK_RATIO, N_EVAL_MASKS, SEEDS
from .masking import make_mask

MODELS_GROUP = {"full": "full", "expression+spatial": "expression_spatial"}      # display name -> checkpoint prefix
ORDER = ["expression-only", "expression+PPI", "expression+spatial", "full"]


def structure(samples):
    rows = []
    for sid, d in samples.items():
        rows.append(dict(slide=sid, group=d.leakage_group_id, cancer=d.cancer, platform=d.platform, n_spots=d.x.shape[0]))
    s = pd.DataFrame(rows)
    g = s.groupby("group").agg(n_slides=("slide", "count"), n_spots=("n_spots", "sum"),
                               cancer=("cancer", lambda v: "/".join(sorted(set(v)))),
                               platform=("platform", lambda v: "/".join(sorted(set(v))))).reset_index()
    g["spot_%"] = 100 * g.n_spots / g.n_spots.sum()
    g["tek_slide_grubu"] = g["group"].astype(str).str.startswith("SINGLE")
    return s, g


@torch.no_grad()
def slide_sums(model, samples, avail, ppi_ei, role, device, n_masks=N_EVAL_MASKS):
    model.eval(); out = {}
    for sid, data in samples.items():
        x = data.x.to(device); ei = data.edge_index.to(device); av = avail[sid].to(device)
        ss, n = 0.0, 0
        for i in range(n_masks):
            mask = make_mask(x.shape[0], mask_ratio=MASK_RATIO, deterministic_seed_str=f"{sid}_{role}_{i}", device=device)
            recon = model(x, ei, effective_mask(mask, av), ppi_ei if model.use_ppi else None)
            sel = score_mask(mask, av)
            ss += ((recon - x) ** 2).double()[sel].sum().item(); n += int(sel.sum().item())
        out[sid] = (ss, n)
    return out


def build(cohorts, sums, cohort, key):
    smp = cohorts[cohort]; rows = []
    for (coh, vn, s), sl in sums.items():
        if coh != cohort:
            continue
        acc = {}
        for sid, (ss, n) in sl.items():
            g = key(smp[sid]); a = acc.setdefault(g, [0.0, 0, 0, 0])
            a[0] += ss; a[1] += n; a[2] += 1; a[3] += smp[sid].x.shape[0]
        for g, (ss, n, ns, nsp) in acc.items():
            rows.append(dict(group=g, model=vn, seed=s, ss=ss, n=n, mse=ss / n, n_slides=ns, n_spots=nsp))
    return pd.DataFrame(rows)


def report(cohorts, sums, cohort, key):
    G = build(cohorts, sums, cohort, key)
    piv = G.pivot_table(index=["group", "seed"], columns="model", values="mse").reset_index()
    piv["fark(full-spatial)"] = piv["full"] - piv["expression+spatial"]
    info = G.drop_duplicates(["group"]).set_index("group")[["n_slides", "n_spots"]]
    sc = G[G.model == "full"].groupby("group").n.first()
    rows = []
    for g, sub in piv.groupby("group"):
        rows.append(dict(group=g, n_slides=info.loc[g, "n_slides"], n_spots=info.loc[g, "n_spots"], n_hedef=int(sc[g]),
                         full=f"{sub['full'].mean():.4f}±{sub['full'].std(ddof=1):.4f}",
                         spatial=f"{sub['expression+spatial'].mean():.4f}±{sub['expression+spatial'].std(ddof=1):.4f}",
                         ort_fark=sub["fark(full-spatial)"].mean(),
                         seed_full_daha_dusuk=f"{int((sub['fark(full-spatial)'] < 0).sum())}/{len(sub)}"))
    T = pd.DataFrame(rows).sort_values("n_spots", ascending=False)
    out = []
    for vn in MODELS_GROUP:
        for s in SEEDS:
            g = G[(G.model == vn) & (G.seed == s)]
            big = g.loc[g.n.idxmax(), "group"]
            gx = g[g.group != big]
            out.append(dict(model=vn, seed=s, pooled=g.ss.sum() / g.n.sum(), macro=g.mse.mean(),
                            pooled_en_buyuk_haric=gx.ss.sum() / gx.n.sum() if len(gx) else np.nan, en_buyuk=big))
    return G, T, pd.DataFrame(out)


def run_group_eval(models, test_s, hd_s, avail, ppi_ei, device):
    """models: {(variant_key, seed): model} (needs 'full' and 'expression_spatial').
    Returns dict of file name -> DataFrame (group_* and groups_structure_* tables) plus the pooled cross-check inputs."""
    cohorts = {"test": test_s, "hd": hd_s}
    sums = {}
    for vn, pref in MODELS_GROUP.items():
        for s in SEEDS:
            m = models[(pref, s)]
            for coh, smp in cohorts.items():
                sums[(coh, vn, s)] = slide_sums(m, smp, avail, ppi_ei, "test" if coh == "test" else "hd", device)
    R = {}
    R[("test", "grup")] = report(cohorts, sums, "test", lambda d: d.leakage_group_id)
    R[("hd", "grup")] = report(cohorts, sums, "hd", lambda d: d.leakage_group_id)
    R[("hd", "kanser")] = report(cohorts, sums, "hd", lambda d: d.cancer)
    R[("test", "kanser")] = report(cohorts, sums, "test", lambda d: d.cancer)
    _, G_test = structure(test_s); _, G_hd = structure(hd_s)
    files = {"groups_structure_test.csv": G_test, "groups_structure_hd.csv": G_hd}
    for (coh, level), (G, T, O) in R.items():
        files[f"group_{coh}_{level}_per_seed_long.csv"] = G
        files[f"group_{coh}_{level}_summary_full_vs_spatial.csv"] = T
        files[f"group_{coh}_{level}_pooled_macro_per_seed.csv"] = O
    return files, R, sums


def main_table(eval_rows, val_rows, base_av):
    """Table 1 (corrected protocol). eval_rows: DataFrame variant/seed/test_av/hd_av for the 4 variants (new protocol);
    val_rows: DataFrame variant/seed/best_val_loss; base_av: baselines three-scoring table (tam_av columns)."""
    rows = []
    for v in ORDER:
        n = eval_rows[eval_rows.variant == v]; vv = val_rows[val_rows.variant == v]
        assert len(n) == 3 and len(vv) == 3
        rows.append(dict(method=v, n_seeds=3, val_mean=vv.best_val_loss.mean(), val_sd=vv.best_val_loss.std(ddof=1),
                         test_mean=n.test_av.mean(), test_sd=n.test_av.std(ddof=1),
                         hd_mean=n.hd_av.mean(), hd_sd=n.hd_av.std(ddof=1)))
    for k, lab in [("A", "Baseline A (gen ortalaması)"), ("B", "Baseline B (spot ortalaması)")]:
        t = base_av[(base_av.baseline == k) & (base_av.cohort == "test")].iloc[0]
        h = base_av[(base_av.baseline == k) & (base_av.cohort == "hd")].iloc[0]
        rows.append(dict(method=lab, n_seeds=0, val_mean=np.nan, val_sd=np.nan, test_mean=t.tam_av, test_sd=np.nan,
                         hd_mean=h.tam_av, hd_sd=np.nan))
    return pd.DataFrame(rows)
