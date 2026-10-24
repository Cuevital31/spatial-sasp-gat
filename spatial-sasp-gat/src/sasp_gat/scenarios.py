"""Pre-registered masking scenarios (logic from Revision-1 cells 17 and 17S).

17A  validation, 25% random masking
17B  random masking at 10 / 25 / 50% (test and external HD)
17C  structural neighbourhood masking: centres ~ Bernoulli(p_c = 0.05) per spot-gene pair; hidden = centre OR its
     1-hop neighbours (same gene), scored = centre AND measured.
Methods: Baseline A, Baseline B, NeighborMean (1-hop), Ring2Mean (exactly 2 hops), four model variants x 3 seeds.
The 25% arm reuses the primary-protocol mask strings, so it reproduces the primary results. Models were trained only
with 25% random masking; the other scenarios are distribution shifts. No model selection or hyper-parameter tuning.
"""
import numpy as np
import pandas as pd
import torch

from .baselines import nb_predict, pred_baseline_b, train_gene_means
from .config import N_GENES, N_EVAL_MASKS
from .evaluation import DISPLAY
from .masking import make_mask, seed_gen

P_CENTRE_17C = 0.05
BASELINES = ["Baseline A", "Baseline B", "NeighborMean", "Ring2Mean"]
METHOD_ORDER = BASELINES + ["expression-only", "expression+PPI", "expression+spatial", "full"]
MODEL_NAMES = ["full", "expression-only", "expression+spatial", "expression+PPI"]


class Predictors:
    """All methods evaluated on the same hidden-input mask."""

    def __init__(self, models, ppi_edge_index, mu32, device):
        self.models, self.ppi, self.device = models, ppi_edge_index, device
        self.mu = mu32.to(device).double()
        self.mu_fallback_b = mu32.to(device).mean().item()

    def predict_all(self, x, ei, hide, g):
        x64 = x.double()
        P, H = {}, {}
        P["Baseline A"] = self.mu.unsqueeze(0).expand_as(x64)
        P["Baseline B"] = pred_baseline_b(x64, hide, self.mu_fallback_b)
        P["NeighborMean"], H["NeighborMean"] = nb_predict(x64, ~hide, g["src"], g["dst"], self.mu)
        P["Ring2Mean"], H["Ring2Mean"] = nb_predict(x64, ~hide, g["r2s"], g["r2t"], self.mu)
        with torch.no_grad():
            for (v, seed), m in self.models.items():
                P[(DISPLAY[v], seed)] = m(x, ei, hide, self.ppi if m.use_ppi else None).double()
        return P, H


def get_masks(scn, sid, role, i, n, av, g, device):
    """scn = ('rand', ratio) or ('nbhd', p_centre). Returns hide, score, stats."""
    kind, par = scn
    if kind == "rand":
        pct = int(round(par * 100))
        tag = f"{sid}_{role}_{i}" if pct == 25 else f"{sid}_{role}_r{pct}_{i}"        # 25%: primary-protocol strings
        mask = make_mask(n, mask_ratio=par, deterministic_seed_str=tag, device=device)
        hide = mask | (~av).unsqueeze(0)
        score = mask & av.unsqueeze(0)
        return hide, score, dict(n_cent=int(mask.sum()), n_H=int(mask.sum()))
    C = (torch.rand(n, N_GENES, generator=seed_gen(f"{sid}_{role}_nbhd_{i}")) < par).to(device)
    cnt = torch.zeros(n, N_GENES, device=device).index_add_(0, g["dst"], C[g["src"]].float())
    H = C | (cnt > 0)
    hide = H | (~av).unsqueeze(0)
    score = C & av.unsqueeze(0)
    return hide, score, dict(n_cent=int(C.sum()), n_H=int(H.sum()))


def run_scenario(scn_name, scn, coh, samples, role, pred, avail, GR, device, rows, stats):
    """Appends per slide x mask x method sums of squares to `rows` and sampling statistics to `stats`."""
    for sid, d in samples.items():
        x = d.x.to(device); ei = d.edge_index.to(device); av = avail[sid].to(device); g = GR[sid]; n = g["n"]
        for i in range(N_EVAL_MASKS):
            hide, score, st = get_masks(scn, sid, role, i, n, av, g, device)
            n_sc = int(score.sum().item())
            if scn[0] == "nbhd":
                vis_nb = torch.zeros(n, N_GENES, device=device).index_add_(0, g["dst"], (~hide)[g["src"]].float())
                assert (vis_nb[score] == 0).all(), f"{sid}: scored centre has a visible 1-hop neighbour"
            P, H = pred.predict_all(x, ei, hide, g)
            if scn[0] == "nbhd":
                assert not H["NeighborMean"][score].any(), f"{sid}: NeighborMean not fully fallback in 17C"
            stats.append(dict(scn=scn_name, cohort=coh, slide=sid, mask=i, n_pos=n * N_GENES, n_hidden=int(hide.sum().item()),
                              n_cent=st["n_cent"], n_H=st["n_H"], n_scored=n_sc))
            x64 = x.double()
            for k, p in P.items():
                ss = ((p - x64) ** 2)[score].sum().item()
                meth, seed = (k[0], k[1]) if isinstance(k, tuple) else (k, -1)
                nfb = int((score & ~H[k]).sum().item()) if k in H else 0
                rows.append(dict(scn=scn_name, cohort=coh, slide=sid, group=d.leakage_group_id, mask=i,
                                 method=meth, seed=seed, ss=ss, n=n_sc, n_fb=nfb))


def summarize(rows, scn_name, coh):
    g = pd.DataFrame(rows)
    g = g[(g.scn == scn_name) & (g.cohort == coh)]
    out = []
    for m in METHOD_ORDER:
        z = g[g.method == m]
        if m in MODEL_NAMES:
            s = z.groupby("seed")[["ss", "n"]].sum(); per = (s.ss / s.n).to_numpy()
            out.append(dict(method=m, mse=per.mean(), sd=per.std(ddof=1), fallback_pct=np.nan))
        else:
            out.append(dict(method=m, mse=z.ss.sum() / z.n.sum(), sd=np.nan,
                            fallback_pct=100 * z.n_fb.sum() / z.n.sum() if m in ("NeighborMean", "Ring2Mean") else np.nan))
    return pd.DataFrame(out)


def stat_summary(stats, scn_name, coh):
    s = pd.DataFrame(stats); s = s[(s.scn == scn_name) & (s.cohort == coh)]
    return dict(scn=scn_name, cohort=coh, merkez_veya_maske_pct=100 * s.n_cent.sum() / s.n_pos.sum(),
                gizli_pct=100 * s.n_hidden.sum() / s.n_pos.sum(), n_hedef=int(s.n_scored.sum()))


def run_all_scenarios(pred, avail, val_s, test_s, hd_s, GR, device):
    """Runs 17A, 17B (10/25/50) and 17C. Returns RES17, STAT17, SUM17 (dict (scn, cohort) -> table)."""
    rows, stats, SUM = [], [], {}
    plan = [("rand25", ("rand", 0.25), [("test", test_s, "test"), ("hd", hd_s, "hd")]),
            ("val25", ("rand", 0.25), [("val", val_s, "val")]),
            ("rand10", ("rand", 0.10), [("test", test_s, "test"), ("hd", hd_s, "hd")]),
            ("rand50", ("rand", 0.50), [("test", test_s, "test"), ("hd", hd_s, "hd")]),
            ("nbhd05", ("nbhd", P_CENTRE_17C), [("test", test_s, "test"), ("hd", hd_s, "hd")])]
    for name, scn, cohorts in plan:
        for coh, smp, role in cohorts:
            run_scenario(name, scn, coh, smp, role, pred, avail, GR, device, rows, stats)
            SUM[(name, coh)] = summarize(rows, name, coh)
    return pd.DataFrame(rows), pd.DataFrame(stats), SUM


def group_tables(RES17, SUM17, STAT17):
    """Group-level (leakage_group_id) tables derived from RES17 (Rev1 cell 17S, section 3)."""
    g = RES17.groupby(["scn", "cohort", "group", "method", "seed"])[["ss", "n", "n_fb"]].sum().reset_index()
    g["mse"] = g.ss / g.n
    is_m = g.method.isin(MODEL_NAMES)
    gm = (g[is_m].groupby(["scn", "cohort", "group", "method"])
          .agg(mse=("mse", "mean"), mse_sd_seed=("mse", lambda v: v.std(ddof=1)), n_targets=("n", "first")).reset_index())
    gm["fallback_pct"] = np.nan; gm["kind"] = "model: 3 model seed ortalaması"
    gb = g[~is_m].copy()
    gb["mse_sd_seed"] = np.nan; gb["n_targets"] = gb.n
    gb["fallback_pct"] = np.where(gb.method.isin(["NeighborMean", "Ring2Mean"]), 100 * gb.n_fb / gb.n, np.nan)
    gb["kind"] = "baseline: 3 maske havuzlanmış"
    cols = ["scn", "cohort", "group", "method", "mse", "mse_sd_seed", "n_targets", "fallback_pct", "kind"]
    GROUP = pd.concat([gb[cols], gm[cols]], ignore_index=True)
    per_seed_macro = g.groupby(["scn", "cohort", "method", "seed"]).mse.mean().reset_index()
    MACRO = (per_seed_macro.groupby(["scn", "cohort", "method"])
             .agg(macro_mse=("mse", "mean"), macro_sd_seed=("mse", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)).reset_index())
    MACRO["method"] = pd.Categorical(MACRO.method, METHOD_ORDER, ordered=True)
    MACRO = MACRO.sort_values(["scn", "cohort", "method"]).reset_index(drop=True)
    pv = GROUP.pivot_table(index=["scn", "cohort", "group"], columns="method", values="mse")
    defs = [("full", "Ring2Mean"), ("full", "NeighborMean"), ("full", "expression+spatial"), ("full", "expression-only"),
            ("expression+spatial", "Ring2Mean")]
    crow = []
    for a, b in defs:
        d = (pv[a] - pv[b]).rename("diff_a_minus_b").reset_index()
        d["contrast"] = f"{a} - {b}"; crow.append(d)
    CONTR = pd.concat(crow, ignore_index=True)
    CONTR_SUM = (CONTR.groupby(["scn", "cohort", "contrast"])
                 .agg(n_groups=("diff_a_minus_b", "size"), n_a_lower=("diff_a_minus_b", lambda v: int((v < 0).sum())),
                      mean_diff=("diff_a_minus_b", "mean")).reset_index())
    grp_struct = (RES17[RES17.method == "Baseline A"].groupby(["scn", "cohort", "group"])
                  .agg(n_slides=("slide", "nunique"), n_targets_all_masks=("n", "sum")).reset_index())
    SUM_LONG = pd.concat([t.assign(scn=s, cohort=c) for (s, c), t in SUM17.items()], ignore_index=True)
    STAT_POOLED = pd.DataFrame([stat_summary(STAT17.to_dict("records"), s, c) for (s, c) in SUM17.keys()])
    return {"res17_per_slide_mask_seed.csv": RES17,
            "stat17_sampling_structure_per_slide_mask.csv": STAT17,
            "stat17_sampling_structure_pooled.csv": STAT_POOLED,
            "sum17_pooled_summary.csv": SUM_LONG,
            "group_level_all_scenarios.csv": GROUP,
            "group_level_macro_summary.csv": MACRO,
            "group_level_contrasts.csv": CONTR,
            "group_level_contrasts_summary.csv": CONTR_SUM,
            "group_structure.csv": grp_struct}
