"""Training-free baselines used in the paper (logic from Revision-1 cells 2HOP-2 / 16B).
Baseline A: training-set gene means (mu_new, availability-aware).
Baseline B: per-spot mean of the visible genes.
NeighborMean / Ring2Mean: mean of visible values over 1-hop / exact-2-hop neighbours, fallback mu.
"""
import torch

from .config import N_GENES


def train_gene_means(train_samples, avail):
    """Availability-aware per-gene training mean mu_new (unmeasured slide-gene pairs excluded)."""
    sum_g = torch.zeros(N_GENES, dtype=torch.float64); cnt_g = torch.zeros(N_GENES, dtype=torch.float64)
    for sid, d in train_samples.items():
        av = avail[sid]
        sum_g += torch.where(av, d.x.double().sum(0), torch.zeros(N_GENES, dtype=torch.float64))
        cnt_g += torch.where(av, torch.full((N_GENES,), float(d.x.shape[0]), dtype=torch.float64),
                             torch.zeros(N_GENES, dtype=torch.float64))
    return (sum_g / cnt_g).float()


def nb_predict(x64, valid, src, dst, mu_vec):
    vf = valid.double(); xv = x64 * vf
    num = torch.zeros_like(xv).index_add_(0, dst, xv[src])
    den = torch.zeros_like(xv).index_add_(0, dst, vf[src])
    has = den > 0
    return torch.where(has, num / den.clamp_min(1.0), mu_vec.unsqueeze(0).expand_as(num)), has


def pred_baseline_b(x64, hide, mu_fallback):
    vis = (~hide).double(); nvis = vis.sum(1, keepdim=True)
    pred = ((x64 * vis).sum(1, keepdim=True) / nvis.clamp_min(1.0)).expand(-1, N_GENES).clone()
    pred[(nvis == 0).squeeze(-1), :] = mu_fallback
    return pred


# ---------------------------------------------------------------------------------------------
# Baselines A / B with the three scoring definitions (logic from Revision-1 cell 13, unchanged)
# ---------------------------------------------------------------------------------------------
from .availability import effective_mask, score_mask
from .config import MASK_RATIO, N_EVAL_MASKS
from .masking import make_mask


def legacy_gene_means(train_samples):
    """Original definition: mean over all training spots, artificial zeros included."""
    return torch.cat([d.x for d in train_samples.values()], dim=0).mean(dim=0)


@torch.no_grad()
def eval_baseline(kind, samples, avail, role, mu, use_av_input, device, n_masks=N_EVAL_MASKS):
    """Returns mse_old (all masked targets), mse_av (masked AND measured targets), n_old, n_av.
    kind 'A': constant gene-mean vector. kind 'B': mean of the spot's visible genes
    (use_av_input=True: visible = randomly visible AND measured); fallback = mean of mu if nothing is visible."""
    mu = mu.to(device)
    fallback = mu.mean().item()
    ss_old = ss_av = 0.0
    n_old = n_av = 0
    n_fb = 0
    for sid, data in samples.items():
        x = data.x.to(device); n_spots = x.shape[0]; av = avail[sid].to(device)
        for i in range(n_masks):
            mask = make_mask(n_spots, mask_ratio=MASK_RATIO, deterministic_seed_str=f"{sid}_{role}_{i}", device=device)
            if kind == "A":
                pred = mu.unsqueeze(0).expand(n_spots, -1)
            else:
                visible = ~effective_mask(mask, av) if use_av_input else ~mask
                n_vis = visible.sum(dim=1, keepdim=True).float()
                sum_vis = (x * visible.float()).sum(dim=1, keepdim=True)
                zero_vis = (n_vis == 0).squeeze(-1)
                n_fb += int(zero_vis.sum().item())
                denom = n_vis.clone(); denom[zero_vis] = 1.0
                pred = (sum_vis / denom).expand(-1, N_GENES).clone()
                if zero_vis.any():
                    pred[zero_vis, :] = fallback
            d2 = (pred - x) ** 2
            sel_av = score_mask(mask, av)
            ss_old += d2[mask].sum().item();   n_old += mask.sum().item()
            ss_av += d2[sel_av].sum().item();  n_av += sel_av.sum().item()
    return dict(mse_old=ss_old / n_old, mse_av=ss_av / n_av, n_old=int(n_old), n_av=int(n_av), n_fallback=n_fb)


# ---------------------------------------------------------------------------------------------
# Training-free spatial baselines: NeighborMean and 1-hop inverse-distance-weighted mean
# (logic from Revision-1 cell 16B v3; coordinates are used only for distances)
# ---------------------------------------------------------------------------------------------
def build_neighbour_edges(samples_by_cohort, h5ad_dir, device):
    """Returns {slide: {method: (src, dst, w)}} for 'NeighborMean' (w=1) and 'IDW_1hop' (w=1/d).
    The spatial graph is the dataset's edge_index; coordinates come from <h5ad_dir>/<slide>.h5ad (obsm['spatial']),
    whose obs_names must equal the dataset's obs_names."""
    import warnings
    import anndata as ad
    import numpy as np
    warnings.filterwarnings("ignore", message="Variable names are not unique")
    edges = {}
    for coh, smp in samples_by_cohort.items():
        for sid, d in smp.items():
            a = ad.read_h5ad(f"{h5ad_dir}/{sid}.h5ad", backed="r")
            assert list(a.obs_names) == list(d.obs_names), f"{sid}: obs_names differ"
            xy = np.asarray(a.obsm["spatial"], dtype=np.float64); a.file.close()
            n = d.x.shape[0]
            assert xy.shape == (n, 2) and np.isfinite(xy).all()
            ei = d.edge_index.cpu().numpy(); src, dst = ei[0], ei[1]
            assert (src != dst).all(), f"{sid}: self-loop"
            dist = np.linalg.norm(xy[src] - xy[dst], axis=1)
            assert dist.min() > 0, f"{sid}: zero distance"
            meth = {"NeighborMean": (src, dst, np.ones_like(dist)), "IDW_1hop": (src, dst, 1.0 / dist)}
            edges[sid] = {k: (torch.as_tensor(s, device=device), torch.as_tensor(t, device=device),
                              torch.as_tensor(w, device=device, dtype=torch.float64)) for k, (s, t, w) in meth.items()}
    return edges


def neighbour_predict(x, mask, av, src, dst, w, mu):
    """Weighted mean of neighbours' values that are visible AND measured; fallback = availability-aware train mean mu."""
    valid = (~mask) & av.unsqueeze(0)
    xv = x.double() * valid.double()
    vf = valid.double()
    num = torch.zeros_like(xv).index_add_(0, dst, w.unsqueeze(1) * xv[src])
    den = torch.zeros_like(xv).index_add_(0, dst, w.unsqueeze(1) * vf[src])
    has = den > 0
    pred = torch.where(has, num / den.clamp_min(1e-300), mu.unsqueeze(0).expand_as(num))
    return pred, has


def eval_neighbour_methods(samples_by_cohort, edges, avail, mu, device, n_masks=N_EVAL_MASKS):
    """Per slide x mask sums of squares for every method; returns a DataFrame
    (cohort, slide, group, mask, method, ss, n, n_fb)."""
    import pandas as pd
    mu = mu.to(device).double()
    rows = []
    for coh, smp in samples_by_cohort.items():
        role = "test" if coh == "test" else "hd"
        for sid, d in smp.items():
            x = d.x.to(device); av = avail[sid].to(device)
            for i in range(n_masks):
                mask = make_mask(x.shape[0], mask_ratio=MASK_RATIO, deterministic_seed_str=f"{sid}_{role}_{i}", device=device)
                sel = score_mask(mask, av)
                for k, (s_, t_, w_) in edges[sid].items():
                    pred, has = neighbour_predict(x, mask, av, s_, t_, w_, mu)
                    d2 = (pred - x.double()) ** 2
                    rows.append(dict(cohort=coh, slide=sid, group=d.leakage_group_id, mask=i, method=k,
                                     ss=d2[sel].sum().item(), n=int(sel.sum().item()), n_fb=int((sel & ~has).sum().item())))
    return pd.DataFrame(rows)
