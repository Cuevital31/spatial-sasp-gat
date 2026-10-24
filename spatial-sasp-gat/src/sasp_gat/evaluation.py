"""Pooled-MSE evaluation on the locked test and external HD cohorts (logic from Revision-1 cell 11)."""
import torch

from .availability import effective_mask, score_mask
from .config import MASK_RATIO, N_EVAL_MASKS
from .masking import make_mask
from .model import DualGraphSASPModel


def load_model(path, device, expect_seed=None, expect_avail=None):
    """Loads a checkpoint with weights_only=True (state_dict + plain metadata only)."""
    ck = torch.load(path, map_location=device, weights_only=True)
    if expect_seed is not None:
        assert ck["seed"] == expect_seed
    if expect_avail is not None:
        assert bool(ck.get("availability_aware", False)) == expect_avail, f"availability flag mismatch: {path}"
    m = DualGraphSASPModel(use_spatial=ck["use_spatial"], use_ppi=ck["use_ppi"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


@torch.no_grad()
def eval_both(model, samples, avail, ppi_edge_index, device, role, input_av, n_masks=N_EVAL_MASKS,
              mask_ratio=MASK_RATIO):
    """Two scores on the same masks.
    mse_old: all randomly masked targets; mse_av: only masked AND measured targets (the reported score).
    input_av=False: legacy input (random mask only); True: unmeasured genes are also hidden at the input."""
    model.eval()
    ss_old = ss_av = 0.0
    n_old = n_av = 0
    for sid, data in samples.items():
        x = data.x.to(device); ei = data.edge_index.to(device); av = avail[sid].to(device)
        for i in range(n_masks):
            mask = make_mask(x.shape[0], mask_ratio=mask_ratio,
                             deterministic_seed_str=f"{sid}_{role}_{i}", device=device)
            in_mask = effective_mask(mask, av) if input_av else mask
            recon = model(x, ei, in_mask, ppi_edge_index if model.use_ppi else None)
            d2 = ((recon - x) ** 2).double()
            sel_av = score_mask(mask, av)
            ss_old += d2[mask].sum().item();   n_old += int(mask.sum().item())
            ss_av += d2[sel_av].sum().item();  n_av += int(sel_av.sum().item())
    return dict(mse_old=ss_old / n_old, mse_av=ss_av / n_av, n_old=n_old, n_av=n_av)


# ---------------------------------------------------------------------------------------------
# All twelve availability-aware checkpoints (4 variants x 3 seeds)
# ---------------------------------------------------------------------------------------------
VARIANT_KEYS = ["full", "expression_only", "expression_spatial", "expression_PPI"]      # checkpoint file prefixes
DISPLAY = {"full": "full", "expression_only": "expression-only",
           "expression_spatial": "expression+spatial", "expression_PPI": "expression+PPI"}
FLAGS = {"full": (True, True), "expression_only": (False, False),
         "expression_spatial": (True, False), "expression_PPI": (False, True)}


def load_all_models(ckpt_dir, device, seeds=(42, 123, 2026), variants=VARIANT_KEYS):
    """{(variant_key, seed): model in eval mode}; checks name, seed, availability flag and architecture flags."""
    from pathlib import Path
    models = {}
    for v in variants:
        for s in seeds:
            p = Path(ckpt_dir) / f"{v}_avail_seed{s}.pt"
            m, ck = load_model(p, device, expect_seed=s, expect_avail=True)
            assert ck["variant_name"] == DISPLAY[v], (p.name, ck["variant_name"])
            assert (ck["use_spatial"], ck["use_ppi"]) == FLAGS[v], p.name
            models[(v, s)] = m
    return models
