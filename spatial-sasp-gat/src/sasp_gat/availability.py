"""Availability-aware masking and loss.

Definitions (a slide-gene pair is 'unmeasured' if the gene was not measured on that slide):
  hide  (model input) = random mask  OR  unmeasured
  score (loss / eval) = random mask  AND measured

Zero-filling of unmeasured entries is only a data-representation device so that the tensors have
a fixed shape. It is kept separate from the loss/evaluation exclusion implemented here: an
unmeasured entry is never visible to the model and never scored. These functions consume no RNG.
"""
import pandas as pd
import torch

from .config import N_GENES


def build_availability(slide_ids, gene_order, log_csv):
    """Return (AVAIL dict slide -> bool[N_GENES], LOGGED list of (slide, gene, index))
    from the unmeasured-pair table (columns: sample_id, gene, action)."""
    zlog = pd.read_csv(log_csv)
    avail = {s: torch.ones(N_GENES, dtype=torch.bool) for s in slide_ids}
    logged = []
    for r in zlog.itertuples():
        sid, gene = str(r.sample_id), str(r.gene)
        assert sid in avail, f"slide in log not in dataset: {sid}"
        assert gene in gene_order, f"gene in log not in gene_order: {gene}"
        j = list(gene_order).index(gene)
        avail[sid][j] = False
        logged.append((sid, gene, j))
    return avail, logged


def effective_mask(mask, avail_row):
    """Mask passed to the model: random mask OR unmeasured input."""
    return mask | (~avail_row).unsqueeze(0)


def score_mask(mask, avail_row):
    """Positions that enter the loss / score: randomly masked AND measured."""
    return mask & avail_row.unsqueeze(0)


def masked_mse_loss_av(recon, x_true, mask, avail_row):
    sel = ((recon - x_true) ** 2)[score_mask(mask, avail_row)]
    if sel.numel() == 0:
        return torch.tensor(0.0, device=recon.device, requires_grad=True)
    return sel.mean()


def forward_av(model, x, edge_index, mask, avail_row, ppi_edge_index=None):
    return model(x, edge_index, effective_mask(mask, avail_row), ppi_edge_index)
