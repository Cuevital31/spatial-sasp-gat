"""Two-hop (receptive-field) masking: inference-only stress test (logic from Revision-1 cells 2HOP-1/2HOP-2).

For every centre (slide, gene position) drawn with probability p_c, the centre and its 1- and 2-hop spatial
neighbours are hidden for that gene, so that no visible spot lies within the 2-layer receptive field of a
scored centre. Hidden = ball(radius 2) OR unmeasured; scored = centre AND measured. No model is retrained:
the existing full-data checkpoints are evaluated under this masking distribution, which differs from the
25% Bernoulli masking used in training (a distribution-shift caveat that is reported with the results).
"""
import numpy as np
import scipy.sparse as sp
import torch

from .config import N_GENES
from .masking import seed_gen


def ring2_pairs(s, t, n):
    """Pairs at shortest-path distance exactly 2 (no self-return, 1-hop neighbours removed)."""
    A = sp.csr_matrix((np.ones(len(s), dtype=np.int32), (s, t)), shape=(n, n))
    A.sum_duplicates(); A.data[:] = 1
    B = A @ A
    B.setdiag(0); B.eliminate_zeros(); B.data[:] = 1
    B = B - B.multiply(A); B.eliminate_zeros()
    B = B.tocoo()
    return B.row.astype(np.int64), B.col.astype(np.int64)


def build_graph_cache(samples_by_cohort, device):
    """samples_by_cohort: dict cohort -> {slide: Data}. Returns GR dict used by the masking functions."""
    GR = {}
    for coh, smp in samples_by_cohort.items():
        for sid, d in smp.items():
            n = d.x.shape[0]; ei = d.edge_index.cpu().numpy()
            s, t = ei[0].astype(np.int64), ei[1].astype(np.int64)
            assert (s != t).all(), f"{sid}: self-loop"
            k1 = np.sort(s * n + t); k2 = np.sort(t * n + s)
            assert (k1 == k2).all() and len(np.unique(k1)) == len(k1), f"{sid}: edges not symmetric/unique"
            r2s, r2t = ring2_pairs(s, t, n)
            assert (r2s != r2t).all()
            GR[sid] = dict(n=n, coh=coh, src=torch.as_tensor(s, device=device), dst=torch.as_tensor(t, device=device),
                           r2s=torch.as_tensor(r2s, device=device), r2t=torch.as_tensor(r2t, device=device))
    return GR


def spread(M, g):
    """Count, for every position, of neighbours (along graph g) where M is True."""
    return torch.zeros(M.shape, device=M.device).index_add_(0, g["dst"], M[g["src"]].float())


def masks_2hop(sid, role, i, av, g, p, device):
    n = g["n"]
    C = (torch.rand(n, N_GENES, generator=seed_gen(f"{sid}_{role}_nbhd2_{i}")) < p).to(device)
    R1 = C | (spread(C, g) > 0)
    H = R1 | (spread(R1, g) > 0)
    hide = H | (~av).unsqueeze(0)
    score = C & av.unsqueeze(0)
    return hide, score, C, H


def invariants(sid, hide, score, g):
    """No visible spot within 0, 1 or 2 hops of any scored centre (receptive-field leakage check)."""
    V = ~hide
    v1 = spread(V, g) > 0
    v2 = spread(v1, g) > 0
    assert not V[score].any(), f"{sid}: scored centre is visible"
    assert not v1[score].any(), f"{sid}: 1-hop neighbour of a scored centre is visible"
    assert not v2[score].any(), f"{sid}: visible spot within 2 hops of a scored centre (leakage)"
