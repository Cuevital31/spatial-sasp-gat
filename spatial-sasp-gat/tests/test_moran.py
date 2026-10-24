import numpy as np, pandas as pd, pytest, scipy.sparse as sp
from sasp_gat import moran as mo


def ring_W(n):
    W = sp.lil_matrix((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = 1; W[(i + 1) % n, i] = 1
    return W.tocsr()


def test_moran_matches_naive_and_known_values():
    n = 12; W = ring_W(n); rng = np.random.default_rng(0); X = rng.normal(size=(n, 5))
    I = mo.moran_cols(W, X)
    for j in range(5): assert abs(I[j] - mo.moran_naive(W, X[:, j])) < 1e-12
    smooth = np.sin(2 * np.pi * np.arange(n) / n)[:, None]
    assert mo.moran_cols(W, smooth)[0] > 0.8                       # smooth signal on a ring: strongly positive
    alt = ((-1.0) ** np.arange(n))[:, None]
    assert mo.moran_cols(W, alt)[0] < -0.99                        # alternating: strongly negative
    assert np.isnan(mo.moran_cols(W, np.ones((n, 1)))[0])          # constant column: undefined, not zero


def test_panel_summaries_shapes():
    genes = [f"S{i}" for i in range(54)]; ctrl = [f"C{i}" for i in range(54)]
    P = pd.DataFrame([dict(rule=r, panel=p, sasp_gene=genes[i], control_gene=ctrl[i]) for r in ("md", "mdv") for p in range(200) for i in range(54)])
    SL = [f"A{i}" for i in range(13)] + [f"H{i}" for i in range(12)]; COH = {s: "TEST" if s[0] == "A" else "HD" for s in SL}
    rng = np.random.default_rng(1)
    G = pd.DataFrame([dict(slide=s, cohort=COH[s], gene=g, role="x", moran_I=rng.normal()) for s in SL for g in genes + ctrl])
    PS, SD, SC = mo.panel_summaries(G, P, genes, SL, COH)
    assert len(PS) == 10000 and len(SD) == 50 and len(SC) == 4
    assert SD.paired_panel_win_fraction.between(0, 1).all()
