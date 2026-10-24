import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from sasp_gat.twohop import ring2_pairs
from sasp_gat.baselines import nb_predict
from sasp_gat.scenarios import get_masks, METHOD_ORDER
from sasp_gat.config import N_GENES
from sasp_gat.twohop import build_graph_cache


def test_ring2_small_graph_exact_distance_two():
    s = np.array([0, 1, 1, 2, 2, 3, 0, 2]); t = np.array([1, 0, 2, 1, 3, 2, 2, 0])   # chain 0-1-2-3 + shortcut 0-2
    a, b = ring2_pairs(s, t, 4)
    assert set(zip(a.tolist(), b.tolist())) == {(0, 3), (3, 0), (1, 3), (3, 1)}


def test_ring2mean_hand_example():
    xs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float64)
    cs = np.array([0, 1, 1, 2, 2, 3, 3, 4]); ct = np.array([1, 0, 2, 1, 3, 2, 4, 3])
    a, b = [torch.as_tensor(v) for v in ring2_pairs(cs, ct, 5)]
    mu = torch.tensor([99.0], dtype=torch.float64)
    hid = torch.zeros(5, 1, dtype=torch.bool)
    p0, _ = nb_predict(xs, ~hid, a, b, mu)
    assert abs(p0[2, 0].item() - 3.0) < 1e-12 and abs(p0[0, 0].item() - 3.0) < 1e-12
    hid2 = hid.clone(); hid2[2, 0] = True
    p2, h2 = nb_predict(xs, ~hid2, a, b, mu)
    assert (not h2[0, 0].item()) and p2[0, 0].item() == 99.0


def _grid(n_side=12):
    n = n_side * n_side; idx = lambda r, c: r * n_side + c; s, t = [], []
    for r in range(n_side):
        for c in range(n_side):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < n_side and cc < n_side:
                    s += [idx(r, c), idx(rr, cc)]; t += [idx(rr, cc), idx(r, c)]
    class D: pass
    d = D(); d.x = torch.rand(n, N_GENES); d.edge_index = torch.tensor([s, t]); return d


def test_17c_scored_centres_have_no_visible_neighbour_and_25pct_arm_uses_primary_strings():
    d = _grid(); GR = build_graph_cache({"test": {"S": d}}, "cpu")
    av = torch.ones(N_GENES, dtype=torch.bool); av[4] = False
    hide, score, st = get_masks(("nbhd", 0.05), "S", "test", 0, GR["S"]["n"], av, GR["S"], "cpu")
    g = GR["S"]
    vis_nb = torch.zeros(g["n"], N_GENES).index_add_(0, g["dst"], (~hide)[g["src"]].float())
    assert (vis_nb[score] == 0).all() and score[:, 4].sum() == 0
    from sasp_gat.masking import make_mask
    h25, s25, _ = get_masks(("rand", 0.25), "S", "test", 0, g["n"], av, g, "cpu") if False else get_masks(("rand", 0.25), "S", "test", 0, g["n"], av, g, "cpu")
    assert torch.equal(s25 | (~av).unsqueeze(0).expand_as(s25) & False, s25)
    ref = make_mask(g["n"], mask_ratio=0.25, deterministic_seed_str="S_test_0")
    assert torch.equal(s25, ref & av.unsqueeze(0))


def test_method_order_complete():
    assert METHOD_ORDER[:4] == ["Baseline A", "Baseline B", "NeighborMean", "Ring2Mean"] and len(METHOD_ORDER) == 8
