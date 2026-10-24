import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from sasp_gat.baselines import neighbour_predict, train_gene_means, legacy_gene_means, eval_baseline
from sasp_gat.config import N_GENES


def test_neighbour_predict_hand_example():
    x = torch.tensor([[2.0], [5.0], [8.0]], dtype=torch.float64)       # chain 0-1-2, distances 1 and 3
    mk = torch.zeros(3, 1, dtype=torch.bool); av = torch.ones(1, dtype=torch.bool)
    s = torch.tensor([0, 1, 1, 2]); t = torch.tensor([1, 0, 2, 1])
    d = torch.tensor([1.0, 1.0, 3.0, 3.0], dtype=torch.float64); mu = torch.tensor([99.0], dtype=torch.float64)
    p_mean, _ = neighbour_predict(x, mk, av, s, t, torch.ones_like(d), mu)
    p_idw, _ = neighbour_predict(x, mk, av, s, t, 1.0 / d, mu)
    assert abs(p_mean[1, 0].item() - 5.0) < 1e-12
    assert abs(p_idw[1, 0].item() - (2 / 1 + 8 / 3) / (1 + 1 / 3)) < 1e-12
    mk2 = mk.clone(); mk2[0, 0] = True                                  # node 0 hidden -> node 1 uses node 2 only
    assert neighbour_predict(x, mk2, av, s, t, torch.ones_like(d), mu)[0][1, 0].item() == 8.0
    mk3 = torch.tensor([[True], [False], [True]])                       # all neighbours hidden -> fallback
    p3, h3 = neighbour_predict(x, mk3, av, s, t, torch.ones_like(d), mu)
    assert (not h3[1, 0].item()) and p3[1, 0].item() == 99.0
    av0 = torch.zeros(1, dtype=torch.bool)                              # unmeasured gene -> never used from neighbours
    assert not neighbour_predict(x, mk, av0, s, t, torch.ones_like(d), mu)[1].any()


def test_neighbour_predict_ignores_hidden_values():
    torch.manual_seed(0)
    x = torch.rand(20, N_GENES, dtype=torch.float64); mk = torch.rand(20, N_GENES) < 0.25
    av = torch.ones(N_GENES, dtype=torch.bool); av[3] = False
    ei = torch.stack([torch.arange(20), (torch.arange(20) + 1) % 20]); ei = torch.cat([ei, ei.flip(0)], 1)
    mu = torch.zeros(N_GENES, dtype=torch.float64); w = torch.ones(ei.shape[1], dtype=torch.float64)
    p0, _ = neighbour_predict(x, mk, av, ei[0], ei[1], w, mu)
    hidden = mk | (~av).unsqueeze(0); x2 = x.clone(); x2[hidden] = 1e6
    p1, _ = neighbour_predict(x2, mk, av, ei[0], ei[1], w, mu)
    assert (p0 - p1).abs().max().item() < 1e-9


class _D:
    def __init__(self, x): self.x = x


def test_gene_means_availability_aware_differs_only_where_unmeasured():
    torch.manual_seed(1)
    tr = {"a": _D(torch.rand(10, N_GENES)), "b": _D(torch.rand(10, N_GENES))}
    tr["a"].x[:, 5] = 0.0                                               # artificial zero-fill
    av = {"a": torch.ones(N_GENES, dtype=torch.bool), "b": torch.ones(N_GENES, dtype=torch.bool)}; av["a"][5] = False
    new, old = train_gene_means(tr, av), legacy_gene_means(tr)
    diff = (new - old).abs()
    assert diff[5] > 1e-3 and torch.cat([diff[:5], diff[6:]]).max() < 1e-6


def test_baseline_a_and_b_run_and_exclude_unmeasured():
    torch.manual_seed(2)
    d = _D(torch.rand(30, N_GENES)); d.x[:, 7] = 0.0
    av = {"S": torch.ones(N_GENES, dtype=torch.bool)}; av["S"][7] = False
    mu = torch.rand(N_GENES)
    for kind in "AB":
        r = eval_baseline(kind, {"S": d}, av, "test", mu, use_av_input=True, device="cpu")
        assert r["n_av"] < r["n_old"] and r["mse_av"] > 0
