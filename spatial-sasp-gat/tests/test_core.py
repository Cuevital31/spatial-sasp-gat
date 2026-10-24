"""Synthetic-data tests (no dataset needed). Run: pytest -q  (requires torch, torch_geometric, scipy)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from sasp_gat.availability import effective_mask, score_mask, masked_mse_loss_av, forward_av
from sasp_gat.config import N_GENES, N_MODEL_PARAMETERS
from sasp_gat.masking import make_mask
from sasp_gat.model import DualGraphSASPModel
from sasp_gat.twohop import build_graph_cache, masks_2hop, invariants


def _grid_data(n_side=12):
    n = n_side * n_side; idx = lambda r, c: r * n_side + c; s, t = [], []
    for r in range(n_side):
        for c in range(n_side):
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < n_side and cc < n_side:
                    s += [idx(r, c), idx(rr, cc)]; t += [idx(rr, cc), idx(r, c)]
    class D: pass
    d = D(); d.x = torch.rand(n, N_GENES); d.edge_index = torch.tensor([s, t])
    return d


def test_parameter_count():
    assert sum(p.numel() for p in DualGraphSASPModel().parameters()) == N_MODEL_PARAMETERS


def test_deterministic_mask_is_reproducible_and_seed_string_dependent():
    a = make_mask(50, deterministic_seed_str="S_test_0"); b = make_mask(50, deterministic_seed_str="S_test_0")
    c = make_mask(50, deterministic_seed_str="S_test_1")
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert abs(a.float().mean().item() - 0.25) < 0.05


def test_availability_hide_vs_score():
    mask = make_mask(200, deterministic_seed_str="x"); av = torch.ones(N_GENES, dtype=torch.bool); av[5] = False
    assert effective_mask(mask, av)[:, 5].all()
    assert score_mask(mask, av)[:, 5].sum() == 0
    other = torch.arange(N_GENES) != 5
    assert torch.equal(effective_mask(mask, av)[:, other], mask[:, other])
    assert torch.equal(score_mask(mask, av)[:, other], mask[:, other])


def test_loss_ignores_unmeasured_column():
    x = torch.rand(100, N_GENES); mask = make_mask(100, deterministic_seed_str="y")
    av = torch.ones(N_GENES, dtype=torch.bool); av[7] = False
    recon = x.clone(); recon[:, 7] += 100.0
    assert masked_mse_loss_av(recon, x, mask, av).item() == 0.0


def test_availability_functions_do_not_consume_rng():
    mask = make_mask(30, deterministic_seed_str="z"); av = torch.ones(N_GENES, dtype=torch.bool)
    torch.manual_seed(123); s0 = torch.get_rng_state()
    effective_mask(mask, av); score_mask(mask, av)
    assert torch.equal(s0, torch.get_rng_state())


def test_model_sees_unmeasured_gene_as_hidden():
    class Probe:
        use_ppi = False
        def __call__(self, x, ei, mask, ppi):
            self.seen = mask; return torch.zeros_like(x)
    d = _grid_data(); mask = make_mask(d.x.shape[0], deterministic_seed_str="p")
    av = torch.ones(N_GENES, dtype=torch.bool); av[3] = False
    p = Probe(); forward_av(p, d.x, d.edge_index, mask, av, None)
    assert p.seen[:, 3].all()


def test_forward_shapes_all_variants():
    d = _grid_data(); ppi = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 1]])
    mask = make_mask(d.x.shape[0], deterministic_seed_str="f")
    for sp_, pp in [(False, False), (True, False), (False, True), (True, True)]:
        m = DualGraphSASPModel(use_spatial=sp_, use_ppi=pp).eval()
        assert m(d.x, d.edge_index, mask, ppi if pp else None).shape == d.x.shape


def test_twohop_no_receptive_field_leakage():
    d = _grid_data(14); GR = build_graph_cache({"test": {"S": d}}, "cpu")
    av = torch.ones(N_GENES, dtype=torch.bool); av[2] = False
    for i in range(3):
        hide, score, C, H = masks_2hop("S", "test", i, av, GR["S"], 0.03, "cpu")
        invariants("S", hide, score, GR["S"])
        assert score[:, 2].sum() == 0 and hide[:, 2].all()
