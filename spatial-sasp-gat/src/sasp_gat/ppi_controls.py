"""PPI control analyses (logic from Revision-1 cells 18-20): network audit, degree-preserving rewiring,
edge-free control, and locked test / external-HD evaluation of the six control checkpoints. No training here:
the control checkpoints (full + edge-free / rewired, 3 seeds) were trained in the revision with the primary protocol
(see notebooks/01_train_full.ipynb for the training routine, which is the same code path)."""
import hashlib
import json

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .config import N_GENES, SEEDS
from .scenarios import get_masks

CONTROLS = ["edge_free", "rewired"]
NAME = {"edge_free": "full + edge-free", "rewired": "full + rewired"}
REAL_NAME = "full (STRING)"


def real_undirected_edges(ppi_graph, gene_order):
    idx = {g: i for i, g in enumerate(gene_order)}
    real = sorted({tuple(sorted((idx[u], idx[v]))) for u, v in ppi_graph.edges()})
    assert len(real) == 680 and all(u != v for u, v in real)
    return real


def to_edge_index(und, device):
    arr = list(und) + [(v, u) for u, v in und]
    return torch.tensor(arr, dtype=torch.long).t().contiguous().to(device)


def rewire(real, seed, nswap_factor=10):
    """Degree-preserving rewiring (double edge swap, 10 x |E| swaps)."""
    g = nx.Graph(); g.add_nodes_from(range(N_GENES)); g.add_edges_from(real)
    nx.double_edge_swap(g, nswap=nswap_factor * len(real), max_tries=20_000_000, seed=seed)
    return sorted(tuple(sorted(e)) for e in g.edges())


def graph_hash(ei):
    return hashlib.sha256(json.dumps(sorted(map(tuple, ei.t().tolist()))).encode()).hexdigest()


def network_audit(real):
    g = nx.Graph(); g.add_nodes_from(range(N_GENES)); g.add_edges_from(real)
    deg = np.array([g.degree(i) for i in range(N_GENES)])
    n_pairs = N_GENES * (N_GENES - 1) // 2
    return dict(nodes=N_GENES, edges=len(real), density=len(real) / n_pairs, isolated=int((deg == 0).sum()),
                components=nx.number_connected_components(g), deg_min=int(deg.min()), deg_median=float(np.median(deg)),
                deg_mean=float(deg.mean()), deg_max=int(deg.max()), clustering=float(nx.average_clustering(g))), deg


def rewired_graphs(real, deg, device, seeds=SEEDS):
    out = {}
    for s in seeds:
        und = rewire(real, s)
        g = nx.Graph(); g.add_nodes_from(range(N_GENES)); g.add_edges_from(und)
        assert len(und) == len(real) and all(u != v for u, v in und)
        assert [g.degree(i) for i in range(N_GENES)] == deg.tolist()
        out[s] = dict(und=und, ei=to_edge_index(und, device), overlap=len(set(und) & set(real)))
    return out


def string_network(gene_order, ntype, score=400, base="https://version-12-0.string-db.org", timeout=90):
    """Live STRING query (functional or physical). Returns (edge set in gene_order indices, n_rows, unmapped names).
    The answer reflects the current STRING server, not necessarily the 2026-09-26 retrieval."""
    import urllib.parse, urllib.request
    idx = {g: i for i, g in enumerate(gene_order)}
    data = urllib.parse.urlencode({"identifiers": "\r".join(gene_order), "species": 9606, "required_score": score,
                                   "network_type": ntype, "add_nodes": 0,
                                   "caller_identity": "spatial_sasp_gat_revision"}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{base}/api/tsv/network", data=data), timeout=timeout) as r:
        txt = r.read().decode("utf-8")
    lines = [l for l in txt.strip().split("\n") if l.strip()]
    df = pd.DataFrame([l.split("\t") for l in lines[1:]], columns=lines[0].split("\t"))
    a, b = df["preferredName_A"], df["preferredName_B"]
    unmapped = sorted((set(a) | set(b)) - set(idx))
    edges = {tuple(sorted((idx[u], idx[v]))) for u, v in zip(a, b) if u in idx and v in idx and u != v}
    return edges, len(df), unmapped


def load_control_models(ctrl_dir, device, graphs, expected_sha=None):
    """{(control, seed): (model, edge_index, best_val_loss)}; checks flags, parameter count, graph hash vs expected graphs."""
    from pathlib import Path
    from .model import DualGraphSASPModel
    from .data import sha256_file
    cm = {}
    for c in CONTROLS:
        for seed in SEEDS:
            p = Path(ctrl_dir) / f"full_{c}_avail_seed{seed}.pt"
            ck = torch.load(p, map_location=device, weights_only=True)
            assert ck["ppi_control"] == c and ck["seed"] == seed and ck["use_spatial"] is True and ck["use_ppi"] is True
            assert ck["availability_aware"] is True
            ei = ck["ppi_edge_index"].to(device)
            if graphs is not None:
                assert graph_hash(ei) == graph_hash(graphs[(c, seed)].to(device)) == ck["graph_hash"], p.name
            m = DualGraphSASPModel(use_spatial=True, use_ppi=True).to(device)
            m.load_state_dict(ck["state_dict"]); m.eval()
            assert sum(q.numel() for q in m.parameters()) == 114198
            cm[(c, seed)] = (m, ei, ck["best_val_loss"])
    return cm


def evaluate_controls(scn_name, scn, coh, samples, role, models, cm, ppi_edge_index, avail, GR, device, rows):
    for sid, d in samples.items():
        x = d.x.to(device); ei_sp = d.edge_index.to(device); av = avail[sid].to(device); g = GR[sid]; n = g["n"]
        for i in range(3):
            hide, score, _ = get_masks(scn, sid, role, i, n, av, g, device)
            n_sc = int(score.sum().item()); x64 = x.double()
            with torch.no_grad():
                outs = [(REAL_NAME, seed, models[("full", seed)](x, ei_sp, hide, ppi_edge_index)) for seed in SEEDS]
                outs += [(NAME[c], seed, cm[(c, seed)][0](x, ei_sp, hide, cm[(c, seed)][1])) for (c, seed) in cm]
            for meth, seed, o in outs:
                ss = ((o.double() - x64) ** 2)[score].sum().item()
                rows.append(dict(scn=scn_name, cohort=coh, slide=sid, group=d.leakage_group_id, mask=i, method=meth, seed=seed,
                                 ss=ss, n=n_sc))


def per_seed(df):
    micro = df.groupby("seed")[["ss", "n"]].sum(); micro = micro.ss / micro.n
    gg = df.groupby(["seed", "group"])[["ss", "n"]].sum(); gg = (gg.ss / gg.n).groupby("seed").mean()
    return micro, gg


def summarize_controls(R2, ref_rows):
    """R2: control/real rows; ref_rows: RES17 rows for expression+spatial / expression-only (same scn/cohort)."""
    ALL = pd.concat([R2, ref_rows[R2.columns]], ignore_index=True)
    ORDER = [REAL_NAME, NAME["rewired"], NAME["edge_free"], "expression+spatial", "expression-only"]
    summ, paired = [], []
    for scn_name in ["rand25", "nbhd05"]:
        for coh in ["test", "hd"]:
            blk = ALL[(ALL.scn == scn_name) & (ALL.cohort == coh)]
            ps = {m: per_seed(blk[blk.method == m]) for m in ORDER}
            for m in ORDER:
                mi, ma = ps[m]
                summ.append(dict(scn=scn_name, cohort=coh, method=m, micro_mean=mi.mean(), micro_sd=mi.std(ddof=1),
                                 macro_mean=ma.mean(), macro_sd=ma.std(ddof=1)))
            for m in [NAME["rewired"], NAME["edge_free"]]:
                for kind, k in [("micro", 0), ("macro", 1)]:
                    dlt = ps[m][k] - ps[REAL_NAME][k]
                    paired.append(dict(scn=scn_name, cohort=coh, control=m, metric=kind, diff_mean=dlt.mean(), diff_sd=dlt.std(ddof=1),
                                       d42=dlt[42], d123=dlt[123], d2026=dlt[2026], n_control_worse=int((dlt > 0).sum())))
    return pd.DataFrame(summ), pd.DataFrame(paired)
