"""Gene-gene (PPI) graph handling.

The STRING v12.0 functional network (required_score 400; 54 nodes, 680 undirected edges) is stored in
data_tables/ppi_string_v12_edges.csv (columns gene_a, gene_b, weight). Edge weights are the STRING
combined scores; they are provided for reference and are NOT used by the model (GATv2 on edge_index only).

Nodes are indexed by the SASP54 canonical order (gene_order). The STRING node_list order is independent
of that order and must not be used for indexing.
"""
import pandas as pd
import torch

from .config import N_GENES


def read_genes(path):
    df = pd.read_csv(path).sort_values("order")
    genes = df["gene"].tolist()
    assert len(genes) == N_GENES and len(set(genes)) == N_GENES
    return genes


def build_ppi_edge_index(gene_order, edges_csv, device=None):
    """Directed edge_index (2, 2E): every undirected edge in file order, then its reverse.
    The file order is the original graph iteration order (kept so the construction is identical to the
    one used for the reported results; the connectivity itself does not depend on edge order)."""
    e = pd.read_csv(edges_csv)
    node_to_idx = {g: i for i, g in enumerate(gene_order)}
    pairs = [(node_to_idx[u], node_to_idx[v]) for u, v in zip(e["gene_a"], e["gene_b"])]
    pairs += [(v, u) for u, v in pairs]
    idx = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    assert idx.min().item() >= 0 and idx.max().item() < N_GENES
    return idx.to(device) if device is not None else idx


def ppi_edge_index_from_networkx(G, gene_order, device=None):
    """Same construction directly from the networkx graph stored in the dataset package."""
    node_to_idx = {g: i for i, g in enumerate(gene_order)}
    pairs = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    pairs += [(v, u) for u, v in pairs]
    idx = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return idx.to(device) if device is not None else idx
