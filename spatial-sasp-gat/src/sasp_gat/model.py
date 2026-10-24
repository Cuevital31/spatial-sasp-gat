"""Dual-graph model. Architecture copied unchanged from the original methods notebook
(New-SASP-methods, cell 9, everything before '# 5) SMOKE TEST').

Variant naming (use_spatial, use_ppi):
  (False, False) expression-only
  (True,  False) expression+spatial
  (False, True)  expression+PPI
  (True,  True)  full (dual-graph)
Every variant receives the visible expression (x_masked) and the visibility indicator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .config import N_GENES, EPS


class SpotSpatialEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, heads=4, use_spatial=True):
        super().__init__()
        self.use_spatial = use_spatial
        if use_spatial:
            self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=0.1)
            self.gat2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1, dropout=0.1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x, edge_index):
        if self.use_spatial:
            h = F.elu(self.gat1(x, edge_index))
            h = self.gat2(h, edge_index)
            return h
        else:
            return self.mlp(x)


class GenePPIEncoder(nn.Module):
    def __init__(self, n_genes=N_GENES, emb_dim=32, hidden_dim=32, out_dim=32, heads=2):
        super().__init__()
        self.gene_embedding = nn.Embedding(n_genes, emb_dim)
        self.gat1 = GATv2Conv(emb_dim, hidden_dim, heads=heads, dropout=0.1)
        self.gat2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1, dropout=0.1)

    def forward(self, ppi_edge_index):
        h0 = self.gene_embedding.weight
        h = F.elu(self.gat1(h0, ppi_edge_index))
        h = self.gat2(h, ppi_edge_index)
        return h


class DualGraphSASPModel(nn.Module):
    def __init__(self, n_genes=N_GENES, d_spot=64, d_gene=32,
                 use_spatial=True, use_ppi=True):
        super().__init__()
        self.n_genes = n_genes
        self.use_spatial = use_spatial
        self.use_ppi = use_ppi

        in_dim = n_genes * 2  # [masked_expression ; visibility_mask]
        self.spot_encoder = SpotSpatialEncoder(in_dim, hidden_dim=64, out_dim=d_spot,
                                               use_spatial=use_spatial)
        if use_ppi:
            self.gene_encoder = GenePPIEncoder(n_genes=n_genes, out_dim=d_gene)

        fused_dim = d_spot + (d_gene if use_ppi else 0)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(fused_dim, n_genes),
        )

    def forward(self, x, edge_index, mask, ppi_edge_index=None):
        visible = (~mask).float()
        x_masked = x * visible

        model_input = torch.cat([x_masked, visible], dim=-1)
        h_spatial = self.spot_encoder(model_input, edge_index)

        if self.use_ppi:
            gene_embeddings = self.gene_encoder(ppi_edge_index)
            weighted_sum = x_masked @ gene_embeddings
            denom = x_masked.sum(dim=1, keepdim=True) + EPS
            ppi_context = weighted_sum / denom
            h_fused = torch.cat([h_spatial, ppi_context], dim=-1)
        else:
            h_fused = h_spatial

        recon = self.head(h_fused)
        return recon
