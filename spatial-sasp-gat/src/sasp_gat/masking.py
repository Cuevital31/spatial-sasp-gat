"""Random masking (unchanged from the original pipeline).

True = hidden. With deterministic_seed_str (validation/test/HD) the seed is derived from the string,
so masks are identical across epochs and runs; without it (training) the global CPU RNG is used.
The mask is generated on CPU (torch.Generator is CPU-only) and then moved to the target device.
"""
import hashlib
import torch

from .config import N_GENES


def make_mask(n_spots, n_genes=N_GENES, mask_ratio=0.25,
              deterministic_seed_str=None, device=None):
    if deterministic_seed_str is not None:
        seed = int(hashlib.md5(deterministic_seed_str.encode()).hexdigest(), 16) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        mask = torch.rand(n_spots, n_genes, generator=gen) < mask_ratio
    else:
        mask = torch.rand(n_spots, n_genes) < mask_ratio
    return mask.to(device) if device is not None else mask


def seed_gen(tag):
    """CPU generator seeded from an md5 tag (used by the two-hop masks)."""
    return torch.Generator().manual_seed(int(hashlib.md5(tag.encode()).hexdigest(), 16) % (2 ** 31))
