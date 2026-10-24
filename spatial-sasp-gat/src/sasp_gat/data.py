"""Loading the packaged dataset (PyG Data objects per slide) with integrity checks."""
import hashlib
import pickle

from .config import DATASET_SHA256, SPLIT_COUNTS, N_GENES


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_package(pkl_path, check_hash=True):
    """The package is a pickle; only load files whose SHA-256 matches DATASET_SHA256."""
    if check_hash:
        assert sha256_file(pkl_path) == DATASET_SHA256, "dataset package hash mismatch"
    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)
    gene_order = list(pkg["gene_order"])
    assert len(gene_order) == N_GENES and len(set(gene_order)) == N_GENES
    return pkg, pkg["pyg_dataset"], gene_order


def split_samples(pyg_dataset):
    sp = {k: {s: d for s, d in pyg_dataset.items() if d.split == k}
          for k in ("train", "val", "test", "external_hd")}
    counts = tuple(len(sp[k]) for k in ("train", "val", "test", "external_hd"))
    assert counts == SPLIT_COUNTS, f"unexpected split sizes: {counts}"
    return sp["train"], sp["val"], sp["test"], sp["external_hd"]
