import numpy as np, pandas as pd, scipy.sparse as sp, pytest
anndata = pytest.importorskip("anndata")
from sasp_gat import hd_gap as hg


def _write(path, X, obs):
    anndata.AnnData(sp.csr_matrix(X), obs=pd.DataFrame(index=obs), var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])])).write_h5ad(path)


def test_row_vector_audit_finds_shared_rows_only(tmp_path):
    rng = np.random.default_rng(0); A = rng.poisson(2.0, (30, 120)).astype(np.float32); B = rng.poisson(2.0, (30, 120)).astype(np.float32)
    C = A.copy(); C[:5] = rng.poisson(2.0, (5, 120)); C = C[::-1].copy()          # C shares 25 rows with A under a different order
    files = {}
    for n, X in (("A", A), ("B", B), ("C", C)):
        f = tmp_path / f"{n}.h5ad"; _write(f, X, [f"{n}{i}" for i in range(30)]); files[n] = f
    T = hg.row_vector_audit(files, ["A", "B", "C"], {"A": "train", "B": "test", "C": "external_hd"}, min_nnz=20, verbose=False)
    assert len(T) == 1 and {T.slide_a[0], T.slide_b[0]} == {"A", "C"} and T.shared_row_vectors[0] == 25
    T2 = hg.row_vector_audit(files, ["A", "B"], {"A": "train", "B": "test"}, min_nnz=20, verbose=False)
    assert len(T2) == 0


def test_low_nnz_rows_are_ignored(tmp_path):
    X = np.zeros((10, 50), dtype=np.float32); X[:, 0] = 1                          # identical sparse rows in both slides
    files = {}
    for n in ("A", "B"):
        f = tmp_path / f"{n}.h5ad"; _write(f, X, [f"{n}{i}" for i in range(10)]); files[n] = f
    assert len(hg.row_vector_audit(files, ["A", "B"], {"A": "train", "B": "test"}, min_nnz=20, verbose=False)) == 0
