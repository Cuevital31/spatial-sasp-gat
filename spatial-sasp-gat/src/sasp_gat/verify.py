"""Helpers to compare freshly regenerated tables with the tables stored during the revision."""
import numpy as np
import pandas as pd


def compare_csv(new_df, stored_path, rtol=1e-6, atol=1e-9):
    """Returns (ok, max_abs_diff, message). Rows are sorted by all non-float columns before comparing;
    float columns are compared with np.allclose, everything else must be equal."""
    old = pd.read_csv(stored_path)
    if list(old.columns) != list(new_df.columns):
        return False, np.nan, f"columns differ: stored {list(old.columns)} vs new {list(new_df.columns)}"
    if old.shape != new_df.shape:
        return False, np.nan, f"shape differs: stored {old.shape} vs new {new_df.shape}"
    new = new_df.copy()
    for c in old.columns:                      # categorical / object columns are compared as plain strings (sort order must not depend on dtype)
        if not pd.api.types.is_float_dtype(old[c]):
            new[c] = new[c].astype(str); old[c] = old[c].astype(str)
    key = [c for c in old.columns if not pd.api.types.is_float_dtype(old[c])]
    dup = ""
    if key and (old.duplicated(key).any() or new.duplicated(key).any()):
        dup = " [warning: duplicated key rows; row matching may depend on order]"
    if key:
        old = old.sort_values(key, kind="mergesort").reset_index(drop=True)
        new = new.sort_values(key, kind="mergesort").reset_index(drop=True)
    worst, bad = 0.0, []
    for c in old.columns:
        if pd.api.types.is_float_dtype(old[c]):
            a, b = old[c].to_numpy(float), pd.to_numeric(new[c], errors="coerce").to_numpy(float)
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                bad.append(c)
            d = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
            worst = max(worst, float(d))
        else:
            if not (old[c].astype(str).values == new[c].astype(str).values).all():
                bad.append(c)
    return (not bad), worst, (("" if not bad else f"differing columns: {bad}") + dup)


def report_compare(results):
    """results: {name: (ok, maxdiff, msg)} -> prints a table; returns True if all ok."""
    allok = True
    for name, (ok, d, msg) in results.items():
        print(f"{'MATCH' if ok else 'DIFFERENT':9s} {name:55s} max|diff|={d:.2e} {msg}")
        allok &= ok
    print("ALL MATCH" if allok else "SOME TABLES DIFFER")
    return allok


def roundtrip(df):
    """Write to CSV text and read back, so dtypes (nullable ints, blanks) match a table read from disk."""
    import io
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision="round_trip")
