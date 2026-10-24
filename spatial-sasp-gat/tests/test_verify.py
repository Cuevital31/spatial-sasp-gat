import tempfile, numpy as np, pandas as pd
from sasp_gat.verify import compare_csv


def _stored(df):
    f = tempfile.mktemp(suffix=".csv"); df.to_csv(f, index=False); return f


def base():
    return pd.DataFrame(dict(scn=["a"] * 6, method=["full", "Ring2Mean", "Baseline A"] * 2, seed=[42, 42, 42, 123, 123, 123],
                             v=[1., 2., 3., 4., 5., 6.]))


def test_row_order_and_dtypes_do_not_matter():
    old = base(); f = _stored(old)
    new = old.sample(frac=1, random_state=3).reset_index(drop=True)
    new["method"] = pd.Categorical(new.method, ["Baseline A", "Ring2Mean", "full"], ordered=True)   # categorical, different category order
    new["seed"] = new.seed.astype(str)                                                              # numeric ids stored as text
    ok, d, msg = compare_csv(new, f); assert ok and d == 0.0, msg


def test_numeric_id_sort_order_regression():
    # seeds 42/123/2026 sort differently as numbers and as text: both sides must be aligned the same way
    old = pd.DataFrame(dict(seed=[42, 123, 2026], v=[1., 2., 3.])); f = _stored(old)
    new = old.assign(seed=old.seed.astype(str)).iloc[::-1]
    assert compare_csv(new, f)[0]


def test_pandas_string_dtype_is_accepted():
    old = base(); f = _stored(old); new = old.copy(); new["method"] = new.method.astype("string")
    assert compare_csv(new, f)[0]


def test_real_differences_are_detected():
    old = base(); f = _stored(old)
    new = old.copy(); new.loc[0, "v"] += 1e-3; ok, d, msg = compare_csv(new, f); assert not ok and "v" in msg
    new = old.copy(); new.loc[0, "method"] = "other"; assert not compare_csv(new, f)[0]
    assert not compare_csv(old.iloc[:-1], f)[0]                                                    # shape
    assert not compare_csv(old.rename(columns={"v": "w"}), f)[0]                                   # columns


def test_nan_and_tolerance():
    old = base(); old.loc[2, "v"] = np.nan; f = _stored(old)
    new = old.copy(); new.loc[0, "v"] += 5e-7
    assert compare_csv(new, f, rtol=1e-6, atol=1e-9)[0] is True
    assert compare_csv(new, f, rtol=1e-9, atol=1e-12)[0] is False


def test_duplicate_keys_are_flagged():
    old = pd.DataFrame(dict(k=["a", "a"], v=[1., 2.])); f = _stored(old)
    ok, d, msg = compare_csv(old, f); assert "duplicated key" in msg
