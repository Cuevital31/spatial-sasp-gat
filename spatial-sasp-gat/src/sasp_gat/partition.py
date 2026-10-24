"""Partition / leakage audit behind Table 1 (Revision-1 cell 11B v1.1). Read-only: writes nothing, stops at the first failed check.
Two separate questions: (1) are the known biological sample IDs spread over splits, (2) do the conservative leakage groups
(sub-series prefix / title similarity) cross splits. Passing (2) does not prove that slides with unresolved identity come from different patients."""
import hashlib
import pickle

import pandas as pd

ORDER = ["train", "val", "test", "external_hd"]
COLS = ["slide_id", "cancer", "platform", "cohort", "study", "biological_sample_id", "leakage_group_id", "confidence_level", "split_assignment"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def partition_audit(F3, F78, FM, PKL, verbose=True):
    log = []

    def ok(name, cond, info=""):
        log.append((name, bool(cond)))
        if verbose:
            print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f"  [{info}]" if info else ""))
        if not cond:
            raise AssertionError(f"check failed: {name} [{info}]")
    A, O, M = pd.read_csv(F3), pd.read_csv(F78), pd.read_csv(FM)
    assert set(COLS) <= set(A.columns) and set(COLS) <= set(O.columns)
    assert {"slide_id", "cancer", "platform", "cohort", "study", "biological_sample_id", "leakage_group_id", "confidence_level"} <= set(M.columns)
    ok("v3: 76 slides, unique slide_id", len(A) == 76 and A.slide_id.is_unique)
    ok("v3: split sizes train 38 / val 13 / test 13 / external_hd 12", A.split_assignment.value_counts().reindex(ORDER).tolist() == [38, 13, 13, 12])
    ok("v3: no missing values (except biological_sample_id)", A[[c for c in COLS if c != "biological_sample_id"]].notna().all().all())
    ok("v3: Visium HD only in external_hd and vice versa", ((A.platform == "Visium HD") == (A.split_assignment == "external_hd")).all())
    ok("v3: cohort label consistent with platform", ((A.cohort == "SECONDARY_HD_ROBUSTNESS") == (A.platform == "Visium HD")).all())
    g_n = A.groupby("leakage_group_id").split_assignment.nunique(); ok("every leakage_group_id in exactly ONE split", (g_n == 1).all(), f"{len(g_n)} groups")
    b = A.dropna(subset=["biological_sample_id"]); b_n = b.groupby("biological_sample_id").split_assignment.nunique()
    ok("every resolved biological_sample_id in exactly ONE split", (b_n == 1).all(), f"{len(b_n)} ids, {len(b)} slides")
    ok("slides with a resolved ID do not contradict the leakage group", (b.groupby("biological_sample_id").leakage_group_id.nunique() == 1).all())
    dropped = sorted(set(O.slide_id) - set(A.slide_id)); ok("slides removed from the 78 file are exactly TENX23 and TENX28", dropped == ["TENX23", "TENX28"] and set(A.slide_id) <= set(O.slide_id))
    C = A.merge(O, on="slide_id", suffixes=("", "_old"), validate="one_to_one")
    ok("cancer/platform/cohort/study/split identical to the 78 file on the 76 common slides", all((C[c] == C[c + "_old"]).all() for c in ["cancer", "platform", "cohort", "study", "split_assignment"]))
    ok("mapping v2: unique slide_id (78 rows)", len(M) == 78 and M.slide_id.is_unique)
    ok("every v3 slide appears exactly once in the mapping", set(A.slide_id) <= set(M.slide_id) and len(M[M.slide_id.isin(A.slide_id)]) == 76)
    m_out = sorted(set(M.slide_id) - set(A.slide_id)); ok("mapping slides outside the package are exactly TENX23 and TENX28", m_out == ["TENX23", "TENX28"])
    MM = A.merge(M, on="slide_id", how="left", suffixes=("", "_map"), validate="one_to_one"); conflicts = {}
    for c in ["cancer", "platform", "cohort", "study", "biological_sample_id", "leakage_group_id", "confidence_level"]:
        a_, b_ = MM[c], MM[c + "_map"]; bad = ~((a_ == b_) | (a_.isna() & b_.isna())); conflicts[c] = MM.slide_id[bad].tolist()
    if verbose:
        print("  conflicts between split v3 and mapping v2 (never silently overwritten):", {c: len(v) for c, v in conflicts.items()})
    with open(PKL, "rb") as f:
        obj = pickle.load(f)
    DS = obj["pyg_dataset"]; ps = {s: DS[s].split for s in DS}
    ok("PKL slide set = v3 (76) and DS.split == v3 split", set(ps) == set(A.slide_id) and all(ps[s] == sp for s, sp in zip(A.slide_id, A.split_assignment)))
    ok("PKL leakage_group_id/cancer/platform = v3 (76/76)", all(str(DS[s].leakage_group_id) == g and str(DS[s].cancer) == c and str(DS[s].platform) == p
                                                               for s, g, c, p in zip(A.slide_id, A.leakage_group_id, A.cancer, A.platform)))
    rows = []
    for sp in ORDER:
        d = A[A.split_assignment == sp]; cc = d.cancer.value_counts()
        rows.append(dict(split=sp, n_slides=len(d), n_leakage_groups=int(d.leakage_group_id.nunique()), n_resolved_biological_ids=int(d.biological_sample_id.nunique()),
                         n_slides_with_resolved_id=int(d.biological_sample_id.notna().sum()), platform="; ".join(f"{k} {v}" for k, v in d.platform.value_counts().items()),
                         cancer_types="; ".join(f"{k} {v}" for k, v in cc.items()), n_studies=int(d.study.nunique())))
    T1 = pd.DataFrame(rows)
    T2 = pd.crosstab(A.split_assignment, A.cancer).reindex(ORDER).fillna(0).astype(int).reset_index().rename(columns={"split_assignment": "split"})
    T3 = pd.crosstab(A.split_assignment, A.confidence_level).reindex(ORDER).fillna(0).astype(int).reset_index().rename(columns={"split_assignment": "split"})
    T4 = (A.groupby(["split_assignment", "study"]).agg(n_slides=("slide_id", "size"), cancers=("cancer", lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().items())))
          .reset_index().rename(columns={"split_assignment": "split"}))
    T4["split"] = pd.Categorical(T4.split, ORDER, ordered=True); T4 = T4.sort_values(["split", "n_slides"], ascending=[True, False]).reset_index(drop=True); T4["split"] = T4.split.astype(str)
    SL = A[COLS].copy(); SL["split_assignment"] = pd.Categorical(SL.split_assignment, ORDER, ordered=True)
    SL = SL.sort_values(["split_assignment", "cancer", "slide_id"]).reset_index(drop=True); SL["split_assignment"] = SL.split_assignment.astype(str)
    EX = M[M.slide_id.isin(m_out)].reset_index(drop=True)
    miss = pd.DataFrame(dict(split=ORDER, cancer_types_missing=[", ".join(sorted(set(A.cancer) - set(A[A.split_assignment == s].cancer))) or "-" for s in ORDER]))
    return dict(T1=T1, T2=T2, T3=T3, T4=T4, slides=SL, excluded=EX, missing_cancers=miss, n_checks=len(log), conflicts=conflicts)
