# Manuscript result → original cell → release module

`Rev1` = Revision-1.ipynb, `Meth` = New-SASP-methods.ipynb, `Start` = New_SASP-start.ipynb. Cell numbers are 0-based notebook indices.
Status: **ported** = logic moved to `src/`; **verified** = regenerated and compared with the tables stored during the revision (shows that the code reproduces the authors' own earlier outputs, not an independent re-derivation); **to port** = original cell(s) identified, not yet moved.

| Manuscript item | Original cell(s) | Release location | Status |
|---|---|---|---|
| Model architecture | Meth 9 (up to `# 5) SMOKE TEST`) | `src/sasp_gat/model.py` | ported (AST-identical; 114,198 parameters confirmed on synthetic build) |
| Random masking | Meth 9 `make_mask` | `masking.py` | ported (AST-identical) |
| Availability-aware mask/loss | Rev1 9, 22 | `availability.py` | ported (functions AST-identical) |
| Training / early stopping | Rev1 9, 10 | `training.py` | ported (explicit `avail` argument instead of global); validation-loss evaluation path **verified** (see below); the training loop itself is not re-run |
| Full model, locked test + HD (Table 4) | Rev1 11 | `evaluation.py`, `notebooks/02` | ported; **verified 2026-10-24** for the three full checkpoints (see below) |
| Two-hop masking (Sec 2.6, Table 7, S2) | Rev1 93, 94 | `twohop.py`, `notebooks/02` | ported; **verified 2026-10-24** for the full model (protocol hash, per-slide graph/availability ids, MSE); other methods in Rev1 94 to port |
| Baselines A/B (3 scorings), NeighborMean, IDW 1-hop (Table 4) | Rev1 14, 19 | `baselines.py`, `notebooks/03` | ported; **verified 2026-10-24** (see log). Ring2Mean (training-free) is in `scenarios.py` and verified through notebooks 04–07 |
| Ablations (Table 3), availability-aware checkpoints | Rev1 12 (training), 13 (evaluation) | `training.py` (`01` with VARIANTS), `evaluation.py`, `notebooks/03` | ported; evaluation **verified 2026-10-24** (see log; training loop not re-run). Old-protocol checkpoints are not part of the release |
| Group-level evaluation (Table 6, S2) | Rev1 15, 25, 26 | `group_eval.py`, `notebooks/04` | ported; **verified 2026-10-24** (log 3) (compared with stored `revision_tables_2026-10-18` in notebook 04) |
| Masking scenarios 10/25/50, neighbourhood (Table 7, S2) | Rev1 25, 26 | `scenarios.py`, `notebooks/04` | ported; **verified 2026-10-24** (log 3) (compared with stored `revision_masking_robustness_2026-10-19`) |
| PPI graph validation, controls (Table 5, S4) | Rev1 27–29 | `ppi_controls.py`, `notebooks/05` | ported; **verified 2026-10-24** (log 3). The physical-subnetwork check is a live STRING API query and may drift from the stored result |
| Per-gene MSE, cancer-type Full vs Ring2Mean (Tables 8–10, S1) | Rev1 36, 37, 39 | `tables.py`, `notebooks/05` | ported; **verified 2026-10-24** (log 3) |
| Matched control panels (S3) | Rev1 45–64 | `panels.py`, `notebooks/06` | panel construction, training-free evaluation, re-evaluation of the stored trained panel runs and spatial contribution ported; **verified 2026-10-24** (log 3). Panel *training* is not re-run (documentation only: `panels.train_and_eval`) |
| Structure preservation (Sec 3.10, Table 12, S6, Figs 3–5) | Rev1 68–70, 73 | `structure.py`, `notebooks/07`, `scripts/figures_structure_as_run_in_revision.py` | metrics ported; **verified 2026-10-24** (log 3). The figure script is the original cell, not re-run |
| HD-gap / depth / coordinate scale (Table 10, S5) | Rev1 78–80, 88–91, 96 | `hd_gap.py`, `notebooks/08` | ported; **S5 tables verified 2026-10-24** (log 3); depth-QC / TENX128-155 printed values still to be compared with the manuscript. Depth-QC and TENX128/155 outputs have no stored CSV (printed only) |
| Moran's I (Sec 3.9) | Rev1 82, 83 | `moran.py`, `notebooks/09` | ported; **verified 2026-10-24** (log 3) (compared with stored `p7m_*`, `p7m2_*`) |
| Partition table and leakage audit (Table 1) | Rev1 85 | `partition.py`, `notebooks/10` | ported (read-only audit); **verified 2026-10-24** (log 3) |
| Dataset package construction | Start 76–104 | – | to port (optional; the packaged file is the supported input) |

Superseded, diagnostic and internal cells of the revision notebooks are not part of the release.

## Verification log (full-model checkpoints only)
Run on Colab (torch 2.11.0+cu130, torch_geometric 2.8.1, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3), 2026-10-24, notebooks `00` and `02`.
* Checkpoint SHA-256 prefixes, metadata and validation loss reproduced for seeds 42/123/2026 (e.g. seed 42: 0.196794 recomputed = saved).
* Locked test, availability-aware pooled MSE per seed 0.173005 / 0.169362 / 0.172484 (mean 0.1716, SD 0.0020, ddof=1); 1,789,306 scored of 1,791,286 masked targets (1,980 unmeasured excluded).
* External HD per seed 0.421227 / 0.417067 / 0.423972 (mean 0.4208, SD 0.0035); 1,058,104 scored targets, none excluded.
* Two-hop (p_c = 0.03, protocol sha256[:16] 42437a308bce07be): test 0.244996 / 0.247204 / 0.257259 (mean 0.2498, SD 0.0065), HD 0.495701 / 0.497450 / 0.496172 (mean 0.4964, SD 0.0009); 214,924 (test) and 127,151 (HD) scored centres.
Scope: this verifies the full dual-graph model only. Ablations, PPI controls, masking scenarios, panels, structure preservation and HD-gap analyses are not covered.

## Verification log 2 (ablations and training-free baselines), 2026-10-24, notebook `03`
* Nine availability-aware ablation checkpoints: validation loss reproduced (|diff| < 1e-4) and locked test/HD pooled MSE equal to the stored seed-level table (max |diff| 7.4e-11 test, 1.1e-10 HD). Means ± SD over 3 seeds (ddof=1), test / HD: expression-only 0.2283 ± 0.0009 / 0.4889 ± 0.0012; expression+spatial 0.1793 ± 0.0017 / 0.4271 ± 0.0068; expression+PPI 0.2296 ± 0.0008 / 0.4897 ± 0.0007.
* Baselines A and B (three scoring definitions): original baseline CSV reproduced exactly; stored table5 reproduced (max |diff| 5.6e-17).
* NeighborMean and 1-hop IDW (anndata 0.13.4 for coordinates): pooled MSE test 0.16342, HD 0.48773 for both methods (they coincide to the reported precision), equal to the stored summary (max |diff| 5.6e-17); sensitivity check: hidden inputs have no effect (≤ 1.8e-15), visible inputs do.


## Comparison tolerances used (np.allclose, rtol / atol)
| Tables | rtol / atol | Reason |
|---|---|---|
| CPU-only regenerations (Moran's I, panels, partition) | 1e-9 / 1e-12 (CSV round trip) | deterministic |
| Default (GPU evaluations, notebooks 04, 05, 08) | 1e-6 / 1e-9 | observed max differences up to 3e-4 in per-seed sums of squares are relative <~1e-6 |
| GPU gene-level panel tables (06) | 1e-5 / 1e-4 | float32; RI is a ratio and moves by up to 2e-5 where the Ring2Mean denominator is tiny |
| Metrics from regenerated profiles (07) | 1e-5 / 1e-5 for continuous metrics; ARI differences are listed explicitly | KMeans-based ARI is not robust to ~1e-6 profile noise |
Note: the Table-4/7 rows for STAGATE (adapted) and Steamboat (adapted) and the non-full methods of the two-hop comparison are not part of this release.

## Not covered by this release
* STAGATE (adapted) and Steamboat (adapted) comparators (Tables 4 and 7): no code or result tables in this repository.
* Two-hop comparison for methods other than the full model, and the selection of p_c (Rev1 93/94).
* Training of the PPI-control models (edge-free, rewired) and of the 246 panel runs.
* Construction of the 76-slide package from raw HEST data (Start 76–104); the packaged file is the supported input.
* Re-training: GPU training is not bit-reproducible; checkpoints are re-evaluated, not re-trained (also for the matched-panel runs).
* Superseded/diagnostic cells, and the 22B SD-audit cell.
* Figures 3–5: the original plotting cell is included verbatim and was not re-run.

## Verification log 3 (batch notebooks 04–07), 2026-10-24, Colab Tesla T4, torch 2.11.0+cu130
* **04 (scenarios, group tables):** 9 scenario tables and 14 group tables match the stored ones (max |diff| 1.7e-4 for per-seed long tables = GPU float noise; others <= 1.6e-9). 25% arm: test full 0.17162, HD full 0.42076.
* **05 (PPI controls, per-gene, slide/cancer, final tables):** live STRING functional@400 network identical to the stored 680-edge network (physical subnetwork: 126 edges, subset of it); regenerated rewired graphs identical (hash) to those of the control checkpoints; all 9 compared tables MATCH (max |diff| 7.4e-5).
* **06 (matched panels):** train gene statistics, 200 md + 200 mdv panels, training-free evaluation, expression-only re-evaluation and spatial-contribution tables match. Expression+spatial gene-level table: relative differences of ss/mse/NMSE <= 5.5e-8; `RI` (a ratio) differs by up to 1.8e-5 absolute in 33 control-gene rows (SASP54 rows: none), consistent with float32 non-determinism. Comparison tolerance for GPU tables: rtol 1e-5, atol 1e-4.
* **07 (structure preservation):** metrics computed from the stored profiles reproduce both stored tables exactly (difference 0; observed/Ring2Mean profiles identical, 25/25). Metrics from model profiles regenerated on a Tesla T4 (profile difference <= 1.43e-6): continuous metrics (PCA R2, score Pearson/Spearman) differ by <= 9e-7, all counts and all 150 full-minus-Ring2Mean signs are unchanged, medians agree at 4 decimals. ARI depends on KMeans labels and is not robust to this noise: in a first run no value changed at 4 decimals (max 1.9e-6); in a second run one slide/seed (HD TENX162, seed 2026, k=5) moved by 1.5e-3 (seed mean and full-minus-Ring2Mean by 4.9e-4; HD k=5 q75 of full-minus-Ring2Mean 0.0507 -> 0.0504). With scikit-learn 1.6.1 (Colab default) the TENX162 difference appeared in both repeated runs with the final zip (sha256[:16] a2363495aaebcb38); the first run used a scikit-learn that had been re-installed in the session (version not recorded), and showed no such difference, so a version effect is possible but not established. Reported numbers come from the stored profiles; regenerating profiles on a GPU cannot be guaranteed to reproduce ARI (k=5) to the fourth decimal.
* Comparison helper fixes made during verification (no change to any analysis): categorical/numeric id columns compared as strings on both sides; pandas-3 string dtype handled via `pd.api.types.is_float_dtype`.
* **08 (HD gap, QC):** the four `p8_*` tables match (max |diff| 1.0e-9; pandas 3.1.0 runtime). Depth QC (Spearman between slide MSE and median total counts / genes detected: TEST -0.835 / -0.797, HD -0.529 / -0.529) and the TENX128/TENX155 barcode comparison (2279 common barcodes, 2 only in TENX155, 7.6% identical rows on the common barcodes) have no stored CSV; they are printed and still to be compared with the manuscript wording. HD pooled MSE (x1e-3, 12 slides, rand25): full 420.76, expression+spatial 427.06, Ring2Mean 436.46; for nbhd05 Ring2Mean (441.26) is below full (447.12).
* Environment note: results were obtained under pandas 2.2.3 (04-06) and pandas 3.1.0 (07-08); the comparisons are insensitive to the version. Do not re-install scikit-learn/networkx inside a running Colab kernel (it replaced pandas under the kernel once).
* **08 addendum (TENX128/TENX155, 2026-10-24):** compared by count vector: shapes (2279, 18085) and (2281, 18085), identical gene order, 2279 common barcodes (2 only in TENX155); 2241 of 2279 TENX128 count vectors also occur in TENX155 (2068 under a different barcode, 173 under the same barcode); spatial coordinates differ on the common barcodes (max |diff| 2.4e4 px); median total counts 1561 and median genes detected 1025 are identical in both slides, which is explained by the shared vectors. Exact-duplicate audit over all 76 slides (rows with >= 50 nonzero genes): the only pair sharing vectors is TENX128-TENX155 (2133 shared vectors, 98.3% / 98.2%); no pair from different splits shares any. Not determined: whether the two records are the same section processed twice, a duplicate HEST record or an assignment error.
* **09 (Moran's I):** all 9 compared tables match (5 `p7m_*` + 4 `p7m2_*`, max |diff| 8.9e-16; CPU only). Median Moran's I of SASP54 vs md control panels: TEST 0.137 vs 0.093, HD 0.060 vs 0.047; gene-level Spearman(Moran's I, spatial gain) is positive for SASP54 (TEST 0.52 NMSE / 0.46 RI; HD 0.58 / 0.45) and for control panels (md/test 0.60).
* **10 (partition / leakage audit):** 15/15 checks pass: 76 slides (38/13/13/12); each of 38 leakage groups and each of 11 resolved biological IDs (18 slides) lies in exactly one split; TENX23 and TENX28 are the only slides removed from the 78-slide file; no conflicts between split v3 and mapping v2; package pickle (sha256 692ed037...) agrees with v3. External HD: 12 slides, 12 leakage groups, 0 resolved biological IDs (group-level separation only). One study (CRC high-definition profiling) contributes 1 COAD slide to validation and 3 COAD slides to external HD (different leakage groups).
* Notebooks 04-10 were run on Colab and compared with the stored revision results (see above); the depth-QC and TENX128/155 outputs of 08 are printed values without stored reference.
