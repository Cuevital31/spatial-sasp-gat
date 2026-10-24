# Spatial SASP-GAT — code

Code accompanying *Spatial SASP-GAT: Dual-Graph Attention Networks for Masked SASP Gene Reconstruction in Spatial Transcriptomics* (Functional & Integrative Genomics).

> **Status of this release.** `src/` contains the model, masking, availability-aware loss, training, evaluation, two-hop masking, baselines and the analysis modules for masking scenarios, PPI controls, matched control panels, structure preservation, HD-gap, Moran's I and the partition audit. Notebooks `00`–`10` re-evaluate released checkpoints and regenerate the reported tables from stored results; each regenerated table is compared with the stored revision table (see `docs/source_cell_mapping.md` for status, tolerances and what is not covered, e.g. no re-training and no raw-data package construction).

## What the model does
A masked-gene reconstruction model on 54 SASP genes (`data_tables/sasp54_genes_canonical_order.csv`). A spot-level 2-layer GATv2 encoder runs on the spatial spot graph; a gene-level 2-layer GATv2 encoder runs on the STRING v12.0 functional network (54 nodes, 680 undirected edges, combined score ≥ 0.4). Edge weights are stored for reference and are **not** used by the model. PPI nodes are indexed by the SASP54 canonical order (the STRING `node_list` order is independent and must not be used).

## Availability-aware protocol
A slide–gene pair can be unmeasured (4 pairs: `data_tables/availability_unmeasured.csv`). The model input hides *random-masked OR unmeasured* entries; the loss and all reported scores include *random-masked AND measured* entries only. Zero-filling of unmeasured entries is only a data-representation device and is kept separate from this exclusion.

## Layout
```
src/sasp_gat/   model.py masking.py availability.py ppi.py data.py training.py evaluation.py twohop.py baselines.py config.py
                scenarios.py group_eval.py ppi_controls.py tables.py panels.py structure.py hd_gap.py moran.py partition.py verify.py
notebooks/      00_setup_and_data  01_train_full  02_evaluate_main_and_twohop  03_ablations_and_baselines
                04_masking_scenarios_and_groups  05_ppi_controls_per_gene_final_tables  06_matched_control_panels
                07_structure_preservation  08_hd_gap_and_qc  09_morans_I  10_partition_audit  (run 04 before 05/06/08)
data_tables/    sasp54 gene order, PPI edges, slide partition, unmeasured pairs
tests/          synthetic-data unit tests (pytest)
docs/           source_cell_mapping.md  (manuscript result -> original cell -> module/notebook, with status)
```

## Data
* **Checkpoints:** the three full-model checkpoints (seeds 42/123/2026, ≈457 KB each) are included under `checkpoints/` (full SHA-256 in `checkpoints/SHA256SUMS.txt`; the 16-character prefixes are checked by `notebooks/02`). They are loaded with `weights_only=True`.
* **Packaged dataset:** `MODEL_READY_DATASET_v1_2026-09-26.pkl` (76 slides derived from HEST-1k, ≈75 MB, SHA-256 in `src/sasp_gat/config.py`) is **not** in the repository; its archive location will be added when available.
* **`notebooks/00_setup_and_data` is not a raw-data pipeline.** It loads the already packaged dataset and runs consistency checks (hash, splits, gene order, PPI edges, unmeasured pairs). The pipeline that builds the 76-slide package from raw HEST-1k data (Start cells 76–104) has **not yet been ported**; this is a known reproducibility gap.
* **Security note:** the package is a Python pickle. Only load a pickle whose SHA-256 matches `DATASET_SHA256`, and never unpickle files from untrusted sources. Checkpoints are loaded with `torch.load(..., weights_only=True)`.
* **Software versions:** the notebooks print the installed torch / torch_geometric versions; `requirements.txt` lists the versions of the Colab runtime on which the full-model and ablation results were verified (pandas 2.2.3); notebooks 07–08 were also run under pandas 3.1.0 with identical conclusions, and the comparison helper supports both.
* Third-party data (STRING, HEST-1k and the underlying studies) remain under their own licences and terms; please cite and comply with them.

## Splits
76 slides: train 38, validation 13, locked test 13, external Visium HD 12 (a platform-robustness cohort, not an independent patient-level validation). `data_tables/slide_partition.csv` lists the assignment and the evidence level of each leakage group. TENX23 and TENX28 were excluded as targeted gene-panel assays (see the manuscript for the exclusion criterion).

## What this repository does and does not reproduce
**Included:** model, masking, availability-aware loss, evaluation, two-hop masking, training-free baselines, the analysis modules and notebooks `00`–`10`, the three full-model checkpoints, and small data tables. **Not included, and needed for most notebooks:** the model-ready dataset package (`MODEL_READY_DATASET_v1_2026-09-26.pkl`, ≈75 MB, archive location to be added), the raw HEST `.h5ad` files, the nine ablation checkpoints, the six PPI-control checkpoints, the 246 panel-run checkpoints, the stored revision tables that the notebooks compare against, the 125 stored reconstruction profiles and the locked protocol JSON files (`package9_protocol_v1.json`, `protocol_2hop_v1.json`). Without them only the architecture, the tests and the CSV-level checks in notebook `00` run.

**What "verified" means.** A notebook marked verified regenerated its tables from the inputs above and matched the tables stored during the revision (the tolerances are listed in `docs/source_cell_mapping.md`). This shows that the released code reproduces the authors' own earlier outputs; it is not an independent re-derivation of the results.

**Not covered by this release**
* STAGATE (adapted) and Steamboat (adapted) comparators (Tables 4 and 7): adapter code and result tables are not part of this repository.
* Two-hop masking: only the full model is evaluated here; the other methods of the two-hop comparison (expression-only, expression+PPI, expression+spatial, baselines) and the selection of the two-hop probability are not ported.
* Training of the edge-free / rewired PPI-control models and of the 246 matched-panel runs; these checkpoints are re-evaluated, not re-trained. Notebook `01` trains the full model and the three ablation variants only.
* Re-training in general: not re-run for this release; GPU training is not bit-reproducible, so a re-trained checkpoint is not expected to match the released one exactly, and how close it comes has not been tested.
* Construction of the 76-slide package from raw HEST data; Figures 3–5 (see below); the 22B SD-audit cell.
* TENX128/TENX155 (both external HD): about 98% of their spot count vectors are identical (2241 of 2279 TENX128 vectors occur in TENX155) but are assigned to different barcodes and spatial coordinates; whether this is the same section processed twice, a duplicate record or an assignment error was not determined. No other pair of the 76 slides shares an identical count vector, and no pair from different splits does (exact-match audit, notebook `08`). The depth-QC output of notebook `08` is printed, not stored.

**Environment.** The notebooks were run on Colab (Tesla T4; torch 2.11.0, torch_geometric 2.8.1, scikit-learn 1.6.1, pandas 2.2.3 or 3.1.0). The 24 unit tests need torch and torch_geometric and were run on Colab and, independently, on CPU with other library versions (torch 2.14.1, numpy 2.4.4, pandas 3.0.2); `pytest` from the repository root works (see `tests/conftest.py`). Tolerances for comparing GPU results are given in `docs/source_cell_mapping.md`.

**Random numbers in training.** Training masks are drawn at every step from PyTorch's global CPU random number generator, which is seeded with `torch.manual_seed(seed)` (seeds 42, 123, 2026) before the model is built. Validation, test and HD masks are generated deterministically from string tags (`<slide>_<role>_<i>`). GPU kernels are not deterministic, so re-training can give different checkpoints.

* The two-hop masking is an inference-only stress test on the existing checkpoints; its masking distribution differs from the 25% Bernoulli masking used in training.
* Tests: `pip install pytest` and run `pytest -q` from the repository root (synthetic data only).

## Figure script
`scripts/figures_structure_as_run_in_revision.py` is the original Colab cell that produced Figures 3–5, kept verbatim for transparency. It was **not re-run** for this release and depends on the original Google-Drive layout (`/content/drive/MyDrive/New-SASP-idea/...`); other users must edit the paths. All numeric results behind the figures are regenerated by `notebooks/07_structure_preservation.ipynb`.

## Running the notebooks
Each notebook starts by deleting any older unpacked copy of the repository and unpacking the zip given in `REPO_ZIP` (or clone the repository to `/content/spatial-sasp-gat` instead). Reset the runtime when you replace the zip, otherwise Python keeps the old modules in memory (the notebook prints a warning in that case). The model-ready dataset package (≈75 MB) is not part of this repository; the notebooks that need it expect it under `BASE` (see *Data*).

## Licence
MIT (code). Data and third-party resources: see above.
