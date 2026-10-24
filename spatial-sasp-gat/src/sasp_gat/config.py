"""Constants shared by all modules. Values are taken from the final (revision) pipeline."""

N_GENES = 54            # SASP54 panel (canonical order: data_tables/sasp54_genes_canonical_order.csv)
EPS = 1e-8              # denominator stabiliser in the PPI context pooling
MASK_RATIO = 0.25       # primary random-masking ratio (training and primary evaluation)

SEEDS = (42, 123, 2026)
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 800
PATIENCE = 30
N_EVAL_MASKS = 3        # deterministic masks per slide for validation / test / HD evaluation

# SHA-256 of the packaged dataset (MODEL_READY_DATASET_v1_2026-09-26.pkl)
DATASET_SHA256 = "692ed03726c1877bc72b56d53ee5af8046f70caffca3ae26c96d476587494bc2"

# Expected split sizes (slides): train / val / test / external HD
SPLIT_COUNTS = (38, 13, 13, 12)
N_PPI_UNDIRECTED_EDGES = 680
N_MODEL_PARAMETERS = 114198   # full dual-graph model

# Two-hop (receptive-field) masking protocol, locked before any metric was computed
TWOHOP_P_CENTRE = 0.03
TWOHOP_PROTOCOL_SHA256_16 = "42437a308bce07be"
