"""Training with availability-aware loss and early stopping (logic unchanged from the revision pipeline).

Order of RNG consumption is preserved: set torch.manual_seed(seed) first, then build the model.
Training masks are drawn from the global CPU RNG at every step (one fresh 25% mask per slide per epoch);
validation uses 3 deterministic masks per slide f'{slide}_val_{i}'. The best epoch is the lowest
availability-aware validation MSE.
"""
import copy
from pathlib import Path

import torch

from .availability import forward_av, masked_mse_loss_av, score_mask
from .config import MASK_RATIO, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, N_EVAL_MASKS
from .masking import make_mask


def train_one_epoch_av(model, samples, avail, ppi_edge_index, optimizer, device, mask_ratio=MASK_RATIO):
    model.train()
    total_loss, total_scored = 0.0, 0
    for sid, data in samples.items():
        x = data.x.to(device); edge_index = data.edge_index.to(device); av = avail[sid].to(device)
        mask = make_mask(x.shape[0], mask_ratio=mask_ratio, device=device)
        optimizer.zero_grad()
        recon = forward_av(model, x, edge_index, mask, av, ppi_edge_index if model.use_ppi else None)
        loss = masked_mse_loss_av(recon, x, mask, av)
        loss.backward(); optimizer.step()
        n_scored = int(score_mask(mask, av).sum().item())
        total_loss += loss.item() * n_scored; total_scored += n_scored
    return total_loss / max(total_scored, 1)


@torch.no_grad()
def evaluate_multimask_av(model, samples, avail, ppi_edge_index, device, role="val",
                          n_masks=N_EVAL_MASKS, mask_ratio=MASK_RATIO):
    model.eval()
    total_loss, total_scored = 0.0, 0
    for sid, data in samples.items():
        x = data.x.to(device); edge_index = data.edge_index.to(device); av = avail[sid].to(device)
        for i in range(n_masks):
            mask = make_mask(x.shape[0], mask_ratio=mask_ratio,
                             deterministic_seed_str=f"{sid}_{role}_{i}", device=device)
            recon = forward_av(model, x, edge_index, mask, av, ppi_edge_index if model.use_ppi else None)
            loss = masked_mse_loss_av(recon, x, mask, av)
            n_scored = int(score_mask(mask, av).sum().item())
            total_loss += loss.item() * n_scored; total_scored += n_scored
    return total_loss / max(total_scored, 1)


def train_with_early_stopping_av(model, train_samples, val_samples, avail, ppi_edge_index, device,
                                 lr=LR, weight_decay=WEIGHT_DECAY, max_epochs=MAX_EPOCHS,
                                 patience=PATIENCE, log_every=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, best_epoch, no_improve, history = float("inf"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch_av(model, train_samples, avail, ppi_edge_index, optimizer, device)
        va = evaluate_multimask_av(model, val_samples, avail, ppi_edge_index, device, role="val")
        history.append((epoch, tr, va))
        if va < best_val:
            best_val, best_state, best_epoch, no_improve = va, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            no_improve += 1
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(f"  epoch {epoch:4d} | train={tr:.4f} | val={va:.4f} | best_val={best_val:.4f} @ {best_epoch}")
        if no_improve >= patience:
            print(f"  Early stopping @ epoch {epoch} (patience={patience})")
            break
    return best_state, best_val, best_epoch, history


def vkey(name):
    """Checkpoint file-name key of a variant (e.g. 'expression+PPI' -> 'expression_PPI')."""
    return name.replace("+", "_").replace("-", "_")


def checkpoint_path(out_dir, variant, seed):
    """Refuses to overwrite an existing file."""
    p = Path(out_dir) / f"{variant}_avail_seed{seed}.pt"
    assert not p.exists(), f"refusing to overwrite: {p}"
    return p


def save_checkpoint(path, best_state, variant, use_spatial, use_ppi, seed, best_val, best_epoch,
                    total_epochs, logged):
    torch.save({"state_dict": best_state, "use_spatial": use_spatial, "use_ppi": use_ppi,
                "variant_name": variant, "seed": seed, "best_val_loss": best_val,
                "best_epoch": best_epoch, "total_epochs_run": total_epochs,
                "availability_aware": True, "avail_log": logged}, path)
