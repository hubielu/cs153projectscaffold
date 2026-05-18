from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from src.data.synthetic import SyntheticConfig, SyntheticIVDataset, in_channels, metadata_dim
from src.models.encoder_decoder import FET2DEncoderDecoder

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=19700101)
    return p.parse_args()

def main(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = SyntheticConfig(**ckpt["cfg"])
    torch.manual_seed(args.seed)
    dataset = SyntheticIVDataset(cfg)
    n_dev = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_dev
    _, dev_set = random_split(dataset, [n_train, n_dev], generator=torch.Generator().manual_seed(args.seed))
    model = FET2DEncoderDecoder(
        in_channels=in_channels(cfg),
        num_points=cfg.num_points,
        d_metadata=metadata_dim(cfg),
        num_params=cfg.num_params,
        d_latent=ckpt["args"]["d_latent"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(dev_set, batch_size=128, shuffle=False)
    all_pred, all_true, all_mat = [], [], []
    with torch.no_grad():
        for batch in loader:
            iv = batch["iv"].to(args.device)
            md = batch["metadata"].to(args.device)
            p_hat, _, _ = model(iv, md)
            all_pred.append(p_hat.cpu().numpy())
            all_true.append(batch["params"].numpy())
            all_mat.append(md.cpu().numpy()[:, : cfg.num_materials].argmax(axis=1))
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    mat = np.concatenate(all_mat, axis=0)
    overall_rmse = np.sqrt(((pred - true) ** 2).mean(axis=0))
    print("Overall per-parameter RMSE:")
    for i, r in enumerate(overall_rmse):
        print(f"  param[{i}]  {r:.4f}")
    print("\nPer-material per-parameter RMSE:")
    for m in range(cfg.num_materials):
        mask = mat == m
        if mask.sum() == 0:
            continue
        rmse = np.sqrt(((pred[mask] - true[mask]) ** 2).mean(axis=0))
        print(f"  material {m} (n={mask.sum()}): " + " ".join(f"{r:.3f}" for r in rmse))


if __name__ == "__main__":
    main(parse_args())
