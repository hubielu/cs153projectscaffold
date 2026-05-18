from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.synthetic import SyntheticConfig, SyntheticIVDataset, in_channels, metadata_dim
from src.models.encoder_decoder import FET2DEncoderDecoder

def reconstruction_error(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            iv = batch["iv"].to(device)
            md = batch["metadata"].to(device)
            _, iv_hat, _ = model(iv, md)
            err = ((iv_hat - iv) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errs.append(err)
    return np.concatenate(errs)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()

def main(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = SyntheticConfig(**ckpt["cfg"])
    dataset = SyntheticIVDataset(cfg)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model = FET2DEncoderDecoder(
        in_channels=in_channels(cfg),
        num_points=cfg.num_points,
        d_metadata=metadata_dim(cfg),
        num_params=cfg.num_params,
        d_latent=ckpt["args"]["d_latent"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])

    errs = reconstruction_error(model, loader, args.device)
    order = np.argsort(errs)[::-1]
    print(f"Top {args.top_k} highest-reconstruction-error devices (candidate anomalies):")
    for rank, idx in enumerate(order[: args.top_k]):
        print(f"  {rank:2d}.  device {idx:5d}   err {errs[idx]:.4f}")


if __name__ == "__main__":
    main(parse_args())
