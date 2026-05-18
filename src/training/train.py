from __future__ import annotations
import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from src.data.synthetic import SyntheticConfig, SyntheticIVDataset, in_channels, metadata_dim
from src.models.encoder_decoder import FET2DEncoderDecoder
from src.models.losses import CompositeLoss

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-latent", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=1024)
    p.add_argument("--num-materials", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="runs/dev")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=19700101)
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    cfg = SyntheticConfig(n_samples=args.n_samples, num_materials=args.num_materials, seed=args.seed)
    dataset = SyntheticIVDataset(cfg)
    n_dev = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_dev
    train_set, dev_set = random_split(dataset, [n_train, n_dev], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

    model = FET2DEncoderDecoder(
        in_channels=in_channels(cfg),
        num_points=cfg.num_points,
        d_metadata=metadata_dim(cfg),
        num_params=cfg.num_params,
        d_latent=args.d_latent,
    ).to(args.device)

    loss_fn = CompositeLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        tic = time.time()
        train_losses = []
        for batch in train_loader:
            iv = batch["iv"].to(args.device)
            metadata = batch["metadata"].to(args.device)
            params = batch["params"].to(args.device)

            params_hat, iv_hat, z = model(iv, metadata)
            losses = loss_fn(params_hat, params, iv_hat, iv, latent=z)
            loss = losses["loss"]

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            dev_losses = []
            for batch in dev_loader:
                iv = batch["iv"].to(args.device)
                metadata = batch["metadata"].to(args.device)
                params = batch["params"].to(args.device)
                params_hat, iv_hat, z = model(iv, metadata)
                losses = loss_fn(params_hat, params, iv_hat, iv, latent=z)
                dev_losses.append(float(losses["loss"]))

        train_mean = sum(train_losses) / len(train_losses)
        dev_mean = sum(dev_losses) / max(1, len(dev_losses))
        history.append({"epoch": epoch, "train": train_mean, "dev": dev_mean})
        print(f"epoch {epoch:3d}  train {train_mean:.4f}  dev {dev_mean:.4f}  ({time.time()-tic:.1f}s)")

    ckpt_path = out_dir / "model.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "cfg": cfg.__dict__}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train(parse_args())
