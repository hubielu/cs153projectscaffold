from __future__ import annotations
import torch
from src.data.synthetic import SyntheticConfig, SyntheticIVDataset, in_channels, metadata_dim
from src.models.encoder_decoder import FET2DEncoderDecoder
from src.models.losses import CompositeLoss


def test_forward_and_loss_runs():
    cfg = SyntheticConfig(n_samples=16, num_points=16)
    ds = SyntheticIVDataset(cfg)
    batch = {k: torch.stack([ds[i][k] for i in range(8)]) for k in ds[0]}
    model = FET2DEncoderDecoder(
        in_channels=in_channels(cfg),
        num_points=cfg.num_points,
        d_metadata=metadata_dim(cfg),
        num_params=cfg.num_params,
        d_latent=16,
    )
    params_hat, iv_hat, z = model(batch["iv"], batch["metadata"])
    assert params_hat.shape == (8, cfg.num_params)
    assert iv_hat.shape == batch["iv"].shape
    assert z.shape == (8, 16)
    loss_fn = CompositeLoss()
    out = loss_fn(params_hat, batch["params"], iv_hat, batch["iv"], latent=z)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
