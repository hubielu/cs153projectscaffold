from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticConfig:
    n_samples: int = 1024
    num_points: int = 32
    num_idvg: int = 2
    num_feats: int = 2
    num_params: int = 9
    num_materials: int = 3
    noise_sigma: float = 0.02
    seed: int = 19700101


class SyntheticIVDataset(Dataset):
    """Generate (iv, metadata, params) triples that loosely mimic FET behavior."""

    def __init__(self, cfg: SyntheticConfig = SyntheticConfig()) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.params = rng.uniform(0.1, 0.9, size=(cfg.n_samples, cfg.num_params)).astype(np.float32)
        self.material_ids = rng.integers(0, cfg.num_materials, size=cfg.n_samples)
        v = np.linspace(0.0, 1.0, cfg.num_points, dtype=np.float32)
        iv = np.zeros((cfg.n_samples, cfg.num_points, cfg.num_idvg * cfg.num_feats), dtype=np.float32)
        for i in range(cfg.n_samples):
            for k in range(cfg.num_idvg):
                vth = self.params[i, 1] + 0.05 * k
                gain = self.params[i, 0] + 0.1 * k
                slope = max(self.params[i, 2], 0.05)
                base = gain / (1.0 + np.exp(-(v - vth) / slope))
                base = base * (1.0 + 0.1 * self.material_ids[i])
                base = base + rng.normal(0.0, cfg.noise_sigma, size=cfg.num_points).astype(np.float32)
                deriv = np.gradient(base).astype(np.float32)
                iv[i, :, 2 * k] = base
                iv[i, :, 2 * k + 1] = deriv
        self.iv = iv
        
    def __len__(self) -> int:
        return self.cfg.n_samples

    def metadata_vector(self, idx: int) -> np.ndarray:
        m = np.zeros(self.cfg.num_materials, dtype=np.float32)
        m[self.material_ids[idx]] = 1.0
        lch = np.array([self.params[idx, 0]], dtype=np.float32)
        return np.concatenate([m, lch], axis=0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "iv": torch.from_numpy(self.iv[idx]),
            "metadata": torch.from_numpy(self.metadata_vector(idx)),
            "params": torch.from_numpy(self.params[idx]),
        }


def metadata_dim(cfg: SyntheticConfig) -> int:
    return cfg.num_materials + 1  # one-hot material + 1 numeric feature (Lch)


def in_channels(cfg: SyntheticConfig) -> int:
    return cfg.num_idvg * cfg.num_feats
