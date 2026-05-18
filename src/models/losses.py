from __future__ import annotations
import torch
import torch.nn as nn

def _finite_diff(x: torch.Tensor) -> torch.Tensor:
    """First difference along the voltage axis (dim=1)."""
    return x[:, 1:, :] - x[:, :-1, :]

class CompositeLoss(nn.Module):
    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        lambda1: float = 0.5,
        lambda2: float = 0.25,
        c: float = 0.0,
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.c = c
        self.mse = nn.MSELoss()

    def forward(
        self,
        params_pred: torch.Tensor,
        params_true: torch.Tensor,
        iv_pred: torch.Tensor,
        iv_true: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_params = self.mse(params_pred, params_true)

        loss_iv = self.mse(iv_pred, iv_true)
        d1 = self.mse(_finite_diff(iv_pred), _finite_diff(iv_true))
        d2 = self.mse(_finite_diff(_finite_diff(iv_pred)), _finite_diff(_finite_diff(iv_true)))
        loss_iv_total = loss_iv + self.lambda1 * d1 + self.lambda2 * d2

        loss = self.a * loss_params + self.b * loss_iv_total

        if self.c > 0.0 and latent is not None:
            # simple L2 regularizer on the latent (placeholder for VAE-style KL)
            loss_latent = latent.pow(2).mean()
            loss = loss + self.c * loss_latent
        else:
            loss_latent = torch.zeros((), device=loss.device)

        return {
            "loss": loss,
            "loss_params": loss_params.detach(),
            "loss_iv": loss_iv.detach(),
            "loss_iv_d1": d1.detach(),
            "loss_iv_d2": d2.detach(),
            "loss_latent": loss_latent.detach(),
        }
