from __future__ import annotations
import torch
import torch.nn as nn


class IVEncoder(nn.Module):
   

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        d_latent: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(2 * d_model)
        self.to_latent = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.proj(x)             # (B, T, d_model)
        h, _ = self.gru(h)           # (B, T, 2*d_model)
        h = self.norm(h)
        h = h.mean(dim=1)            # mean-pool over voltage points
        return self.to_latent(h)     # (B, d_latent)


class ParameterDecoder(nn.Module):
    

    def __init__(
        self,
        d_latent: int,
        d_metadata: int,
        num_params: int,
        d_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_metadata, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, num_params),
        )

    def forward(self, z: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, metadata], dim=-1)
        return self.net(h)


class IVReconstructor(nn.Module):


    def __init__(
        self,
        d_latent: int,
        d_metadata: int,
        num_points: int,
        out_channels: int,
        d_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_metadata, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, num_points * out_channels),
        )

    def forward(self, z: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, metadata], dim=-1)
        out = self.net(h)
        return out.view(-1, self.num_points, self.out_channels)


class FET2DEncoderDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_points: int,
        d_metadata: int,
        num_params: int,
        d_latent: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = IVEncoder(in_channels=in_channels, d_latent=d_latent)
        self.param_decoder = ParameterDecoder(
            d_latent=d_latent, d_metadata=d_metadata, num_params=num_params
        )
        self.iv_decoder = IVReconstructor(
            d_latent=d_latent,
            d_metadata=d_metadata,
            num_points=num_points,
            out_channels=in_channels,
        )

    def forward(
        self, iv: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(iv)
        params_hat = self.param_decoder(z, metadata)
        iv_hat = self.iv_decoder(z, metadata)
        return params_hat, iv_hat, z
