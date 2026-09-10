"""AVNet model family: shared CNN → velocity (+ attitude) heads.

Fixes carried over from the QDeepOdo reference (do not regress):
- ``Flatten(0)`` → ``Flatten(start_dim=1)`` (batch dim no longer collapsed).
- GRU hidden init ``randn`` → zeros (deterministic).
- Batch support + explicit input-shape validation.

Input: ``(B, 200, 6)`` @100 Hz = 2 s window, 6 channels
``[linAcc xyz, gyro yaw/pitch/roll]`` normalized.
Outputs: ``v_pred (B,1)``, ``logσ_v (B,1)``, ``att_pred (B,3)``,
``logσ_att (B,3)``, ``h (B, feat)``.

Reference: ref/QDeepOdo deepodo_6axis_imu_model.py + deepori_model.py;
paper West=200, 1 Hz output — we train per-window at 10 Hz stride.
"""

from __future__ import annotations

import torch
import torch.nn as nn

WINDOW = 200
IN_CHANNELS = 6


def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
    """Validate (B, 200, 6) or (B, 6, 200) and return (B, 6, 200)."""
    if x.dim() != 3:
        raise ValueError(f"expected 3D (B,W,C) or (B,C,W), got {x.shape}")
    if x.shape[1] == IN_CHANNELS and x.shape[2] == WINDOW:
        return x
    if x.shape[1] == WINDOW and x.shape[2] == IN_CHANNELS:
        return x.permute(0, 2, 1)
    raise ValueError(f"expected (B,{WINDOW},{IN_CHANNELS}) or (B,{IN_CHANNELS},{WINDOW}), got {x.shape}")


class AVNet(nn.Module):
    """Full-size merged model (velocity + attitude heads, ~13.6M params)."""

    def __init__(self, window: int = WINDOW, in_ch: int = IN_CHANNELS, feat_dim: int = 512, dropout: float = 0.1) -> None:
        """Build shared CNN backbone → FC bottleneck → GRU → heads.

        Args:
            window: Input window length (200).
            in_ch: Input channels (6).
            feat_dim: Bottleneck + GRU width (512).
            dropout: Dropout between FC layers.
        """
        super().__init__()
        # Conv geometry: 200 →(k11)→ 190 →(pool)→ 95 →(k9)→ 87 →(pool)→ 43; 43*256=11008
        self.conv1 = nn.Conv1d(in_ch, 128, kernel_size=11)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=9)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten(start_dim=1)  # FIX: reference used Flatten(0)
        conv_out = self._conv_out(window)
        self.fc1 = nn.Linear(conv_out, 1024)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, feat_dim)
        self.relu_fc2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gru_cell = nn.GRUCell(feat_dim, feat_dim)

        self.head_vel = nn.Linear(feat_dim, 1)
        self.head_logsig_vel = nn.Linear(feat_dim, 1)
        self.head_att = nn.Linear(feat_dim, 3)
        self.head_logsig_att = nn.Linear(feat_dim, 3)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _conv_out(self, window: int) -> int:
        """Feature count after the conv trunk (for the fc1 input dim)."""
        t = window - 11 + 1
        t = t // 2
        t = t - 9 + 1
        t = t // 2
        return t * 256

    def forward(
        self, x: torch.Tensor, hx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward one batch.

        Args:
            x: (B, 200, 6) or (B, 6, 200) normalized IMU window.
            hx: Optional (B, feat_dim) GRU state; zeros when None.

        Returns:
            (v_pred, logσ_v, att_pred, logσ_att, h_next).
        """
        x = _to_channels_first(x)
        B = x.shape[0]
        h = self.pool1(self.relu1(self.conv1(x)))
        h = self.pool2(self.relu2(self.conv2(h)))
        h = self.flatten(h)
        h = self.dropout(self.relu_fc1(self.fc1(h)))
        h = self.relu_fc2(self.fc2(h))

        if hx is None:
            hx = torch.zeros(B, h.shape[1], device=x.device, dtype=x.dtype)  # FIX: was randn
        h_next = self.gru_cell(h, hx)

        return (
            self.head_vel(h_next),
            self.head_logsig_vel(h_next),
            self.head_att(h_next),
            self.head_logsig_att(h_next),
            h_next,
        )

    def forward_window(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: (B,200,6) → v (B,)."""
        return self.forward(x)[0].squeeze(-1)


class AVNetLite(nn.Module):
    """Lite model for TFLite: ~460k params, <1.2 MB FP16, <8 ms.

    Conv trunk with BN + dilated conv, FC bottleneck to 128, 2-layer GRU
    (64), velocity + attitude heads with learned σ.
    """

    def __init__(self, window: int = WINDOW, in_ch: int = IN_CHANNELS, dropout: float = 0.1) -> None:
        """Build the lite model.

        Args:
            window: Input window length (200).
            in_ch: Input channels (6).
            dropout: Dropout after the FC bottleneck.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # 200→100
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # 100→50 (dilation: effective 46)
        self.flatten = nn.Flatten(start_dim=1)
        conv_out = self._conv_out(window)
        self.fc1 = nn.Linear(conv_out, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(128, 64, num_layers=2, batch_first=True)
        self.head_vel = nn.Linear(64, 1)
        self.head_logsig_vel = nn.Linear(64, 1)
        self.head_att = nn.Linear(64, 3)
        self.head_logsig_att = nn.Linear(64, 3)

    def _conv_out(self, window: int) -> int:
        """Feature count after the conv trunk (for the fc1 input dim).

        padding=4 convs preserve length; dilation=2 on conv2 shrinks the
        effective kernel to 4+2*8+1=17 → reach −16 samples → 100→(k17)→84
        → pool → 42... but with padding the length-preserving conv keeps
        100 then pool2 gives 46 after the dilation reach. Empirically the
        legacy model used 2944 = 46*64; keep that constant pinned.
        """
        t = window // 2          # pool1 (padding=same conv)
        t = t // 2 - 4           # pool2 minus dilation reach (100→50, reach −4 → 46)
        return t * 64

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward one batch.

        Args:
            x: (B, 200, 6) or (B, 6, 200) normalized IMU window.

        Returns:
            (v_pred, logσ_v, att_pred, logσ_att, h) — h is (B, 64).
        """
        x = _to_channels_first(x)
        h = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        h = self.pool2(self.relu2(self.bn2(self.conv2(h))))
        h = self.flatten(h)
        h = self.dropout(self.relu_fc1(self.fc1(h)))
        # GRU expects (B, seq, feat) — single step, so unsqueeze the time dim
        h_seq, _ = self.gru(h.unsqueeze(1))
        h = h_seq.squeeze(1)
        return (
            self.head_vel(h),
            self.head_logsig_vel(h),
            self.head_att(h),
            self.head_logsig_att(h),
            h,
        )


def count_params(m: nn.Module) -> int:
    """Total trainable parameter count (logged to reports/model_params.txt)."""
    return sum(p.numel() for p in m.parameters())
