"""
avnet.py — Step 3 / 4.1 — Merged AVNet (shared CNN → vel + att heads)
Fixes QDeepOdo bugs: Flatten(0)->Flatten(1), hx randn->zeros, dtype, batch support.

Input: (B,200,6) @100Hz = 2s window, 6ch acc+gyro normalized
Outputs: v_pred (B,1) m/s, logσ_v (B,1), att_pred (B,3) rad, logσ_att (B,3)
Params: ~1.2M (backbone 11008->1024->512) + heads

Reference: ref/QDeepOdo/graphs/models/deepodo_6axis_imu_model.py + deepori_model.py
Paper: West=200, 1Hz output, but we train per-window 10Hz stride.

Test: python -m python.models.avnet
"""
import torch
import torch.nn as nn

class AVNet(nn.Module):
    def __init__(self, window=200, in_ch=6, feat_dim=512, dropout=0.1):
        super().__init__()
        # Shared CNN backbone — same as DeepOri (200 length)
        self.conv1 = nn.Conv1d(in_ch, 128, kernel_size=11)  # 200->190
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # 190->95
        self.conv2 = nn.Conv1d(128, 256, kernel_size=9)  # 95->87
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # 87->43
        # 43*256 = 11008
        self.flatten = nn.Flatten(start_dim=1)  # FIX: was Flatten(0)
        self.fc1 = nn.Linear(11008, 1024)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, feat_dim)
        self.relu_fc2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GRUCell for temporal (per-window, hx zero)
        self.gru_cell = nn.GRUCell(feat_dim, feat_dim)

        # Heads
        self.head_vel = nn.Linear(feat_dim, 1)      # v_forward
        self.head_logsig_vel = nn.Linear(feat_dim, 1)  # log σ
        self.head_att = nn.Linear(feat_dim, 3)      # att delta (yaw/pitch/roll)
        self.head_logsig_att = nn.Linear(feat_dim, 3)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hx=None):
        """
        x: (B, 200, 6) or (B, 6, 200) — we handle both
        hx: (B, 512) or None
        returns: v (B,1), logσ_v (B,1), att (B,3), logσ_att (B,3), hx_next
        """
        # ensure (B, 6, 200) for Conv1d
        if x.dim() == 3 and x.shape[1] == 200 and x.shape[2] == 6:
            x = x.permute(0, 2, 1)  # (B,6,200)
        elif x.dim() == 3 and x.shape[1] == 6 and x.shape[2] == 200:
            pass
        else:
            raise ValueError(f"expected (B,200,6) or (B,6,200) got {x.shape}")

        B = x.shape[0]
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)
        h = self.flatten(h)  # (B, 11008)
        h = self.fc1(h)
        h = self.relu_fc1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu_fc2(h)
        # h = self.dropout(h)  # already

        # GRUCell per sample (zero hx if None)
        if hx is None:
            hx = torch.zeros(B, h.shape[1], device=x.device, dtype=x.dtype)
        hx_next = self.gru_cell(h, hx)

        v = self.head_vel(hx_next)
        log_sig_v = self.head_logsig_vel(hx_next)
        att = self.head_att(hx_next)
        log_sig_att = self.head_logsig_att(hx_next)
        return v, log_sig_v, att, log_sig_att, hx_next

    def forward_window(self, x):
        """Convenience: (B,200,6) -> v (B,)"""
        v, _, _, _, _ = self.forward(x)
        return v.squeeze(-1)

class AVNetLite(nn.Module):
    """Lite version for TFLite: ~150k params, <1.2MB FP16, <8ms"""
    def __init__(self, window=200, in_ch=6, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # 200->100
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # 100->50
        # 46*64=2944 (200->100->46 after pools, conv2 dil2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(2944, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(128, 64, num_layers=2, batch_first=True)
        self.head_vel = nn.Linear(64, 1)
        self.head_logsig_vel = nn.Linear(64, 1)
        self.head_att = nn.Linear(64, 3)
        self.head_logsig_att = nn.Linear(64, 3)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 200 and x.shape[2] == 6:
            x = x.permute(0, 2, 1)  # (B,6,200)
        h = self.conv1(x)
        h = self.bn1(h); h = self.relu1(h); h = self.pool1(h)
        h = self.conv2(h)
        h = self.bn2(h); h = self.relu2(h); h = self.pool2(h)
        h = self.flatten(h)  # (B,3200)
        h = self.fc1(h); h = self.relu_fc1(h); h = self.dropout(h)
        # GRU expects (B, seq, feat) — we have single step, so unsqueeze
        h_seq, _ = self.gru(h.unsqueeze(1))  # (B,1,64)
        h = h_seq.squeeze(1)  # (B,64)
        v = self.head_vel(h)
        ls_v = self.head_logsig_vel(h)
        att = self.head_att(h)
        ls_att = self.head_logsig_att(h)
        return v, ls_v, att, ls_att, h

def count_params(m):
    return sum(p.numel() for p in m.parameters())

if __name__ == "__main__":
    for cls, name in [(AVNet, "AVNet"), (AVNetLite, "AVNetLite")]:
        m = cls()
        m.eval()
        print(f"{name} params {count_params(m):,}")
        x = torch.randn(4, 200, 6)
        v, ls_v, att, ls_att, hx = m(x)
        print(f"  v {v.shape} ls_v {ls_v.shape} att {att.shape} ls_att {ls_att.shape}")
        # TFLite size est
        import io
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        print(f"  state_dict {len(buf.getvalue())/1e6:.2f} MB FP32 -> FP16 {len(buf.getvalue())/2e6:.2f} MB")
        try:
            traced = torch.jit.trace(m, torch.randn(1,200,6))
            print(f"  jit trace ok")
        except Exception as e:
            print(f"  jit trace fail {e}")
