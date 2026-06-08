import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionCell(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout_rate):
        super(PredictionCell, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        a, _ = self.lstm(x)
        out = self.linear_1(a[:, -1, :])
        out = self.activation(out)
        out = self.linear_2(out)
        return out


class Encoder(nn.Module):
    """Autoregressive encoder.

    `partial_aware` controls how NaN rows are handled:
      - False (legacy default): any row containing NaN is treated as part of the
        trailing-missing block. latent_map sees only the prefix of fully-observed
        rows; PredictionCell autoregresses through the entire any-NaN suffix.
        Preserves exact behavior of pre-i1 checkpoints.
      - True (i1 mode): rows with at least one real feature go through latent_map
        with NaN columns zero-filled. Only fully-NaN trailing rows go through
        PredictionCell. Lets partial-month real data inform the latent."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, partial_aware=False):
        super(Encoder, self).__init__()
        self.latent_map = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.prediction_cell = PredictionCell(hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate)
        self.hidden_dim = hidden_dim
        self.partial_aware = partial_aware

    def forward(self, x):
        seq_len = x.size(1)
        batch_size = x.size(0)

        if self.partial_aware:
            # Trailing block = consecutive fully-NaN rows from the end.
            fully_nan = x.isnan().all(dim=2)                          # [B, L]
            rev = fully_nan.flip(dims=[1]).long()
            n_trailing = rev.cumprod(dim=1).sum(dim=1)                # [B]
        else:
            # Legacy: any-NaN row counts toward the trailing missing block.
            n_trailing = x.isnan().any(dim=2).sum(dim=1)              # [B]
        body_len = seq_len - n_trailing                               # [B]

        latents_out = []
        for idx in range(batch_size):
            bl = int(body_len[idx].item())
            nt = int(n_trailing[idx].item())
            if bl > 0:
                # In partial_aware mode the body may contain partial-NaN rows;
                # zero-fill them. In legacy mode the body is by construction
                # fully-observed, so nan_to_num is a no-op there.
                body = torch.nan_to_num(x[idx:idx+1, :bl, :], nan=0.0) if self.partial_aware else x[idx:idx+1, :bl, :]
                latent, _ = self.latent_map(body)
            else:
                # All-NaN sample — seed with a zero latent and let PredictionCell autoregress.
                latent = torch.zeros(1, 1, self.hidden_dim, device=x.device, dtype=x.dtype)
                bl = 1
                nt = max(0, nt - 1)
            for i in range(nt):
                updated = self.prediction_cell(latent[:, :(bl + i), :])
                latent = torch.cat([latent[:, :(bl + i), :], updated.unsqueeze(1)], dim=1)
            latents_out.append(latent)
        latents = torch.cat(latents_out, dim=0)
        mask = torch.ones(latents.size(0), latents.size(1), dtype=torch.bool, device=latents.device)
        return latents, mask


class PreAttnEncoder(nn.Module):
    """Non-autoregressive encoder: one batched LSTM pass over non-NaN prefixes via
    pack_padded_sequence. NaN-tail positions receive zero latents; the returned mask
    marks them invalid so attention can ignore them via softmax(-inf).

    If bidirectional=True, a linear projection squashes the 2*hidden_dim LSTM output
    back to hidden_dim so downstream attention's input dim is independent of
    direction choice.

    `partial_aware` controls trailing-missing detection:
      - False (legacy default): any row with any NaN counts toward the trailing
        block. lengths = seq_len - (# any-NaN rows). Preserves pre-i2 checkpoints.
      - True (i2 mode): only fully-NaN rows count. Partial-NaN rows stay in the
        packed sequence with NaN columns zero-filled, so real columns inform
        the latent."""
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.0, bidirectional=False, partial_aware=False):
        super(PreAttnEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.partial_aware = partial_aware
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True, bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)
        lstm_out_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.out_proj = nn.Linear(lstm_out_dim, hidden_dim) if bidirectional else nn.Identity()

    def forward(self, x):
        seq_len = x.size(1)
        if self.partial_aware:
            fully_nan = x.isnan().all(dim=2)
            rev = fully_nan.flip(dims=[1]).long()
            n_trailing = rev.cumprod(dim=1).sum(dim=1)
        else:
            n_trailing = x.isnan().any(dim=2).sum(dim=1)
        lengths = (seq_len - n_trailing).clamp(min=1)
        x_clean = torch.nan_to_num(x, nan=0.0)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_clean, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        x_a, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len,
        )
        x_a = self.dropout(x_a)
        x_a = self.out_proj(x_a)
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        return x_a, mask


class OneStepAttn(nn.Module):
    """Attention alignment module. Accepts an optional mask of shape [B, Lx];
    positions where mask is False are excluded from the softmax via -inf energies.
    If a window is entirely masked, returns zero context.

    use_relu_energies defaults to True for backwards compat with v1 checkpoints,
    where energies were ReLU-clamped before softmax (atypical for Bahdanau and
    likely harmful — clamps negative scores to identical zeros). New v2 runs
    should set this to False for standard Bahdanau attention."""
    def __init__(self, n_a, n_s, n_align, use_relu_energies=True):
        super(OneStepAttn, self).__init__()
        self.densor1 = nn.Linear(n_a + n_s, n_align)
        self.densor2 = nn.Linear(n_align, 1)
        self.use_relu_energies = use_relu_energies

    def forward(self, a, s_prev, mask=None):
        s_prev = s_prev.unsqueeze(1).repeat(1, a.size(1), 1)
        concat = torch.cat((a, s_prev), dim=-1)
        e = torch.tanh(self.densor1(concat))
        energies = self.densor2(e)
        if self.use_relu_energies:
            energies = F.relu(energies)
        if mask is not None:
            energies = energies.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        alphas = F.softmax(energies, dim=1)
        # All-masked windows: softmax(-inf everywhere) = NaN; fall back to zero context.
        alphas = torch.nan_to_num(alphas, nan=0.0)
        context = torch.bmm(alphas.transpose(1, 2), a).squeeze(1)
        return context
