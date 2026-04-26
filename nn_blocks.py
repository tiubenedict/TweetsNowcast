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
    """Autoregressive encoder: LSTM on non-NaN prefix, PredictionCell fills the
    NaN tail. Every output position is treated as valid by downstream attention
    (the returned mask is all-True) since the encoder itself has imputed the tail."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.latent_map = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.prediction_cell = PredictionCell(hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate)

    def forward(self, x):
        n_missing = x.isnan().any(dim=2).sum(dim=1)
        seq_len = x.size(1)
        batch_size = x.size(0)
        latents = []
        for idx in range(batch_size):
            n_miss = n_missing[idx].item()
            latent, _ = self.latent_map(x[idx:idx+1, :(seq_len - n_miss), :])
            if n_miss == 0:
                latents.append(latent)
                continue
            latent = latent.clone()
            for i in range(n_miss):
                updated = self.prediction_cell(latent[:, :(seq_len - n_miss + i), :])
                latent = torch.cat([latent[:, :(seq_len - n_miss + i), :], updated.unsqueeze(1)], dim=1)
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        mask = torch.ones(latents.size(0), latents.size(1), dtype=torch.bool, device=latents.device)
        return latents, mask


class PreAttnEncoder(nn.Module):
    """Non-autoregressive encoder: one batched LSTM pass over non-NaN prefixes via
    pack_padded_sequence. NaN-tail positions receive zero latents; the returned mask
    marks them invalid so attention can ignore them via softmax(-inf).

    If bidirectional=True, a linear projection squashes the 2*hidden_dim LSTM output
    back to hidden_dim so downstream attention's input dim is independent of
    direction choice."""
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.0, bidirectional=False):
        super(PreAttnEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            batch_first=True, bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout_rate)
        lstm_out_dim = 2 * hidden_dim if bidirectional else hidden_dim
        self.out_proj = nn.Linear(lstm_out_dim, hidden_dim) if bidirectional else nn.Identity()

    def forward(self, x):
        n_missing = x.isnan().any(dim=2).sum(dim=1)
        lengths = x.size(1) - n_missing
        # pack_padded_sequence requires lengths >= 1; clamp to protect against all-NaN samples.
        lengths = torch.clamp(lengths, min=1)
        x_clean = torch.nan_to_num(x, nan=0.0)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_clean, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        x_a, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1),
        )
        x_a = self.dropout(x_a)
        x_a = self.out_proj(x_a)
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
        return x_a, mask


class OneStepAttn(nn.Module):
    """Attention alignment module. Accepts an optional mask of shape [B, Lx];
    positions where mask is False are excluded from the softmax via -inf energies.
    If a window is entirely masked, returns zero context."""
    def __init__(self, n_a, n_s, n_align):
        super(OneStepAttn, self).__init__()
        self.densor1 = nn.Linear(n_a + n_s, n_align)
        self.densor2 = nn.Linear(n_align, 1)

    def forward(self, a, s_prev, mask=None):
        s_prev = s_prev.unsqueeze(1).repeat(1, a.size(1), 1)
        concat = torch.cat((a, s_prev), dim=-1)
        e = torch.tanh(self.densor1(concat))
        energies = F.relu(self.densor2(e))
        if mask is not None:
            energies = energies.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        alphas = F.softmax(energies, dim=1)
        # All-masked windows: softmax(-inf everywhere) = NaN; fall back to zero context.
        alphas = torch.nan_to_num(alphas, nan=0.0)
        context = torch.bmm(alphas.transpose(1, 2), a).squeeze(1)
        return context
