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
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.latent_map = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.prediction_cell = PredictionCell(hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # Compute n_missing for each sample in the batch
        # Result: (batch_size,)
        n_missing = x.isnan().any(dim=2).sum(dim=1)  # shape: (batch_size, timesteps, input_dim)
        seq_len = x.size(1)
        batch_size = x.size(0)
        latents = []
        for idx in range(batch_size):
            n_miss = n_missing[idx].item()
            # print('n_miss:',n_miss)
            # Only use the non-missing part for LSTM
            latent, _ = self.latent_map(x[idx:idx+1, :(seq_len - n_miss), :])
            if n_miss == 0:
                latents.append(latent)
                continue
            latent = latent.clone()
            for i in range(n_miss):
                # print('run prediction_cell')
                updated = self.prediction_cell(latent[:, :(seq_len - n_miss + i), :])
                latent = torch.cat([latent[:, :(seq_len - n_miss + i), :], updated.unsqueeze(1)], dim=1)
            latents.append(latent)
        # Pad latents to the same length if needed, or stack if all same length
        latents = torch.cat(latents, dim=0)
        return latents


class PreAttnEncoder(nn.Module):
    """Non-autoregressive alternative to Encoder: one batched LSTM pass, supports bidirectionality.
    Kept alongside Encoder for A/B comparison — trades ragged-edge imputation for vectorization."""
    def __init__(self, dim_x, n_a, dropout_rate=0.2, bidirectional_encoder=False):
        super(PreAttnEncoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        self.lstm = nn.LSTM(input_size=dim_x, hidden_size=n_a, batch_first=True, bidirectional=bidirectional_encoder)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        a, _ = self.lstm(x)
        return a


class OneStepAttn(nn.Module):
    """Attention alignment module"""
    def __init__(self, n_a, n_s, n_align):
        super(OneStepAttn, self).__init__()
        self.densor1 = nn.Linear(n_a + n_s, n_align)
        self.densor2 = nn.Linear(n_align, 1)

    def forward(self, a, s_prev):
        s_prev = s_prev.unsqueeze(1).repeat(1, a.size(1), 1)
        concat = torch.cat((a, s_prev), dim=-1)
        e = torch.tanh(self.densor1(concat))
        energies = F.relu(self.densor2(e))
        alphas = F.softmax(energies, dim=1)
        context = torch.bmm(alphas.transpose(1, 2), a).squeeze(1)
        return context
