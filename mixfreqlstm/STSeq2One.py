import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

class PreAttnEncoder(nn.Module):
    """Pre-attention Encoder module"""
    def __init__(self, dim_x, n_a, dropout_rate=0.2, bidirectional_encoder=False):
        super(PreAttnEncoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        self.lstm = nn.LSTM( input_size=dim_x, hidden_size=n_a, batch_first=True, bidirectional=bidirectional_encoder)
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


class STMFSeq2One(nn.Module):
    def __init__(
        self,
        Lx,
        dim_x,
        Ty,
        dim_y,
        n_a,
        n_s,
        n_align,
        fc_y,
        dropout_rate,
        freq_ratio=3,
        bidirectional_encoder=False
    ):
        super(STMFSeq2One, self).__init__()
        self.Lx = Lx
        self.Ty = Ty
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_a = n_a
        self.n_s = n_s
        self.n_align = n_align
        self.fc_y = fc_y
        self.freq_ratio = freq_ratio
        self.bidirectional_encoder = bidirectional_encoder

        # Encoder
        self.pre_attn = PreAttnEncoder(dim_x, n_a, dropout_rate, bidirectional_encoder)

        # Attention alignment model
        self.one_step_attention = OneStepAttn(n_a, n_s, n_align)

        # Decoder
        self.post_attn = nn.LSTMCell(input_size=n_a + dim_y, hidden_size=n_s)
        self.ffn1 = nn.Linear(n_s, fc_y)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn2 = nn.Linear(fc_y, dim_y) ### add regularizer?

    def initialize_state(self, batch_size, dim):
        return torch.zeros(batch_size, dim) ### faster to initialize specific device?

    def forward(self, x_encoder_in, y_decoder_in):
        batch_size = x_encoder_in.size(0)
        # Stage 1: Pre-attention encoding
        a = self.pre_attn(x_encoder_in)
        # Stage 2: Attention-based decoding
        s = self.initialize_state(batch_size, self.n_s).to(x_encoder_in.device)
        c = self.initialize_state(batch_size, self.n_s).to(x_encoder_in.device)

        for t in range(self.Ty):
            a_idx = int((t + 1) * self.freq_ratio - 1)
            a_to_attend = a[:, (a_idx - self.freq_ratio + 1):(a_idx + 1), :]
            context = self.one_step_attention(a_to_attend, s)

            post_attn_input = torch.cat((context, y_decoder_in[:, t, :]), dim=-1)
            s, c = self.post_attn(post_attn_input, (s, c))

        y_pred = F.relu(self.ffn1(s))
        y_pred = self.dropout(y_pred)
        y_pred = self.ffn2(y_pred)

        return y_pred
    
class MTMFSeq2SeqLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=0.01):
        super(MTMFSeq2SeqLightning, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss()

    def forward(self, x_encoder_in, y_decoder_in):
        return self.model(x_encoder_in, y_decoder_in)

    def training_step(self, batch, batch_idx):
        x_encoder_in, y_decoder_in, y_target = batch
        y_pred = self(x_encoder_in, y_decoder_in)
        loss = self.criterion(y_pred, y_target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)