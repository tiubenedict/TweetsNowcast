import lightning.pytorch as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from nn_blocks import Encoder, OneStepAttn, PreAttnEncoder


class STMFSeq2One(nn.Module):
    def __init__(self, dim_x, dim_y, num_layers, n_a=4, n_s=8, n_align=4, fc_y=4,
                 dropout_rate=0, freq_ratio=3,
                 encoder_type="autoregressive", bidirectional=False):
        super(STMFSeq2One, self).__init__()
        self.n_s = n_s
        self.num_layers = num_layers
        self.freq_ratio = freq_ratio
        self.encoder_type = encoder_type

        if encoder_type == "autoregressive":
            self.encoder_x = Encoder(input_dim=dim_x, hidden_dim=n_a, num_layers=num_layers, dropout_rate=dropout_rate)
        elif encoder_type == "pre_attn":
            self.encoder_x = PreAttnEncoder(input_dim=dim_x, hidden_dim=n_a, dropout_rate=dropout_rate, bidirectional=bidirectional)
        else:
            raise ValueError(f"encoder_type must be 'autoregressive' or 'pre_attn', got {encoder_type!r}")

        # y-encoder stays autoregressive: y (quarterly target) has its own ragged-tail
        # to impute, and the direction knob is for x's encoder only.
        self.encoder_y = Encoder(input_dim=dim_y, hidden_dim=n_a, num_layers=1, dropout_rate=dropout_rate)

        self.one_step_attention = OneStepAttn(n_a, n_s, n_align)

        self.post_attn_cells = nn.ModuleList([
            nn.LSTMCell(input_size=n_a + n_a if i == 0 else n_s, hidden_size=n_s)
            for i in range(num_layers)
        ])
        self.ffn1 = nn.Linear(n_s, fc_y)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn2 = nn.Linear(fc_y, dim_y)

    def forward(self, x_encoder_in, y_decoder_in):
        batch_size = x_encoder_in.size(0)
        Ty = y_decoder_in.size(1)

        x_a, x_mask = self.encoder_x(x_encoder_in)
        y_a, _ = self.encoder_y(y_decoder_in)

        s = [torch.zeros(batch_size, self.n_s, device=x_encoder_in.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.n_s, device=x_encoder_in.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(Ty):
            a_idx = int((t + 1) * self.freq_ratio - 1)
            a_to_attend = x_a[:, (a_idx - self.freq_ratio + 1):(a_idx + 1), :]
            mask_to_attend = x_mask[:, (a_idx - self.freq_ratio + 1):(a_idx + 1)]
            context = self.one_step_attention(a_to_attend, s[-1], mask=mask_to_attend)
            post_attn_input = torch.cat((context, y_a[:, t, :]), dim=-1)
            input_t = post_attn_input
            for i, cell in enumerate(self.post_attn_cells):
                s[i], c[i] = cell(input_t, (s[i], c[i]))
                input_t = s[i]
            outputs.append(s[-1])

        y_pred = F.relu(self.ffn1(torch.stack(outputs, dim=1)))
        y_pred = self.dropout(y_pred)
        y_pred = self.ffn2(y_pred)

        return y_pred


class STMFSeq2OneLightning(pl.LightningModule):
    def __init__(self, dim_x, dim_y, learning_rate, weight_decay, num_layers,
                 n_a=4, n_s=8, n_align=4, fc_y=4, dropout_rate=0.0,
                 encoder_type="autoregressive", bidirectional=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = STMFSeq2One(
            dim_x=dim_x, dim_y=dim_y, num_layers=num_layers,
            n_a=n_a, n_s=n_s, n_align=n_align, fc_y=fc_y, dropout_rate=dropout_rate,
            encoder_type=encoder_type, bidirectional=bidirectional,
        )
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freq_ratio = self.model.freq_ratio

    def forward(self, x_encoder_in, y_encoder_in):
        return self.model(x_encoder_in, y_encoder_in)

    def training_step(self, batch, batch_idx):
        x_encoder_in, y_encoder_in, y_target = batch
        y_pred = self(x_encoder_in, y_encoder_in)
        loss = self.criterion(y_pred, y_target)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_encoder_in, y_encoder_in, y_target = batch
        y_pred = self(x_encoder_in, y_encoder_in)
        loss_y = self.criterion(y_pred[:, -1:, :], y_target[:, -1:, :])
        self.log("val_loss_y", loss_y, on_epoch=True, prog_bar=True)
        return loss_y

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5),
            "monitor": "val_loss_y",
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
