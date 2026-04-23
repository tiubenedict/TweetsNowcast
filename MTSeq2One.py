import lightning.pytorch as pl
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from nn_blocks import Encoder, OneStepAttn


class Decoder(nn.Module):
    def __init__(self, n_latent, dim, hidden_dim, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_latent, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, latent):
        return self.mlp(latent)


class MTMFSeq2One(nn.Module):
    def __init__(self, dim_x, dim_y, num_layers=1, n_a=4, n_s=8, n_align=4, fc_x=4, fc_y=4, dropout_rate=0.0, freq_ratio=3):
        super(MTMFSeq2One, self).__init__()
        self.n_s = n_s
        self.num_layers = num_layers
        self.freq_ratio = freq_ratio

        self.encoder_x = Encoder(input_dim=dim_x, hidden_dim=n_a, num_layers=num_layers, dropout_rate=dropout_rate)
        self.decoder_x = Decoder(n_latent=n_a, dim=dim_x, hidden_dim=fc_x)
        self.encoder_y = Encoder(input_dim=dim_y, hidden_dim=n_a, num_layers=1, dropout_rate=dropout_rate)
        self.one_step_attention = OneStepAttn(n_a, n_s, n_align)

        self.post_attn_cells = nn.ModuleList([
            nn.LSTMCell(input_size=n_a + n_a if i == 0 else n_s, hidden_size=n_s)
            for i in range(num_layers)
        ])
        self.decoder_y = Decoder(n_latent=n_s, dim=dim_y, hidden_dim=fc_y)

    def forward(self, x_encoder_in, y_decoder_in):
        batch_size = x_encoder_in.size(0)
        Ty = y_decoder_in.size(1)

        x_a = self.encoder_x(x_encoder_in)
        y_a = self.encoder_y(y_decoder_in)

        s = [torch.zeros(batch_size, self.n_s, device=x_encoder_in.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.n_s, device=x_encoder_in.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(Ty):
            a_idx = int((t + 1) * self.freq_ratio - 1)
            a_to_attend = x_a[:, (a_idx - self.freq_ratio + 1):(a_idx + 1), :]
            context = self.one_step_attention(a_to_attend, s[-1])
            post_attn_input = torch.cat((context, y_a[:, t, :]), dim=-1)
            input_t = post_attn_input
            for i, cell in enumerate(self.post_attn_cells):
                s[i], c[i] = cell(input_t, (s[i], c[i]))
                input_t = s[i]
            outputs.append(s[-1])
        x_pred = self.decoder_x(x_a)
        y_pred = self.decoder_y(torch.stack(outputs, dim=1))
        return x_pred, y_pred


class MTMFSeq2OneLightning(pl.LightningModule):
    def __init__(self, dim_x, dim_y, learning_rate, weight_decay, alpha, num_layers,
                 n_a=4, n_s=8, n_align=4, fc_x=4, fc_y=4, dropout_rate=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = MTMFSeq2One(
            dim_x=dim_x, dim_y=dim_y, num_layers=num_layers,
            n_a=n_a, n_s=n_s, n_align=n_align, fc_x=fc_x, fc_y=fc_y, dropout_rate=dropout_rate,
        )
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.freq_ratio = self.model.freq_ratio

    def forward(self, x_encoder_in, y_encoder_in):
        return self.model(x_encoder_in, y_encoder_in)

    def training_step(self, batch, batch_idx):
        x_encoder_in, y_encoder_in, x_target, y_target = batch
        x_pred, y_pred = self(x_encoder_in, y_encoder_in)
        loss_x = self.criterion(x_pred, x_target)
        loss_y = self.criterion(y_pred, y_target)
        loss = loss_x + self.alpha * loss_y
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_encoder_in, y_encoder_in, x_target, y_target = batch
        x_pred, y_pred = self(x_encoder_in, y_encoder_in)
        loss_x = self.criterion(x_pred[:, -self.freq_ratio:, :], x_target[:, -self.freq_ratio:, :])
        loss_y = self.criterion(y_pred[:, -1:, :], y_target[:, -1:, :])
        loss = loss_x + loss_y
        self.log("val_loss_x", loss_x, on_epoch=True, prog_bar=True)
        self.log("val_loss_y", loss_y, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

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
