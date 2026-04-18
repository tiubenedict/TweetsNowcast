import lightning.pytorch as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

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

class OneStepAttn(nn.Module):
    """Attention alignment module"""
    def __init__(self, n_a, n_s, n_align):
        super(OneStepAttn, self).__init__()
        self.densor1 = nn.Linear(n_a + n_s, n_align)
        self.densor2 = nn.Linear(n_align, 1)

    def forward(self, a, s_prev):
        """
        Performs one step of attention
        Args:
            a: hidden state from the pre-attention LSTM, shape = (batch_size, Lx, n_a)
            s_prev: previous hidden state of the post-attention LSTM, shape = (batch_size, n_s)
        Returns:
            context: context vector, input of the next post-attention LSTM cell
        """
        s_prev = s_prev.unsqueeze(1).repeat(1, a.size(1), 1)  # (batch_size, Lx, n_s)
        concat = torch.cat((a, s_prev), dim=-1)  # (batch_size, Lx, n_a + n_s)
        e = torch.tanh(self.densor1(concat))
        energies = F.relu(self.densor2(e))  # (batch_size, Lx, 1)
        alphas = F.softmax(energies, dim=1)  # (batch_size, Lx, 1)
        context = torch.bmm(alphas.transpose(1,2), a).squeeze(1)  # (batch_size, n_a)
        return context

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
        ### latent: (batch_size, seq_len, n_a)
        prediction = self.mlp(latent)
        # Return last 3 rows as x_prediction for current quarter
        # x_prediction = decoded[:, -3:, :]
        return prediction
     
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

        # self.lstm = nn.LSTM(input_size=n_a + n_a, hidden_size=n_s, batch_first=True) # n_a (x) + n_a (y)
        self.post_attn_cells = nn.ModuleList([
            nn.LSTMCell(input_size=n_a + n_a if i == 0 else n_s, hidden_size=n_s)
            for i in range(num_layers)
        ])
        self.decoder_y = Decoder(n_latent=n_s, dim=dim_y, hidden_dim=fc_y)

    def forward(self, x_encoder_in, y_decoder_in):
        batch_size = x_encoder_in.size(0)
        Ty = y_decoder_in.size(1)

        # Stage 1: Pre-attention encoding
        x_a = self.encoder_x(x_encoder_in)
        y_a = self.encoder_y(y_decoder_in)
        # Stage 2: Attention-based decoding
        # s = torch.zeros(1,batch_size, self.n_s, device=x_encoder_in.device)
        # c = torch.zeros(1,batch_size, self.n_s, device=x_encoder_in.device)
        s = [torch.zeros(batch_size, self.n_s, device=x_encoder_in.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.n_s, device=x_encoder_in.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(Ty):
            a_idx = int((t + 1) * self.freq_ratio - 1)
            a_to_attend = x_a[:, (a_idx - self.freq_ratio + 1):(a_idx + 1), :]
            context = self.one_step_attention(a_to_attend, s[-1]) # s: (batch_size, n_a)
            post_attn_input = torch.cat((context, y_a[:, t, :]), dim=-1)  # (batch_size, 1, n_a + n_a)
            # out, (s, c) = self.lstm(post_attn_input, (s, c))
            # outputs.append(out.squeeze(1))  # (batch_size, n_s)
            input_t = post_attn_input
            for i, cell in enumerate(self.post_attn_cells):
                s[i], c[i] = cell(input_t, (s[i], c[i]))
                input_t = s[i]
            outputs.append(s[-1])
        x_pred = self.decoder_x(x_a)
        y_pred = self.decoder_y(torch.stack(outputs, dim=1))  # (batch_size, Ty, dim_y)
        return x_pred, y_pred
   
class MTMFSeq2OneLightning(pl.LightningModule):
    def __init__(self, dim_x, dim_y,learning_rate, weight_decay, alpha, num_layers):
        super().__init__()
        self.save_hyperparameters()
        self.model = MTMFSeq2One(dim_x=dim_x, dim_y=dim_y, num_layers=num_layers)
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
        # self.log("train_loss_x", loss_x, on_epoch=True, prog_bar=False)
        # self.log("train_loss_y", loss_y, on_epoch=True, prog_bar=False)
        
        # with open("/home/btiu/Documents/Research/TweetsNowcast/output.log", "a") as logfile:
        #     print('training',loss.item(), x_encoder_in.shape, y_encoder_in.shape, x_target.shape, y_target.shape, file=logfile)
        return loss

    def validation_step(self, batch, batch_idx):
        x_encoder_in, y_encoder_in, x_target, y_target = batch
        x_pred, y_pred = self(x_encoder_in, y_encoder_in)
        loss_x = self.criterion(x_pred[:, -self.freq_ratio:, :], x_target[:, -self.freq_ratio:, :])
        loss_y = self.criterion(y_pred[:, -4:, :], y_target[:, -4:, :])
        loss = loss_x + loss_y
        self.log("val_loss_x", loss_x, on_epoch=True, prog_bar=True)
        self.log("val_loss_y", loss_y, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        # with open("/home/btiu/Documents/Research/TweetsNowcast/output.log", "a") as logfile:
        #     print('validation',loss_y.item(), loss_x.item(), x_encoder_in.shape, y_encoder_in.shape, x_target.shape, y_target.shape, file=logfile)
        return loss

    # def configure_optimizers(self):
    #     return optim.AdamW(
    #         self.parameters(),
    #         lr=self.hparams.learning_rate,
    #         weight_decay=self.hparams.weight_decay
    #     )
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5),
            "monitor": "val_loss_y",  # metric to monitor
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

# class MTMFSeq2OneOLD(nn.Module):
#     def __init__(self, dim_x, dim_y, num_layers=1, dropout_rate=0.0, n_a=4, fc_x=4, n_s=8, n_align=4, fc_y=4, freq_ratio=3):
#         super(MTMFSeq2OneOLD, self).__init__()
#         self.n_s = n_s
#         self.num_layers = num_layers
#         self.freq_ratio = freq_ratio
#         self.encoder_x = Encoder(input_dim=dim_x, hidden_dim=n_a, num_layers=num_layers, dropout_rate=dropout_rate)
#         self.decoder_x = Decoder(n_latent=n_a, dim=dim_x, hidden_dim=fc_x)
#         self.encoder_y = Encoder(input_dim=dim_y, hidden_dim=n_a, num_layers=1, dropout_rate=dropout_rate)
#         self.one_step_attention = OneStepAttn(n_a, n_s, n_align)
#         self.lstm = nn.LSTM(input_size=n_a + n_a, hidden_size=n_s, batch_first=True) # n_a (x) + n_a (y)
        
#         self.decoder_y = Decoder(n_latent=n_s, dim=dim_y, hidden_dim=fc_y)
#     def forward(self, x_encoder_in, y_decoder_in):
#         batch_size = x_encoder_in.size(0)
#         Ty = y_decoder_in.size(1)
#         # Stage 1: Pre-attention encoding
#         x_a = self.encoder_x(x_encoder_in)
#         x_pred = self.decoder_x(x_a)
#         y_a = self.encoder_y(y_decoder_in)
#         # Stage 2: Attention-based decoding
#         s = torch.zeros(1,batch_size, self.n_s, device=x_encoder_in.device)
#         c = torch.zeros(1,batch_size, self.n_s, device=x_encoder_in.device)

#         outputs = []
#         for t in range(Ty):
#             a_idx = int((t + 1) * self.freq_ratio - 1)
#             a_to_attend = x_a[:, (a_idx - self.freq_ratio + 1):(a_idx + 1), :]
#             context = self.one_step_attention(a_to_attend, s.squeeze(0)) # s: (batch_size, n_a)
#             post_attn_input = torch.cat((context, y_a[:, t, :]), dim=-1).unsqueeze(1)  # (batch_size, 1, n_a + n_a)
#             out, (s, c) = self.lstm(post_attn_input, (s, c))
#             outputs.append(out.squeeze(1))  # (batch_size, n_s)
#         y_pred = self.decoder_y(torch.stack(outputs, dim=1))  # (batch_size, Ty, dim_y)
#         return x_pred, y_pred
   
# class MTMFSeq2OneLightningOLD(pl.LightningModule):
#     def __init__(self, dim_x, dim_y,learning_rate, weight_decay, alpha, num_layers):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = MTMFSeq2OneOLD(dim_x=dim_x, dim_y=dim_y, num_layers=num_layers)
#         self.criterion = torch.nn.MSELoss()
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.alpha = alpha
#         self.freq_ratio = self.model.freq_ratio
#     def forward(self, x_encoder_in, y_encoder_in):
#         return self.model(x_encoder_in, y_encoder_in)
#     def training_step(self, batch, batch_idx):
#         x_encoder_in, y_encoder_in, x_target, y_target = batch
#         x_pred, y_pred = self(x_encoder_in, y_encoder_in)
#         loss_x = self.criterion(x_pred, x_target)
#         loss_y = self.criterion(y_pred, y_target)
#         loss = loss_x + self.alpha * loss_y
#         self.log("train_loss", loss, on_epoch=True, prog_bar=True)
#         # self.log("train_loss_x", loss_x, on_epoch=True, prog_bar=False)
#         # self.log("train_loss_y", loss_y, on_epoch=True, prog_bar=False)
        
#         # with open("/home/btiu/Documents/Research/TweetsNowcast/output.log", "a") as logfile:
#         #     print('training',loss.item(), x_encoder_in.shape, y_encoder_in.shape, x_target.shape, y_target.shape, file=logfile)
#         return loss
#     def validation_step(self, batch, batch_idx):
#         x_encoder_in, y_encoder_in, x_target, y_target = batch
#         x_pred, y_pred = self(x_encoder_in, y_encoder_in)
#         loss_x = self.criterion(x_pred[:, -self.freq_ratio:, :], x_target[:, -self.freq_ratio:, :])
#         loss_y = self.criterion(y_pred[:, -2:, :], y_target[:, -2:, :])
#         loss = loss_x + loss_y
#         self.log("val_loss_x", loss_x, on_epoch=True, prog_bar=True)
#         self.log("val_loss_y", loss_y, on_epoch=True, prog_bar=True)
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True)
#         # with open("/home/btiu/Documents/Research/TweetsNowcast/output.log", "a") as logfile:
#         #     print('validation',loss_y.item(), loss_x.item(), x_encoder_in.shape, y_encoder_in.shape, x_target.shape, y_target.shape, file=logfile)
#         return loss
#     # def configure_optimizers(self):
#     #     return optim.AdamW(
#     #         self.parameters(),
#     #         lr=self.hparams.learning_rate,
#     #         weight_decay=self.hparams.weight_decay
#     #     )
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(
#             self.parameters(),
#             lr=self.hparams.learning_rate,
#             weight_decay=self.hparams.weight_decay
#         )
#         scheduler = {
#             "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5),
#             "monitor": "val_loss_y",  # metric to monitor
#             "interval": "epoch",
#             "frequency": 1
#         }
#         return {"optimizer": optimizer, "lr_scheduler": scheduler}