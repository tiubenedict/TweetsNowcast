import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from STSeq2One import STMFSeq2One, MTMFSeq2SeqLightning
from data_utils import get_dataloader_for_vintage


def train_one_vintage(vintage, with_econ, with_tweets, data_window, kmpair={}, device="cpu"):
    from pytorch_lightning import seed_everything
    seed_everything(42, workers=True)
    dataloader, target_scaler, econ_scaler, tweets_scaler = get_dataloader_for_vintage(vintage, with_econ, with_tweets, data_window, kmpair={}, mode="train")
    if with_econ and with_tweets:
        econ_n_feat = econ_scaler.n_features_in_
        tweets_n_feat = tweets_scaler.n_features_in_
        dim_x = econ_n_feat + tweets_n_feat
    elif with_econ:
        econ_n_feat = econ_scaler.n_features_in_
        dim_x = econ_n_feat
    elif with_tweets:
        tweets_n_feat = tweets_scaler.n_features_in_
        dim_x = tweets_n_feat
    # Initialize the model
    # dim_x = 4                       #trainX_in.shape[-1] # 9, 4, 13
    dim_y = 1                       #trainY_in.shape[-1] # 1
    Lx = data_window                #trainX_in.shape[1] # 15, 27
    Ty = data_window // 3 - 1       #trainY_in.shape[1] # 4, 8
    n_a = 4
    n_s = 8
    n_align = 4
    fc_y = 4
    dropout_rate = 0
    freq_ratio = 3
    bidirectional_encoder = False
    model = STMFSeq2One(dim_x=dim_x,dim_y=dim_y,Lx=Lx,Ty=Ty,n_a=n_a,n_s=n_s,n_align=n_align,fc_y=fc_y,dropout_rate=dropout_rate,freq_ratio=freq_ratio,bidirectional_encoder=bidirectional_encoder)
    lightning_model = MTMFSeq2SeqLightning(model)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/vintage_{vintage}",
        filename="model",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    # logger = TensorBoardLogger("tb_logs", name=f"vintage_{vintage}")

    trainer = pl.Trainer(max_epochs=150, accelerator=device, callbacks=[checkpoint_callback], enable_progress_bar=True, log_every_n_steps=1, deterministic=True)
    tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(lightning_model, dataloader)
    # lightning_model.hparams.lr = lr_finder.suggestion()
    trainer.fit(lightning_model, dataloader)
    return f"Vintage {vintage} done."