import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
# from lightning.pytorch.tuner.tuning import Tuner
# from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from STSeq2One import STMFSeq2OneLightning
from MTSeq2One import MTMFSeq2OneLightning
from data_utils import get_dataloader_for_vintage

def train_model(config, vintage, with_econ, with_tweets, kmpair, task, ckpt_path=None, logger_enabled=False, device='cpu', train_bias=False, walk_n=2):
    pl.seed_everything(42, workers=True)
    train_loader, val_loader, _,_,_ = get_dataloader_for_vintage(vintage, with_econ, with_tweets, kmpair=kmpair, data_window = config['data_window'], task=task, train_bias=train_bias, walk_n=walk_n)
    # Architecture knobs — read from config with defaults matching historical fixed values.
    arch_kwargs = dict(
        n_a=config.get('n_a', 4),
        n_s=config.get('n_s', 8),
        n_align=config.get('n_align', 4),
        dropout_rate=config.get('dropout_rate', 0.0),
        encoder_type=config.get('encoder_type', 'autoregressive'),
        bidirectional=config.get('bidirectional', False),
        attention_relu=config.get('attention_relu', True),
        loss_fn=config.get('loss_fn', 'mse'),
        huber_delta=config.get('huber_delta', 1.0),
    )
    if task == "singletask":
        model = STMFSeq2OneLightning(
            dim_x=train_loader.dataset.tensors[0].shape[-1],
            dim_y=train_loader.dataset.tensors[1].shape[-1],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            num_layers=config['num_layers'],
            fc_y=config.get('fc_y', 4),
            **arch_kwargs,
        )
        tune_callback = TuneReportCheckpointCallback(metrics={'val_loss_y': 'val_loss_y', 'train_loss': 'train_loss'})
        checkpoint_callback = ModelCheckpoint(monitor="val_loss_y", mode="min", save_top_k=1, save_last=True)
    elif task == "multitask":
        model = MTMFSeq2OneLightning(
            dim_x=train_loader.dataset.tensors[0].shape[-1], # type: ignore
            dim_y=train_loader.dataset.tensors[1].shape[-1], # type: ignore
            num_layers=config['num_layers'],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            alpha=config['alpha'],
            fc_x=config.get('fc_x', 4),
            fc_y=config.get('fc_y', 4),
            **arch_kwargs,
        )
        tune_callback = TuneReportCheckpointCallback(metrics={'val_loss_y': 'val_loss_y', 'val_loss_x': 'val_loss_x', 'val_loss': 'val_loss','train_loss': 'train_loss'})
        checkpoint_callback = ModelCheckpoint(monitor="val_loss_y", mode="min", save_top_k=1, save_last=True)
    # checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/vintage_{vintage}", filename="model", save_top_k=1, monitor="val_loss_y", mode="min", save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        callbacks=[tune_callback] + ([checkpoint_callback, lr_monitor] if logger_enabled else []),
        enable_checkpointing=True,
        deterministic=True,
        accelerator=device,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=logger_enabled,
        log_every_n_steps=1 if logger_enabled else 50
    )
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, train_loader)
    # model.hparams.learning_rate = lr_finder.suggestion()
    # print(f"Suggested learning rate: {model.hparams.learning_rate}")
    if ckpt_path is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path = ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader)