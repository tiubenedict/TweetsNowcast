import torch
import lightning.pytorch as pl
from dateutil.relativedelta import relativedelta
from data_utils import sliding_windows_ST, sliding_windows_MT, NowcastingLSTM_MQ, get_dataloader_for_vintage
from STSeq2One import STMFSeq2One, STMFSeq2OneLightning
from MTSeq2One import MTMFSeq2One, MTMFSeq2OneLightning


def _make_refit_trainer(epochs, device):
    """Trainer for stage-2 refit: trains on all data, no validation loop, no checkpointing."""
    return pl.Trainer(
        max_epochs=epochs, deterministic=True, accelerator=device,
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        num_sanity_val_steps=0, enable_checkpointing=False,
    )


def evaluate_one_vintage_ST(vintage, with_econ, with_tweets, kmpair, target, ckpt_path, config, device="cpu",
                            refit=False, refit_epochs=None):
    ### Train Scaler + Backcast/Test Data
    data_model = NowcastingLSTM_MQ()
    imputation = config.get('imputation', 'legacy') if isinstance(config, dict) else 'legacy'
    # DFM extension only in legacy mode. i1/i2 modes let the LSTM encoder handle NaN tails directly.
    extend_at_infer = (imputation == 'legacy')
    _, target_scaler, econ_scaler, tweets_scaler = data_model.load_data(vintage=vintage,window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=True, extend=False, DFM_order=(1,0,1,0), optimize_order = False, target=target)
    test_data, _, _, _ = data_model.load_data(vintage=vintage, window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=False, extend=extend_at_infer, DFM_order=(1,0,1,0), optimize_order = False, target=target)
    test_data = test_data.loc[vintage + relativedelta(months = -((vintage.month - 1) % 3) - config['data_window'] - (3 if vintage.month % 3 == 1 else 0), day=31):,:] # get first month of same qtr last year, but get final day. Extend one quarter if first month of quarter.
    test_data.iloc[:,:1] = target_scaler.transform(test_data.iloc[:,:1])
    if with_econ and with_tweets:
        econ_n_feat = econ_scaler.n_features_in_
        tweets_n_feat = tweets_scaler.n_features_in_
        test_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(test_data.iloc[:,1:tweets_n_feat+1])
        test_data.iloc[:,tweets_n_feat+1:] = econ_scaler.transform(test_data.iloc[:,tweets_n_feat+1:])
    elif with_econ:
        econ_n_feat = econ_scaler.n_features_in_
        test_data.iloc[:,1:econ_n_feat+1] = econ_scaler.transform(test_data.iloc[:,1:econ_n_feat+1])
    elif with_tweets:
        tweets_n_feat = tweets_scaler.n_features_in_
        test_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(test_data.iloc[:,1:tweets_n_feat+1])

    x_encoder_in, y_decoder_in, _, _ = sliding_windows_ST(test_data, train=False, seq_length=config['data_window'])
    testX_in = torch.Tensor(x_encoder_in)#.to(device)
    testY_in = torch.Tensor(y_decoder_in)#.to(device)

    if refit:
        # Stage-2: retrain the winning config on ALL released quarters (train_bias=True),
        # for refit_epochs (the stage-1 best epoch). No val loop — refit_mode points the
        # LR scheduler at train_loss. Replaces the frozen stage-1 checkpoint.
        pl.seed_everything(42, workers=True)
        train_loader, _, _, _, _ = get_dataloader_for_vintage(
            vintage, with_econ, with_tweets, kmpair=kmpair,
            data_window=config['data_window'], task='singletask',
            train_bias=True, imputation=imputation,
        )
        lightning_model = STMFSeq2OneLightning(
            dim_x=testX_in.shape[-1], dim_y=testY_in.shape[-1],
            learning_rate=config['learning_rate'], weight_decay=config['weight_decay'],
            num_layers=config['num_layers'],
            n_a=config.get('n_a', 4), n_s=config.get('n_s', 8),
            n_align=config.get('n_align', 4), fc_y=config.get('fc_y', 4),
            dropout_rate=config.get('dropout_rate', 0.0),
            encoder_type=config.get('encoder_type', 'autoregressive'),
            bidirectional=config.get('bidirectional', False),
            attention_relu=config.get('attention_relu', True),
            loss_fn=config.get('loss_fn', 'mse'), huber_delta=config.get('huber_delta', 1.0),
            imputation=imputation, refit_mode=True,
        )
        _make_refit_trainer(refit_epochs or config.get('epochs', 150), device).fit(lightning_model, train_loader)
    else:
        lightning_model = STMFSeq2OneLightning.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=STMFSeq2One(
                dim_x=testX_in.shape[-1], dim_y=testY_in.shape[-1],
                num_layers=config['num_layers'],
                n_a=config.get('n_a', 4), n_s=config.get('n_s', 8),
                n_align=config.get('n_align', 4), fc_y=config.get('fc_y', 4),
                dropout_rate=config.get('dropout_rate', 0.0),
                encoder_type=config.get('encoder_type', 'autoregressive'),
                bidirectional=config.get('bidirectional', False),
                attention_relu=config.get('attention_relu', True),
                imputation=imputation,
            ),
        )
    lightning_model = lightning_model.to(device)
    lightning_model.eval()
    with torch.no_grad():
        start_row = 3 if vintage.month % 3 == 1 else 0
        # start_row = 0       # start_row = 3 if vintage.month % 3 == 1 else 0 ### whether to start from the first row or the fourth row
        if vintage.month % 3 == 1:
            backcastY = lightning_model(testX_in[:,:-3],testY_in[:,:-1])
            testY_in[:,-2] = backcastY[0,-1]
            # testY_in[:,-2] = backcastY[0][-1]
        nowcastY = lightning_model(testX_in[:,start_row:],testY_in[:,start_row//3:])
        nowcastY = target_scaler.inverse_transform(nowcastY[0].cpu().numpy())
    return (vintage, nowcastY.flatten()[-1])

def evaluate_one_vintage_MT(vintage, with_econ, with_tweets, kmpair, target, ckpt_path, config, device="cpu",
                            refit=False, refit_epochs=None):
    ### Train Scaler + Backcast/Test Data
    data_model = NowcastingLSTM_MQ()
    imputation = config.get('imputation', 'legacy') if isinstance(config, dict) else 'legacy'
    _, target_scaler, econ_scaler, tweets_scaler = data_model.load_data(vintage=vintage,window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=True, extend=False, DFM_order=(1,0,1,0), optimize_order = False, target=target)
    test_data, _, _, _ = data_model.load_data(vintage=vintage, window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=False, extend=False, DFM_order=(1,0,1,0), optimize_order = False, target=target)
    test_data = test_data.loc[vintage + relativedelta(months = -((vintage.month - 1) % 3) - config['data_window'] - (3 if vintage.month % 3 == 1 else 0), day=31):,:] # get first month of same qtr last year, but get final day. Extend one quarter if first month of quarter.
    test_data.iloc[:,:1] = target_scaler.transform(test_data.iloc[:,:1])
    if with_econ and with_tweets:
        econ_n_feat = econ_scaler.n_features_in_
        tweets_n_feat = tweets_scaler.n_features_in_
        test_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(test_data.iloc[:,1:tweets_n_feat+1])
        test_data.iloc[:,tweets_n_feat+1:] = econ_scaler.transform(test_data.iloc[:,tweets_n_feat+1:])
    elif with_econ:
        econ_n_feat = econ_scaler.n_features_in_
        test_data.iloc[:,1:econ_n_feat+1] = econ_scaler.transform(test_data.iloc[:,1:econ_n_feat+1])
    elif with_tweets:
        tweets_n_feat = tweets_scaler.n_features_in_
        test_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(test_data.iloc[:,1:tweets_n_feat+1])

    x_encoder_in, y_decoder_in, _, _ = sliding_windows_MT(test_data, train=False, seq_length=config['data_window'])
    testX_in = torch.Tensor(x_encoder_in)#.to(device)
    testY_in = torch.Tensor(y_decoder_in)#.to(device)

    if refit:
        # Stage-2: retrain winning config on all released quarters (train_bias=True).
        pl.seed_everything(42, workers=True)
        train_loader, _, _, _, _ = get_dataloader_for_vintage(
            vintage, with_econ, with_tweets, kmpair=kmpair,
            data_window=config['data_window'], task='multitask',
            train_bias=True, imputation=imputation,
        )
        lightning_model = MTMFSeq2OneLightning(
            dim_x=testX_in.shape[-1], dim_y=testY_in.shape[-1],
            learning_rate=config['learning_rate'], weight_decay=config['weight_decay'],
            alpha=config['alpha'], num_layers=config['num_layers'],
            n_a=config.get('n_a', 4), n_s=config.get('n_s', 8),
            n_align=config.get('n_align', 4),
            fc_x=config.get('fc_x', 4), fc_y=config.get('fc_y', 4),
            dropout_rate=config.get('dropout_rate', 0.0),
            encoder_type=config.get('encoder_type', 'autoregressive'),
            bidirectional=config.get('bidirectional', False),
            attention_relu=config.get('attention_relu', True),
            loss_fn=config.get('loss_fn', 'mse'), huber_delta=config.get('huber_delta', 1.0),
            imputation=imputation, refit_mode=True,
        )
        _make_refit_trainer(refit_epochs or config.get('epochs', 150), device).fit(lightning_model, train_loader)
    else:
        lightning_model = MTMFSeq2OneLightning.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=MTMFSeq2One(
                dim_x=testX_in.shape[-1], dim_y=testY_in.shape[-1],
                num_layers=config['num_layers'],
                n_a=config.get('n_a', 4), n_s=config.get('n_s', 8),
                n_align=config.get('n_align', 4),
                fc_x=config.get('fc_x', 4), fc_y=config.get('fc_y', 4),
                dropout_rate=config.get('dropout_rate', 0.0),
                encoder_type=config.get('encoder_type', 'autoregressive'),
                bidirectional=config.get('bidirectional', False),
                attention_relu=config.get('attention_relu', True),
                imputation=imputation,
            ),
        )
    lightning_model = lightning_model.to(device)
    lightning_model.eval()
    with torch.no_grad():
        start_row = 3 if vintage.month % 3 == 1 else 0
        # start_row = 0       # start_row = 3 if vintage.month % 3 == 1 else 0 ### whether to start from the first row or the fourth row
        if vintage.month % 3 == 1:
            latent, _ = lightning_model.model.encoder_x(testX_in[:,:-3]) # encoder now returns (x_a, mask); mask unused for decoder_x backcast
            backcastX = lightning_model.model.decoder_x(latent)
            nan_mask = torch.isnan(testX_in[0, -4])                 # Find NaNs in the second to the last row of testX_in
            testX_in[0, -4][nan_mask] = backcastX[0, -1][nan_mask]  # Take last row of backcast and replace NaNs
            _, backcastY = lightning_model(testX_in[:,:-3],testY_in[:,:-1])
            testY_in[:,-2] = backcastY[0,-1]
        nowcastX_0, nowcastY_0 = lightning_model(testX_in[:,start_row:],testY_in[:,start_row//3:])
        nan_mask = torch.isnan(testX_in[0, -3:])                    # Find NaNs in the current quarter of testX_in
        testX_in[0, -3:][nan_mask] = nowcastX_0[0,-3:][nan_mask]    # Take last row of backcast and replace NaNs
        _, nowcastY_1 = lightning_model(testX_in[:,start_row:],testY_in[:,start_row//3:])
        nowcastY_0 = target_scaler.inverse_transform(nowcastY_0[0].cpu().numpy())
        nowcastY_1 = target_scaler.inverse_transform(nowcastY_1[0].cpu().numpy())
    # return (vintage, nowcastY_0.flatten()[-1], nowcastY_1.flatten()[-1])
    return (vintage, nowcastY_1.flatten()[-1])