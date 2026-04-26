import torch
from dateutil.relativedelta import relativedelta
from data_utils import sliding_windows_ST, sliding_windows_MT, NowcastingLSTM_MQ
from STSeq2One import STMFSeq2One, STMFSeq2OneLightning
from MTSeq2One import MTMFSeq2One, MTMFSeq2OneLightning

def evaluate_one_vintage_ST(vintage, with_econ, with_tweets, kmpair, target, ckpt_path, config, device="cpu"):
    ### Train Scaler + Backcast/Test Data
    data_model = NowcastingLSTM_MQ()
    _, target_scaler, econ_scaler, tweets_scaler = data_model.load_data(vintage=vintage,window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=True, extend=False, DFM_order=(1,0,1,0), optimize_order = False, target=target)
    test_data, _, _, _ = data_model.load_data(vintage=vintage, window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=False, extend=True, DFM_order=(1,0,1,0), optimize_order = False, target=target)
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

    # lightning_model = MTMFSeq2OneLightning.load_from_checkpoint(checkpoint_path=f"lightning_logs/version{version}/checkpoints/model.ckpt", model=model) # checkpoints/vintage_{vintage}/model{version}.ckpt
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

def evaluate_one_vintage_MT(vintage, with_econ, with_tweets, kmpair, target, ckpt_path, config, device="cpu"):
    ### Train Scaler + Backcast/Test Data
    data_model = NowcastingLSTM_MQ()
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

    # lightning_model = MTMFSeq2OneLightning.load_from_checkpoint(checkpoint_path=f"lightning_logs/version{version}/checkpoints/model.ckpt", model=model) # checkpoints/vintage_{vintage}/model{version}.ckpt
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