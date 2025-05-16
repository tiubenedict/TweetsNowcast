import torch
from dateutil.relativedelta import relativedelta
from data_utils import sliding_windows, NowcastingLSTM_MQ
from STSeq2One import STMFSeq2One, MTMFSeq2SeqLightning

def evaluate_one_vintage(vintage, with_econ, with_tweets, data_window, kmpair={}, version="", device="cpu"):
    ### Train Scaler + Backcast/Test Data
    data_model = NowcastingLSTM_MQ(target='GDP')
    train_data, target_scaler, econ_scaler, tweets_scaler = data_model.load_data(vintage=vintage,window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=True, extend=False, DFM_order=(1,0,1,0), optimize_order = False)
    backcast_data, _, _, _ = data_model.load_data(vintage=vintage, window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=False, extend=True, DFM_order=(1,0,1,0), optimize_order = False)
    backcast_data = backcast_data.loc[vintage + relativedelta(months = -((vintage.month - 1) % 3) - data_window, day=31):,:]#.dropna()#.reset_index()  # get first month of same qtr last year, but get final day. Based on window = 12
    backcast_data.iloc[:,:1] = target_scaler.transform(backcast_data.iloc[:,:1])

    if with_econ and with_tweets:
        econ_n_feat = econ_scaler.n_features_in_
        tweets_n_feat = tweets_scaler.n_features_in_
        backcast_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(backcast_data.iloc[:,1:tweets_n_feat+1])
        backcast_data.iloc[:,tweets_n_feat+1:] = econ_scaler.transform(backcast_data.iloc[:,tweets_n_feat+1:])
        dim_x = econ_n_feat + tweets_n_feat
    elif with_econ:
        econ_n_feat = econ_scaler.n_features_in_
        backcast_data.iloc[:,1:econ_n_feat+1] = econ_scaler.transform(backcast_data.iloc[:,1:econ_n_feat+1])
        dim_x = econ_n_feat
    elif with_tweets:
        tweets_n_feat = tweets_scaler.n_features_in_
        backcast_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(backcast_data.iloc[:,1:tweets_n_feat+1])
        dim_x = tweets_n_feat
    
    x_encoder_backcast_in, y_decoder_backcast_in, _, _ = sliding_windows(backcast_data, seq_length=data_window+3)
    backcastX_in = torch.Tensor(x_encoder_backcast_in)#.to(device)
    backcastY_in = torch.Tensor(y_decoder_backcast_in)#.to(device)

    ### Load Model
    # dim_x = 9                 #trainX_in.shape[-1] # 9 # Number of features
    dim_y = 1                   #trainY_in.shape[-1] # 1 # Number of targets
    Lx = data_window            #trainX_in.shape[1] # 15 or 27 # Window length of x_input sequence (num months)
    Ty = data_window // 3 - 1   #trainY_in.shape[1] # 4 or 8 # Window length of y_input sequence (num quarters)
    n_a = 4
    n_s = 8
    n_align = 4
    fc_y = 4
    dropout_rate = 0
    freq_ratio = 3
    bidirectional_encoder = False
    model = STMFSeq2One(dim_x=dim_x,dim_y=dim_y,Lx=Lx,Ty=Ty,n_a=n_a,n_s=n_s,n_align=n_align,fc_y=fc_y,dropout_rate=dropout_rate,freq_ratio=freq_ratio,bidirectional_encoder=bidirectional_encoder)
    lightning_model = MTMFSeq2SeqLightning.load_from_checkpoint(checkpoint_path=f"checkpoints/vintage_{vintage}/model{version}.ckpt", model=model)

    lightning_model.eval()
    with torch.no_grad():
            ### Check if vintage is first month in a quarter
        if vintage.month % 3 == 1:
            backcastY = lightning_model(backcastX_in[:,:-3],backcastY_in[:,:-1])
            backcastY_in[:,-1] = backcastY.item() # backcastY_in = torch.cat([backcastY_in, backcastY.unsqueeze(1)], dim=1)
        nowcastY = lightning_model(backcastX_in[:,3:],backcastY_in[:,1:])
        nowcastY = target_scaler.inverse_transform(nowcastY)
    return (vintage, nowcastY)