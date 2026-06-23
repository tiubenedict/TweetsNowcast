from functools import reduce
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta
import torch
from torch.utils.data import DataLoader, TensorDataset

import os
import sys
# sys.path.append('..')
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# os.chdir('..')

from NowcastingPipelineM import NowcastingPH_M
import dynamicfactoranalysis.dynamicfactoranalysis as dfa

import warnings
from statsmodels.tools.sm_exceptions import  ValueWarning
warnings.simplefilter('ignore', ValueWarning)

class NowcastingLSTM_MQ(NowcastingPH_M):
    def set_classname(self, **kwargs):
        self.prefix = f'LSTM{self.kwargs.get("lag_order")} x ' + ('DFM_Opt' if self.kwargs.get("optimize_order") else f'DFM{self.kwargs.get("DFM_order")}') if self.kwargs.get("extend") else f'LSTM{self.kwargs.get("lag_order")}'
    # def load_tweets(self, vintage, window, kmpair, freq='M', extend=False, **kwargs):
    #     vintage = pd.to_datetime(vintage)
    #     tweets = pd.read_csv('data/PH_Tweets_v3.csv')
    #     tweets['date'] = pd.to_datetime(tweets['date']) + pd.offsets.MonthEnd(0)
    #     tweets = tweets.set_index('date')

    #     if len(kmpair) == 0:
    #         kmpair = {keyword: list(tweets.columns.drop('keyword')) for keyword in tweets['keyword'].unique()}
    #     data = [tweets[tweets['keyword'] == keyword][kmpair[keyword]].add_suffix(f'_{keyword}') for keyword in kmpair.keys()]
    #     tweets = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), data)
    #     tweets = tweets.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
    #     # tweets = tweets.loc[pd.to_datetime(vintage)  - relativedelta(months =  (pd.to_datetime(vintage).month - 1)%3 + window) : pd.to_datetime(vintage), :]
    #     # tweets = super().load_tweets(vintage, freq='M', **kwargs)
    #     # DFM_order = self.kwargs.get('DFM_order')                                             ### temporary measure to solve stationarity error
    #     # kwargs['DFM_order'] = (1, DFM_order[1], DFM_order[2], DFM_order[3])                   ### temporary measure to solve stationarity error
    #     tweets = self.extend_data(tweets, vintage, **kwargs) if extend else tweets
    #     tweets.index = pd.PeriodIndex(tweets.index, freq=freq)
    #     return tweets
    def extend_data(self, df, vintage, DFM_order, optimize_order=False, **kwargs):
        ### Instead of extending until year end, just extend until current quarter end
        factor_order, error_order, k_factors, factor_lag = DFM_order
        # drop row if not enough non-missing (max safety)
        # df = df.dropna(thresh = k_factors * (1 + factor_lag))

        if optimize_order:
            model = dfa.DynamicFactorModelOptimizer(
                endog=df, k_factors_max=k_factors, factor_lag_max=factor_lag, factor_order_max=factor_order, 
                error_order_max=error_order, verbose=True,**kwargs).fit(**kwargs)
        else:
            model = dfa.DynamicFactorModel(
                endog=df, k_factors=k_factors, factor_lag=factor_lag, factor_order=factor_order, 
                error_order=error_order, **kwargs)
        results = model.fit(disp=False, maxiter=10, method='powell', ftol=1e-3, **kwargs)
        # results = model.fit(disp=False, maxiter=1000, method='powell', ftol=1e-5, **kwargs)
        
        df_extended = pd.DataFrame()
        for col in df.columns:
            col_extended = pd.concat([df[[col]].dropna(), 
                                    results.predict(start=df[col].dropna().index[-1], end=vintage + pd.offsets.QuarterEnd(0))[[col]].iloc[1:]])
            df_extended = pd.concat([df_extended, col_extended], axis=1)
        df_extended.index.name = df.index.name

        return df_extended

def sliding_windows_ST(df, seq_length, train=True, freq_ratio = 3):
    '''
    seq_length = number of rows of historical data + number of rows of input data (i.e. 12 + 3)
    '''
    x_encoder_in = []
    y_encoder_in = []
    x_target = []
    y_target = []
    def pad_nans(data, target_length, columns=None):
        if data.shape[0] < target_length:
            n_pad = target_length - data.shape[0]
            pad = np.full((n_pad, data.shape[1]), np.nan)
            data_np = np.vstack([data.values, pad])
            data = pd.DataFrame(data_np, columns=columns if columns is not None else data.columns)
        return data
    if train:
        # Loop bound ensures _x_in always has the full seq_length+freq_ratio rows
        # (pandas iloc silently truncates past the end → inhomogeneous shapes
        # at np.array time). Originally masked by dropna shortening the matrix.
        for i in range(0, len(df) - seq_length - freq_ratio + 1, freq_ratio):
            _x_in = df.iloc[i:(i+seq_length+freq_ratio),1:] ## 15 months length, skips every 3 rows
            _y_in = df.iloc[i+freq_ratio-1:i+seq_length:freq_ratio, :1] ## gets every rth (3rd) row but stops before the current low-freq/quarter vintage
            # _x_out = df.iloc[i+seq_length, 1:] ## < 16th month x?
            _y_out = df.iloc[i+freq_ratio-1:i+seq_length+freq_ratio:freq_ratio, :1] ## < final low freq/quarter vintage
            # Skip windows where Y is missing — necessary when data isn't pre-dropna'd (i1/i2 imputation modes).
            # No-op when data has been dropna'd (legacy path).
            if _y_in.isna().any().any() or _y_out.isna().any().any():
                continue
            data_length_y = seq_length // freq_ratio + 1
            _y_in = pad_nans(_y_in, data_length_y, columns=df.columns[:1])
            x_encoder_in.append(_x_in)
            y_encoder_in.append(_y_in)
            # x_target.append(_x_out)
            y_target.append(_y_out)
    else:
        _x_in = df.iloc[:,1:]
        _y_in = df.iloc[freq_ratio-1::freq_ratio, :1]           ## gets every rth (3rd) row 
        data_length_y = seq_length // freq_ratio + 1
        _y_in = pad_nans(_y_in, data_length_y, columns=df.columns[:1])
        x_encoder_in.append(_x_in)
        y_encoder_in.append(_y_in)

    # return np.array(x_encoder_in),np.array(y_decoder_in), np.array(x_target), np.array(y_target)
    return torch.tensor(np.array(x_encoder_in), dtype=torch.float32), torch.tensor(np.array(y_encoder_in), dtype=torch.float32), torch.tensor(np.array(x_target), dtype=torch.float32), torch.tensor(np.array(y_target), dtype=torch.float32)


def sliding_windows_qtr_jump(data_x, seq_length, freq_ratio = 3):
    '''
    seq_length = number of rows of historical data + number of rows of input data (i.e. 12 + 3)
    '''
    x_encoder_in = []
    y_encoder_in = []
    x_target = []
    y_target = []

    for i in range(0,len(data_x)-seq_length+1, freq_ratio):
        if i + seq_length > len(data_x) - 1:
            break
        _x_in = data_x.iloc[i:(i+seq_length),1:] ## 15 months length
        _y_in = data_x.iloc[i+freq_ratio-1:i+seq_length-freq_ratio:freq_ratio, :1] ## gets every rth (3rd) row but stops before the current low-freq/quarter vintage
        _x_out = data_x.iloc[i+seq_length, 1:] ## < 16th month x?
        _y_out = data_x.iloc[i+seq_length-1, :1] ## < final low freq/quarter vintage
        x_encoder_in.append(_x_in)
        y_encoder_in.append(_y_in)
        x_target.append(_x_out)
        y_target.append(_y_out)
    
    # return np.array(x_encoder_in),np.array(y_decoder_in), np.array(x_target), np.array(y_target)
    return torch.tensor(np.array(x_encoder_in), dtype=torch.float32), torch.tensor(np.array(y_encoder_in), dtype=torch.float32), torch.tensor(np.array(x_target), dtype=torch.float32), torch.tensor(np.array(y_target), dtype=torch.float32)

def sliding_windows_MT(df, seq_length, train=True, freq_ratio = 3):
    '''
    seq_length = number of rows of historical data + number of rows of input data (i.e. 12 + 3)
    Note: in current version, start of data must be 1st month of a quarter (i.e. Jan, Apr, Jul, Oct) for the sliding window to work correctly.
    '''
    x_encoder_in = []
    y_encoder_in = []
    x_target = []
    y_target = []
    def pad_nans(data, target_length, columns=None):
        if data.shape[0] < target_length:
            n_pad = target_length - data.shape[0]
            pad = np.full((n_pad, data.shape[1]), np.nan)
            data_np = np.vstack([data.values, pad])
            data = pd.DataFrame(data_np, columns=columns if columns is not None else data.columns)
        return data
    if train:
        # print(len(df)-seq_length)
        # Bound = largest i with a valid df.index[i+seq_length] reference month.
        # Earlier `- freq_ratio + 1` was over-conservative: it reserved the full
        # freq_ratio look-ahead for EVERY window, so the most-recent quarter kept
        # only its M1-reference window and lost the M2/M3 vintages (whose targets
        # are present). pad_nans + the NaN-_y_out filter below already drop any
        # window that genuinely needs absent future data, so this looser bound is
        # safe and makes validation symmetric (3 vintages per held-out quarter).
        for i in range(0, len(df) - seq_length):
            month_idx = (df.iloc[[i+seq_length]].index[0].month - 1) % 3
            steps = 3 - month_idx
            # print("i:", i, "month_idx: ", month_idx, " steps: ", steps)
            # print(i + seq_length + steps, "len(df) - 1: ", len(df))
            # if i + seq_length + steps > len(df) - 1:
            #     break
            _x_in = df.iloc[i-month_idx:(i+seq_length),1:]                      ## M1: window + 0, M2: window + 1, M3: window + 2
            _y_in = df.iloc[i+steps-1:i+seq_length:freq_ratio, :1]              ## gets every rth (3rd) row but stops before the current low-freq/quarter vintage
            _x_out = df.iloc[i-month_idx:i+seq_length+freq_ratio-month_idx, 1:] ## < 13th to 15th month
            _y_out = df.iloc[i+steps-1:i+seq_length+steps:freq_ratio, :1]       ## < Next period low freq / (Q5) quarter vintage
            # Skip windows where Y is missing — necessary when data isn't pre-dropna'd (i1/i2 imputation modes).
            # No-op when data has been dropna'd (legacy path).
            if _y_in.isna().any().any() or _y_out.isna().any().any():
                continue
            data_length_x = seq_length + freq_ratio
            _x_in = pad_nans(_x_in, data_length_x, columns=df.columns[1:])
            _x_out = pad_nans(_x_out, data_length_x, columns=df.columns[1:])
            data_length_y = seq_length // freq_ratio + 1
            _y_in = pad_nans(_y_in, data_length_y, columns=df.columns[:1])
            _y_out = pad_nans(_y_out, data_length_y, columns=df.columns[:1])
            # display(_x_in)
            # display(_y_in)
            # display(_x_out)
            # display(_y_out)
            x_encoder_in.append(_x_in)
            y_encoder_in.append(_y_in)
            x_target.append(_x_out)
            y_target.append(_y_out)
    else:
        _x_in = df.iloc[:,1:]                                   ## Get all data except the target column
        _y_in = df.iloc[freq_ratio-1::freq_ratio, :1]            ## Get every rth (3rd) row for low-freq/quarter vintage (5, 4, 5)
        month_idx = (df.index[-1].month - 1) % 3                ## Pad the input data to ensure it has the same length as the training data
        data_length_x = len(df) + freq_ratio - 1 - month_idx    ## seq_length=12: 18, 15, 15 (16+3-1-0=18, 14+3-1-1=15, 15+3-1-2=15) / 15 15 15
        # print("month_idx:", month_idx, "data_length_x:", data_length_x)
        _x_in = pad_nans(_x_in, data_length_x, columns=df.columns[1:])
        data_length_y = len(_y_in) + (1 if len(df) % 3 != 0 else 0) ## seq_length=12: 6, 5, 5 (5+1, 4+1, 5+0) / 5 5 5
        _y_in = pad_nans(_y_in, data_length_y, columns=df.columns[:1])
        # display(_x_in)
        # display(_y_in)
        x_encoder_in.append(_x_in)
        y_encoder_in.append(_y_in)

    return torch.tensor(np.array(x_encoder_in), dtype=torch.float32), torch.tensor(np.array(y_encoder_in), dtype=torch.float32), torch.tensor(np.array(x_target), dtype=torch.float32), torch.tensor(np.array(y_target), dtype=torch.float32)

def get_dataloader_for_vintage(vintage, with_econ, with_tweets, kmpair, data_window, task, mode="train", train_bias=False, walk_n=2, imputation="legacy"):
    data_model = NowcastingLSTM_MQ()
    data, target_scaler, econ_scaler, tweets_scaler = data_model.load_data(vintage=vintage,window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=True,scaled=True, extend=False, DFM_order=(1,0,1,0), optimize_order = False, target='PHL_GDP_SA')
    # Imputation mode controls how NaN tails are handled:
    #   "legacy"  : drop NaN rows; encoder never sees NaN. DFM imputation is applied at inference only (in evaluate_vintage).
    #   "i1_pc"   : keep NaN rows; modified Encoder uses PredictionCell on fully-NaN tail + zero-fill on partial-NaN.
    #   "i2_attn" : keep NaN rows; modified PreAttnEncoder zero-fills NaN and masks the fully-NaN trailing positions.
    if imputation == "legacy":
        data = data.dropna()
    # Else: keep NaN tail; the encoder handles it. Sliding-window NaN-Y filter prevents NaN targets.
    # Forward-walk split:
    #   train_bias=False (default): drop last walk_n*3 rows so the last walk_n quarters are held out of training.
    #                               Validation last-1 target lands on the most recent observed quarter.
    #                               walk_n=2: training ends 1 quarter before val target (gap of 1 quarter).
    #                               walk_n=1: training ends adjacent to val target (no gap, more recent training data).
    #   train_bias=True: include everything in training (leaky; explicit opt-in for comparison runs only).
    drop = walk_n * 3
    train_data = data.iloc[:] if train_bias else data.iloc[:-drop]
    if imputation == "legacy":
        val_data = data.iloc[-data_window-drop:]
    else:
        # i1/i2: data extends beyond the latest released quarter (NaN target tail).
        # Trim that tail before slicing val_data, otherwise every stride-3 val
        # window's _y_out lands on a NaN-target row → all windows get filtered
        # → empty val_loader → ReduceLROnPlateau fails to find val_loss_y.
        # Legacy's dropna had this effect implicitly.
        target_col = data.columns[0]
        last_valid = data[target_col].last_valid_index()
        if last_valid is not None:
            val_data = data.loc[:last_valid].iloc[-data_window-drop:]
        else:
            val_data = data.iloc[-data_window-drop:]
    # val_data, _, _, _ = data_model.load_data(vintage=vintage+pd.offsets.QuarterEnd(0)+pd.DateOffset(months=2),window=1000, kmpair=kmpair, with_econ=with_econ, with_tweets=with_tweets, target_release_lag=False, scaled=False, extend=False, DFM_order=(1,0,1,0), optimize_order = False, target='PHL_GDP_SA')
    # val_data = val_data.dropna().iloc[-data_window-3:]
    # val_data.iloc[:,:1] = target_scaler.transform(val_data.iloc[:,:1])
    # if with_econ and with_tweets:
    #     econ_n_feat = econ_scaler.n_features_in_
    #     tweets_n_feat = tweets_scaler.n_features_in_
    #     val_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(val_data.iloc[:,1:tweets_n_feat+1])
    #     val_data.iloc[:,tweets_n_feat+1:] = econ_scaler.transform(val_data.iloc[:,tweets_n_feat+1:])
    # elif with_econ:
    #     econ_n_feat = econ_scaler.n_features_in_
    #     val_data.iloc[:,1:econ_n_feat+1] = econ_scaler.transform(val_data.iloc[:,1:econ_n_feat+1])
    # elif with_tweets:
    #     tweets_n_feat = tweets_scaler.n_features_in_
    #     val_data.iloc[:,1:tweets_n_feat+1] = tweets_scaler.transform(val_data.iloc[:,1:tweets_n_feat+1])

    if task == 'singletask':
        trainX_in, trainY_in, _, trainY_out = sliding_windows_ST(train_data, train=True, seq_length=data_window)
        valX_in, valY_in, _, valY_out = sliding_windows_ST(val_data, train=True, seq_length=data_window)
        train_dataset = TensorDataset(trainX_in, trainY_in, trainY_out)
        val_dataset = TensorDataset(valX_in, valY_in, valY_out)
    elif task == 'multitask':
        trainX_in, trainY_in, trainX_out, trainY_out = sliding_windows_MT(train_data, train=True, seq_length=data_window)
        valX_in, valY_in, valX_out, valY_out = sliding_windows_MT(val_data, train=True, seq_length=data_window)
        train_dataset = TensorDataset(trainX_in, trainY_in, trainX_out, trainY_out)
        val_dataset = TensorDataset(valX_in, valY_in, valX_out, valY_out)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False)
    
    return train_loader, val_loader, target_scaler, econ_scaler, tweets_scaler