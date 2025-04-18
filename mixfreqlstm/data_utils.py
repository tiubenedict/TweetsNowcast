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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from NowcastingPipelineM import NowcastingPH_M
import dynamicfactoranalysis.dynamicfactoranalysis as dfa

class NowcastingLSTM_MQ(NowcastingPH_M):
    def set_classname(self, **kwargs):
        self.prefix = f'LSTM{self.kwargs.get("lag_order")} x ' + ('DFM_Opt' if self.kwargs.get("optimize_order") else f'DFM{self.kwargs.get("DFM_order")}') if self.kwargs.get("extend") else f'LSTM{self.kwargs.get("lag_order")}'
    def load_tweets(self, vintage, window, kmpair, freq='M', extend=False, **kwargs):
        vintage = pd.to_datetime(vintage)
        tweets = pd.read_csv('data/PH_Tweets_v3.csv')
        tweets['date'] = pd.to_datetime(tweets['date']) + pd.offsets.MonthEnd(0)
        tweets = tweets.set_index('date')

        if len(kmpair) == 0:
            kmpair = {keyword: list(tweets.columns.drop('keyword')) for keyword in tweets['keyword'].unique()}
        data = [tweets[tweets['keyword'] == keyword][kmpair[keyword]].add_suffix(f'_{keyword}') for keyword in kmpair.keys()]
        tweets = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), data)
        tweets = tweets.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
        # tweets = tweets.loc[pd.to_datetime(vintage)  - relativedelta(months =  (pd.to_datetime(vintage).month - 1)%3 + window) : pd.to_datetime(vintage), :]
        # tweets = super().load_tweets(vintage, freq='M', **kwargs)
        # DFM_order = self.kwargs.get('DFM_order')                                             ### temporary measure to solve stationarity error
        # kwargs['DFM_order'] = (1, DFM_order[1], DFM_order[2], DFM_order[3])                   ### temporary measure to solve stationarity error
        tweets = self.extend_data(tweets, vintage, **kwargs) if extend else tweets
        tweets.index = pd.PeriodIndex(tweets.index, freq=freq)
        return tweets
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

def sliding_windows(data_x, seq_length, freq_ratio = 3):
    '''
    seq_length = number of rows of historical data + number of rows of input data (i.e. 12 + 3)
    '''
    x_encoder_in = []
    y_decoder_in = []
    x_target = []
    y_target = []

    for i in range(0,len(data_x)-seq_length+1, freq_ratio):
        _x_in = data_x.iloc[i:(i+seq_length),1:] ## 15 months length, skips every 3 rows
        _y_in = data_x.iloc[i+freq_ratio-1:i+seq_length-freq_ratio:freq_ratio, :1] ## gets every rth (3rd) row but stops before the current low-freq/quarter vintage
        # _x_out = data_x.iloc[i+seq_length, 1:] ## < 16th month x?
        _y_out = data_x.iloc[i+seq_length-1, :1] ## < final low freq/quarter vintage
        x_encoder_in.append(_x_in)
        y_decoder_in.append(_y_in)
        # x_target.append(_x_out)
        y_target.append(_y_out)
    
    return np.array(x_encoder_in),np.array(y_decoder_in), np.array(x_target), np.array(y_target)

def get_dataloader_for_vintage(vintage, mode="train"):
    data_model = NowcastingLSTM_MQ(target='GDP')
    train_data, target_scaler, econ_scaler, tweets_scaler = data_model.load_data(vintage=vintage,window=1000, kmpair={'PE': ['CRVADER_BVN','CR_BxP_0'],'PU+': ['CRVADER_BVN','CR_BxP_0']}, with_econ=False, with_tweets=True, target_release_lag=True,scaled=True, extend=False, DFM_order=(1,0,1,0), optimize_order = False)
    train_data = train_data.dropna()

    x_encoder_in, y_decoder_in, x_target, y_target = sliding_windows(train_data, seq_length=27)
    trainX_in = torch.Tensor(x_encoder_in)#.to(device)
    trainY_in = torch.Tensor(y_decoder_in)#.to(device)
    trainX_out = torch.Tensor(y_target)#.to(device)
    trainY_out = torch.Tensor(y_target)#.to(device)

    dataset = TensorDataset(trainX_in, trainY_in, trainY_out)
    return DataLoader(dataset, batch_size=200, shuffle=(mode=="train"), num_workers=2), target_scaler, econ_scaler, tweets_scaler

