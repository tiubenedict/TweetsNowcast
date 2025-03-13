import os
import math
import numpy as np
import pandas as pd
import datetime as dt
from functools import reduce
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
import dynamicfactoranalysis as dfa

import concurrent
# For multiprocess to work in MacOS
import multiprocessing as mp
mp_fork = mp.get_context('fork')

class NowcastingPipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_quarter(self, date):
        date = pd.to_datetime(date)
        return f'{date.year}Q{math.ceil(date.month/3)}'

    def load_target(self, vintage, growth, quarterly, freq, target_release_lag=True, **kwargs):
        raise NotImplementedError
    
    def quarter_to_annual(self, vintage, nowcasts, **kwargs):
        vintage = pd.to_datetime(vintage)
        annual_target = self.load_target(vintage, growth=False, quarterly=False, freq='Y', **kwargs)
        quarter_target = self.load_target(vintage, growth=False, quarterly=True, freq='Q', **kwargs)

        if len(nowcasts) + 1 <= math.ceil(vintage.month / 3):
            return np.nan, np.nan
        try:
            nowcasts_ = nowcasts.copy()
            for m in range(4):
                quarter = 3 * (m + 1)
                prev_quarter = vintage - relativedelta(years=1) + relativedelta(month=quarter)
                nowcasts_[m] = quarter_target.loc[prev_quarter, 'target'] * (1 + nowcasts[m] / 100)
                curr_quarter = vintage + relativedelta(month=quarter)
                nowcasts_[m] = quarter_target.loc[curr_quarter, 'target'] if curr_quarter in quarter_target.index else nowcasts_[m]

            prev_year = vintage - relativedelta(years=1)
            return 100 * (np.sum(nowcasts_) / annual_target.loc[prev_year, 'target'] - 1), nowcasts[math.ceil(vintage.month / 3) - 1]
        except:
            return np.nan, nowcasts[math.ceil(vintage.month / 3) - 1]

    def fit_model(self, vintage, window, **kwargs):
        raise NotImplementedError

    def error_message(self, vintage, error):
        vintage = pd.to_datetime(vintage)
        print(f'Vintage: {vintage.date()} \t {error}')

    def process_vintage(self, vintage_now, window, annual_target, quarter_target, **kwargs):
        annual = annual_target.loc[vintage_now, 'target']
        quarter = quarter_target.loc[vintage_now, 'target']
        try:
            nowcasts, model_desc = self.fit_model(vintage_now, window, **kwargs)
            nowcast_annual, nowcast_quarter = self.quarter_to_annual(vintage_now, nowcasts, **kwargs)
            
            res = [vintage_now, self.to_quarter(vintage_now), model_desc, nowcast_annual, annual, nowcast_quarter, quarter] + list(nowcasts)
            print(f'Vintage: {vintage_now.date()} \t {model_desc}')
                
        except Exception as ex:
            res = [vintage_now, self.to_quarter(vintage_now), 'No model', np.nan, annual, np.nan, quarter, np.nan, np.nan, np.nan, np.nan]
            self.error_message(vintage_now, ex)
        return res
    
    def run(self, start, end, delta, window, multiprocess, **kwargs):
        self.prefix = self.__class__.__name__
        print(f'Running {self.prefix}')

        annual_target = self.load_target(end + relativedelta(years=1), growth=True, quarterly=False, freq='Y', target_release_lag=False, **self.kwargs, **kwargs)
        quarter_target = self.load_target(end + relativedelta(years=1), growth=True, quarterly=True, freq='Q', target_release_lag=False, **self.kwargs, **kwargs)

        summary = []
        vintage_now = start
        
        if multiprocess > 1:
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp_fork, max_workers=multiprocess) as executor:
                while vintage_now < end:
                    summary.append(executor.submit(self.process_vintage, vintage_now, window, annual_target, quarter_target, **self.kwargs, **kwargs))
                    vintage_now += delta
                summary = [process.result() for process in summary]
        else:
            while vintage_now < end:
                summary.append(self.process_vintage(vintage_now, window, annual_target, quarter_target, **self.kwargs, **kwargs))
                vintage_now += delta

        summary = pd.DataFrame(summary, columns=['date', 'Quarter', 'Model', 'Nowcast_A', 'Actual_A', 'Nowcast_Q', 'Actual_Q'] + [f'ForecastQ{q}' for q in range(1,5)])
        summary['date'] = pd.to_datetime(summary['date'])

        return summary

class NowcastingPH_M(NowcastingPipeline):
    def extend_data(self, df, vintage, DFM_order, optimize_order=False, **kwargs):
        factor_order, error_order, k_factors, factor_lag = DFM_order
        # drop row if not enough non-missing (max safety)
        df = df.dropna(thresh = k_factors * (1 + factor_lag))

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
                                    results.predict(start=df[col].dropna().index[-1], end=vintage + pd.offsets.YearEnd(0))[[col]].iloc[1:]])
            df_extended = pd.concat([df_extended, col_extended], axis=1)
        df_extended.index.name = df.index.name

        return df_extended
    def lag_data(self, df, lag_order):
        target_lag, tweet_lag, econ_lag = lag_order
        lagged_df = ([df] + [df[['target']].shift(l * 3).add_suffix(f'.Q{l}') for l in range(1, target_lag + 1)] + 
                        [df[[col for col in df.columns if 'TWT' in col]].shift(l).add_suffix(f'.L{l}') for l in range(1, tweet_lag + 1)] +
                        [df[[col for col in df.columns if 'ECN' in col]].shift(l).add_suffix(f'.L{l}') for l in range(1, econ_lag + 1)])
        df = pd.concat(lagged_df, axis=1)
        
        df = df.loc[:, ~df.T.duplicated(keep='first')]
        return df
    def load_target(self, vintage, target='GDP', growth=True, quarterly=True, freq='M', target_release_lag=True, **kwargs):
        vintage = pd.to_datetime(vintage)
        df = pd.read_excel('data/PH_Econ_Q.xlsx', sheet_name='data')[['date', target]].rename(columns={target: 'target'}).dropna()
        df['date'] = pd.to_datetime(df['date'], format="%m/%Y") + pd.offsets.MonthEnd(0)
        df = df.set_index('date')

        meta = pd.read_excel('data/PH_Econ_Q.xlsx', sheet_name='release')
        target_release_lag = meta.set_index('Variable Name').to_dict('dict')['Lag'][target] if target_release_lag else 0

        if quarterly:
            df = df.resample('QE').sum()
            df = 100 * (df / df.shift(4) - 1) if growth else df
        else:
            df = df.resample('YE').sum()
            df = 100 * (df / df.shift(1) - 1) if growth else df

        df = df.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
        df.loc[pd.to_datetime(vintage) - relativedelta(days=target_release_lag-1) :, :] = np.nan
        df.index = pd.PeriodIndex(df.index, freq=freq)

        return df.dropna()

    def load_econ_m(self, vintage, window, freq='M', extend=False, **kwargs):
        vintage = pd.to_datetime(vintage)
        econ_m = pd.read_excel('data/PH_Econ_M.xlsx', sheet_name='data')
        econ_m['date'] = pd.to_datetime(econ_m['date'], format='%Y-%m-%d') + pd.offsets.MonthEnd(0)
        econ_m = econ_m.set_index('date')

        meta = pd.read_excel('data/PH_Econ_M.xlsx', sheet_name='release')
        meta['Lag'] = meta['Lag'] - 1
        lag_dict = meta.set_index('Variable Name').to_dict('dict')['Lag']
        econ_m = econ_m[meta.loc[meta['Include'] == 1, 'Variable Name']]

        econ_m = econ_m.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
        # econ_m = econ_m.loc[pd.to_datetime(vintage)  - relativedelta(months =  (pd.to_datetime(vintage).month - 1)%3 + window) : pd.to_datetime(vintage), :]
        for col in econ_m.columns:
            econ_m.loc[pd.to_datetime(vintage) - relativedelta(days=lag_dict[col]) :, col] = np.nan
        
        econ_m = self.extend_data(econ_m, vintage, **kwargs) if extend else econ_m # test
        econ_m.index = pd.PeriodIndex(econ_m.index, freq=freq) 
        return econ_m

    def load_econ_q(self, vintage, window, freq='Q', extend=False, **kwargs):
        vintage = pd.to_datetime(vintage)
        econ_q = pd.read_excel('data/PH_Econ_Q.xlsx', sheet_name='data')
        econ_q['date'] = pd.to_datetime(econ_q['date'], format="%m/%Y") + pd.offsets.MonthEnd(0)
        econ_q = econ_q.set_index('date')

        meta = pd.read_excel('data/PH_Econ_Q.xlsx', sheet_name='release')
        meta['Lag'] = meta['Lag'] - 1
        lag_dict = meta.set_index('Variable Name').to_dict('dict')['Lag']
        econ_q = econ_q[meta.loc[meta['Include'] == 1, 'Variable Name']]

        econ_q = econ_q.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
        # econ_q = econ_q.loc[pd.to_datetime(vintage)  - relativedelta(months =  (pd.to_datetime(vintage).month - 1)%3 + window) : pd.to_datetime(vintage), :]
        for col in econ_q.columns:
            econ_q.loc[pd.to_datetime(vintage) - relativedelta(days=lag_dict[col]) :, col] = np.nan
        econ_q.index = pd.PeriodIndex(econ_q.index, freq=freq)
        # econ_q = self.extend_data(econ_q, vintage, **kwargs) if extend else econ_q ### will error out when econ_q is empty, so commented out for now
        return econ_q
        
    def load_econ(self, vintage, window, freq='M', **kwargs):
        vintage = pd.to_datetime(vintage)
        econ_m = self.load_econ_m(vintage, window, freq='M', **kwargs)
        econ_q = self.load_econ_q(vintage, window, freq='M', **kwargs)

        data = ([econ_m] if kwargs.get('econ_monthly', True) else []) + ([econ_q] if kwargs.get('econ_quarterly', True) else [])
        econ = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), data)
        econ.index = pd.PeriodIndex(econ.index, freq=freq)
        
        return econ

    def load_tweets(self, vintage, window, kmpair, freq='M', extend=False, **kwargs):
        vintage = pd.to_datetime(vintage)
        tweets = pd.read_csv('data/PH_Tweets_v4.csv')
        tweets['date'] = pd.to_datetime(tweets['date'], format='%Y-%m-%d') + pd.offsets.MonthEnd(0)
        tweets = tweets.set_index('date')

        if len(kmpair) == 0:
            kmpair = {keyword: list(tweets.columns.drop('keyword')) for keyword in tweets['keyword'].unique()}
        data = [tweets[tweets['keyword'] == keyword][kmpair[keyword]].add_suffix(f'_{keyword}') for keyword in kmpair.keys()]
        tweets = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), data)

        # tweets = tweets.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
        tweets = tweets.loc[pd.to_datetime(vintage)  - relativedelta(months =  (pd.to_datetime(vintage).month - 1)%3 + window) : pd.to_datetime(vintage), :]
        tweets = self.extend_data(tweets, vintage, **kwargs) if extend else tweets
        tweets.index = pd.PeriodIndex(tweets.index, freq=freq)
        
        cols = ['C_00_PE', 'L_00_PE', 'R_00_PE', 'C_00_PU+', 'L_00_PU+', 'R_00_PU+']
        for col in cols:
            if list(tweets.columns).count(col) > 1:
                tweets[col] = tweets[col].clip(lower=1)
                tweets[col] = tweets[col].pct_change()
            # tweets[col] = scaler.fit_transform(tweets[col].values.reshape(-1, 1))
        tweets.loc[:,:] = StandardScaler().fit_transform(tweets)
        return tweets

    def load_data(self, vintage, target='GDP', window=0, scaled=True, **kwargs):
        vintage = pd.to_datetime(vintage)
        df = self.load_target(vintage, target, growth=True, quarterly=True, freq='M', **kwargs)
        target_scaler = StandardScaler(with_mean=scaled, with_std=scaled).fit(df[['target']])
        df['target'] = target_scaler.transform(df[['target']])

        if kwargs.get('with_tweets', True):
            tweets = self.load_tweets(vintage, window, freq='M', **kwargs).add_prefix('TWT.')
            tweets_scaler = StandardScaler(with_mean=scaled, with_std=scaled).fit(tweets)
            tweets.loc[:,:] = tweets_scaler.transform(tweets)
            # tweets = tweets.reindex(pd.period_range(tweets.index[0], tweets.index[-1] + (3 - tweets.index[-1].month) % 3, 
            #                                         freq=tweets.index.freq, name=tweets.index.name), fill_value=np.nan)
            # tweets = pd.concat([tweets.shift(l).add_suffix(f'.L{l}') for l in range(3)], axis=1)
            # tweets = tweets.loc[tweets.index.month % 3 == 0, :]
        else:
            tweets_scaler = []

        if kwargs.get('with_econ', True):
            econ = self.load_econ(vintage, window, freq='M', **kwargs).add_prefix('ECN.')
            econ_scaler = StandardScaler(with_mean=scaled, with_std=scaled).fit(econ)
            econ.loc[:,:] = econ_scaler.transform(econ)
            # econ = econ.reindex(pd.period_range(econ.index[0], econ.index[-1] + (3 - econ.index[-1].month) % 3, 
            #                                         freq=econ.index.freq, name=econ.index.name), fill_value=np.nan)
            # econ = pd.concat([econ.shift(l).add_suffix(f'.L{l}') for l in range(3)], axis=1)
            # econ = econ.loc[econ.index.month % 3 == 0, :]
            econ = econ.drop(columns=[target, target + '_YoY'], errors='ignore')
        else:
            econ_scaler = []

        data = [df] + ([tweets] if kwargs.get('with_tweets', True) else []) + ([econ] if kwargs.get('with_econ', True) else [])
        df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), data)
        df.index = pd.PeriodIndex(df.index, freq='M')
        df = df.tail(window) if window > 0 else df
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.T.duplicated(keep='first')]
        
        df['target'] = df['target'].bfill() ## Add Quarterly target variable  to M2 and M1 models also
        
        return df, target_scaler, econ_scaler, tweets_scaler 
    
    def rmse(self, y_pred, y_true):
        return np.sqrt(np.nanmean(np.power(y_true - y_pred, 2)))
    
    def process_summary(self, summary, **kwargs):
        summary['Period'] = np.where(summary['date'].dt.year < 2020, 1, 0)
        summary['Year'] = summary['date'].dt.year
        summary['Month_Q'] = summary['date'].dt.month % 3
        summary['Month_A'] = summary['date'].dt.month

        for freq, periods in zip(['A', 'Q'], [['Month_A', 'Month_Q', 'Year', 'Period'], ['Month_Q', 'Quarter', 'Period']]):
            summary[f'Difference_{freq}'] = summary[f'Nowcast_{freq}'] - summary[f'Actual_{freq}']
            summary[f'Overall_RMSE_{freq}'] = self.rmse(summary[f'Nowcast_{freq}'], summary[f'Actual_{freq}'])
            for period in periods:
                summary = summary.set_index(period)
                summary[f'{period}_RMSE_{freq}'] = summary.groupby(period).apply(lambda df: self.rmse(df[f'Nowcast_{freq}'], df[f'Actual_{freq}']))
                summary = summary.reset_index()
        summary = summary.drop(columns=['Period', 'Year', 'Quarter', 'Month_A', 'Month_Q'])
        summary = summary.rename(columns={'Month_A_RMSE_A': 'Month_RMSE_A', 'Month_Q_RMSE_A': 'Quarter_RMSE_A', 'Month_Q_RMSE_Q': 'Month_RMSE_Q'})
        summary = summary[['date', 'Model'] + [col for col in summary.columns if '_A' in col] + 
                          [col for col in summary.columns if '_Q' in col] + [f'ForecastQ{q}' for q in range(1,5)]]
        return summary
    def set_classname(self, **kwargs):
        raise NotImplementedError
    def run(self, start=dt.datetime(2017,1,31), end=dt.datetime(2023,1,1), delta=pd.offsets.MonthEnd(1), window=0, multiprocess=0, **kwargs):
        summary = super().run(start, end, delta, window, multiprocess, **kwargs)
        summary = self.process_summary(summary)

        self.set_classname(**self.kwargs, **kwargs)
        if kwargs.get('save_aggregate', True):
            if not os.path.exists(f'Results'):
                os.makedirs(f'Results')
            suffix = ('T' if kwargs.get('with_tweets', True) else '') + ('E' if kwargs.get('with_econ', True) else '')
            # summary.to_csv(f'Results/{self.prefix}_W{window}_{self.kwargs.get("target")}_{suffix}_n{str(format(self.kwargs.get("n_components"), ".02f"))[2:]}_summary.csv', index=False)
            summary.to_csv(f'Results/{self.prefix}-M_W{window}_{self.kwargs.get("target")}_{suffix}_summary.csv', index=False)

        return summary