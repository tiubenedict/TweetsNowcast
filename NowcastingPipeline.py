import os
import math
import numpy as np
import pandas as pd
import datetime as dt
from functools import reduce
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta

class NowcastingPipeline:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_quarter(self, date):
        date = pd.to_datetime(date)
        return f'{date.year}Q{math.ceil(date.month/3)}'

    def rmse(self, y_pred, y_true):
        return np.sqrt(np.nanmean(np.power(y_true - y_pred, 2)))
    
    def load_gdp(self, vintage, growth, quarterly, freq, gdp_release_lag=0, **kwargs):
        raise NotImplementedError
    
    def load_data(self, vintage, window, **kwargs):
        raise NotImplementedError
    
    def quarter_to_annual(self, vintage, nowcasts):
        vintage = pd.to_datetime(vintage)
        annual_gdp = self.load_gdp(vintage, growth=False, quarterly=False, freq='Y')
        quarter_gdp = self.load_gdp(vintage, growth=False, quarterly=True, freq='Q')

        if len(nowcasts) + 1 <= math.ceil(vintage.month / 3):
            return np.nan, np.nan
        elif len(nowcasts) < 4:
            return np.nan, nowcasts[math.ceil(vintage.month / 3) - 1]
        else:
            try:
                nowcasts_ = nowcasts.copy()
                for m in range(4):
                    quarter = 3 * (m + 1)
                    prev_quarter = vintage - relativedelta(years=1) + relativedelta(month=quarter)
                    nowcasts_[m] = quarter_gdp.loc[prev_quarter, 'GDP'] * (1 + nowcasts[m] / 100)
                    curr_quarter = vintage + relativedelta(month=quarter)
                    nowcasts_[m] = quarter_gdp.loc[curr_quarter, 'GDP'] if curr_quarter in quarter_gdp.index else nowcasts_[m]

                prev_year = vintage - relativedelta(years=1)
                return 100 * (np.sum(nowcasts_) / annual_gdp.loc[prev_year, 'GDP'] - 1), nowcasts[math.ceil(vintage.month / 3) - 1]
            except:
                return np.nan, nowcasts[math.ceil(vintage.month / 3) - 1]

    def fit_model(self, vintage, window, **kwargs):
        raise NotImplementedError

    def error_message(self, vintage, error):
        print(f'Vintage: {vintage.date()} \t {error}')
    
    def run(self, start, end, delta, window, **kwargs):
        self.prefix = self.__class__.__name__
        print(f'Running {self.prefix}')

        annual_gdp = self.load_gdp(end + relativedelta(years=1), growth=True, quarterly=False, freq='Y', gdp_release_lag=0)
        quarter_gdp = self.load_gdp(end + relativedelta(years=1), growth=True, quarterly=True, freq='Q', gdp_release_lag=0)

        summary = []
        vintage_now = start
        while vintage_now < end:
            annual = annual_gdp.loc[vintage_now, 'GDP']
            quarter = quarter_gdp.loc[vintage_now, 'GDP']
            
            try:
                nowcasts, model_desc = self.fit_model(vintage_now, window, **self.kwargs, **kwargs)
                nowcast_annual, nowcast_quarter = self.quarter_to_annual(vintage_now, nowcasts)
                
                summary.append([vintage_now, self.to_quarter(vintage_now), model_desc, nowcast_annual, annual, nowcast_quarter, quarter] + list(nowcasts))
                print(f'Vintage: {vintage_now.date()} \t {model_desc}')
                    
            except Exception as ex:
                summary.append([vintage_now, self.to_quarter(vintage_now), 'No model', np.nan, annual, np.nan, quarter, np.nan, np.nan, np.nan, np.nan])                
                self.error_message(vintage_now, ex)

            vintage_now += delta

        summary = pd.DataFrame(summary, columns=['date', 'Quarter', 'Model', 'Nowcast_A', 'Actual_A', 'Nowcast_Q', 'Actual_Q'] + [f'ForecastQ{q}' for q in range(1,5)])
        summary['date'] = pd.to_datetime(summary['date'])
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
        summary = summary[['date'] + [col for col in summary.columns if '_A' in col] + 
                          [col for col in summary.columns if '_Q' in col] + [f'ForecastQ{q}' for q in range(1,5)]]

        return summary

class NowcastingPH(NowcastingPipeline):
    def load_gdp(self, vintage, growth=True, quarterly=True, freq='M', gdp_release_lag=41, **kwargs):
        gdp = pd.read_csv('data/GDP_2014USD.csv')[['date', 'PH']].rename(columns={'PH': 'GDP'}).dropna()
        gdp['date'] = pd.to_datetime(gdp['date']) + pd.offsets.MonthEnd(0)
        gdp = gdp.set_index('date')

        if quarterly:
            gdp = gdp.resample('Q').sum()
            gdp = 100 * (gdp / gdp.shift(4) - 1) if growth else gdp
        else:
            gdp = gdp.resample('Y').sum()
            gdp = 100 * (gdp / gdp.shift(1) - 1) if growth else gdp

        gdp = gdp.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage) - relativedelta(days=gdp_release_lag), :]
        gdp.index = pd.PeriodIndex(gdp.index, freq=freq)

        return gdp.dropna()

    def load_econ(self, vintage, **kwargs):
        econ_m = pd.read_csv('data/PH_Econ_M.csv')
        econ_m['date'] = pd.to_datetime(econ_m['date']) + pd.offsets.MonthEnd(0)
        econ_m = econ_m.set_index('date')

        econ_q = pd.read_csv('data/PH_Econ_Q.csv')
        econ_q['date'] = pd.to_datetime(econ_q['date']) + pd.offsets.MonthEnd(0)
        econ_q = econ_q.set_index('date')

        data = ([econ_m] if kwargs.get('econ_monthly', True) else []) + ([econ_q] if kwargs.get('econ_quarterly', True) else [])
        econ = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), data)
        econ = econ[[col for col in econ.columns if col + '_YoY' not in econ.columns]]
        
        econ = econ.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
        econ.index = pd.PeriodIndex(econ.index, freq='M')
        
        return econ

    def load_tweets(self, vintage, keywords=['PE'], metrics=['TBweight_cl2rt', 'VADERweight_cl2rt'], **kwargs):
        tweets = pd.read_csv('data/PH_Tweets.csv')
        tweets['date'] = pd.to_datetime(tweets['date']) + pd.offsets.MonthEnd(0)
        tweets = tweets.set_index('date')

        keywords = keywords if len(keywords) > 0 else list(tweets['keyword'].unique())
        metrics = metrics if len(metrics) > 0 else list(tweets.columns.drop(['keyword']))
        tweets = tweets[metrics + ['keyword']]
        tweets_keyword = [tweets[tweets['keyword'] == keyword].drop(columns=['keyword']).add_suffix(f'_{keyword}') for keyword in keywords]
        tweets = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), tweets_keyword)

        tweets = tweets.loc[dt.datetime(2010,1,1) : pd.to_datetime(vintage), :]
        tweets.index = pd.PeriodIndex(tweets.index, freq='M')

        return tweets

    def load_data(self, vintage, window=0, scaled=True, **kwargs):
        gdp = self.load_gdp(vintage, **kwargs)
        gdp_scaler = StandardScaler(with_mean=scaled, with_std=scaled).fit(gdp[['GDP']])
        gdp['GDP'] = gdp_scaler.transform(gdp[['GDP']])

        tweets = self.load_tweets(vintage, **kwargs).add_prefix('TWT.')
        while tweets.index[-1].month % 3 != 0:
            tweets.loc[tweets.index[-1] + pd.offsets.MonthEnd(1), :] = np.nan
        tweets = pd.concat([tweets.shift(l).add_suffix(f'.L{l}') for l in range(3)], axis=1)
        tweets = tweets.loc[tweets.index.month % 3 == 0, :]

        econ = self.load_econ(vintage, **kwargs).add_prefix('ECN.')
        econ_scaler = StandardScaler(with_mean=scaled, with_std=scaled).fit(econ)
        econ.loc[:,:] = econ_scaler.transform(econ)
        while econ.index[-1].month % 3 != 0:
            econ.loc[econ.index[-1] + pd.offsets.MonthEnd(1), :] = np.nan
        econ = pd.concat([econ.shift(l).add_suffix(f'.L{l}') for l in range(3)], axis=1)
        econ = econ.loc[econ.index.month % 3 == 0, :]

        data = [gdp] + ([tweets] if kwargs.get('with_tweets', True) else []) + ([econ] if kwargs.get('with_econ', True) else [])
        df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer', sort=True), data)
        df.index = pd.PeriodIndex(df.index, freq='Q')
        df = df.tail(math.ceil(window / 3)) if window > 0 else df
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.T.duplicated(keep='first')]

        return df, gdp_scaler, econ_scaler

    def run(self, start=dt.datetime(2017,1,31), end=dt.datetime(2023,1,1), delta=pd.offsets.MonthEnd(1), window=0, **kwargs):
        summary = super().run(start, end, delta, window, **kwargs)

        if kwargs.get('save_aggregate', True):
            if not os.path.exists(f'Results'):
                os.makedirs(f'Results')
            suffix = ('T' if kwargs.get('with_tweets', True) else '') + ('E' if kwargs.get('with_econ', True) else '')
            summary.to_csv(f'summary/{self.prefix}_W{window}_{suffix}_summary.csv', index=False)

        return summary