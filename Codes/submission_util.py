import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve, 
    auc
    )
# TODO: SimOS 에서의 정답을 알고있다. 그러므로 eval metric 계산할 수 있다. 

import submission_config as subconfig

## Params
DACON_SID_CNT = 2000
SIMOS_START = subconfig.SIMOS_START
SIMOS_END = subconfig.SIMOS_END

## Import data
krx_df = pd.read_csv(subconfig.krx_df_PATH)
adjclose_df = pd.read_pickle(subconfig.adjclose_df_PATH)
return_df = pd.read_pickle(subconfig.return_df_PATH)

def get_simos_data(return_df, adjclose_df):
    holidays = return_df.isnull().all(axis=1)
    tradingdays = ~holidays

    holidays = holidays.index[holidays]
    tradingdays = tradingdays.index[tradingdays]

    return_df = return_df.loc[tradingdays, :]
    adjclose_df = adjclose_df.loc[tradingdays, :]

    return_df = return_df.loc[SIMOS_START:SIMOS_END, :]
    adjclose_df = adjclose_df.loc[SIMOS_START:SIMOS_END, :]

    return return_df, adjclose_df

# TODO: Confusing if global variables are not capitalized
simos_return_df, simos_adjclose_df = get_simos_data(return_df, adjclose_df) # simos period, only trading days

## for filtering
def get_tradables(adjclose_df, trading_date=subconfig.PORTFOLIO_DATE):
    sid_list = adjclose_df.columns

    notnull = adjclose_df.loc[trading_date, :].notnull()
    notzero = adjclose_df.loc[trading_date, :] != 0

    return sid_list[notnull * notzero]

def is_tradables(sid_list, tradables):
    tradables = set(tradables)

    return np.array([True if sid in tradables else False for sid in sid_list])

def get_daconsids(krx_df):
    krx_df.columns = ['date', 'code', 'name', 'volume', 'open', 'high', 'low', 'close']
    dacon_sid_list = [ii[1:] for ii in krx_df['code'].unique()] # 060310 형식으로 바꿔줌

    return dacon_sid_list

def is_daconsids(sid_list, daconsids):
    daconsids = set(daconsids)

    return np.array([True if sid in daconsids else False for sid in sid_list])

class Submission:
    holding_return_s = (simos_adjclose_df.loc[SIMOS_END, :] - simos_adjclose_df.loc[SIMOS_START, :]).divide(simos_adjclose_df.loc[SIMOS_START, :])  
    holding_return_s = holding_return_s.fillna(0)

    # simos_winners = 
    # TODO: Add data science evaluation metrics

    # TODO: Make not-instance-specific variables to class variables
    def __init__(self, alpha_series:pd.Series, alpha_name:str, top=200, bottom=200):
        self.alpha_series = alpha_series
        self.alpha_name = alpha_name
        self.top = top
        self.bottom = bottom

        self.sid_list = self.alpha_series.index
        self.tradables = get_tradables(adjclose_df)
        self.daconsids = get_daconsids(krx_df)
    
        self.is_selectables = is_tradables(self.sid_list, self.tradables) * is_daconsids(self.sid_list, self.daconsids)
        self.submission_df = None
        self.alpha_winners = None
        self.alpha_losers = None

        # for excess return
        self.long_hpr = None
        self.short_hpr = None
        self.final_return = None

        # for variance
        self.long_returns = None
        self.short_returns = None
        
    def get_rank(self, export_path=None):
        selectables = self.alpha_series[self.is_selectables]
        top_s = selectables.nlargest(self.top)
        bottom_s = selectables.nsmallest(self.bottom)
        
        self.alpha_winners = top_s.index
        self.alpha_losers = bottom_s.index
        
        submission_df = pd.DataFrame(
            data={'rank': [-1]*DACON_SID_CNT},
            index=self.daconsids
        )
        submission_df.index.name = 'sid'

        submission_df['rank'][top_s.index] = np.arange(1, self.top+1)
        submission_df['rank'][bottom_s.index] = np.arange(DACON_SID_CNT, DACON_SID_CNT - self.bottom, -1)

        submission_df['rank'][submission_df['rank'] == -1] = np.arange(self.top+1, DACON_SID_CNT - self.bottom + 1)

        self.submission_df = submission_df

        if export_path:
            submission_df.index = ['A' + idx for idx in submission_df.index]
            submission_df.index.name = '종목코드'
            submission_df.columns = ['순위']
            submission_df.to_csv(export_path / f'{self.alpha_name}.csv', encoding='utf-8')
            
            print(f'Saved to {export_path / self.alpha_name}.csv')
            return submission_df

        return submission_df

    def get_excess_return(self, risk_free_rate=0.035, days_of_trading=15):
        self.long_hpr = Submission.holding_return_s[self.alpha_winners].sum()
        self.short_hpr = Submission.holding_return_s[self.alpha_losers].sum()

        self.final_return = (self.long_hpr - self.short_hpr) / 400

        annualized_final_return = self.final_return * 250 / days_of_trading
        excess_return = annualized_final_return - risk_free_rate

        return excess_return
    
    def get_volatility(self, days_of_trading=15):
        self.long_returns = simos_return_df.loc[:, self.alpha_winners].mean(axis=1)
        self.short_returns = simos_return_df.loc[:, self.alpha_losers].mean(axis=1)

        annualized_portfolio_returns = (self.long_returns - self.short_returns) / 2 * 250
        annualized_mean_returns = annualized_portfolio_returns.mean()
        
        annualized_portfolio_volatility = np.sqrt((annualized_portfolio_returns - annualized_mean_returns).pow(2)[2:].sum() / (days_of_trading-2))

        return annualized_portfolio_volatility

    def get_Sharpe(self):
        return self.get_excess_return() / self.get_volatility()

    
class Score:
    holding_return_s = (simos_adjclose_df.loc[SIMOS_END, :] - simos_adjclose_df.loc[SIMOS_START, :]).divide(simos_adjclose_df.loc[SIMOS_START, :])  
    holding_return_s = holding_return_s.fillna(0)

    def __init__(self, submission_csv_filepath, alpha_name, top=200, bottom=200, encoding='utf-8'):
        self.alpha_name = alpha_name
        self.top = top
        self.bottom = bottom

        with open(submission_csv_filepath, 'r', encoding=encoding) as f:
            submission_df = pd.read_csv(f, index_col=0)
        
        submission_df.index = [idx[1:] for idx in submission_df.index]
        submission_df.index.name = 'sid'
        submission_df.columns = ['rank']

        self.alpha_series = submission_df['rank']
        self.sid_list = self.alpha_series.index

        # TODO: Add validations

        self.submission_df = None
        self.alpha_winners = self.alpha_series.nsmallest(self.top).index
        self.alpha_losers = self.alpha_series.nlargest(self.bottom).index

        # for excess return
        self.long_hpr = None
        self.short_hpr = None
        self.final_return = None

        # for variance
        self.long_returns = None
        self.short_returns = None
    
    def get_excess_return(self, risk_free_rate=0.035, days_of_trading=15):
        self.long_hpr = Score.holding_return_s[self.alpha_winners].sum()
        self.short_hpr = Score.holding_return_s[self.alpha_losers].sum()

        self.final_return = (self.long_hpr - self.short_hpr) / 400

        annualized_final_return = self.final_return * 250 / days_of_trading
        excess_return = annualized_final_return - risk_free_rate

        return excess_return

    def get_volatility(self, days_of_trading=15):
        self.long_returns = simos_return_df.loc[:, self.alpha_winners].mean(axis=1)
        self.short_returns = simos_return_df.loc[:, self.alpha_losers].mean(axis=1)

        annualized_portfolio_returns = (self.long_returns - self.short_returns) / 2 * 250
        annualized_mean_returns = annualized_portfolio_returns.mean()
        
        annualized_portfolio_volatility = np.sqrt((annualized_portfolio_returns - annualized_mean_returns).pow(2)[2:].sum() / (days_of_trading-2))

        return annualized_portfolio_volatility

    def get_Sharpe(self):
        sharpe = self.get_excess_return() / self.get_volatility()
        print(f'Sharpe of {self.alpha_name}: {sharpe}')

        return sharpe