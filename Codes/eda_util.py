import pandas as pd
import numpy as np

def agg_pct(pct):
    def percentile_(x):
        return x.quantile(pct)
   
    percentile_.__name__ = f'percentile_{pct*100:02.0f}'
   
    return percentile_

basic_stats = [
    'count',
    'mean',
    'median',
    'max',
    'min',
    'std',
    'var',
    'skew',
    'kurt',
]
 
pct_ranks = [
    agg_pct(0.01),
    agg_pct(0.05),
    agg_pct(0.1),
    agg_pct(0.2),
    agg_pct(0.3),
    agg_pct(0.4),
    agg_pct(0.5),
    agg_pct(0.6),
    agg_pct(0.7),
    agg_pct(0.8),
    agg_pct(0.9),
    agg_pct(0.95),
    agg_pct(0.99),
]

statistics = basic_stats + pct_ranks

def get_statistics(pd_data, statistics:list, excluded_cols=[]):
    pd_type = type(pd_data)
   
    if pd_type is pd.Series:
        s_stats = pd_data.agg(statistics)
       
        return s_stats
    elif pd_type is pd.DataFrame:
        excluded_cols = excluded_cols
        cols = [col for col in pd_data.columns if col not in excluded_cols]
        df_stats = pd_data.agg({
            col: statistics for col in cols
        })
       
        return df_stats
    else:
        print('Wrong data type. Only takes pd.DataFrame or pd.Series')
 
def compare_s_stats(s_list, statistics:list, colname_list=None):
    s_stats = [get_statistics(s, statistics) for s in s_list]
   
    stats_df = pd.DataFrame(s_stats).T
   
    if colname_list:
        stats_df.columns = colname_list
    else:
        stats_df.columns = np.arange(len(s_list))
   
    return stats_df