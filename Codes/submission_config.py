from pathlib import Path

## Path configs

BASE_PATH = Path('.').resolve()
DATA_PATH = BASE_PATH / 'data'
OUTPUT_PATH = BASE_PATH / 'output'

krx_df_PATH = DATA_PATH / 'train.csv'
return_df_PATH = DATA_PATH / 'return_20140101_20230705.pickle'
adjclose_df_PATH = DATA_PATH / 'adjClose_20140101_20230705.pickle'
adjhigh_df_PATH = DATA_PATH / 'adjHigh_20140101_20230705.pickle'
adjlow_df_PATH = DATA_PATH / 'adjLow_20140101_20230705.pickle'
adjopen_df_PATH = DATA_PATH / 'adjOpen_20140101_20230705.pickle'
volume_df_PATH = DATA_PATH / 'volume_df_20140101_20230705.pickle'
dollarvolume_df_PATH = DATA_PATH / 'dollarvolume_df_20140101_20230705.pickle'
marketcap_df_PATH = DATA_PATH / 'marketcap_df_20140101_20230705.pickle'

## Param configs

# train (custom)
TRAIN_START = '2021-06-01'

# SimOS
PORTFOLIO_DATE = '2023-05-30' 
SIMOS_START = '2023-05-31'
SIMOS_END = '2023-06-21'

# RealOS
REALOS_PORTFOLIO_DATE = '2023-07-28' 
REALOS_START = '2023-07-31'

WINDOWS = {
    'rdvadv': 20,
}