import requests
import polygon
import loky
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import os

os.makedirs(
    "/student/wangsaizhuo/Data/my_data_dir/main/us/aggs", exist_ok=True
)

client = polygon.RESTClient("NbOaYW2hvhGCEKNKaUeVI3iNg1P40Sie")

us_ticker_list_fpath = (
    "/student/wangsaizhuo/Data/my_data_dir/main/us/instruments/all.txt"
)
with open(us_ticker_list_fpath) as f:
    us_ticker_list = f.read().splitlines()
us_stocks = [x.split("\t")[0] for x in us_ticker_list]


def get_aggs_for_all_stocks(freq, start, end):
    print(f"Getting {freq} bars for all stocks from {start} to {end}")
    all_aggs = {}
    os.makedirs('/student/wangsaizhuo/Data/my_data_dir/main/us/aggs/single_file', exist_ok=True)

    def stock_worker(stock) -> pd.Series:
        client = polygon.RESTClient("NbOaYW2hvhGCEKNKaUeVI3iNg1P40Sie")
        aggs = client.get_aggs(
            ticker=stock,
            multiplier=1,
            timespan=freq,
            from_=start,
            to=end,
            limit=50000,
        )
        # for each field, return a series
        time_indices = [
            datetime.fromtimestamp(x.timestamp / 1000.0) for x in aggs
        ]
        time_indices = [x.strftime("%Y%m%dT%H%M%S") for x in time_indices]
        open_prices = [x.open for x in aggs]
        high_prices = [x.high for x in aggs]
        low_prices = [x.low for x in aggs]
        close_prices = [x.close for x in aggs]
        volumes = [x.volume for x in aggs]
        vwaps = [x.vwap for x in aggs]
        # amounts = [x.volume * x.vwap for x in aggs]
        ntrades = [x.transactions for x in aggs]
        series_dict = {
            "open": pd.Series(open_prices, index=time_indices, name=stock),
            "close": pd.Series(close_prices, index=time_indices, name=stock),
            "high": pd.Series(high_prices, index=time_indices, name=stock),
            "low": pd.Series(low_prices, index=time_indices, name=stock),
            "volume": pd.Series(volumes, index=time_indices, name=stock),
            "vwap": pd.Series(vwaps, index=time_indices, name=stock),
            # "amount": pd.Series(amounts, index=time_indices, name=stock),
            "ntrades": pd.Series(ntrades, index=time_indices, name=stock),
        }
        stock_df = pd.DataFrame(series_dict)
        stock_df.to_csv(
            f"/student/wangsaizhuo/Data/my_data_dir/main/us/aggs/single_file/{stock}.csv"
        )
        return series_dict

    with loky.get_reusable_executor(128) as e:
        all_series = []
        for ret_series in tqdm(
            e.map(stock_worker, us_stocks), total=len(us_stocks), desc="Querying polygon"
        ):
            all_series.append(ret_series)
            
        # For each data field, make a dataframe with each column as a stock
        for field in all_series[0].keys():
            print(f'Concating field "{field}"')
            all_aggs[field] = pd.concat([x[field] for x in all_series], axis=1)
    return all_aggs


import os

get_aggs_for_all_stocks(
    freq="day", start="2004-01-01", end="2024-05-24"
)



import os
day_vp_dir = '/student/wangsaizhuo/Data/my_data_dir/main/us/aggs/single_file'
files = os.listdir(day_vp_dir)
from tqdm import tqdm
data_dict = {}
for f in tqdm(files, desc='Reading files'):
    if not f.endswith('.csv'):
        continue
    data = pd.read_csv(os.path.join(day_vp_dir, f), index_col=0)
    stock = f.split('.')[0]
    for col in data.columns:
        data_dict.setdefault(col, {})[stock] = data[col]
        
df_dict = {}
os.makedirs('/student/wangsaizhuo/Data/my_data_dir/main/us/aggs/full', exist_ok=True)
for field in data_dict.keys():
    data = pd.DataFrame(data_dict[field])
    data.to_csv(f'/student/wangsaizhuo/Data/my_data_dir/main/us/aggs/full/{field}.csv')
    print(f"Saved {field}")
    df_dict[field] = data