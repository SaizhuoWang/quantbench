# For a frequency, get aggregate bars for all stocks
import requests
import polygon
import pandas as pd
from datetime import datetime

client = polygon.RESTClient("NbOaYW2hvhGCEKNKaUeVI3iNg1P40Sie")
us_ticker_list_fpath = (
    "/student/wangsaizhuo/Data/my_data_dir/main/us/instruments/all.txt"
)
with open(us_ticker_list_fpath) as f:
    us_ticker_list = f.read().splitlines()
us_stocks = [x.split("\t")[0] for x in us_ticker_list]

from datetime import timedelta


def get_aggs_for_all_stocks(freq, start, end):
    print(f"Getting {freq} bars for all stocks from {start} to {end}")
    all_aggs = {}

    def stock_worker(stock) -> pd.Series:
        client = polygon.RESTClient("NbOaYW2hvhGCEKNKaUeVI3iNg1P40Sie")
        total_aggs = []
        get_start = start
        get_end = end

        while True:
            aggs = client.get_aggs(
                ticker=stock,
                multiplier=1,
                timespan=freq,
                from_=get_start,
                to=end,
                limit=50000,
            )
            total_aggs.extend(aggs)
            time_indices = [
                datetime.fromtimestamp(x.timestamp / 1000.0) for x in aggs
            ]
            if time_indices[-1].strftime("%Y-%m-%d") >= end:
                break
            get_start = time_indices[-1] + timedelta(minutes=1)
            print(f"{stock} starts at {get_start}")

        aggs = total_aggs

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
        df = pd.DataFrame(series_dict)
        df.to_csv(f"./us_1min/{stock}.csv")
        # return df
        # return series_dict

    import loky
    from tqdm import tqdm

    with loky.get_reusable_executor(500) as e:
        all_series = [
            x
            for x in tqdm(e.map(stock_worker, us_stocks), total=len(us_stocks))
        ]
        for i, series_dict in enumerate(all_series):
            print(f"Got {len(series_dict)} bars for {us_stocks[i]}")
            for k, v in series_dict.items():
                v.to_csv(f"./{us_stocks[i]}_{k}.csv")
                break
            break
        # # For each data field, make a dataframe with each column as a stock
        # for field in all_series[0].keys():
        #     all_aggs[field] = pd.concat([x[field] for x in all_series], axis=1)
    return all_series


aggs = get_aggs_for_all_stocks("minute", "2014-01-01", "2024-05-31")
