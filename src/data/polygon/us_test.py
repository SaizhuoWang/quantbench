import pandas as pd
from datetime import datetime
import loky
from tqdm import tqdm
from polygon import RESTClient  # Importing RESTClient

us_ticker_list_fpath = '/student/wangsaizhuo/Data/my_data_dir/main/us/instruments/all.txt'
with open(us_ticker_list_fpath) as f:
    us_ticker_list = f.read().splitlines()
us_stocks = [x.split('\t')[0] for x in us_ticker_list]

def get_aggs_for_all_stocks(freq, start, end):
    print(f"Getting {freq} bars for all stocks from {start} to {end}")
    all_aggs = {}

    def stock_worker(stock):
        # Create a RESTClient instance inside the worker function
        client = RESTClient("NbOaYW2hvhGCEKNKaUeVI3iNg1P40Sie")
        aggs = client.get_aggs(stock, 1, freq, start, end)
        # Process data as before
        time_indices = [
            datetime.fromtimestamp(x.timestamp / 1000.0).strftime(
                "%Y%m%dT%H%M%S"
            )
            for x in aggs
        ]
        series_dict = {
            "open": pd.Series(
                [x.open for x in aggs], index=time_indices, name=stock
            ),
            "close": pd.Series(
                [x.close for x in aggs], index=time_indices, name=stock
            ),
            "high": pd.Series(
                [x.high for x in aggs], index=time_indices, name=stock
            ),
            "low": pd.Series(
                [x.low for x in aggs], index=time_indices, name=stock
            ),
            "volume": pd.Series(
                [x.volume for x in aggs], index=time_indices, name=stock
            ),
            "vwap": pd.Series(
                [x.vwap for x in aggs], index=time_indices, name=stock
            ),
            "amount": pd.Series(
                [x.volume * x.vwap for x in aggs],
                index=time_indices,
                name=stock,
            ),
            "ntrades": pd.Series(
                [x.transactions for x in aggs],
                index=time_indices,
                name=stock,
            ),
        }
        return series_dict

    with loky.get_reusable_executor(128) as e:
        all_jobs = [e.submit(stock_worker, x) for x in us_stocks]
        all_series = [
            job.result() for job in tqdm(all_jobs, desc="Fetching stock data")
        ]
        for field in all_series[0].keys():
            all_aggs[field] = pd.concat([x[field] for x in all_series], axis=1)

    return all_aggs


us_minute_data = get_aggs_for_all_stocks(
    freq="minute", start="2021-01-01", end="2021-01-12"
)
for k, v in us_minute_data.items():
    v.to_csv(f'./{k}.csv')