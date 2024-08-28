polygonio_key = "NbOaYW2hvhGCEKNKaUeVI3iNg1P40Sie"

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

polygonio_key = "YOUR_KEY_HERE"  # I've just added this to make the code complete; you should already have this defined elsewhere


def get_stock_fund(ticker, date):
    base_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?date={date}&apiKey={polygonio_key}"
    resp = requests.get(base_url)
    if resp.status_code != 200:
        raise Exception(f"Failed to get stock fund data for {ticker} on {date}")
    return_dict = resp.json()
    cap = return_dict["results"]["market_cap"]
    return cap


def fetch_ticker_data(ticker, date_str):
    try:
        return ticker, get_stock_fund(ticker, date_str)
    except:
        return ticker, np.nan


def get_stock_caps(ticker_list, start_date, end_date):
    date_list = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame(index=date_list, columns=ticker_list)

    with tqdm(total=len(date_list), desc="Fetching Data") as pbar:
        for date in date_list:
            date_str = date.strftime("%Y-%m-%d")

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(fetch_ticker_data, ticker, date_str)
                    for ticker in ticker_list
                ]

                for future in as_completed(futures):
                    ticker, value = future.result()
                    df.loc[date][ticker] = value

            pbar.update(1)

    return df


tickers = pd.read_csv(
    "/student/wangsaizhuo/Data/my_data_dir/main/us/instruments/all.txt",
    delimiter="\t",
    header=None,
)
tickers = tickers[0].tolist()
tickers[:10]

caps = get_stock_caps(tickers, start_date="2015-01-01", end_date="2023-05-01")


caps.to_csv("us_caps.csv")
