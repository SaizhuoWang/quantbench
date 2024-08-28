eod_token = "64cab1616fad59.83458373"
import requests

historical_cap_base_url = "https://eodhistoricaldata.com/api/historical-market-cap/{ticker}?api_token={token}&from={from_date}"
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

ticker_codes = pd.read_csv(
    "/student/wangsaizhuo/Data/my_data_dir/main/us/instruments/all.txt",
    delimiter="\t",
    header=None,
)[0]


def fetch_data(ticker):
    cap_data = {}
    try:
        cap_query_url = historical_cap_base_url.format(
            ticker=f"{ticker}.US", token=eod_token, from_date="2015-01-01"
        )
        cap_query_response = requests.get(
            cap_query_url, params={"from": "2015-01-01"}
        )
        cap_df = pd.DataFrame(cap_query_response.json()).T
        cap_data[ticker] = cap_df
    except:
        print(f"Error in fetching data for {ticker}")
        cap_data[ticker] = None
    return cap_data


# Assuming we want to use 10 threads. Adjust the max_workers parameter as needed.
num_threads = 10
cap_dict_parallel = {}

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for ticker, result in tqdm(
        zip(ticker_codes, executor.map(fetch_data, ticker_codes))
    ):
        cap_dict_parallel.update(result)

import pickle

# Serialize the result to a pickle file
with open("cap_dict_us_parallel.pkl", "wb") as f:
    pickle.dump(cap_dict_parallel, f)
