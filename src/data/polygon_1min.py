# Step 1: Define the function to fetch and parse a single CSV from a URL

import typing as tp
from typing import Dict

import numpy as np
import pandas as pd
import requests


def fetch_and_parse_csv(url: str) -> pd.DataFrame:
    """Fetches a CSV from the given URL, parses it into a pandas DataFrame,
    converts the 't' column to human-readable datetime format, and sets 't' as
    the index.

    Parameters:
    - url (str): The URL of the CSV file.

    Returns:
    - pd.DataFrame: A DataFrame containing the parsed CSV data with 't' as the index.

    """
    # Directly read the CSV from the URL
    resp = requests.get(url)

    df = pd.DataFrame(resp.json()["results"])

    # Convert the 't' column to human-readable datetime format
    df["t"] = pd.to_datetime(df["t"], unit="ms")

    # Set the 't' column as the index
    df.set_index("t", inplace=True)

    return df


from concurrent.futures import ThreadPoolExecutor


def parallel_fetch_and_parse(
    stock_urls: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """Fetches and parses multiple CSVs in parallel and shows a progress bar.

    Parameters:
    - stock_urls (Dict[str, str]): A dictionary mapping stock names to their respective CSV URLs.

    Returns:
    - Dict[str, pd.DataFrame]: A dictionary mapping stock names to their respective DataFrames.

    """
    results = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(fetch_and_parse_csv, url): stock
            for stock, url in stock_urls.items()
        }

        # Adding tqdm progress bar
        for future in tqdm(
            futures, total=len(stock_urls), desc="Fetching and Parsing"
        ):
            stock = futures[future]
            try:
                results[stock] = future.result()
            except Exception as e:
                print(f"Failed to fetch and parse CSV for {stock}. Error: {e}")
    return results


import os


def dump_dataframes(dfs: Dict[str, pd.DataFrame], output_dir: str):
    """Dumps the processed DataFrames to separate files.

    Parameters:
    - dfs (Dict[str, pd.DataFrame]): Dictionary mapping stock names to their respective DataFrames.
    - output_dir (str): Directory where the resulting files will be saved.

    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Factors to be extracted
    factors = ["o", "h", "l", "c", "v", "vw", "n"]

    for factor in factors:
        # Create a new DataFrame for the factor
        factor_df = pd.DataFrame()

        for stock, df in dfs.items():
            try:
                factor_df[stock] = df[factor]
            except KeyError:
                print(f"Failed to extract factor {factor} for {stock}")
                factor_df[stock] = np.nan

        # Save the DataFrame to a CSV file
        factor_df.to_csv(os.path.join(output_dir, f"{factor}.csv"))


def get_all_tickers(region: str) -> tp.List[str]:
    ticker_codes = pd.read_csv(
        f"/student/wangsaizhuo/Data/my_data_dir/main/{region}/instruments/all.txt",
        delimiter="\t",
        header=None,
    )[0]
    return ticker_codes


from tqdm import tqdm

url_template = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2015-01-01/2023-08-01?adjusted=true&sort=asc&apiKey=66W2UFpWBuv07P6xg7miaf3WBmDrsx4G"

ticker_codes = get_all_tickers("us")
ticker_url_dict = {}
for ticker in ticker_codes:
    ticker_url_dict[ticker] = url_template.format(ticker=ticker)

dfs = parallel_fetch_and_parse(ticker_url_dict)
os.makedirs(
    "/student/wangsaizhuo/Data/my_data_dir/main/us/stocks/day", exist_ok=True
)
dump_dataframes(dfs, "/student/wangsaizhuo/Data/my_data_dir/main/us/stocks/day")
