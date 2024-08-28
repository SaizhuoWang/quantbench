import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def fetch_data_for_ticker(ticker, base_url, headers):
    """Helper function to fetch data for a single ticker."""
    response = requests.get(base_url.format(ticker, ticker), headers=headers)
    try:
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            print(tables)
            return ticker, tables
            if len(tables) == 3:
                return ticker, {
                    "Table1": tables[0],
                    "Table2": tables[1],
                    "Table3": tables[2],
                }
    except:
        return ticker, None
    return ticker, None


def get_stock_holders_parallel_with_progress(tickers):
    """Retrieve top holders' data from Yahoo Finance for a list of tickers and
    display a progress bar.

    Parameters
    ----------
    tickers : list of str
        List of stock tickers.

    Returns
    -------
    dict
        Dictionary where each key is a ticker and its value is another dictionary with keys "Table1",
        "Table2", and "Table3" representing the tables.

    Notes
    -----
    The function mimics a browser request to access Yahoo Finance data. Web scraping may be against the
    terms of service of some websites, including Yahoo Finance. Always review and comply with the website's
    robots.txt file and terms of service.

    """
    base_url = "https://finance.yahoo.com/quote/{}/holders?p={}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    results = {}

    with ProcessPoolExecutor(max_workers=10) as executor:
        # Start the load operations and mark each future with its ticker
        future_to_ticker = {
            executor.submit(
                fetch_data_for_ticker, ticker, base_url, headers
            ): ticker
            for ticker in tickers
        }

        for future in tqdm(
            as_completed(future_to_ticker),
            total=len(tickers),
            desc="Fetching data",
        ):
            ticker = future_to_ticker[future]
            _, data = future.result()
            if data:
                results[ticker] = data

    # for ticker in tqdm(tickers, desc="Fetching data"):
    #     _, data = fetch_data_for_ticker(ticker, base_url, headers)
    #     if data:
    #         results[ticker] = data
    return results


region = input("Enter region (us, hk, cn, etc.): ")
tickers = pd.read_csv(
    f"/student/wangsaizhuo/Data/my_data_dir/main/{region}/instruments/all.txt",
    delimiter="\t",
    header=None,
)
tickers = tickers[0].tolist()
tickers[:10]

ticker_files = get_stock_holders_parallel_with_progress(tickers)

import pickle

with open(f"{region}_ticker_files.pkl", "wb") as f:
    pickle.dump(ticker_files, f)


# Extract significant keywords from shareholder names
def extract_keywords(name):
    pattern = re.compile(
        r"\b(Inc|LLC|LP|Llc|Ltd|Co|Corp|Group|Management|Associates|Partners|Capital|Bank|Advisors|Investments|Investment|Asset|Trust|Services|Holdings|Global|Financial|Equity|Enterprises|Company|Brothers|Bancorp)\b",
        re.IGNORECASE,
    )
    name = pattern.sub("", name)
    return set(name.split())


def parallel_matrix_computation(node_lists, matrix_size):
    final_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    for i, node_list in tqdm(enumerate(node_lists)):
        final_matrix[np.ix_(node_list, node_list)] = 1
    final_matrix[np.diag_indices_from(final_matrix)] = 0
    return final_matrix


def main(args):
    # Read crawler data
    with open(
        f"data/shareholder/yf_crawl/{args.region}_ticker_files.pkl", "rb"
    ) as f:
        ticker_files = pickle.load(f)

    # Read shareholder lists
    good_tickers = 0
    holders_dict = {}
    for ticker, tables in ticker_files.items():
        try:
            holders_dict[ticker] = tables[1]["Holder"].to_list()
            good_tickers += 1
        except:
            continue
    print(
        f"good_tickers = {good_tickers}, ratio = {good_tickers / len(ticker_files)}"
    )

    # Dump holder list
    with open(
        f"data/shareholder/processed/{args.region}_holders_dict.pkl", "wb"
    ) as f:
        pickle.dump(holders_dict, f)

    # Create keyword to tickers mapping
    with open(
        f"data/shareholder/processed/{args.region}_holders_dict.pkl", "rb"
    ) as f:
        holders_dict = pickle.load(f)
    tickers = list(holders_dict.keys())

    # Create keyword to tickers mapping
    keyword_to_tickers = {}
    for ticker, shareholders in holders_dict.items():
        for shareholder in shareholders:
            keywords = extract_keywords(shareholder)
            for keyword in keywords:
                if keyword not in keyword_to_tickers:
                    keyword_to_tickers[keyword] = []
                keyword_to_tickers[keyword].append(tickers.index(ticker))

    connected_components = list(keyword_to_tickers.values())
    parallel_results = parallel_matrix_computation(
        node_lists=connected_components, matrix_size=len(tickers)
    )
    # final_adj_matrix = combine_matrices(parallel_results)
    final_adj_matrix = parallel_results

    # Dump data
    data_dict = {
        "node_list": tickers,
        "adj_matrix": final_adj_matrix,
    }
    with open(
        f"data/shareholder/processed/{args.region}_data_dict.pkl", "wb"
    ) as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us")
    args = parser.parse_args()
    main(args)
