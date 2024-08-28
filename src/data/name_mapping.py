# Get ticker-code to company name mapping
# 2 information sources: Yahoo Finance (HTML Crawling) and EODHD (paid API)
# We will implement both here.


import os
import typing as tp
from multiprocessing import Pool

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm
from utils import read_ticker_list

from ..constants import EODHD_API_KEY, REGION_XCHG_MAPPING


def get_region_ticker_names_from_eodhd(region: str, ticker_list: tp.List[str]):
    eodhd_endpoint = "https://eodhistoricaldata.com/api/exchange-symbol-list/{xchg_code}?api_token={api_key}"
    xchg_list = REGION_XCHG_MAPPING["EODHD"].get(region, [])
    if len(xchg_list) == 0:
        print(f"Region {region} not supported by EODHD")
        return {}
    print(
        f"Retriving ticker names from EODHD for region {region}, exchanges: {xchg_list}"
    )
    ret = {}
    for xchg_code in xchg_list:
        url = eodhd_endpoint.format(xchg_code=xchg_code, api_key=EODHD_API_KEY)
        df = pd.read_csv(url, keep_default_na=False, na_values=["_"])
        stocks = df.loc[df["Type"] == "Common Stock"]
        ticker_codes = [x.split(".")[0].split("_")[0] for x in ticker_list]
        if region == "cn":
            ticker_codes = ["{:06}".format(int(x)) for x in ticker_codes]
            int_ticker_codes = [int(x) for x in ticker_codes]
            intersected_stocks = stocks.loc[
                stocks["Code"].isin(int_ticker_codes)
            ]
        else:
            intersected_stocks = stocks.loc[stocks["Code"].isin(ticker_codes)]
        new_data = intersected_stocks.set_index("Code")["Name"].to_dict()
        if region == "cn":
            new_data = {
                "{:06}_X{}".format(int(k), xchg_code): v
                for k, v in new_data.items()
            }
        ret.update(new_data)
    return ret


def get_ticker_company_name_from_yf(ticker: str) -> tp.Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    proxy = {"https": "socks5://localhost:9050"}

    # time.sleep(1)
    try:
        yf_base_url = (
            "https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
        )
        resp = requests.get(
            yf_base_url.format(ticker=ticker), headers=headers, proxies=proxy
        )
        soup = BeautifulSoup(resp.content, "lxml")
        name = (
            soup.head.title.text.split("Company Profile")[0]
            .rsplit("(", 1)[0]
            .strip()
        )
        # print(name)
        return ticker, name
    except:
        return ticker, "Requested symbol wasn't found"


def get_region_ticker_names_from_yahoo(region: str, ticker_list: tp.List[str]):
    """A web scraper to get ticker names from Yahoo Finance.

    Fake user agent is used to avoid blocking.

    """
    ticker_name_map = {}
    counter = 0
    if region == "cn":

        def remap_postfix(x):
            if x.endswith("_XSHE"):
                return x.replace("_XSHE", ".SZ")
            elif x.endswith("_XSHG"):
                return x.replace("_XSHG", ".SS")
            else:
                return x

        ticker_list = [remap_postfix(x) for x in ticker_list]
    with Pool(processes=30) as pool:
        for ticker, company_name in tqdm(
            pool.imap_unordered(get_ticker_company_name_from_yf, ticker_list),
            total=len(ticker_list),
            desc=f"Processing region {region}",
        ):
            # if counter < 10:
            #     print(f"{ticker}: {company_name}")
            counter += 1
            if (
                "Requested symbol wasn't found" in company_name
                or "Symbol Lookup from Yahoo Finance" in company_name
            ):
                continue
            if region == "cn":
                ticker = ticker.replace(".SZ", "_XSHE").replace(".SS", "_XSHG")
            ticker_name_map[ticker] = company_name

    return ticker_name_map


def main(args):
    ticker_list_path = os.path.join(
        args.main_data_dir, args.region, "instruments", "all.txt"
    )
    ticker_list = read_ticker_list(ticker_list_path)
    ticker_name_map = {}
    ticker_name_map.update(
        get_region_ticker_names_from_eodhd(args.region, ticker_list)
    )
    remaining_tickers = list(set(ticker_list) - set(ticker_name_map.keys()))
    print(
        f"{len(ticker_name_map)} tickers found in EODHD. Remaining tickers: {len(remaining_tickers)}"
    )
    ticker_name_map.update(
        get_region_ticker_names_from_yahoo(args.region, remaining_tickers)
    )
    with open(
        os.path.join(args.output_dir, f"{args.region}_ticker_name_map.yaml"),
        "w",
    ) as f:
        yaml.safe_dump(ticker_name_map, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us")
    parser.add_argument(
        "--main_data_dir", type=str, default="./data/market_data"
    )
    parser.add_argument("--output_dir", type=str, default="./data/stocks/names")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        for region in ["cn", "us", "hk", "uk", "jp", "fr"]:
            args.region = region
            main(args)
    else:
        main(args)
