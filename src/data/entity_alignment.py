import os

import pandas as pd
import yaml

from ..constants import REGION_XCHG_MAPPING
from ..utils import jsonl_generator, read_ticker_list

CSV_PATH = "/student/wangsaizhuo/Codes/q4l/examples/benchmark/data/stocks/names/US_LIST_OF_SYMBOLS.csv"
ALIAS_DIR = "/student/wangsaizhuo/Data/Wikidata/wikidump_20230716/aliases"


def load_company_codes(csv_path):
    """Loads and processes the CSV to create a map of company names to their
    codes."""
    df = pd.read_csv(csv_path, index_col=0)

    code_map = {}
    for row in df.iterrows():
        code = row[0]
        company_name = row[1]["Name"].lower()
        valid_exchanges = ["NASDAQ", "NYSE", "NYSE MKT", "AMEX", "NYSE ARCA"]

        if (
            row[1]["Type"] == "Common Stock"
            and row[1]["Exchange"] in valid_exchanges
        ):
            code_map[company_name] = code

    return code_map


def map_alias(
    fpath,
    code_map,
):
    """Maps aliases to codes."""
    ret = {}
    for d in jsonl_generator(fpath):
        alias_pruned = d["alias"].replace(",", "").replace(".", "").lower()
        if alias_pruned in code_map and code_map[alias_pruned] not in ret:
            ret[code_map[alias_pruned]] = {"name": d["alias"], "qid": d["qid"]}
    return ret


# There are 2 ways of entity alignment:
# 1. Use company names to match entity names and aliases
# 2. Retrieve all stock entities, and backward align them to ticker list
# We will take the union of the results from these two methods.


def main(args):
    # Make some paths
    alias_dir = os.path.join(args.wikidump_dir, "aliases")
    ticker_list_fpath = os.path.join(
        args.stock_data_dir, args.region, "instruments", "all.txt"
    )
    ticker_name_fpath = os.path.join(
        args.ticker_name_map_dir, f"{args.region}_ticker_name_map.yaml"
    )

    ticker_list = read_ticker_list(ticker_list_fpath)
    with open(ticker_name_fpath, "r") as f:
        yaml.safe_load(f)
    with open(args.exchange_info, "r") as f:
        exchange_info = yaml.safe_load(f)
    [os.path.join(alias_dir, x) for x in os.listdir(alias_dir)]

    ret = {}

    # First, query exchange_info
    wiki_xchg_list = REGION_XCHG_MAPPING["Wikidata"][args.region]
    ticker_list_stripped = {
        ticker.replace(".", "_").split("_")[0]: ticker for ticker in ticker_list
    }
    for xchg in wiki_xchg_list:
        stock_entity_list = exchange_info[xchg]
        for stock_entity in stock_entity_list:
            ticker_code = stock_entity["ticker"]
            if ticker_code in ticker_list_stripped:
                ret[ticker_list_stripped[ticker_code]] = {
                    "qid": stock_entity["qid"],
                    "name": stock_entity["name"],
                }

    # Pure name match is too fragile, so we will not use it.
    # # Then, query aliases
    # remaining_tickers = list(set(ticker_list) - set(ret.keys()))
    # print(
    #     f"Remaining tickers: {len(remaining_tickers)}, ratio = {len(remaining_tickers) / len(ticker_list)}"
    # )
    # ticker_list_stripped = {
    #     ticker.replace(".", "_").split("_")[0]: ticker
    #     for ticker in remaining_tickers
    # }
    # code_name_map = {
    #     ticker_name_map.get(ticker, "N/A")
    #     .replace(".", "")
    #     .replace(",", ""): ticker
    #     for ticker in ticker_list_stripped
    # }
    # code_name_map.pop("N/A", None)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=400) as executor:
    #     results = list(
    #         tqdm(
    #             executor.map(
    #                 partial(map_alias, code_map=code_name_map), alias_files
    #             ),
    #             total=len(alias_files),
    #         )
    #     )
    # for result in results:
    #     ret.update(result)
    print(
        f"Successfully mapped {len(ret)} companies to Wikidata QIDs. Ratio: {len(ret) / len(ticker_list)}"
    )
    with open(os.path.join(args.output_dir, f"{args.region}.yaml"), "w") as f:
        yaml.dump(ret, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us")
    parser.add_argument(
        "--stock_data_dir", type=str, default="data/market_data"
    )
    parser.add_argument("--wikidump_dir", type=str, default="data/wikidump")
    parser.add_argument(
        "--ticker_name_map_dir", type=str, default="data/stocks/names"
    )
    parser.add_argument(
        "--server_urls", type=str, default="wikiquery_server_urls.txt"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/wikidata/stock_entity_alignment"
    )
    parser.add_argument(
        "--exchange_info", type=str, default="data/wikidata/exchange_info.yaml"
    )
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        for region in ["us", "cn", "uk", "fr", "hk", "jp"]:
            args.region = region
            main(args)
    else:
        main(args)
