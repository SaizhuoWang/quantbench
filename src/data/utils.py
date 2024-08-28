import typing as tp

import pandas as pd
import ujson as json


def jsonl_generator(fname):
    """Returns generator for jsonl file."""
    for line in open(fname, "r"):
        line = line.strip()
        if len(line) < 3:
            d = {}
        elif line[-1] == ",":
            d = json.loads(line[:-1])
        else:
            d = json.loads(line)
        yield d


def read_ticker_list(fpath: str) -> tp.List[str]:
    record_df = pd.read_csv(
        fpath,
        header=None,
        delimiter="\t",
        index_col=0,
        keep_default_na=False,
        na_values=["_"],
    )
    tickers = record_df.index.to_list()
    return tickers
