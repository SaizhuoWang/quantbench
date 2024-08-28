"""This script fetches all QIDs which have a relationship with a specific value.

For example: all entities which played 'quarterback' on a football team (corresponding to P413 and a value of Q622747)

to run:
python3.6 fetch_with_rel_and_value.py --data $DATA --out_dir $OUT

"""

# Fetch 1-hop relations
import argparse
import os
import sys
import typing as tp
from functools import partial
from multiprocessing import Pool

import ujson
import yaml
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils import jsonl_generator

NUM_PROCESSES = 400

def filter_get_one_hop_relation(universe, filename):
    filtered = []
    for item in jsonl_generator(filename):
        if item["qid"] in universe and item["value"] in universe:
            filtered.append(item)
    return filtered


def find_intermediate_nodes(universe, filename):
    ret = {}
    for item in jsonl_generator(filename):
        from_stock = item["qid"] in universe
        to_stock = item["value"] in universe
        if (from_stock and not to_stock) or (not from_stock and to_stock):
            intermediate_node_qid = (
                item["qid"] if not from_stock else item["value"]
            )
            stock_node_qid = item["qid"] if from_stock else item["value"]
            ll = ret.setdefault(intermediate_node_qid, [])
            stock_role = "src" if from_stock else "dst"
            ll.append((stock_node_qid, item["property_id"], stock_role))
    return ret


def get_ticker_name(ticker_list: tp.List[str], filename: str):
    try:
        filtered = []
        for item in jsonl_generator(filename):
            if item["qid"] in ticker_list:
                filtered.append(item)
        return filtered
    except:
        print("Error in file: ", filename)
        return []


def update_bucket(bucket: tp.Dict, filtered: tp.Dict):
    for k, v in filtered.items():
        if k in bucket:
            bucket[k].extend(v)
        else:
            bucket[k] = v


def filter_2hop_rel(
    intermediate: tp.List, ticker_universe: tp.List, filename: str
):
    filtered = []
    for item in jsonl_generator(filename):
        if (
            item["qid"] in intermediate and item["value"] in ticker_universe
        ) or (item["value"] in intermediate and item["qid"] in ticker_universe):
            filtered.append(item)
    return filtered


def fetch_one_hop_relations(
    entity_rel_files: tp.List[str],
    entity_alignment_fpath: str,
    ticker_list_fpath: str,
    output_dir: str,
):
    # First, load stock universe
    with open(entity_alignment_fpath, "r") as f:
        stock_entity_alignment = yaml.safe_load(f)
    # Dump `stock_records.jsonl`
    with open(os.path.join(output_dir, "stock_records.jsonl"), "w") as f:
        for ticker, info in stock_entity_alignment.items():
            stock_record = {
                "qid": info["qid"],
                "label": info["name"],
                "symbol": ticker,
            }
            f.write(ujson.dumps(stock_record) + "\n")
    # Fetch 1-hop relations
    stock_pool = [info["qid"] for info in stock_entity_alignment.values()]
    pool = Pool(processes=NUM_PROCESSES)
    relations = []
    for output in tqdm(
        pool.imap_unordered(
            partial(filter_get_one_hop_relation, stock_pool),
            entity_rel_files,
            chunksize=1,
        ),
        total=len(entity_rel_files),
    ):
        relations.extend(output)
    with open(
        os.path.join(output_dir, "intra_stock_relations.jsonl"), "w"
    ) as f:
        for relation in relations:
            f.write(ujson.dumps(relation) + "\n")


def fetch_intermediate_nodes(
    entity_rel_files: tp.List[str],
    label_files: tp.List[str],
    entity_alignment_fpath: str,
    ticker_list_fpath: str,
    output_dir: str,
):
    pool = Pool(processes=NUM_PROCESSES)
    # First, load stock universe
    with open(entity_alignment_fpath, "r") as f:
        stock_entity_alignment = yaml.safe_load(f)
    stock_pool = [info["qid"] for info in stock_entity_alignment.values()]

    non_stock_neighbors = {}
    for output in tqdm(
        pool.imap_unordered(
            partial(find_intermediate_nodes, stock_pool),
            entity_rel_files,
            chunksize=1,
        ),
        total=len(entity_rel_files),
        desc="Finding intermediate entities",
    ):
        update_bucket(non_stock_neighbors, output)

    dedup_buckets = {k: list(set(v)) for k, v in non_stock_neighbors.items()}
    intermediate_nodes = [k for k, v in dedup_buckets.items() if len(v) > 1]
    label_info = []
    for output in tqdm(
        pool.imap_unordered(
            partial(get_ticker_name, intermediate_nodes),
            label_files,
            chunksize=1,
        ),
        total=len(label_files),
        desc="Getting labels of intermediate nodes",
    ):
        for item in output:
            try:
                label_info.append(
                    {
                        "qid": item["qid"],
                        "label": item["label"],
                        "neighbors": dedup_buckets[item["qid"]],
                    }
                )
            except:
                continue
    with open(
        os.path.join(output_dir, "intermediate_nodes.jsonl"),
        "w",
    ) as f:
        for item in label_info:
            f.write(ujson.dumps(item) + "\n")
    # Find 2-hop relations
    rel_2hops = []
    for output in tqdm(
        pool.imap_unordered(
            partial(filter_2hop_rel, intermediate_nodes, stock_pool),
            entity_rel_files,
            chunksize=1,
        ),
        total=len(entity_rel_files),
        desc="Getting 2-hop relations",
    ):
        rel_2hops.extend(output)
    with open(
        os.path.join(output_dir, "two_hop_relations.jsonl"),
        "w",
    ) as f:
        for item in rel_2hops:
            f.write(ujson.dumps(item) + "\n")


def main(args):
    # Get some dir and paths first
    entity_rel_files = [
        os.path.join(args.wikidump_dir, "entity_rels", x)
        for x in os.listdir(os.path.join(args.wikidump_dir, "entity_rels"))
    ]
    label_files = [
        os.path.join(args.wikidump_dir, "labels", x)
        for x in os.listdir(os.path.join(args.wikidump_dir, "labels"))
    ]
    entity_alignment_fpath = os.path.join(
        args.wiki_data_dir, "stock_entity_alignment", f"{args.region}.yaml"
    )
    ticker_list_fpath = os.path.join(
        args.stock_data_dir, args.region, "instruments", "all.txt"
    )
    output_dir = os.path.join(args.wiki_data_dir, "stock_graph", args.region)
    os.makedirs(output_dir, exist_ok=True)

    # Fetch one-hop relations
    fetch_one_hop_relations(
        entity_rel_files=entity_rel_files,
        entity_alignment_fpath=entity_alignment_fpath,
        ticker_list_fpath=ticker_list_fpath,
        output_dir=output_dir,
    )

    # Fetch intermediate nodes
    fetch_intermediate_nodes(
        entity_rel_files=entity_rel_files,
        label_files=label_files,
        entity_alignment_fpath=entity_alignment_fpath,
        ticker_list_fpath=ticker_list_fpath,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="us", help="country")
    parser.add_argument("--wikidump_dir", type=str, required=True)
    parser.add_argument(
        "--stock_data_dir", type=str, default="data/market_data"
    )
    parser.add_argument("--wiki_data_dir", type=str, default="data/wikidata")
    parser.add_argument("--auto", action="store_true", help="use auto config")
    parser.add_argument("--processes", type=int, default=400)
    args = parser.parse_args()
    NUM_PROCESSES = args.processes

    if args.auto:
        for region in ["cn", "us", "hk", "jp", "uk", "fr"]:
            args.region = region
            main(args)
    else:
        main(args)
