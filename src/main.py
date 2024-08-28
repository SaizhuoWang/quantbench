import collections
import copy
import datetime
import importlib
import itertools
import os
import sys
import time
import typing as tp
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import dgl
import hydra
import jsonlines
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from dateutil import rrule
from dgl import DGLGraph
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies.single_device import SingleDeviceStrategy
from omegaconf import OmegaConf
from q4l.config import ExperimentConfig, GlobalConfig, JobConfig, TimeInterval
from q4l.data.dataset import Q4LDataModule
from q4l.data.graph import Edge, HyperEdge, StockGraph
from q4l.eval.plotting import plot_return_curve
from q4l.exp.pipeline import q4l_task_wrapper_fn
from q4l.exp.repeat import RepeatExpManager
from q4l.exp.rolling import (
    RollingSubdirCollector,
    RollingTaskRunner,
    subdir_contain,
)
from q4l.model.base import (
    NonDLModel,
    QuantModel,
    SpatiotemporalModel,
    TimeSeriesModel,
)
from q4l.model.trainer import Q4LTrainer
from q4l.qlib import init as qlib_init
from q4l.qlib.model.ens.group import RollingGroup
from q4l.qlib.workflow import R
from q4l.utils.log import LoggingContext, get_logger
from q4l.utils.misc import (
    create_instance,
    generate_evaluations,
    make_qlib_init_config,
)
from torch.profiler import schedule
from torchinfo import summary
from q4l.exp.repeat import ChildRunCollector
import mlflow
from q4l.constants import (
    TAG_KEY,
    TEMPORAL_EMBEDDING_KEY,
    INPUT_KEY,
    SPATIAL_EMBEDDING_KEY,
)
from datetime import timedelta
from lightning.pytorch.loggers import MLFlowLogger
from q4l.model.rl import RLModel


def read_wikidata(
    data_dir: str,
    ticker_index_map: tp.Dict[str, int],
    use_two_hop: bool = False,
) -> tp.List[Edge]:
    date_formatter = lambda x: x.rstrip("Z").strip("+").replace("-00", "-01")
    stock_qid_map = {}

    with jsonlines.open(
        os.path.join(data_dir, "stock_records.jsonl"), "r"
    ) as reader:
        for stock_record in reader:
            stock_qid_map[stock_record["qid"]] = stock_record["symbol"]
    ret_edge_list = []
    ret_data_dict = {}
    with jsonlines.open(
        os.path.join(data_dir, "intra_stock_relations.jsonl"), "r"
    ) as reader:
        for rel in reader:
            src_symbol = stock_qid_map[rel["qid"]]
            dst_symbol = stock_qid_map[rel["value"]]
            if (
                src_symbol not in ticker_index_map
                or dst_symbol not in ticker_index_map
            ):
                continue
            start_time = rel.get("start_time", None)
            end_time = rel.get("end_time", None)
            start_time = (
                datetime.fromisoformat(date_formatter(start_time))
                if start_time
                else None
            )
            end_time = (
                datetime.fromisoformat(date_formatter(end_time))
                if end_time
                else None
            )
            rel_type = rel["property_id"]
            ret_edge_list.append(
                Edge(
                    source=src_symbol,
                    destination=dst_symbol,
                    start_time=start_time,
                    end_time=end_time,
                    info_source="Wiki",
                    edge_type=rel_type,
                )
            )

    # First construct the intermediate graph, and then re-connect two-hop relations
    if use_two_hop:
        intermediate_node_adj_list = defaultdict(list)
        with jsonlines.open(
            os.path.join(data_dir, "two_hop_relations.jsonl"), "r"
        ) as reader:
            for entry in reader:
                stock_qid = (
                    entry["qid"]
                    if entry["qid"] in stock_qid_map
                    else entry["value"]
                )
                intermediate_node_qid = (
                    entry["value"]
                    if entry["qid"] in stock_qid_map
                    else entry["qid"]
                )
                if stock_qid_map[stock_qid] not in ticker_index_map:
                    continue
                data_dict = {
                    "stock_qid": stock_qid,
                    "pid": entry["property_id"],
                }
                if "start_time" in entry:
                    data_dict["start_time"] = datetime.fromisoformat(
                        date_formatter(entry["start_time"])
                    )
                if "end_time" in entry:
                    try:
                        data_dict["end_time"] = datetime.fromisoformat(
                            date_formatter(entry["end_time"])
                        )
                    except:
                        pass
                intermediate_node_adj_list[intermediate_node_qid].append(
                    data_dict
                )
        # Iterate over the intermediate nodes
        for node_qid, neighbor_list in intermediate_node_adj_list.items():
            # Iterate over pairs of neighbors
            neighbor_pairs = itertools.combinations(neighbor_list, 2)
            for pair in neighbor_pairs:
                # Check if the two neighbors are connected
                if pair[0]["stock_qid"] == pair[1]["stock_qid"]:
                    continue
                src_symbol = stock_qid_map[pair[0]["stock_qid"]]
                dst_symbol = stock_qid_map[pair[1]["stock_qid"]]

                start_time_0 = pair[0].get("start_time", datetime(1970, 1, 1))
                start_time_1 = pair[1].get("start_time", datetime(1970, 1, 1))

                # Latest start time
                start_time = (
                    start_time_0
                    if start_time_0 > start_time_1
                    else start_time_1
                )
                end_time_0 = pair[0].get("end_time", datetime(2100, 1, 1))
                end_time_1 = pair[1].get("end_time", datetime(2100, 1, 1))

                # Earliest end time
                end_time = (
                    end_time_0 if end_time_0 < end_time_1 else end_time_1
                )

                ret_edge_list.append(
                    Edge(
                        source=src_symbol,
                        destination=dst_symbol,
                        start_time=start_time,
                        end_time=end_time,
                        info_source="Wiki",
                        edge_type=f"{pair[0]['pid']}_{pair[1]['pid']}",
                    )
                )
                # Reverse direction
                ret_edge_list.append(
                    Edge(
                        source=dst_symbol,
                        destination=src_symbol,
                        start_time=start_time,
                        end_time=end_time,
                        info_source="Wiki",
                        edge_type=f"{pair[1]['pid']}_{pair[0]['pid']}",
                    )
                )

    return ret_edge_list


def read_industry(
    data_path: str, ticker_index_map: tp.Dict[str, int]
) -> tp.List[tp.Dict]:
    """
    Reads the industry CSV file and returns a list of dictionaries that can be used
    with the add_hyperedge function. Creates new hyperedges when changes occur and
    preserves all hyperedges, including those that have ended.

    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing industry data.
    ticker_index_map : Dict[str, int]
        A dictionary mapping ticker symbols to their corresponding indices.

    Returns:
    --------
    List[Dict]
        A list of dictionaries, each containing the information needed for add_hyperedge.
    """

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = df.sort_index()  # Ensure the DataFrame is sorted by date
    df.columns = [col.split(".")[0] for col in df.columns]

    # Extract unique industry categories, ignoring NaNs
    industry_categories = sorted(set(df.values.ravel()) - {np.nan})
    industry_to_index = {ind: i for i, ind in enumerate(industry_categories)}

    # Number of unique tickers
    N = len(ticker_index_map)

    # Create a mapping from ticker to index for faster lookup
    ticker_to_index = pd.Series(ticker_index_map)

    # Pre-allocate a numpy array for all incidence vectors
    all_incidence_vectors = np.zeros(
        (len(df), len(industry_categories), N), dtype=np.int8
    )

    # Vectorized operations for all dates
    for i, industry in enumerate(industry_categories):
        industry_mask = df == industry
        valid_tickers = industry_mask.columns[industry_mask.any()]
        # valid_tickers = [ticker for ticker in valid_tickers]
        valid_indices = (
            ticker_to_index.reindex(valid_tickers).dropna().astype(int)
        )
        valid_tickers = valid_indices.index  # re-align the index
        all_incidence_vectors[:, i, valid_indices] = industry_mask.loc[
            :, valid_tickers
        ].values

    dates = df.index.to_pydatetime()

    # Find changes in incidence vectors
    changes = np.diff(all_incidence_vectors, axis=0)
    change_indices = np.nonzero(np.any(changes != 0, axis=2))

    hyperedges = []

    for industry_idx in range(len(industry_categories)):
        industry_changes = change_indices[1] == industry_idx
        industry_change_dates = change_indices[0][industry_changes]

        if len(industry_change_dates) == 0:
            # No changes for this industry, create a single hyperedge
            hyperedges.append(
                {
                    "incidence_vector": torch.from_numpy(
                        all_incidence_vectors[0, industry_idx]
                    ),
                    "etype": f"industry_{industry_categories[industry_idx]}",
                    "info_source": "Industry",
                    "start_time": dates[0],
                    "end_time": dates[-1],
                }
            )
        else:
            # Create hyperedges for each period between changes
            start_dates = np.concatenate(([0], industry_change_dates + 1))
            end_dates = np.concatenate(
                (industry_change_dates, [len(dates) - 1])
            )

            for start, end in zip(start_dates, end_dates):
                hyperedges.append(
                    {
                        "incidence_vector": torch.from_numpy(
                            all_incidence_vectors[start, industry_idx]
                        ),
                        "etype": f"industry_{industry_categories[industry_idx]}",
                        "info_source": "Industry",
                        "start_time": dates[start],
                        "end_time": dates[end],
                    }
                )

    return hyperedges


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


class QBenchGraphModel(SpatiotemporalModel):
    def __init__(
        self,
        config: ExperimentConfig,
        data: Q4LDataModule,
        **kwargs,
    ):
        self.data = data
        self.config = config
        self.graph_config = config.data.graph
        self.info_sources = []
        self.ticker_list = data.ticker_list
        self.ticker_index_map = {
            ticker: i for i, ticker in enumerate(self.ticker_list)
        }
        self.start_time = datetime.fromisoformat(config.time.start_time)
        self.end_time = datetime.fromisoformat(config.time.end_time)
        self.create_kg(device=kwargs.get("device", "cpu"))
        super().__init__(config, data, **kwargs)

    def create_kg(self, device: str):
        self.kg = StockGraph(device=device)
        self.kg.add_nodes(self.ticker_list)
        if self.graph_config.use_wikidata:
            self.read_wiki_graph()
        if self.graph_config.use_industry:
            self.read_industry_graph()

    def read_wiki_graph(self):
        wikidata_edges = read_wikidata(
            data_dir=os.path.join(
                self.graph_config.wikidata_dir, self.config.data.region
            ),
            ticker_index_map=self.ticker_index_map,
            # use_two_hop=True,
        )
        src_list, dst_list, etype_list, st_list, ed_list, info_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for edge in wikidata_edges:
            src_list.append(self.ticker_index_map[edge.source])
            dst_list.append(self.ticker_index_map[edge.destination])
            etype_list.append(edge.edge_type)
            st_list.append(edge.start_time)
            ed_list.append(edge.end_time)
            info_list.append(edge.info_source)
        self.kg.add_edges(
            src=torch.tensor(src_list),
            dst=torch.tensor(dst_list),
            etypes=etype_list,
            info_source=info_list,
            start_times=st_list,
            end_times=ed_list,
        )
        self.info_sources.append("Wiki")

    def read_industry_graph(self):
        industry_data_path = os.path.join(
            self.graph_config.industry_dir,
            f"{self.config.data.region}_ind.csv",
        )
        industry_hyper_edges = read_industry(
            data_path=industry_data_path,
            ticker_index_map=self.ticker_index_map,
        )
        for hyperedge in industry_hyper_edges:
            self.kg.add_hyperedge(**hyperedge)
        self.info_sources.append("Industry")

    def _build_spatial_model(self, try_kwargs):
        return create_instance(
            self.config.model.components.spatial,
            stock_kg=self.kg,
            try_kwargs=try_kwargs,
        )


def quant_model_factory(config: ExperimentConfig, **kwargs) -> QuantModel:
    logger = get_logger("quant_model_factory")

    if not hasattr(config.model, "model_type"):
        logger.warning(
            "No model type specified in config, using default = temporal."
        )
        model_type = "temporal"
    else:
        model_type = config.model.model_type

    model_class_map = {
        "temporal": TimeSeriesModel,
        "spatial": QBenchGraphModel,
        "non_dl": NonDLModel,
        "rl": RLModel,
    }

    if model_type not in model_class_map:
        # If model_type is not in the map, we assume it is a class name with import path
        try:
            model_mod, model_cls = model_type.rsplit(".", 1)
            model_class = getattr(
                importlib.import_module(model_mod), model_cls
            )
        except ImportError:
            raise ValueError(f"Model type {model_type} not supported.")
    else:
        model_class = model_class_map[model_type]

    return model_class(config=config, **kwargs)


def make_trainer(exp_config, job_config, recorder_wrapper):
    device = (
        job_config.misc.device
        if job_config.misc.device != "cuda:-1" and torch.cuda.is_available()
        else "cpu"
    )

    strategy = SingleDeviceStrategy(device=device)
    trainer_callbacks = []

    if exp_config.model.model_type == "non_dl":
        monitor = "valid_mse"
    else:
        monitor = exp_config.model.basic_info.monitor

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(recorder_wrapper.artifact_uri, "checkpoints"),
        filename="bestmodel_{epoch:02d}",
        monitor=monitor,
        save_last=True,
        save_top_k=1,
        verbose=True,
        mode="max",
    )
    trainer_callbacks.append(ckpt_callback)

    # Specify early-stopping metric
    if exp_config.model.basic_info.patience != np.inf:
        es_callback = EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=exp_config.model.basic_info.patience,
            verbose=True,
        )
        trainer_callbacks.append(es_callback)

    # Make loggers
    tb_logger = TensorBoardLogger(
        save_dir=job_config.machine.tensorboard_dir,
        name=job_config.name.run_name,
    )
    csv_logger = CSVLogger(
        save_dir=recorder_wrapper.artifact_uri, name="csv_logs"
    )
    mlflow_logger = MLFlowLogger(
        tracking_uri=recorder_wrapper.get_uri(),
        run_id=recorder_wrapper.get_recorder().id,
    )
    profiler = PyTorchProfiler(
        filename=f"{job_config.name.run_name}_{job_config.misc.timestamp}.txt",
        dirpath=os.path.join(job_config.machine.log_root, "profiler"),
        record_module_names=True,
        profile_memory=True,
        group_by_input_shape=True,
        with_stack=True,
        with_flops=True,
        schedule=schedule(wait=1, warmup=5, active=5, repeat=3),
        with_modules=True,
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(
                job_config.machine.log_root,
                "pytorch_profiler_tensorboard",
                f"{job_config.name.run_name}_{job_config.misc.timestamp}",
            )
        ),
    )
    trainer = Q4LTrainer(
        default_root_dir=recorder_wrapper.artifact_uri,
        strategy=strategy,
        callbacks=trainer_callbacks,
        logger=[tb_logger, csv_logger, mlflow_logger],
        enable_progress_bar=job_config.misc.debug,
        # gradient_clip_val=2.0,
        profiler=profiler if job_config.misc.profile else None,
        **exp_config.model.trainer,
    )

    return trainer


def tune_hyperparameters(
    exp_config: ExperimentConfig,
    job_config: JobConfig,
    data: Q4LDataModule,
    recorder_wrapper,
):
    logger = get_logger("tune_hyperparameters")
    logger.info(f"Before running, tuning hyperparam using dev set")
    trainer = make_trainer(exp_config, job_config, recorder_wrapper)
    model = quant_model_factory(
        exp_config,
        data=data,
        device=job_config.misc.device,
        recorder_wrapper=recorder_wrapper,
    )
    trainer.fit(model, datamodule=data)

    new_config = copy.deepcopy(exp_config)
    ckpt_callback: ModelCheckpoint = trainer.checkpoint_callback
    best_epoch = (
        int(ckpt_callback.best_model_path.split(".")[0].split("=")[-1]) + 1
    )

    logger.info(f"Finished tuning, best epoch = {best_epoch}")

    trainer_cfg = OmegaConf.to_container(
        new_config.model.trainer, resolve=True
    )
    basic_info_cfg = OmegaConf.to_container(
        new_config.model.basic_info, resolve=True
    )

    # Change number of training epochs
    trainer_cfg["max_epochs"] = best_epoch
    # No early stopping
    basic_info_cfg["patience"] = np.inf
    # No validation set
    trainer_cfg["limit_val_batches"] = 0.0
    # Trainer ckpt loading mode is last
    basic_info_cfg["ckpt_loading_mode"] = "last"

    new_config.model.trainer = OmegaConf.create(trainer_cfg)
    new_config.model.basic_info = OmegaConf.create(basic_info_cfg)

    # compute last day of historical data
    first_test_day = pd.Timestamp(new_config.time.segments.test[0].start)
    last_historical_day = first_test_day - timedelta(days=1)
    new_config.time.fit_end_time = last_historical_day
    new_config.time.segments.train = [
        TimeInterval(
            start=f'${{timestamp:{new_config.time.segments.train[0].start.strftime("%Y-%m-%d")}}}',
            end=f'${{timestamp:{last_historical_day.strftime("%Y-%m-%d")}}}',
        )
    ]

    return new_config


def pipeline_fn(
    exp_config: ExperimentConfig,
    job_config: JobConfig,
    gpu_index: int,
    is_subprocess: bool,
    r_suffix: str,
    recorder_wrapper,
    **kwargs,
):
    if is_subprocess:
        qlib_init(
            **make_qlib_init_config(
                exp_config=exp_config, job_config=job_config
            )
        )
    start_time = time.time()
    job_config.misc.device = f"cuda:{gpu_index}"

    # Get a logger instance
    logger = get_logger("q4l.task_fn")
    seed_everything(job_config.misc.seed)
    logger.info(f"job_config.misc.device = {job_config.misc.device}")

    # Log the dataset name and load the dataset
    logger.info(f"Loading data ...")
    data = Q4LDataModule(exp_config=exp_config, job_config=job_config)
    logger.info(f"Successfully loaded dataset {data}")

    if exp_config.model.basic_info.get("tune_test", False):
        # Do hyperparameter tuning on the dev set, then re-training on the train+dev set
        exp_config = tune_hyperparameters(
            exp_config, job_config, data, recorder_wrapper
        )
        # Re-load the dataset with new config
        data = Q4LDataModule(exp_config=exp_config, job_config=job_config)

    # Create the model and optimizer
    logger.info(f"Creating model ...")
    model: QuantModel = quant_model_factory(
        exp_config,
        data=data,
        device=job_config.misc.device,
        recorder_wrapper=recorder_wrapper,
    )
    logger.info(f"Successfully created model {model}")

    # Create the trainer and train the model
    logger.info("Creating trainer")
    trainer = make_trainer(exp_config, job_config, recorder_wrapper)

    # Train the model
    logger.info("Starts training.")
    if "rl" not in exp_config.model.model_type.lower():
        trainer.fit(model, datamodule=data)
    else:
        trainer.fit(model)
    logger.info("Training finished")

    if exp_config.model.model_type == "rl":
        # For RL models, we do not need to pass in the datamodule
        if exp_config.model.trainer.max_epochs > 0:
            if exp_config.model.basic_info.get("ckpt_loading_mode", "last"):
                trainer.predict(model=model)
            else:
                trainer.predict(model=model, ckpt_path="best")
        else:
            trainer.predict(model=model)
    else:
        if exp_config.model.trainer.max_epochs > 0:
            if exp_config.model.basic_info.get("ckpt_loading_mode", "last"):
                trainer.predict(model=model, datamodule=data)
            else:
                trainer.predict(model=model, ckpt_path="best", datamodule=data)

        else:
            trainer.predict(model=model, datamodule=data)

    recorder_config_dict = OmegaConf.to_container(
        exp_config.collector.single, resolve=True
    )

    # Record the performance
    logger.info("Start recording performance")

    recorder = recorder_wrapper.get_recorder()
    for record_name, recorder_config in recorder_config_dict.items():
        # Some recorder require the parameter `model` and `dataset`.
        # try to automatically pass in them to the initialization function
        # to make defining the tasking easier
        logger.info(f"Running recorder {record_name}")
        r = create_instance(
            recorder_config,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": data},
            recorder=recorder,
            recorder_wrapper=recorder_wrapper,
        )
        r.generate()

    logger.info("Successfully recorded performance, task finished.")
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time} seconds")


def rolling_fn(
    repeat_index: int,
    gpu_index: int,
    config: GlobalConfig,
    is_subprocess: bool,
    timestamp: str,
    recorder_wrapper,
    parent_run_id: str = None,
    **kwargs,
):
    """A rolling experiment."""
    if is_subprocess:
        qlib_init(**make_qlib_init_config(config=config))
    recorder_wrapper.set(
        experiment_id=recorder_wrapper.get_recorder().experiment_id,
        recorder_id=parent_run_id,
    )

    with LoggingContext(
        is_debug=config.job.misc.debug, recorder_wrapper=recorder_wrapper
    ):
        logger = get_logger("rolling_fn")
        logger.info(f"is_subprocess = {is_subprocess}")
        logger.info(
            f"Repeat index = {repeat_index}, is_subprocess = {is_subprocess}"
        )
        logger.info(
            f"exp name = {config.job.name.exp_name}, run name = {config.job.name.run_name}"
        )
        logger = get_logger("quantbench.main.rolling")
        logger.info(f"Start repeat #{repeat_index} on GPU #{gpu_index}")
        logger.info(
            f"Start worker_fn #{repeat_index} on GPU #{config.job.misc.device}"
        )
        rolling_runner = BenchmarkRollingRunner(
            config=config,
            repeat_index=repeat_index,
            gpu_index=gpu_index,
            is_subprocess=is_subprocess,
            timestamp=timestamp,
            recorder_wrapper=recorder_wrapper,
        )
        task_func = partial(
            q4l_task_wrapper_fn,
            actual_func=pipeline_fn,
            gpu_index=gpu_index,
            parent_run_id=parent_run_id,
        )  # Wrap the task function with q4l_task_wrapper to init running environment
        rolling_runner.start(task_func=task_func, is_subprocess=is_subprocess)


class BenchmarkRollingRunner(RollingTaskRunner):
    def __init__(
        self,
        config: GlobalConfig,
        repeat_index: int,
        gpu_index: int,
        is_subprocess: bool,
        timestamp: str,
        recorder_wrapper,
    ):
        super().__init__(config, repeat_index, timestamp, recorder_wrapper)
        self.gpu_index = gpu_index
        self.is_subprocess = is_subprocess

    def rolling_generate(
        self,
        total_ticks: int,
        test_start: int,
        rolling_step: int,
        validation_ratio: float = 0.1,
        mode: str = "sliding",
    ):
        configs = []
        test_tick = test_start
        train_start = 0
        while test_tick < total_ticks:
            this_config: ExperimentConfig = copy.deepcopy(self.exp_config)
            train_end = test_tick - 1
            valid_start = int(
                train_end - (train_end - train_start) * validation_ratio
            )
            train_split = (train_start, valid_start)
            valid_split = (valid_start, train_end)
            test_split = (test_tick, test_tick + rolling_step - 1)

            if test_split[1] >= total_ticks:
                test_split = (test_tick, total_ticks - 1)

            test_tick += rolling_step
            if mode == "sliding":
                train_start += rolling_step
            this_config.time.segments.train = [TimeInterval(*train_split)]
            this_config.time.segments.valid = [TimeInterval(*valid_split)]
            this_config.time.segments.test = [TimeInterval(*test_split)]
            configs.append(this_config)

        return configs

    def _generate_tasks(self) -> tp.List[ExperimentConfig]:
        return super()._generate_tasks()

    def collect(self):
        root_dir = self.recorder_wrapper.artifact_uri
        print(
            f"Rolling collector root dir = {root_dir}.\n"
            f"Collecting rolling results for task {self.repeat_index} ..."
        )

        numerical_collector = ChildRunCollector(
            mlflow_client=mlflow.tracking.MlflowClient(
                self.recorder_wrapper.get_uri()
            ),
            current_run_id=self.recorder_wrapper.get_recorder().id,
            subrun_key_func=lambda run: ("rolling", run.info.run_name),
            artifacts_path={
                "pred": "sig_analysis/pred.pkl",
                "label": "sig_analysis/label.pkl",
            },
            process_list=RollingGroup(),
        )
        rolling_results = numerical_collector()

        # Need to specify these things with prefix directory.
        # This is due to the mechanism in the class `ACRecordTemp`. It will
        # perform a check before generating any anaylsis.
        # So we need to put the things into the corresponding directories
        artifact_prefix = {"pred": "sig_analysis", "label": "sig_analysis"}

        rolling_recorder = self.recorder_wrapper.get_recorder()
        self.logger.info(
            f"Rolling recorder ID = {rolling_recorder.id}, name = {rolling_recorder.name}"
        )
        for k, v in rolling_results.items():
            key = list(v.keys())[0]
            file_suffix = artifact_prefix[k]
            rolling_recorder.save_objects(
                artifact_path=file_suffix, **{f"{k}.pkl": v[key]}
            )
        self.logger.info(f"Collect finished.")
        return rolling_recorder

    def _post_analysis(self):
        self.collect()
        generate_evaluations(
            config=self.config,
            stage_key="rolling",
            recorder_wrapper=self.recorder_wrapper,
            logger=self.logger,
        )
        # Do additional plotting
        if "portfolio_analysis" in self.config.experiment.collector.rolling:
            self.plot_rolling_results()

    def plot_rolling_results(self):
        # Do some plotting, with rolling regions
        with open(
            os.path.join(
                self.recorder_wrapper.artifact_uri, "rolling_list.txt"
            ),
            "r",
        ) as f:
            interval_lines = f.readlines()
            intervals = []
            for line in interval_lines:
                interval_ends = line.strip().split("~")
                start, end = interval_ends[0], interval_ends[1]
                intervals.append((start, end))
        portfolio_report_df = pd.read_csv(
            os.path.join(
                self.recorder_wrapper.artifact_uri,
                "portfolio_analysis",
                "report_normal_1day.csv",
            ),
            index_col=0,
        )
        plot_return_curve(
            df=portfolio_report_df,
            intervals=intervals,
            artifact_uri=self.recorder_wrapper.artifact_uri,
        )

        self.logger.info(
            f"Rolling runner #{self.repeat_index} finished collecting"
        )


# QuantBench entry function
@hydra.main(
    config_path="../config",
    config_name="qbench_base_config",
    version_base=None,
)
def main(config: GlobalConfig):
    # Initialize time stamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    config.job.misc.timestamp = timestamp
    # Create the experiment manager
    repeat_exp_mgr = RepeatExpManager(config=config, exp_fn=rolling_fn)
    repeat_exp_mgr.run()  # Launch the experiment


if __name__ == "__main__":
    main()
