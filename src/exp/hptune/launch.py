import sys

import submitit

import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

# Please refer to quant_cl/main.py for original code.
import copy
import datetime
import os
import typing as tp
from datetime import datetime
from functools import partial

import hydra
import pandas as pd
import numpy as np
import psutil
from exp.utils import get_machine
from hydra import compose, initialize
from main import main as task_main
from omegaconf import OmegaConf
from q4l.config import ExperimentConfig, GlobalConfig, TimeInterval
from q4l.eval.plotting import plot_return_curve
from q4l.exp.pipeline import q4l_task_wrapper_fn
from q4l.exp.repeat import RepeatExpManager
from q4l.exp.rolling import (
    RollingSubdirCollector,
    RollingTaskRunner,
    subdir_contain,
)
from q4l.qlib import init as qlib_init
from q4l.qlib.model.ens.group import RollingGroup
from q4l.qlib.workflow import R
from q4l.utils.log import LoggingContext, get_logger
from q4l.utils.misc import generate_evaluations, make_qlib_init_config

# job submission
ablation_overrides = {
    "seg_random": [],
    "frag_random": [],
    "normal": [],
    "normal_trainagain": [
        "++experiment.model.basic_info.tune_test=true",
    ],
    "segr_trainagain": [
        "++experiment.model.basic_info.tune_test=true",
    ],
    "fragr_trainagain": [
        "++experiment.model.basic_info.tune_test=true",
    ],
}

common_overrides = [
    "experiment/model/loss=ic_loss",
    "experiment/model=lstm",
    "experiment/time=hptune_exp_time",
    "++experiment.data.sampler.x_group=[ohlcvp,fundamental]",
    "++experiment.data.sampler.cs_subsample_ratio=0.75",
    "++experiment.data.benchmark='../data/benchmark/sp500.csv'",
    "++experiment.time.rolling_step=10000",
]


executor = submitit.AutoExecutor(folder="./qbench_slurm_log/hptune")
job_partition = "3090,4090" if get_machine() == "hgx" else "batch"
executor.update_parameters(
    **{
        "slurm_mem": "450G",
        "slurm_time": "infinite",
        "cpus_per_task": 60,
        "gpus_per_node": 2,
        "slurm_partition": job_partition,
        "nodes": 1,
        "tasks_per_node": 1,
        "slurm_job_name": "hptune",
        "slurm_exclude": "dgx040",
    }
)

job_dict = {}
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def change_config(cfg: GlobalConfig, setting: str) -> GlobalConfig:
    """
    Refactors the configuration by modifying time intervals for training and validation sets.
    
    Parameters
    ----------
    cfg : GlobalConfig
        The initial configuration object that contains experiment settings.
    setting : str
        The type of change to be applied, e.g., "seg" for segment-based modification or "frag" for fragment-based modification.
    
    Returns
    -------
    GlobalConfig
        A deep-copied configuration object with modified time intervals.
    """
    final_cfg = copy.deepcopy(cfg)
    
    def create_timestamped_interval(start: pd.Timestamp, end: pd.Timestamp) -> TimeInterval:
        """
        Helper function to format a time interval with timestamps for Hydra compatibility.
        """
        return TimeInterval(
            start=f'${{timestamp:{start.strftime("%Y-%m-%d")}}}',
            end=f'${{timestamp:{end.strftime("%Y-%m-%d")}}}',
        )
    
    if 'seg' in setting:
        train_begin = pd.Timestamp(cfg.experiment.time.fit_start_time)
        train_end = pd.Timestamp(cfg.experiment.time.fit_end_time)
        val_start = pd.Timestamp(cfg.experiment.time.segments.valid[0].start)
        val_end = pd.Timestamp(cfg.experiment.time.segments.valid[0].end)
        val_length = val_end - val_start

        # Randomly select a starting point for the validation set from [train_begin, val_start)
        new_val_start = train_begin + pd.Timedelta(
            np.random.randint(0, (val_start - train_begin).days) - 5, "D"
        )
        new_val_end = new_val_start + val_length

        # Now training set is partitioned into two segments:
        # [train_begin, new_val_start) and [new_val_end, train_end)
        final_cfg.experiment.time.segments.train = [
            create_timestamped_interval(train_begin, new_val_start),
            create_timestamped_interval(new_val_end, train_end),
        ]
        final_cfg.experiment.time.segments.valid = [
            create_timestamped_interval(new_val_start, new_val_end),
        ]

    elif "frag" in setting:
        train_begin = pd.Timestamp(cfg.experiment.time.fit_start_time)
        train_end = pd.Timestamp(cfg.experiment.time.fit_end_time)
        val_start = pd.Timestamp(cfg.experiment.time.segments.valid[0].start)
        val_end = pd.Timestamp(cfg.experiment.time.segments.valid[0].end)
        val_length = val_end - val_start
        n_fragments = 3
        fragment_length = val_length / n_fragments

        interval_end = val_start - pd.Timedelta(5, "D")
        valid_segments = []
        train_segments = []

        for i in range(n_fragments):
            frag_start = train_begin + pd.Timedelta(
                np.random.randint(0, (interval_end - train_begin).days), "D"
            )
            frag_end = frag_start + fragment_length

            train_segments.append(create_timestamped_interval(train_begin, frag_start))
            valid_segments.append(create_timestamped_interval(frag_start, frag_end))

            train_begin = frag_end + pd.Timedelta(1, "D")
            interval_end = val_end - (n_fragments - i - 1) * fragment_length - pd.Timedelta(5, "D")

        # Add the last training segment if any remains
        if train_begin < train_end:
            train_segments.append(create_timestamped_interval(train_begin, train_end))

        final_cfg.experiment.time.segments.train = train_segments
        final_cfg.experiment.time.segments.valid = valid_segments

    return final_cfg


for k, v in ablation_overrides.items():
    with initialize(version_base=None, config_path="../../../config/"):
        setting = k
        cfg = compose(
            config_name="qbench_base_config",
            overrides=v
            + common_overrides
            + [
                f"job/machine={get_machine()}",
                f"++job.name.exp_name=Exp|Eval|HPTune_{timestamp}",
                f"++job.name.run_name={setting}",
                f"++job.resource.total_gpu=2",
                f"++experiment.data.preprocess_with_gpu={get_machine() == 'dgx'}",
                "++experiment.data.use_shm=false",
                "++job.misc.prepare_shm=false",
                "++job.parallel.rolling=1",
                "++job.parallel.repeat=2",
            ],
        )

        # Do some custom modifications to the config
        final_cfg = change_config(cfg, k)

        job_dict[k] = executor.submit(task_main, final_cfg)
        print(f"Submitted {k} to job {job_dict[k].job_id}")
