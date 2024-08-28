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
    "10repeat": ["++job.repeat.total_repeat=10"],
    "20repeat": ["++job.repeat.total_repeat=20"],
    "40repeat": ["++job.repeat.total_repeat=40"],
}

common_overrides = [
    "experiment/data=qbench_data_fe",
    "experiment/model/loss=ic_loss",
    "++experiment.data.sampler.cs_subsample_ratio=0.75",
    "experiment/model=mixer",
    "+experiment/data/loader/alpha@experiment.data.loader.alpha.alpha56ba=alpha56ba",
    "++experiment.data.sampler.x_group=[alpha56ba]",
    "++experiment.data.region=cn",
    "++experiment.data.benchmark='../data/benchmark/csi1000.csv'",
]


executor = submitit.AutoExecutor(folder="./qbench_slurm_log/robustness")
job_partition = "3090,4090" if get_machine() == "hgx" else "batch"
executor.update_parameters(
    **{
        "slurm_mem": "1900G",
        "slurm_time": "infinite",
        "cpus_per_task": 240,
        "gpus_per_node": 8,
        "slurm_partition": job_partition,
        "nodes": 1,
        "tasks_per_node": 1,
        "slurm_job_name": "overfit_exp",
        "slurm_exclude": "dgx040",
    }
)

job_dict = {}
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

for k, v in ablation_overrides.items():
    with initialize(version_base=None, config_path="../../../config/"):
        setting = k
        cfg = compose(
            config_name="qbench_base_config",
            overrides=v
            + common_overrides
            + [
                f"job/machine={get_machine()}",
                f"++job.name.exp_name=Exp|Eval|Overfit_{timestamp}",
                f"++job.name.run_name={setting}",
                f"++job.resource.total_gpu=8",
                f"++experiment.data.preprocess_with_gpu=false",
                "++experiment.data.use_shm=true",
                "++job.misc.prepare_shm=true",
                "++job.parallel.rolling=1",
                "++job.parallel.repeat=8",
            ],
        )
        job_dict[k] = executor.submit(task_main, cfg)
        print(f"Submitted {k} to job {job_dict[k].job_id}")
