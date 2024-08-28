import submitit
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
print(sys.path)

from main import main as task_main
from hydra import compose, initialize


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
from omegaconf import OmegaConf
from q4l.qlib.model.ens.group import RollingGroup
from q4l.qlib.workflow import R

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
from q4l.utils.log import LoggingContext, get_logger
from q4l.utils.misc import generate_evaluations, make_qlib_init_config
from exp.utils import get_machine
from datetime import datetime


# job submission
ablation_overrides = {
    "xgboost": ["experiment/model=xgboost"],
    "catboost": ["experiment/model=catboost"],
    "lightgbm": ["experiment/model=lightgbm"],
}
common_overrides = [
    "experiment/model/loss=ic_loss",
    "++experiment.data.sampler.x_group=[ohlcvp,fundamental]",
]

executor = submitit.AutoExecutor(folder="./qbench_slurm_log/model_comparison")
job_partition = "cpu-2" if get_machine() == "hgx" else "batch"
executor.update_parameters(
    **{
        "slurm_mem": "400G",
        "slurm_time": "infinite",
        "cpus_per_task": 56,
        "slurm_partition": job_partition,
        "nodes": 1,
        "tasks_per_node": 1,
        "slurm_job_name": "tree_arch_comp",
    }
)

job_dict = {}
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

for k, v in ablation_overrides.items():
    with initialize(version_base=None, config_path="../../../config/"):
        model = k
        cfg = compose(
            config_name="qbench_base_config",
            overrides=v
            + common_overrides
            + [
                f"job/machine={get_machine()}",
                f"++job.name.exp_name=Exp|Model|Arch-Tree-{timestamp}",
                f"++job.name.run_name={model}",
                f"++job.resource.total_gpu=8",
                "++job.parallel.rolling=2",
                "++job.parallel.repeat=2",
            ],
        )
        job_dict[k] = executor.submit(task_main, cfg)
        print(f"Submitted {k} to job {job_dict[k].job_id}")

