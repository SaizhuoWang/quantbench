import submitit
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
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



# job submission
ablation_overrides = {
    "small": ["++experiment.data.pool=sp600"],
    "mid": ["++experiment.data.pool=sp400"],
    "large": ["++experiment.data.pool=sp500"],
}

executor = submitit.AutoExecutor(folder="./qbench_slurm_log/univ_comparison")
job_partition = "3090" if get_machine() == "hgx" else "batch"
executor.update_parameters(
    **{
        "slurm_mem": "1800G",
        "slurm_time": "infinite",
        "cpus_per_task": 240,
        "slurm_gres": "gpu:8",
        "slurm_partition": job_partition,
        "nodes": 1,
        "tasks_per_node": 1,
        "slurm_job_name": "universe_comparison",
    }
)

job_dict = {}

for k, v in ablation_overrides.items():
    with initialize(version_base=None, config_path="../../../config/"):
        model = "rgcn"
        cfg = compose(
            config_name="prod",
            overrides=v
            + [
                f"experiment/model={model}",
                f"job/machine={get_machine()}",
                "job/parallel=benchmark",
                "++job.name.exp_name=qbench_nips24_prod_universe_comparison_v4",
                f"++job.resource.total_gpu=8",
                "++experiment.time.rolling_step=10000",  # no rolling
                "++experiment.data.graph.use_wikidata=true",
            ],
        )
        job_dict[k] = executor.submit(task_main, cfg)
        print(f"Submitted {k} to job {job_dict[k].job_id}")

for k, v in job_dict.items():
    print(f"Waiting for {k}")
    v.result()
