import copy
import functools
import importlib
import pandas as pd
import ray
import ray.tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from train_vintage import train_model
import lightning.pytorch as pl
import argparse

parser = argparse.ArgumentParser(description="Run Ray Tune experiments for a specified range of vintage months.")
parser.add_argument("--config", type=str, default=None, help="Python module path to a trial config, e.g. configs.st_nobiasTE_v1. If omitted, uses the legacy hardcoded sweep.")
parser.add_argument("--start_month", type=str, default=None, help="Override config's DEFAULT_START_MONTH (YYYY-MM-DD).")
parser.add_argument("--end_month", type=str, default=None, help="Override config's DEFAULT_END_MONTH (YYYY-MM-DD).")
parser.add_argument("--train_bias", action="store_true", help="Override: include last 3 rows in training data. Default follows config (usually nobias).")
parser.add_argument("--walk_n", type=int, default=None, help="Override forward-walk N (1 or 2). Default follows config.WALK_N.")
parser.add_argument("--run_tag", type=str, default=None, help="Tag appended to the storage path. Defaults to config.NAME when --config is used.")
args = parser.parse_args()

task = "singletask"

if args.config is not None:
    cfg = importlib.import_module(args.config)
    assert cfg.TASK == task, f"Config {args.config} has TASK={cfg.TASK!r}, expected {task!r}"
    kmpair = cfg.KMPAIR
    with_econ = cfg.WITH_ECON
    with_tweets = cfg.WITH_TWEETS
    train_bias = args.train_bias or cfg.TRAIN_BIAS
    walk_n = args.walk_n if args.walk_n is not None else getattr(cfg, "WALK_N", 2)
    start_month = args.start_month or cfg.DEFAULT_START_MONTH
    end_month = args.end_month or cfg.DEFAULT_END_MONTH
    run_tag = args.run_tag or cfg.NAME
    param_space = cfg.PARAM_SPACE
    metric = cfg.METRIC
    num_samples = cfg.NUM_SAMPLES
    max_concurrent = cfg.MAX_CONCURRENT
    scheduler = getattr(cfg, "SCHEDULER", None)
else:
    # Legacy hardcoded sweep (preserved for backwards compat; prefer --config).
    assert args.start_month and args.end_month, "Either --config or both --start_month and --end_month are required."
    kmpair = {'PE': ['VADERstanceweight_log_stl', 'VADERraw']}
    with_econ = True
    with_tweets = True
    train_bias = args.train_bias
    walk_n = args.walk_n if args.walk_n is not None else 2
    start_month = args.start_month
    end_month = args.end_month
    run_tag = args.run_tag or "optuna2"
    param_space = {
        "epochs": ray.tune.choice([150]),
        "learning_rate": ray.tune.loguniform(1e-3, 1e-1),
        "weight_decay": ray.tune.loguniform(1e-4, 1e-2),
        "num_layers": ray.tune.choice([1, 2]),
        "data_window": ray.tune.choice([3, 6, 12, 24, 36, 48, 60, 72]),
    }
    metric = "val_loss_y"
    num_samples = 50
    max_concurrent = 8
    scheduler = None

bias_tag = "bias" if train_bias else "nobias"
tweets_tag = "TE" if with_tweets else "E"
vintage_ids = list(pd.date_range(start=start_month, end=end_month, freq="ME"))
storage_root = f"/home/btiu/Documents/Research/TweetsNowcast/ray_results/{task}_{bias_tag}{tweets_tag}{run_tag}"

import os as _os
def _vintage_complete(vstr: str) -> bool:
    """A vintage is treated as complete if its dir has >= num_samples tune trial subdirs."""
    vpath = f"{storage_root}/{vstr}"
    if not _os.path.isdir(vpath):
        return False
    trial_dirs = [d for d in _os.listdir(vpath) if d.startswith("tune_with_parameters_")]
    return len(trial_dirs) >= num_samples

pl.seed_everything(42, workers=True)
ray.init(log_to_driver=False, logging_level="ERROR")
for vintage_id in vintage_ids:
    vstr = vintage_id.strftime("%Y-%m")
    if _vintage_complete(vstr):
        print(f"[skip] {vstr} already has {num_samples} trials")
        continue
    tune_config_kwargs = dict(
        metric=metric,
        mode="min",
        num_samples=num_samples,
        search_alg=ConcurrencyLimiter(OptunaSearch(metric=metric, mode="min"), max_concurrent=max_concurrent),
    )
    if scheduler is not None:
        # Deepcopy so each vintage gets a fresh scheduler — Ray mutates metric/mode
        # onto it during fit(), which would make subsequent vintages fail.
        tune_config_kwargs["scheduler"] = copy.deepcopy(scheduler)
    tuner = ray.tune.Tuner(
        ray.tune.with_parameters(functools.partial(train_model, vintage=vintage_id, with_econ=with_econ, with_tweets=with_tweets, kmpair=kmpair, task=task, train_bias=train_bias, walk_n=walk_n)),
        param_space=param_space,
        tune_config=ray.tune.TuneConfig(**tune_config_kwargs),
        run_config=ray.tune.RunConfig(
            name=f"{vintage_id.strftime('%Y-%m')}",
            storage_path=f"/home/btiu/Documents/Research/TweetsNowcast/ray_results/{task}_{bias_tag}{tweets_tag}{run_tag}",
            verbose=1,
            checkpoint_config=ray.tune.CheckpointConfig(num_to_keep=1, checkpoint_score_attribute=metric, checkpoint_score_order="min"),
        ),
    )
    results = tuner.fit()
