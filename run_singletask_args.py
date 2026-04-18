# import multiprocess as mp
import functools
import pandas as pd
import ray
import ray.tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from train_vintage import train_model
import lightning.pytorch as pl
import argparse

parser = argparse.ArgumentParser(description="Run Ray Tune experiments for a specified range of vintage months.")
parser.add_argument("--start_month", type=str, required=True, help="Start month for vintage IDs (YYYY-MM-DD format).")
parser.add_argument("--end_month", type=str, required=True, help="End month for vintage IDs (YYYY-MM-DD format).")
args = parser.parse_args()

# device = 'cpu'
task = "singletask"  # "multitask" or "singletask"
# kmpair = {'PE': ['CR_G0','CR_B0']}
# kmpair = {'PE': ['VADERstanceweight_log_stl'], 'PU+': ['CR_lognorm']}
kmpair = {'PE':['VADERstanceweight_log_stl', 'VADERraw']}
with_econ = True
with_tweets = True
vintage_ids = list(pd.date_range(start=args.start_month, end=args.end_month, freq="ME"))
# vintage_ids = list(pd.date_range(start="2017-01-31", end="2023-01-01", freq="ME"))
# vintage_ids = list(pd.date_range(start="2020-01-31", end="2021-01-01", freq="ME"))
# with mp.Pool(processes=3) as pool: #.get_context('spawn')
#     results = pool.map(functools.partial(train_one_vintage, device=device), vintage_ids)
# print("\n".join(results))

pl.seed_everything(42, workers=True)
ray.init(log_to_driver=False, logging_level="ERROR") # runtime_env={"working_dir": "/home/btiu/Documents/Research/TweetsNowcast"}
for vintage_id in vintage_ids:
    tuner = ray.tune.Tuner(
        ray.tune.with_parameters(functools.partial(train_model, vintage=vintage_id, with_econ=with_econ, with_tweets=with_tweets, kmpair=kmpair, task=task)),
        param_space={
            # "learning_rate": ray.tune.grid_search([1e-1, 1e-2]),
            # "weight_decay": ray.tune.grid_search([1e-2, 1e-3]),
            # "num_layers": ray.tune.grid_search([1, 2]),
            # "data_window": ray.tune.grid_search([6, 12, 24, 48, 72]), # remove 6, 24, 48
            "epochs": ray.tune.choice([150]),
            "learning_rate": ray.tune.loguniform(1e-3, 1e-1),
            "weight_decay": ray.tune.loguniform(1e-4, 1e-2),
            "num_layers": ray.tune.choice([1, 2]),
            "data_window": ray.tune.choice([3, 6, 12, 24, 36, 48, 60, 72]),
        },
        tune_config=ray.tune.TuneConfig(
            metric="val_loss_y",
            mode="min",
            num_samples=50,
            search_alg=ConcurrencyLimiter(OptunaSearch(metric="val_loss_y",mode="min"), max_concurrent=8),
        ),
        run_config=ray.tune.RunConfig(
            name=f"{vintage_id.strftime('%Y-%m')}",
            storage_path=f"/home/btiu/Documents/Research/TweetsNowcast/ray_results/{task}_nobiasTEoptuna2",
            verbose=1,
            checkpoint_config=ray.tune.CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss_y", checkpoint_score_order="min"), # Slows down training
        ),
    )
    results = tuner.fit()