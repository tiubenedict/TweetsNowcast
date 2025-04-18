import multiprocess as mp
import functools
from train_vintage import train_one_vintage
import pandas as pd

device = 'cpu'
vintage_ids = list(pd.date_range(start="2017-01-31", end="2023-01-01", freq="ME"))
with mp.Pool(processes=3) as pool: #.get_context('spawn')
    results = pool.map(functools.partial(train_one_vintage, device=device), vintage_ids)
print("\n".join(results))