# Tweets Nowcast

## Data

- `PH_Tweets_v2.csv` adds filtering based on sentiment score
- `PH_Tweets_v3.csv` adds filtering based on stance detected via Meta/BART-MNLI from HuggingFace

## Repo Subtrees

- Dynamic Factor Analysis from https://github.com/briangodwinlim/DynamicFactorAnalysis.git

  - to update, use

  ```shell
  git pull -s subtree dynamicfactoranalysis main
  ```
  - see https://docs.github.com/en/get-started/using-git/about-git-subtree-merges
