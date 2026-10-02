"""Krishi AI yield-modelling package.

Modules
-------
config    paths and constants shared by training, serving and monitoring
data      loads and cleans the training data (base dataset + new harvest records)
models    candidate yield predictors (baselines and gradient-boosting models)
evaluate  rolling, time-based backtest used to compare candidates
pipeline  train / retrain: pick the champion, fit it, save it with its model card
drift     input-drift checks on logged prediction requests
"""
