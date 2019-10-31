# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
from util import Submission

if __name__ == '__main__':

    params_xgb = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'silent': 1,
        'random_state': 71,
        'num_round': 1000,
        'early_stopping_rounds': 20,
    }

    params_xgb_all = dict(params_xgb)
    params_xgb_all['num_round'] = 1000

    # 特徴量の指定
    features = []

    # xgboostによる学習・予測
    runner = Runner('xgb1', ModelXGB, features, params_xgb)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission('xgb1')
