# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# train_xは学習データ、train_yは目的変数、test_xはテストデータ
targets = ['arousal_mean','valence_mean']
train = pd.read_csv('../input/train.csv')
train_x = train.drop(targets, axis=1)

for target in targets:
    train_y = train[target]
    test_x = pd.read_csv('../input/test.csv').drop(targets, axis=1)

    # 学習データを学習データとバリデーションデータに分ける
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=71)
    tr_idx, va_idx = list(kf.split(train_x))[0]
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 特徴量と目的変数をxgboostのデータ構造に変換する
    dtrain = xgb.DMatrix(tr_x, label=tr_y)
    dvalid = xgb.DMatrix(va_x, label=va_y)
    dtest = xgb.DMatrix(test_x)

    # モニタリングをrmseで行い、アーリーストッピングの観察するroundを20とする
    params = {
        'objective': 'reg:squarederror',
        'silent': 1,
        'random_state': 71,
        'eval_metric': 'rmse'
        }
    num_round = 1000
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    model = xgb.train(params, dtrain, num_round, evals=watchlist,
                      early_stopping_rounds=20)

    pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    y = pd.read_csv('../input/test.csv')[target]

    #bench markに基づき評価関数はCCCとMAE
    #Developing a benchmark for emotional analysis of music Anna
    ccc = 2 * np.cov(y,pred)[0,1]/(y.mean()+pred.mean()+np.var(y)+np.var(pred))
    score = np.sqrt(mean_squared_error(y, pred))

    print(f'{target}')
    print(f'concordance correlation coefficient: {ccc:.4f}')
    print(f'root mean squared error: {score:.4f}')
