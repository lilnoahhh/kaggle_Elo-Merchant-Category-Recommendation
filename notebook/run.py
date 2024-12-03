import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from models.lightGBM import train_lightgbm, evaluate_rmse, predict
from models.LGBMRegressor import optimize_and_predict
from utils.split_func import split_data_for_lgbm

# データのパスを取得
data_path0 = "/Users/asapcats099/Downloads/elo-merchant-category-recommendation/data/input/train.csv"
data_path1 = "/Users/asapcats099/Downloads/elo-merchant-category-recommendation/data/input/test.csv"
data_path2 = "/Users/asapcats099/Downloads/elo-merchant-category-recommendation/data/input/historical_transactions.csv"
data_path3 = "/Users/asapcats099/Downloads/elo-merchant-category-recommendation/data/input/new_merchant_transactions.csv"
data_path4 = "/Users/asapcats099/Downloads/elo-merchant-category-recommendation/data/input/merchants.csv"

# データの読み込み
train = pd.read_csv(data_path0)
test = pd.read_csv(data_path1)
history = pd.read_csv(data_path2)
new_merchant = pd.read_csv(data_path3)
merchants = pd.read_csv(data_path4)

# カードIDごとに取引データを集約
history_agg = history.groupby('card_id').agg({
    'purchase_amount': ['sum', 'mean', 'min', 'max', 'count'],
    'month_lag': 'mean'
}).reset_index()

new_merchant_agg = new_merchant.groupby('card_id').agg({
    'purchase_amount': ['sum', 'mean', 'min', 'max', 'count'],
    'month_lag': 'mean'
}).reset_index()

# history_aggのカラムをフラット化
history_agg.columns = ['_'.join(filter(None, col)).strip() for col in history_agg.columns]
# new_merchant_aggのカラムをフラット化
new_merchant_agg.columns = ['_'.join(filter(None, col)).strip() for col in new_merchant_agg.columns]

# データの結合
train_merged = pd.merge(train, history_agg, on='card_id', how='left')
train_merged = pd.merge(train_merged, new_merchant_agg, on='card_id', how='left', suffixes=('_history', '_new'))

test_merged = pd.merge(test, history_agg, on='card_id', how='left')
test_merged = pd.merge(test_merged, new_merchant_agg, on='card_id', how='left', suffixes=('_history', '_new'))

# 'first_active_month'を日付型に変換し、各成分を抽出
train_merged['first_active_month'] = pd.to_datetime(train_merged['first_active_month'])
train_merged['year'] = train_merged['first_active_month'].dt.year
train_merged['month'] = train_merged['first_active_month'].dt.month
train_merged['day'] = train_merged['first_active_month'].dt.day
train_merged['dayofweek'] = train_merged['first_active_month'].dt.dayofweek

test_merged['first_active_month'] = pd.to_datetime(test_merged['first_active_month'])
test_merged['year'] = test_merged['first_active_month'].dt.year
test_merged['month'] = test_merged['first_active_month'].dt.month
test_merged['day'] = test_merged['first_active_month'].dt.day
test_merged['dayofweek'] = test_merged['first_active_month'].dt.dayofweek

# 'first_active_month'を取り除く
train_merged = train_merged.drop('first_active_month', axis=1)
test_merged = test_merged.drop('first_active_month', axis=1)

# 説明変数と目的変数に分割
X_train = train_merged.drop(['target', 'card_id'], axis=1)  # card_idを削除
y_train = train_merged['target']
X_test = test_merged.drop('card_id', axis=1)  # card_idを削除

# すべてのカラムを数値型に変換
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# モデルをトレーニング
_, y_pred = optimize_and_predict(X_train,y_train,X_test,n_trials=100)

# モデルがNoneでないことを確認
if y_pred is None:
    print("Error: Model training failed.")
else:
    # y_predが正しく定義されていることを確認してからsubmissionファイルを作成
    # submissionファイル作成
    submission = pd.DataFrame({
        'card_id': test['card_id'],  # テストデータのcard_id
        'target': y_pred  # 予測値
    })

    # card_idを文字列に変換
    submission['card_id'] = submission['card_id'].astype(str)

    # CSVとして保存 (ファイル名 'submission.csv' を指定)
    submission.to_csv('/Users/asapcats099/Downloads/elo-merchant-category-recommendation/data/output/submission.csv', index=False)

    print(f'Submission file saved as submission.csv')