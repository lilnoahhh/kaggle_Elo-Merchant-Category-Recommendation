import optuna.integration.lightgbm as lgb
import numpy as np
from sklearn.metrics import root_mean_squared_error

# パラメータの設定
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',  # 回帰タスクとして設定
    'metric': 'rmse',  # 評価指標
    'verbose': 1,
    'learning_rate': 0.01,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'force_row_wise': True,
    'feature_pre_filter': False  # データセット作成前に指定
}


from sklearn.model_selection import KFold

def train_lightgbm(train_set,valid_set):
    
    # KFoldでクロスバリデーションを行う
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 交差検証を実行
    cv_results = lgb.cv(
        params,
        train_set,
        num_boost_round=1000,  # 最大学習ラウンド数
        folds=kf,  # KFoldクロスバリデーションを指定
        metrics='rmse',  # 評価指標
        seed=42  # 再現性のためのシード値
    )
    
    # cv_resultsの内容を確認
    print(cv_results.keys())  # 結果のキーを確認
    
    if 'valid rmse-mean' in cv_results:
        best_num_boost_round = len(cv_results['valid rmse-mean'])
        print(f'Best number of boosting rounds: {best_num_boost_round}')
    else:
        print("KeyError: 'valid rmse-mean' not found in cv_results. Available keys are:", cv_results.keys())
        return None
    
    # 最適なモデルをトレーニング
    try:
        model = lgb.train(
            params,
            train_set,
            valid_sets=[train_set, valid_set],
            num_boost_round=best_num_boost_round  # 最適なラウンド数を使用
        )
    except Exception as e:
        print(f"Model training failed with error: {e}")
        return None
    
    return model




# 予測関数
def predict(model, X_test):
    # 結果の予測
    y_pred = model.predict(X_test)
    return y_pred

# RMSEの評価関数
def evaluate_rmse(y_true, y_pred):
    # RMSEを計算
    rmse = np.sqrt(root_mean_squared_error(y_true, y_pred))
    return rmse
