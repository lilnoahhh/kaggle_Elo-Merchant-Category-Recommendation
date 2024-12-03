from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb

def split_data_for_lgbm(X, y, test_size=0.2, random_state=42):
    """
    データをトレーニングセットと検証セットに分割し、LightGBM用のデータセットに変換する関数。

    Parameters:
    X (pd.DataFrame or np.array): 特徴量データ
    y (pd.Series or np.array): 目的変数（ターゲット）
    test_size (float): 検証データの割合
    random_state (int): ランダムシード

    Returns:
    train_set (lgb.Dataset): トレーニング用のLightGBMデータセット
    valid_set (lgb.Dataset): 検証用のLightGBMデータセット
    """
    # データをトレーニングセットと検証セットに分割
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # LightGBMのDataset形式に変換
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, free_raw_data=False)
    
    return train_set, valid_set,X_valid,y_valid
