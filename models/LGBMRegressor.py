from gc import collect
from lightgbm import LGBMRegressor
import optuna
from optuna import Trial, trial, create_study
from IPython.display import clear_output
from sklearn.model_selection import cross_val_score

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
}

def optimize_and_predict(X_train, y_train, X_test, n_trials=100):
    """
    X_train: 訓練用の特徴量
    y_train: 訓練用の目的変数
    X_test: テストデータ
    n_trials: Optunaのトライアルの回数
    """
    
    # 目的関数の定義
    def objective(trial):
        clf = LGBMRegressor(
            boosting_type=params['boosting_type'],
            objective=params['objective'],
            metric=params['metric'],
            verbose=params['verbose'],
            max_depth=trial.suggest_int('max_depth', 2, 32),
            subsample=trial.suggest_float('subsample', 0, 1),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0, 1),
            num_leaves=trial.suggest_int('num_leaves', 50, 150),
            reg_alpha=trial.suggest_float('reg_alpha', 0, 1),
            reg_lambda=trial.suggest_float('reg_lambda', 0, 1),
            learning_rate=trial.suggest_float('learning_rate', 0.001, 0.1),
            feature_fraction=trial.suggest_float('feature_fraction', 0.5, 1.0)
        )
        score = cross_val_score(clf, X_train, y_train, cv=3)
        accuracy = score.mean()
        return accuracy

    # Optuna studyの作成と最適化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    clear_output()

    # 最適なパラメータでモデルを再訓練し、予測を行う
    best_params = study.best_params
    clf = LGBMRegressor(
        boosting_type=params['boosting_type'],
        objective=params['objective'],
        metric=params['metric'],
        verbose=params['verbose'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        num_leaves=best_params['num_leaves'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        learning_rate=best_params['learning_rate'],
        feature_fraction=best_params['feature_fraction']
    )

    # モデルを訓練
    clf.fit(X_train, y_train)

    # テストデータに対して予測
    y_pred = clf.predict(X_test)

    return study, y_pred
