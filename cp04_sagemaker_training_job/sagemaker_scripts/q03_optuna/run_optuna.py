import argparse
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
import json
import os

def main(opt):
    X_all, y_all = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=123)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    def objective(trial):
        param = {
            "silent": 0,
            "verbosity": 0,
            "objective": "reg:linear",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "eval_metrics": "rmse"
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        model = xgb.train(params=param,
                        dtrain=dtrain,
                        num_boost_round=1000,
                        early_stopping_rounds=10,
                        evals=[(dtest, "test")])
        return model.best_score

    os.makedirs(os.path.dirname(opt.storage), exist_ok=True)
    study = optuna.create_study(storage=f"sqlite:///{opt.storage}", load_if_exists=True, study_name="california_housing_xgboost")
    study.optimize(objective, n_trials=200)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))    

    result = {
        "value": trial.value,
        "params": trial.params
    }

    os.makedirs(opt.output_dir, exist_ok=True)
    with open(f"{opt.output_dir}/optuna_result.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna xgboost")
    parser.add_argument("--storage", type=str, default="./study_log/optuna.db")
    parser.add_argument("--output_dir", type=str, default="./result")

    opt = parser.parse_args()

    main(opt)