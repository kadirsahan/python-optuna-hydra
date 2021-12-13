
import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

import optuna
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="config", config_name="optuna-xgb-hydra")
def my_app(cfg : DictConfig):    

    def objective(trial):

        (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
        train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        param = {
            "verbosity": cfg.xgb.verbosity,
            "objective": cfg.xgb.objective,
            # use exact for small dataset.
            "tree_method": cfg.xgb.tree_method,
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", cfg.xgb.booster),
            # L2 regularization weight.
            "lambda": trial.suggest_float(cfg["xgb"]["lambda"].name, cfg["xgb"]["lambda"].min, cfg["xgb"]["lambda"].max, log=cfg["xgb"]["lambda"].log),
            # L1 regularization weight.
            "alpha": trial.suggest_float(cfg["xgb"]["alpha"].name, cfg["xgb"]["alpha"].min, cfg["xgb"]["alpha"].max, log=cfg["xgb"]["alpha"].log),
            # sampling ratio for training data.
            "subsample": trial.suggest_float(cfg.xgb.subsample.name, cfg.xgb.subsample.min, cfg.xgb.subsample.max),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float(cfg.xgb.colsample_bytree.name, cfg.xgb.colsample_bytree.min, cfg.xgb.colsample_bytree.max),
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

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
        return accuracy

    # print(cfg)
    # print(cfg.keys()) # get keys of dictionaries.
    # print(cfg.optuna.keys()) 
    # for key in cfg.optuna.keys():
    #     print(key)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
# def objective(trial): # for each trial this funcion is called.
#     x = trial.suggest_float('x', -10, 10)
#     print("cfg-text")
#     return (x - 2) ** 2




wcfg = my_app() # get parameters from ocnfig file using hydra