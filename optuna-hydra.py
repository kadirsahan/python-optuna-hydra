import optuna
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="config", config_name="config-optuna")
def my_app(cfg : DictConfig):    

    def objective(trial): # for each trial this funcion is called. :) inner one is called :)
        x = trial.suggest_float('x', -10, 10)
        print(cfg)
        return (x - 2) ** 2

    print(cfg)
    print(cfg.keys()) # get keys of dictionaries.
    print(cfg.optuna.keys()) 
    for key in cfg.optuna.keys():
        print(key)
    
    study = optuna.create_study()
    
    study.optimize(objective, n_trials=cfg.optuna.trials)

    print(study.best_params)  # E.g. {'x': 2.002108042}
    
# def objective(trial): # for each trial this funcion is called.
#     x = trial.suggest_float('x', -10, 10)
#     print("cfg-text")
#     return (x - 2) ** 2




wcfg = my_app() # get parameters from ocnfig file using hydra