import optuna
import torch.nn as nn

def bayesainOpt(f, list_args, type_args, range_args, other_args, n_trials) :
    def function(trial) :
        args = []
        for i in range(len(list_args)) :
            suggest = eval("trial.suggest_" + type_args[i])
            args.append(suggest(list_args[i], *range_args[i]))
        return f(*args, *other_args)
    
    study = optuna.create_study(direction='minimize', storage = "sqlite:///db.sqlite3", study_name = str(f))
    study.optimize(function, n_trials=n_trials)