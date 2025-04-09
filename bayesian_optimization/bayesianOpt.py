import optuna
import torch.nn as nn

def bayesainOpt(f, list_args : list[str], type_args : list[type], range_args : list[tuple[float]], other_args : list, n_trials : int, direction : str = "minimize") -> dict[str, any] :
    """
    input
    --------
    f : callable - function on which to apply bayesian optimization

    list_args : list[str] - list of names of the arguments to be tested

    type_args : list[type] - list of types of the arguments (type must be int or float)

    range_args : list[tuple[float]] - list of tuple (one tuple containing both bounds for each argument)

    other_args : list - list of all other arguments to be given to f but not to be optimized upon

    n_trials : int - number of trials to be done by bayesian optimization

    direction : "minimize" (default value) or "maximize" - direction to which optimize

    output
    -------
    best_params : dict[str, Any] - best parameters encountered during optimization
    """
    assert direction in ["miminize","maximize"], "the direction can only be 'minimize' or 'maximize'"
    def function(trial) :
        args = []
        for i in range(len(list_args)) :
            suggest = eval("trial.suggest_" + type_args[i])
            args.append(suggest(list_args[i], *range_args[i]))
        return f(*args, *other_args)
    
    study = optuna.create_study(direction = direction, storage = "sqlite:///db.sqlite3", study_name = str(f))
    study.optimize(function, n_trials = n_trials)
    return study.best_params