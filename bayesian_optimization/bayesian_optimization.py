import optuna

def fit_hyperparameters(self, param_ranges : dict[list], score, n_trials = 50, direction : str = "minimize") :
    assert direction in ["miminize","maximize"], "la direction d'optimisation ne peut être que 'minimize' ou 'maximize'"
    assert param_ranges.keys() == self.parameters.keys(), "param_ranges doit avoir les mêmes clés que parameters"
  
    def f(trial) :
        args = {}
        for param in param_ranges :
            assert isinstance(param_ranges[param], (list[int], list[float])) and len(param_ranges[param]) == 2, "param_ranges doit avoir pour valeurs des listes de taille 2 d'entiers ou de flottants"
            suggest = eval("trial.suggest_" + str(type(param_ranges[param][0])))
            args[param] = suggest(param, *param_ranges[param])
        self.fit(params = args)
        return self.compute_val_metric(score, [])

    study = optuna.create_study(direction = direction)
    study.optimize(f, n_trials = n_trials)
    self.parameters = study.best_params