import optuna
import torch.nn as nn
from fElliot import test_architecture, Architecture


def f_Elliot(trial) :
    latent_dim = trial.suggest_int('latent_dim', 10, 50)
    num_epochs = trial.suggest_int('num_epochs', 30, 200)
    batch_size = trial.suggest_int('batch_size', 10, 50)
    learning_rate = trial.suggest_float('leaning_rate', 1e-5, 1e-4)
    archi = Architecture(0.0008, [28], [45, 49, 43, 19, 36, 36, 10, 10, 13, 21], [nn.Sigmoid()], [nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.ReLU()])
    return test_architecture(batch_size, latent_dim, num_epochs, learning_rate, archi, 5)

study = optuna.create_study(direction='minimize', storage = "sqlite:///db.sqlite3", study_name = "f_Elliot_2")
study.optimize(f_Elliot, n_trials=75)