import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from ot.sliced import sliced_wasserstein_distance
from .architectures import *
from abc import ABC, abstractmethod


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time()  # Start timer
        result = func(*args, **kwargs)  # Execute function
        end_time = time()  # End timer
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result  # Return function result
    return wrapper


# TODO : look for other GAN models for time series that might be implemented easily using current framework
# TODO : check if there exists models optimising directly metrics measuring the time consistency of time series
# TODO : tests with synthetic data -> just noise, a straight line, straight line + noise... etc.
class WrapperGAN(ABC):

    def __init__(self):
        self.colors = None
        self.color_style = "dark"
        self.data = None
        self.train_data = None
        self.val_data = None
        self.train_loader = None
        self.val_loader = None
        self.output_dim = None
        self.generator = None
        self.parameters = {}
        self.metrics = {}

    def set_parameters(self, params):
        for param in params:
            if param not in self.parameters:
                raise Exception(f"L'hyperparamètre {param} n'est pas défini pour ce modèle")
            else:
                self.parameters[param] = params[param]

    def set_metrics(self, metrics):
        suppr = []
        remp = []
        if not isinstance(metrics, dict):
            raise Exception("Les métriques doivent être données comme dictionnaire : {'metric_name' : {'function': metric_function, 'metric_args': metric_args}}")
        for metric in metrics:
            if not isinstance(metrics[metric], dict) or not "function" in metrics[metric] or not "metric_args" in metrics[metric]:
                raise Exception("Les métriques doivent être données comme dictionnaire : {'metric_name' : {'function': metric_function, 'metric_args': metric_args}}")
            if not callable(metrics[metric]["function"]):
                raise Exception("Les métriques doivent être des fonctions")
            if metric in self.metrics:
                remp.append(metric)
        for metric in self.metrics:
            if metric not in metrics:
                suppr.append(metric)
        if len(suppr) > 0:
            print(f"Les métriques {", ".join(suppr)} ont été supprimées.")
        if len(remp) > 0:
            print(f"Les métriques {", ".join(remp)} ont été remplacées.")
        self.metrics = metrics

    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_val_data(self):
        pass

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            # Initialisation He pour les couches ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @abstractmethod
    def set_architecture(self):
        pass

    def modify_architecture(self, model, architecture, layer_sizes, activation=None):
        if not isinstance(model, Architecture):
            raise Exception("Vous ne pouvez pas modifier ce modèle")
        model.modify_architecture(architecture=architecture, layer_sizes=layer_sizes, activation=activation)

    def modify_generator(self, architecture, layer_sizes, activation=None):
        if self.generator is None:
            raise Exception("Vous n'avez pas encore initialisé le générateur")
        self.modify_architecture(self.generator, architecture, layer_sizes, activation)

    def modify_models(self, architectures):
        if not isinstance(architectures, dict):
            raise Exception("Vous pouvez modifier les architectures avec un dict de forme {'discriminator': {'architecture': 'MLP', 'layer_sizes': ...}, 'generator': {...}}")
        if "generator" in architectures:
            if self.generator is None:
                raise Exception("Vous n'avez pas encore initialisé le générateur")
            if not isinstance(architectures["generator"], dict) or "architecture" not in architectures["generator"] or "layer_sizes" not in architectures["generator"]:
                raise Exception(
                    "Vous pouvez modifier les architectures avec un dict de forme {'discriminator': {'architecture': 'MLP', 'layer_sizes': ...}, 'generator': {...}}")
            generator_output_dim = self.output_dim if "hidden_dim" not in self.parameters else self.parameters[
                "hidden_dim"]
            if len(architectures["generator"]["layer_sizes"]) == 0 or architectures["generator"]["layer_sizes"][-1] != generator_output_dim:
                raise Exception(f"La dernière couche doit avoir la dimension de sortie {generator_output_dim}")
            self.modify_generator(architectures["generator"]["architecture"], architectures["generator"]["layer_sizes"])

    @abstractmethod
    def train(self, verbose=False):
        pass

    @abstractmethod
    def fit(self, params=None, verbose=False):
        """
        Sets the parameters values if needed
        Sets the architectures and initialize weights
        Checks that there is proper data to use
        Train the model using train() method
        If verbose, records wasserstein distance, gradients norms and scores throughout training for plotting afterward
        Returns train and validation wasserstein distance after training
        """
        pass

    @abstractmethod
    def generate_samples(self, n_samples):
        pass

    def compute_metric(self, metric, metric_args, samples, n_samples):
        generated_data = self.generate_samples(n_samples)
        return metric(generated_data, samples, **metric_args)

    def compute_train_metric(self, metric, metric_args):
        return self.compute_metric(metric, metric_args, self.train_data, len(self.train_data))

    def compute_val_metric(self, metric, metric_args):
        return self.compute_metric(metric, metric_args, self.val_data, len(self.val_data))

    def compute_train_wass_dist(self):
        if self.train_data is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        parameters = {"n_projections": 1000}
        return self.compute_train_metric(sliced_wasserstein_distance, parameters)

    def compute_val_wass_dist(self):
        if self.val_data is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        parameters = {"n_projections": 1000}
        return self.compute_train_metric(sliced_wasserstein_distance, parameters)

    def plot_histograms(self):
        if self.data is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        generated_data = self.generate_samples(len(self.data))
        generated_df = pd.DataFrame(generated_data, columns=self.data.columns)

        # Créer un graphique distinct pour chaque YIELD
        for i, col in enumerate(self.data.columns):
            plt.figure(figsize=(10, 6))

            # Histogramme des données réelles
            plt.hist(self.data[col], bins='auto', alpha=0.8, color=self.colors[i], label=f"Real {col}", density=True)
            # Histogramme des données générées
            plt.hist(generated_df[col], bins='auto', alpha=0.5, color=self.colors[i], label=f"Generated {col}", density=True)

            plt.title(f'Real vs Generated Data Distribution for {col}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()

    def plot_series(self):
        if self.data is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        synthetic_yields = self.generate_samples(len(self.data))
        sy = pd.DataFrame(synthetic_yields, columns=self.data.columns)
        new = pd.concat([self.data, sy], axis=0)
        new.index = pd.RangeIndex(start=1, stop=2*len(self.data)+1, step=1)
        for i, col in enumerate(new.columns):
            plt.plot(new[col], color=self.colors[i], label=col)
        plt.axvline(x=len(self.data)+0.5, color="red", linestyle="dotted", linewidth=2)
        plt.title("Time Series Continuity : historical and forecasted")
        plt.xlabel("Value")
        plt.ylabel("Time")
        plt.legend()
        plt.show()

    def plot_compare_series(self):
        if self.data is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        synthetic_yields = self.generate_samples(len(self.data))
        sy = pd.DataFrame(synthetic_yields, columns=self.data.columns)
        for i, column in enumerate(self.data.columns):
            plt.figure()
            plt.plot(self.data.index, self.data[column], color=self.colors[i], label=column)
            plt.plot(self.data.index, sy[column], color=self.colors[i], label=f"Generated {column}", alpha=0.6)
            plt.title("Time Series Comparison : historical and forecasted")
            plt.xlabel("Value")
            plt.ylabel("Time")
            plt.legend()
        plt.show()

    @staticmethod
    def plot_results(losses, gradients, metrics):
        fig, axes = plt.subplots(2 + len(metrics), 1, figsize=(10, 10))
        for loss in losses:
            axes[0].plot(losses[loss], label=loss)
        axes[0].set_title('Loss Evolution')
        axes[0].legend()

        for gradient in gradients:
            axes[1].plot(gradients[gradient], label=gradient)
        axes[1].set_title('Gradient Norm Evolution')
        axes[1].legend()

        for i, metric in enumerate(metrics):
            axes[i+2].plot(metrics[metric], label=metric.capitalize())
            axes[i+2].set_title(f'{metric.capitalize()} Evolution')
            axes[i+2].legend()
            plt.show()
