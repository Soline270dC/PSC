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
        start_time = time.time()  # Start timer
        result = func(*args, **kwargs)  # Execute function
        end_time = time.time()  # End timer
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result  # Return function result
    return wrapper


# TODO : look for other GAN models for time series that might be implemented easily using current framework
# TODO : introduce use of other evaluation metrics, check if there exists models optimising directly metrics measuring the time consistency of time series
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
            raise Exception("Les métriques doivent être données comme dictionnaire : {'metrics_name' : metrics_function}")
        for metric in metrics:
            if not callable(metrics[metric]):
                raise Exception("Les métriques doivent être des fonctions")
            if metric in self.metrics:
                remp.append(metric)
        for metric in self.metrics:
            if metric not in metrics:
                suppr.append(metric)
        print(f"Les métriques {", ".join(suppr)} ont été supprimées.")
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

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # Initialisation He pour les couches ReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @abstractmethod
    def set_architecture(self):
        pass

    @abstractmethod
    def train(self, verbose=False):
        pass

    @abstractmethod
    @timeit
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

    def compute_wass_dist(self, real_samples, n_samples):
        # Generate synthetic data
        synthetic_yields = self.generate_samples(n_samples)

        # Ensure the shape matches for sliced Wasserstein calculation
        real_samples = real_samples.numpy()
        swd = sliced_wasserstein_distance(synthetic_yields, real_samples, n_projections=1000)

        return swd

    def compute_train_wass_dist(self):
        # Get the real samples from the training data
        if self.train_loader is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        real_train_samples = []
        for real_tuples in self.train_loader:
            real_train_samples.append(real_tuples[0])
        real_train_samples = torch.cat(real_train_samples, dim=0)

        # Compute Wasserstein distance for training data
        return self.compute_wass_dist(real_train_samples, len(real_train_samples))

    def compute_val_wass_dist(self):
        # Récupérer les données réelles de validation
        if self.train_loader is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        real_val_samples = []
        for real_tuples in self.val_loader:
            real_val_samples.append(real_tuples[0])
        real_val_samples = torch.cat(real_val_samples, dim=0)

        # Calculer la distance de Wasserstein
        return self.compute_wass_dist(real_val_samples, len(real_val_samples))

    def compute_train_metric(self, metric):
        pass

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
            plt.plot(self.data.index, self.data[column], color=self.colors[i], label=column)
            plt.plot(self.data.index, sy[column], color=self.colors[i], label=f"Generated {column}", alpha=0.6)
        plt.title("Time Series Comparison : historical and forecasted")
        plt.xlabel("Value")
        plt.ylabel("Time")
        plt.legend()
        plt.show()

    def plot_results(self, losses, gradients, wass_dists):
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        for loss in losses:
            axes[0].plot(losses[loss], label=loss)
        axes[0].set_title('Loss Evolution')
        axes[0].legend()

        for gradient in gradients:
            axes[1].plot(gradients[gradient], label=gradient)
        axes[1].set_title('Gradient Norm Evolution')
        axes[1].legend()

        axes[2].plot(wass_dists, label='Wasserstein Distance')
        axes[2].set_title('Wasserstein Distance Evolution')
        axes[2].legend()
        plt.show()

