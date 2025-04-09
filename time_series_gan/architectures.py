"""
    Author: Siméon Gheorghin
    Last modified: 30/03/2025
    This file implements neural networks for the main components of GAN (Generative Adversarial Networks) models.
    They are used in GAN, WGAN, TimeGAN and TimeWGAN classes implemented in this package.
    It should be noted that these architectures are very basic and have not been optimised for any task.
    The difference between Critic and Discriminator is the use of Sigmoid before output in Discriminator : Critic is
    used in WGAN and Discriminator in GAN.
"""

import torch.nn as nn
from .WrapperArchi import Architecture


class Generator(Architecture):
    """
        Class implementing a generator used in all GAN models
        Input: random vector of noise from latent space of size latent_dim
        Output: new data, hopefully resembling those used for training :)
    """

    def __init__(self, latent_dim, output_dim):
        super().__init__(latent_dim, output_dim, sigmoid=False, architecture="MLP", layer_sizes=[512, 256, 128, 64, output_dim])


class Discriminator(Architecture):
    """
        Class implementing a discriminator used in GAN and TimeGAN. Note this class should not be used in any kind of
        Wasserstein GAN, see Critic instead.
        Input: vector of data, real or generated
        Output: 0-1 value describing whether the input is legit data (ie. not generated by the generator)
    """

    def __init__(self, input_dim):
        super().__init__(input_dim, 1, sigmoid=True, architecture="MLP")


class Critic(Architecture):
    """
        Class implementing a critic used in WGAN and TimeWGAN. Note this class should not be used in non-Wasserstein
        GANs, see Discriminator instead.
        Input: vector of data, real or generated
        Output: [0;1] value describing the likeliness that the input is legit data (ie. not generated by the generator)
    """

    def __init__(self, input_dim):
        super().__init__(input_dim, 1, sigmoid=False, architecture="MLP")


class Embedder(nn.Module):
    """
        Class implementing an embedder model used in TimeGAN and TimeWGAN. Maps input sequences into a hidden space
        usually of higher dimension to help the model capture more complexity. Must be used with Recovery to
        switch between working space and hidden space.
        Input: vector of data (ie. a sequence of time points)
        Output: vector in hidden space
    """

    def __init__(self, input_dim, seq_length, hidden_dim):
        super(Embedder, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim*seq_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim*self.seq_length)
        return self.model(x)


class Recovery(nn.Module):
    """
        Class implementing a recovery model used in TimeGAN and TimeWGAN. Maps input sequences from a hidden space
        back to the working space. Must be used with Embedder to switch between working space and hidden space.
        Input: vector in hidden space
        Output: vector in data space (ie. working space)
    """

    def __init__(self, hidden_dim, seq_length, output_dim):
        super(Recovery, self).__init__()
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, seq_length*output_dim)
        )

    def forward(self, h):
        ret = self.model(h).reshape(-1, self.seq_length, self.output_dim)
        return ret
