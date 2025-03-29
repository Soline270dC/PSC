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


class Generator(nn.Module):
    """
        Class implementing a generator used in all GAN models
        Input: random vector of noise from latent space of size latent_dim
        Output: new data, hopefully resembling those used for training :)
    """

    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
        Class implementing a discriminator used in GAN and TimeGAN. Note this class should not be used in any kind of
        Wasserstein GAN, see Critic instead.
        Input: vector of data, real or generated
        Output: 0-1 value describing whether the input is legit data (ie. not generated by the generator)
    """

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 1),
            # Warning : must end with an activation function, usually Sigmoid works well
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    """
        Class implementing a critic used in WGAN and TimeWGAN. Note this class should not be used in non-Wasserstein
        GANs, see Discriminator instead.
        Input: vector of data, real or generated
        Output: [0;1] value describing the likeliness that the input is legit data (ie. not generated by the generator)
    """

    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 1)
            # Warning : cannot end with an activation function !!
        )

    def forward(self, x):
        return self.model(x)


class Embedder(nn.Module):
    """
        Class implementing an embedder model used in TimeGAN and TimeWGAN. Maps input sequences into a hidden space
        usually of higher dimension to help the model capture more complexity. Must be used with Recovery to
        switch between working space and hidden space.
        Input: vector of data (ie. a sequence of time points)
        Output: vector in hidden space
    """

    def __init__(self, input_dim, hidden_dim):
        super(Embedder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, x):
        return self.model(x)


class Recovery(nn.Module):
    """
        Class implementing a recovery model used in TimeGAN and TimeWGAN. Maps input sequences from a hidden space
        back to the working space. Must be used with Embedder to switch between working space and hidden space.
        Input: vector in hidden space
        Output: vector in data space (ie. working space)
    """

    def __init__(self, hidden_dim, output_dim):
        super(Recovery, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, h):
        return self.model(h)
