import torch.nn as nn


class Generator(nn.Module):

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

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):

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
        )

    def forward(self, x):
        return self.model(x)


class Embedder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Embedder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.model(x)


class Recovery(nn.Module):

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
