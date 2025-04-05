import torch
import torch.nn as nn


class Architecture(nn.Module):
    """
    Flexible Discriminator supporting multiple architectures for time series analysis.
    """

    def __init__(self, input_dim, output_dim, sigmoid=True, architecture="MLP", layer_sizes=None, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Architecture, self).__init__()

        self.architecture = architecture
        self.activation = activation
        self.sigmoid = sigmoid
        self.input_dim = input_dim

        # Default architecture if none is provided
        if layer_sizes is None:
            layer_sizes = [512, 256, 128, 64, output_dim]
        else:
            if layer_sizes[-1] != output_dim:
                raise Exception(f"La dernière couche doit avoir la dimension de sortie {output_dim}")

        self.build_model(layer_sizes)

    def build_model(self, layer_sizes):
        """Dynamically constructs the model based on the selected architecture."""
        if self.architecture == "MLP":
            self.model = self._build_mlp(self.input_dim, layer_sizes)
        elif self.architecture == "RNN":
            self.model = self._build_rnn(self.input_dim, layer_sizes[0], len(layer_sizes)-1)
        elif self.architecture == "LSTM":
            self.model = self._build_lstm(self.input_dim, layer_sizes[0], len(layer_sizes)-1)
        elif self.architecture == "GRU":
            self.model = self._build_gru(self.input_dim, layer_sizes[0], len(layer_sizes)-1)
        elif self.architecture == "CNN":
            self.model = self._build_cnn(self.input_dim, layer_sizes)
        elif self.architecture == "Transformer":
            self.model = self._build_transformer(self.input_dim, layer_sizes[0], 4, len(layer_sizes)-1)
        elif self.architecture == "TCN":
            self.model = self._build_tcn(self.input_dim, layer_sizes)
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

    def _build_mlp(self, input_dim, layer_sizes):
        layers = []
        prev_dim = input_dim
        for size in layer_sizes[:-1]:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(self.activation)
            prev_dim = size
        layers.append(nn.Linear(prev_dim, layer_sizes[-1]))
        if self.sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_rnn(input_dim, hidden_dim, num_layers):
        return nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, nonlinearity='relu')

    @staticmethod
    def _build_lstm(input_dim, hidden_dim, num_layers):
        return nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    @staticmethod
    def _build_gru(input_dim, hidden_dim, num_layers):
        return nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def _build_cnn(self, input_channels, layer_sizes):
        layers = []
        prev_channels = input_channels
        for out_channels in layer_sizes[:-1]:
            layers.append(nn.Conv1d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            prev_channels = out_channels
        layers.append(nn.Conv1d(prev_channels, layer_sizes[-1], kernel_size=3, stride=1, padding=1))
        if self.sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_transformer(input_dim, embed_dim, num_heads, num_layers):
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_tcn(self, input_dim, layer_sizes):
        layers = []
        prev_channels = input_dim
        dilation = 1
        for out_channels in layer_sizes[:-1]:
            layers.append(nn.Conv1d(prev_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            dilation *= 2
            prev_channels = out_channels
        layers.append(nn.Conv1d(prev_channels, layer_sizes[-1], kernel_size=3, padding=1))
        if self.sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def modify_architecture(self, architecture, layer_sizes, activation=None):
        """Modifies the architecture by rebuilding the model with a new structure."""
        self.architecture = architecture
        if activation is not None:
            self.activation = activation
        self.build_model(layer_sizes)

    def forward(self, x):
        if self.architecture in ["RNN", "LSTM", "GRU"]:
            _, hidden = self.model(x)
            if self.sigmoid:
                return torch.sigmoid(hidden[0] if isinstance(hidden, tuple) else hidden)
            else:
                return hidden[0] if isinstance(hidden, tuple) else hidden
        elif self.architecture == "Transformer":
            if self.sigmoid:
                return torch.sigmoid(self.model(x))
        return self.model(x)
