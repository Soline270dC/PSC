from .WrapperGAN import *
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class ModelTimeGAN(WrapperGAN):

    def __init__(self):
        super().__init__()

    def set_architecture(self):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")
        self.generator = Generator(self.parameters["latent_dim"], self.parameters["hidden_dim"])
        self.embedder = Embedder(self.output_dim, self.parameters["seq_length"], self.parameters["hidden_dim"])
        self.recovery = Recovery(self.parameters["hidden_dim"], self.parameters["seq_length"], self.output_dim)
        self.generator.apply(self.weights_init)
        self.embedder.apply(self.weights_init)
        self.recovery.apply(self.weights_init)

    def preprocess_data(self):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")

        sequences = []
        for i in range(0, len(self.data) - self.parameters["seq_length"]):
            sequences.append(np.array(self.data[i:i + self.parameters["seq_length"]]))
        sequences = np.array(sequences)
        train_data, val_data = train_test_split(sequences, test_size=0.2, shuffle=True, random_state=42)

        self.train_data = train_data.reshape(-1, self.output_dim)
        self.val_data = val_data.reshape(-1, self.output_dim)

        train_data = torch.tensor(train_data, dtype=torch.float32)
        val_data = torch.tensor(val_data, dtype=torch.float32)

        self.train_loader = DataLoader(TensorDataset(train_data), batch_size=self.parameters["batch_size"], shuffle=True, drop_last=True)
        self.val_loader = DataLoader(TensorDataset(val_data), batch_size=self.parameters["batch_size"], shuffle=False, drop_last=True)

    def set_data(self, data):
        self.data = data
        self.output_dim = self.data.shape[1]
        self.colors = sns.color_palette(self.color_style, self.data.shape[1])
        self.preprocess_data()

    def generate_samples(self, n_samples):
        if self.generator is None or self.recovery is None:
            raise Exception("Vous n'avez pas encore initialisé le modèle. Voir fit()")
        self.generator.eval()
        self.recovery.eval()

        n_sequences = n_samples // self.parameters["seq_length"]

        with torch.no_grad():
            # Generate the synthetic data
            generated_data = self.generator(torch.randn(n_sequences + 1, self.parameters["latent_dim"]))

            # Decode the generated data
            decoded_data = self.recovery(generated_data)

        return decoded_data.numpy().reshape(-1, self.output_dim)[:n_samples]

    @timeit
    def fit(self, params=None, architectures=None, verbose=False, save=False):
        if self.data is None:
            raise Exception("Vous n'avez pas fourni de données. Voir set_data()")
        if params:
            self.set_parameters(params)
            if "seq_length" in params:
                self.preprocess_data()
        self.set_architecture()
        if architectures is not None:
            self.modify_models(architectures)
        losses, gradients, metrics = self.train(verbose=verbose)
        if verbose is True:
            self.plot_results(losses, gradients, metrics, save=save)
            self.evaluate_autoencoder(save=save)
            self.plot_series(save=save)
            self.plot_compare_series(save=save)
            self.plot_histograms(save=save)
        if isinstance(verbose, list):
            if "results" in verbose:
                self.plot_results(losses, gradients, metrics, save=save)
            if "autoencoder" in verbose:
                self.evaluate_autoencoder(save=save)
            if "trend_series" in verbose:
                self.plot_series(save=save)
            if "compare_series" in verbose:
                self.plot_compare_series(save=save)
            if "histograms" in verbose:
                self.plot_histograms(save=save)
        return {metric: (self.compute_train_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"]),
                         self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"]))
                for metric in self.metrics}

    def evaluate_autoencoder(self, save=False):
        loss_fn = nn.MSELoss()
        torch_val_data = torch.from_numpy(self.val_data).float()
        x_recon = self.recovery(self.embedder(torch_val_data)).reshape(-1, self.output_dim)
        loss = loss_fn(torch_val_data, x_recon)

        print(f"Autoencoder Reconstruction Loss: {loss.item():.4f}")

        # Plot real vs reconstructed samples for visual inspection
        self.plot_real_vs_reconstructed(torch_val_data.detach().numpy(), x_recon.detach().numpy(), save=save)

    @staticmethod
    def plot_real_vs_reconstructed(real, recon, save=False):
        real = np.concatenate(real, axis=0)[:min(len(real), 100)]
        recon = np.concatenate(recon, axis=0)[:min(len(recon), 100)]

        # Ensure you're plotting the data correctly (assuming shape (time_steps, features))
        plt.figure(figsize=(12, 5))
        plt.plot(real, label="Real Data", alpha=0.7)  # Plot the first feature
        plt.plot(recon, label="Reconstructed Data", linestyle="dashed")  # Plot the first feature
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("Autoencoder: Real vs Reconstructed")
        if save:
            plt.savefig("Autoencoder_performance.png")
        plt.show()
