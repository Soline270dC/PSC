from .WrapperGAN import *


class ModelTimeGAN(WrapperGAN):

    def __init__(self):
        super().__init__()

    def set_architecture(self):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")
        self.generator = Generator(self.parameters["latent_dim"], self.parameters["hidden_dim"])
        self.embedder = Embedder(self.output_dim, self.parameters["hidden_dim"])
        self.recovery = Recovery(self.parameters["hidden_dim"], self.output_dim)
        self.generator.apply(self.weights_init)
        self.embedder.apply(self.weights_init)
        self.recovery.apply(self.weights_init)

    def preprocess_data(self):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")

        sequences = []
        for i in range(0, len(self.data) - self.parameters["seq_length"], self.parameters["offset"]):
            sequences.append(self.data[i:i + self.parameters["seq_length"]])
        sequences = torch.tensor(np.array(sequences), dtype=torch.float32)

        train_size = int(len(sequences) * 0.8)
        train_data, val_data = random_split(sequences, [train_size, len(sequences) - train_size])

        self.train_loader = DataLoader(train_data, batch_size=self.parameters["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=self.parameters["batch_size"], shuffle=False)

        self.get_train_data()
        self.get_val_data()

    def set_data(self, data):
        self.data = data
        self.output_dim = self.data.shape[1]
        self.colors = sns.color_palette(self.color_style, self.data.shape[1])
        self.preprocess_data()

    def get_train_data(self):
        if self.train_loader is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        real_train_samples = []
        for real_tuples in self.train_loader:
            real_train_samples.extend([real_tuples[i] for i in range(len(real_tuples))])
        self.train_data = torch.cat(real_train_samples, dim=0).numpy()

    def get_val_data(self):
        if self.val_loader is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        real_train_samples = []
        for real_tuples in self.val_loader:
            real_train_samples.extend([real_tuples[i] for i in range(len(real_tuples))])
        self.val_data = torch.cat(real_train_samples, dim=0).numpy()

    def generate_samples(self, n_samples):
        if self.generator is None or self.recovery is None:
            raise Exception("Vous n'avez pas encore initialisé le modèle. Voir fit()")
        self.generator.eval()
        self.recovery.eval()

        n_sequences = n_samples // self.parameters["seq_length"]

        with torch.no_grad():
            # Generate the synthetic data
            generated_data = self.generator(torch.randn(n_sequences + 1, self.parameters["seq_length"], self.parameters["latent_dim"]))

            # Decode the generated data
            decoded_data = self.recovery(generated_data).view(-1, self.output_dim)

        return decoded_data.numpy()[:n_samples]

    @timeit
    def fit(self, params=None, architectures=None, verbose=False):
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
            self.plot_results(losses, gradients, metrics)
            self.evaluate_autoencoder()
            self.plot_series()
            self.plot_compare_series()
            self.plot_histograms()
        if isinstance(verbose, list):
            if "results" in verbose:
                self.plot_results(losses, gradients, metrics)
            if "autoencoder" in verbose:
                self.evaluate_autoencoder()
            if "trend_series" in verbose:
                self.plot_series()
            if "compare_series" in verbose:
                self.plot_compare_series()
            if "histograms" in verbose:
                self.plot_histograms()
        return {metric: (self.compute_train_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"]),
                         self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"]))
                for metric in self.metrics}

    def evaluate_autoencoder(self):
        loss_fn = nn.MSELoss()
        torch_val_data = torch.from_numpy(self.val_data)
        x_recon = self.recovery(self.embedder(torch_val_data))
        loss = loss_fn(torch_val_data, x_recon)

        print(f"Autoencoder Reconstruction Loss: {loss.item():.4f}")

        # Plot real vs reconstructed samples for visual inspection
        self.plot_real_vs_reconstructed(torch_val_data.detach().numpy(), x_recon.detach().numpy())

    @staticmethod
    def plot_real_vs_reconstructed(real, recon):
        real = np.concatenate(real, axis=0)
        recon = np.concatenate(recon, axis=0)

        # Ensure you're plotting the data correctly (assuming shape (time_steps, features))
        plt.figure(figsize=(12, 5))
        plt.plot(real, label="Real Data (1st Feature)", alpha=0.7)  # Plot the first feature
        plt.plot(recon, label="Reconstructed Data (1st Feature)", linestyle="dashed")  # Plot the first feature
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("Autoencoder: Real vs Reconstructed")
        plt.show()
