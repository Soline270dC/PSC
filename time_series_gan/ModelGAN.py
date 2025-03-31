from .WrapperGAN import *


class ModelGAN(WrapperGAN):

    def __init__(self):
        super().__init__()
        self.parameters = {"lr_g": 1e-5, "lr_d": 1e-5, "epochs": 100, "batch_size": 32, "latent_dim": 100}

    def set_architecture(self, **kwargs):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")
        self.generator = Generator(self.parameters["latent_dim"], self.output_dim)
        self.generator.apply(self.weights_init)

    def set_data(self, data):
        self.data = data
        Y_tensor = torch.tensor(data.values)
        dataset = TensorDataset(Y_tensor)

        train_size = int((1 - 0.2) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.parameters["batch_size"], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.parameters["batch_size"], shuffle=False)

        self.output_dim = self.data.shape[1]
        self.colors = sns.color_palette(self.color_style, self.data.shape[1])

        self.get_train_data()
        self.get_val_data()

    def get_train_data(self):
        if self.train_loader is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        real_train_samples = []
        for real_tuples in self.train_loader:
            real_train_samples.append(real_tuples[0])
        self.train_data = torch.cat(real_train_samples, dim=0).numpy()

    def get_val_data(self):
        if self.val_loader is None:
            raise Exception("Vous n'avez pas initialisé de données. Voir set_data()")
        real_train_samples = []
        for real_tuples in self.val_loader:
            real_train_samples.append(real_tuples[0])
        self.val_data = torch.cat(real_train_samples, dim=0).numpy()

    def generate_samples(self, n_samples):
        if self.generator is None:
            raise Exception("Vous n'avez pas encore initialisé le modèle. Voir fit()")
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.parameters["latent_dim"])
            generated_data = self.generator(z)
        return generated_data.numpy()

    @timeit
    def fit(self, params=None, verbose=False):
        if self.data is None:
            raise Exception("Vous n'avez pas fourni de données. Voir set_data()")
        if params:
            self.set_parameters(params)
        self.set_architecture()
        losses, gradients, metrics = self.train(verbose=verbose)
        if verbose is True:
            self.plot_results(losses, gradients, metrics)
            self.plot_series()
            self.plot_compare_series()
            self.plot_histograms()
        if isinstance(verbose, list):
            if "results" in verbose:
                self.plot_results(losses, gradients, metrics)
            if "trend_series" in verbose:
                self.plot_series()
            if "compare_series" in verbose:
                self.plot_compare_series()
            if "histograms" in verbose:
                self.plot_histograms()
        return self.compute_train_wass_dist(), self.compute_val_wass_dist()
