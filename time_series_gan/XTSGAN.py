from .WrapperGAN import *
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class XTSGAN(WrapperGAN):

    def __init__(self):
        super().__init__()
        self.parameters = {"lr_g": 1e-5, "lr_c": 1e-5, "epochs": 100, "batch_size": 32,
                           "latent_dim": 100, "seq_length": 10, "n_critic": 2, "lambda_gp": 0.1, "variance_coeff": 0}

    def set_architecture(self):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")
        self.generator = Generator(self.parameters["latent_dim"], self.parameters["seq_length"]*self.output_dim)
        self.critic = Critic(self.output_dim*self.parameters["seq_length"])
        self.generator.apply(self.weights_init)
        self.critic.apply(self.weights_init)

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
        if self.generator is None:
            raise Exception("Vous n'avez pas encore initialisé le modèle. Voir fit()")
        self.generator.eval()

        n_sequences = n_samples // self.parameters["seq_length"]

        with torch.no_grad():
            # Generate the synthetic data
            generated_data = self.generator(torch.randn(n_sequences + 1, self.parameters["latent_dim"]))

        return generated_data.numpy().reshape(-1, self.output_dim)[:n_samples]

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
            self.plot_series(save=save)
            self.plot_compare_series(save=save)
            self.plot_histograms(save=save)
        if isinstance(verbose, list):
            if "results" in verbose:
                self.plot_results(losses, gradients, metrics, save=save)
            if "trend_series" in verbose:
                self.plot_series(save=save)
            if "compare_series" in verbose:
                self.plot_compare_series(save=save)
            if "histograms" in verbose:
                self.plot_histograms(save=save)
        return {metric: (self.compute_train_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"]),
                         self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"]))
                for metric in self.metrics}

    def modify_critic(self, architecture, layer_sizes, activation=None):
        if self.critic is None:
            raise Exception("Vous n'avez pas encore initialisé le critique")
        self.modify_architecture(self.critic, architecture, layer_sizes, activation)

    def modify_models(self, architectures):
        super().modify_models(architectures)
        if "critic" in architectures:
            if self.critic is None:
                raise Exception("Vous n'avez pas encore initialisé le critique")
            if not isinstance(architectures["critic"], dict) or "architecture" not in architectures["critic"] or "layer_sizes" not in architectures["critic"]:
                raise Exception(
                    "Vous pouvez modifier les architectures avec un dict de forme {'critic': {'architecture': 'MLP', 'layer_sizes': ...}, 'generator': {...}}")
            self.modify_critic(architectures["critic"]["architecture"], architectures["critic"]["layer_sizes"])

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1).expand_as(real_samples).float()
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        critic_interpolates = self.critic(interpolates)
        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def train(self, verbose=False):
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.parameters["lr_g"], betas=(0.5, 0.999))
        optimizer_C = optim.Adam(self.critic.parameters(), lr=self.parameters["lr_c"], betas=(0.5, 0.999))
        loss_fn = nn.MSELoss()

        metrics = {metric: [] for metric in self.metrics}
        loss_g_list = []
        loss_c_list = []
        grad_g_list = [1.0]
        grad_c_list = []

        for epoch in range(self.parameters["epochs"]):
            for i, (batch, ) in enumerate(self.train_loader):
                batch_size = batch.shape[0]
                real_tuples = batch.float()

                for _ in range(self.parameters["n_critic"]):
                    # Training Critic
                    z = torch.randn(batch_size, self.parameters["latent_dim"]).float()
                    h_fake = self.generator(z)
                    h_real = real_tuples.reshape(batch_size, -1)
                    real_score = self.critic(h_real.detach())
                    fake_score = self.critic(h_fake.detach())
                    gp = self.gradient_penalty(h_real.detach(), h_fake.detach())
                    loss_c = -torch.mean(real_score) + torch.mean(fake_score) + self.parameters["lambda_gp"] * gp
                    optimizer_C.zero_grad()
                    loss_c.backward()
                    optimizer_C.step()

                # Training Generator
                z_g = torch.randn(batch.size(dim=0), self.parameters["latent_dim"]).float()
                z_g_ = torch.randn(batch.size(dim=0), self.parameters["latent_dim"]).float()
                h_fake_g = self.generator(z_g)
                seq_ = self.generator(z_g_).reshape(batch.size(dim=0), self.parameters["seq_length"], self.output_dim)[:, 0, :]
                seq = h_fake_g.reshape(batch_size, self.parameters["seq_length"], self.output_dim)[:, -1, :]
                distances = F.pairwise_distance(seq, seq_, p=2)
                fake_score_g = self.critic(h_fake_g)
                loss_g = -torch.mean(fake_score_g) + self.parameters["variance_coeff"] * distances.mean()
                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()

            if verbose:
                # Store gradient norms
                grad_g_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.generator.parameters() if p.grad is not None])).item())
                grad_c_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.critic.parameters() if p.grad is not None])).item())

                # Compute Wasserstein distance on validation
                for metric in self.metrics:
                    m = self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"])
                    metrics[metric].append(m)
                loss_g_list.append(loss_g.item())
                loss_c_list.append(loss_c.item())

                print(
                    f"Epoch [{epoch + 1}/{self.parameters["epochs"]}] Loss D: {loss_c.item():.4f}, "
                    f"Loss G: {loss_g.item():.4f}, " + ", ".join(
                        [f"{metric.capitalize()}: {metrics[metric][-1]:.4f}" for metric in self.metrics]))

        losses = {"Critic Loss": loss_c_list, "Generator Loss": loss_g_list}
        gradients = {"Critic Gradient": grad_c_list, "Generator Gradient": grad_g_list}

        return losses, gradients, metrics
