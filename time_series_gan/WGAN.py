from .ModelGAN import *


class WGAN(ModelGAN):

    def __init__(self):
        super().__init__()
        self.parameters = {"lr_g": 1e-5, "lr_c": 1e-5, "epochs": 100, "batch_size": 32, "latent_dim": 100, "n_critic": 2, "lambda_gp": 0.1}

    def set_architecture(self):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")
        self.generator = Generator(self.parameters["latent_dim"], self.output_dim)
        self.critic = Critic(self.output_dim)
        self.generator.apply(self.weights_init)
        self.critic.apply(self.weights_init)

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

    def generate_samples(self, n_samples):
        if self.generator is None:
            raise Exception("Vous n'avez pas encore initialisé le modèle. Voir fit()")
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.parameters["latent_dim"])
            generated_data = self.generator(z)
        return generated_data.numpy()

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
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.parameters["lr_g"], betas=(0.5, 0.9))
        optimizer_C = optim.Adam(self.critic.parameters(), lr=self.parameters["lr_c"], betas=(0.5, 0.9))

        critic_losses, generator_losses = [], []
        critic_gradients, generator_gradients = [], []
        wass_dists = []

        for epoch in range(self.parameters["epochs"]):
            for i, (real_tuples,) in enumerate(self.train_loader):
                batch_size = real_tuples.shape[0]
                real_tuples = real_tuples.float()

                for _ in range(self.parameters["n_critic"]):
                    z = torch.randn(batch_size, self.parameters["latent_dim"]).float()
                    fake_tuples = self.generator(z)

                    real_validity = self.critic(real_tuples)
                    fake_validity = self.critic(fake_tuples.detach())
                    gp = self.gradient_penalty(real_tuples, fake_tuples)
                    critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.parameters["lambda_gp"] * gp

                    optimizer_C.zero_grad()
                    critic_loss.backward()

                    # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

                    optimizer_C.step()

                z = torch.randn(batch_size, self.parameters["latent_dim"]).float()
                fake_tuples = self.generator(z)
                generator_loss = -torch.mean(self.critic(fake_tuples))

                optimizer_G.zero_grad()
                generator_loss.backward()
                # Clip gradients of the generator
                # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
                optimizer_G.step()

            if verbose:
                critic_losses.append(critic_loss.item())
                generator_losses.append(generator_loss.item())

                # Gradient norms
                critic_gradients.append(sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None) / len(
                    list(self.critic.parameters())))
                generator_gradients.append(
                    sum(p.grad.norm().item() for p in self.generator.parameters() if p.grad is not None) / len(
                        list(self.generator.parameters())))

                wass_dists.append(self.compute_val_wass_dist())

                print(
                    f"Epoch [{epoch + 1}/{self.parameters["epochs"]}] C Loss: {critic_loss.item():.4f} G Loss: {generator_loss.item():.4f} | Wass Dist: {wass_dists[-1]:.4f}")

        losses = {"Critic Loss": critic_losses, "Generator Loss": generator_losses}
        gradients = {"Critic Gradient": critic_gradients, "Generator Gradient": generator_gradients}

        return losses, gradients, wass_dists
