from .ModelGAN import *


class GAN(ModelGAN):

    def __init__(self):
        super().__init__()
        self.parameters = {"lr_g": 1e-5, "lr_d": 1e-5, "epochs": 100, "batch_size": 32, "latent_dim": 100}

    def set_architecture(self, **kwargs):
        if self.data is None:
            raise Exception("Vous n'avez pas chargé de données. Voir set_data()")
        self.generator = Generator(self.parameters["latent_dim"], self.output_dim)
        self.discriminator = Discriminator(self.output_dim)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

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

    def train(self, verbose=False):
        # Training loop
        if self.generator is None:
            raise Exception("L'architecture du modèle n'a pas été initialisée. Voir set_architecture()")
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.parameters["lr_g"], betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.parameters["lr_d"], betas=(0.5, 0.999))
        # Loss function
        criterion = nn.BCELoss()
        discrim_losses, generator_losses = [], []
        discrim_gradients, generator_gradients = [], []
        wass_dists = []
        for epoch in range(self.parameters["epochs"]):
            for real_data, in self.train_loader:
                real_data = real_data.float()
                batch_size = real_data.size(0)

                # Generate labels
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.discriminator.zero_grad()

                # Real data loss
                real_outputs = self.discriminator(real_data)
                d_loss_real = criterion(real_outputs, real_labels)

                # Fake data loss
                z = torch.randn(batch_size, self.parameters["latent_dim"])  # Latent noise
                fake_data = self.generator(z).detach()  # Detach to avoid updating G
                fake_outputs = self.discriminator(fake_data)
                d_loss_fake = criterion(fake_outputs, fake_labels)

                # Total Discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

                # ---------------------
                # Train Generator
                # ---------------------
                self.generator.zero_grad()

                # Generate fake data and classify with the Discriminator
                z = torch.randn(batch_size, self.parameters["latent_dim"])
                fake_data = self.generator(z)
                outputs = self.discriminator(fake_data)

                # Generator loss: make the Discriminator classify fake data as real
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                optimizer_G.step()

            if verbose:
                discrim_losses.append(d_loss.item())
                generator_losses.append(g_loss.item())

                # Gradient norms
                discrim_gradients.append(
                    sum(p.grad.norm().item() for p in self.discriminator.parameters() if p.grad is not None) / len(
                        list(self.discriminator.parameters())))
                generator_gradients.append(
                    sum(p.grad.norm().item() for p in self.generator.parameters() if p.grad is not None) / len(
                        list(self.generator.parameters())))

                wass_dists.append(self.compute_val_wass_dist())

                print(f"Epoch [{epoch + 1}/{self.parameters["epochs"]}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | Wass Dist: {wass_dists[-1]:.4f}")

        losses = {"Discriminator Loss": discrim_losses, "Generator Loss": generator_losses}
        gradients = {"Discriminator Gradient": discrim_gradients, "Generator Gradient": generator_gradients}

        return losses, gradients, wass_dists
