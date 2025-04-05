from .ModelGAN import *


class GAN(ModelGAN):

    def __init__(self):
        super().__init__()
        self.discriminator = None
        self.parameters = {"lr_g": 1e-5, "lr_d": 1e-5, "epochs": 100, "batch_size": 32, "latent_dim": 100}

    def set_architecture(self, **kwargs):
        super().set_architecture()
        self.discriminator = Discriminator(self.output_dim)
        self.discriminator.apply(self.weights_init)

    def modify_discriminator(self, architecture, layer_sizes, activation=None):
        if self.discriminator is None:
            raise Exception("Vous n'avez pas encore initialisé le critique")
        self.modify_architecture(self.discriminator, architecture, layer_sizes, activation)

    def modify_models(self, architectures):
        super().modify_models(architectures)
        if "discriminator" in architectures:
            if self.discriminator is None:
                raise Exception("Vous n'avez pas encore initialisé le discriminateur")
            if not isinstance(architectures["discriminator"], dict) or "architecture" not in architectures["discriminator"] or "layer_sizes" not in architectures["discriminator"]:
                raise Exception(
                    "Vous pouvez modifier les architectures avec un dict de forme {'discriminator': {'architecture': 'MLP', 'layer_sizes': ...}, 'generator': {...}}")
            self.modify_generator(architectures["discriminator"]["architecture"], architectures["discriminator"]["layer_sizes"])

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
        metrics = {metric: [] for metric in self.metrics}
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

                for metric in self.metrics:
                    m = self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"])
                    metrics[metric].append(m)

                print(
                    f"Epoch [{epoch + 1}/{self.parameters["epochs"]}], C Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, " + ", ".join(
                        [f"{metric.capitalize()}: {metrics[metric][-1]:.4f}" for metric in self.metrics]))

        losses = {"Discriminator Loss": discrim_losses, "Generator Loss": generator_losses}
        gradients = {"Discriminator Gradient": discrim_gradients, "Generator Gradient": generator_gradients}

        return losses, gradients, metrics
