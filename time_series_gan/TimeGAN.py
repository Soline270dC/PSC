from .ModelTimeGAN import *


class TimeGAN(ModelTimeGAN):

    def __init__(self):
        super().__init__()
        self.parameters = {"lr_g": 1e-5, "lr_d": 1e-5, "lr_e": 5e-3, "lr_r": 5e-3, "epochs": 100, "batch_size": 32, "latent_dim": 100, "hidden_dim": 128, "seq_length": 10, "n_critic": 2, "offset": 5}

    def set_architecture(self):
        super().set_architecture()
        self.discriminator = Discriminator(self.parameters["hidden_dim"])
        self.discriminator.apply(self.weights_init)

    def train(self, verbose=False):
        optimizer_E = optim.Adam(self.embedder.parameters(), lr=self.parameters["lr_e"])
        optimizer_R = optim.Adam(self.recovery.parameters(), lr=self.parameters["lr_r"])
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.parameters["lr_g"], betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.parameters["lr_d"], betas=(0.5, 0.999))
        loss_fn = nn.MSELoss()

        metrics = {metric: [] for metric in self.metrics}
        loss_g_list = []
        loss_d_list = []
        loss_ae_list = []
        grad_g_list = [1.0]
        grad_d_list = []
        grad_e_list = []
        grad_r_list = []

        for epoch in range(self.parameters["epochs"]):
            for i, batch in enumerate(self.train_loader):
                real_data = batch

                # Pre-training Embedder/Recovery
                h_real = self.embedder(real_data)
                x_tilde = self.recovery(h_real)
                loss_ae = loss_fn(real_data, x_tilde)
                optimizer_E.zero_grad()
                optimizer_R.zero_grad()
                loss_ae.backward()
                optimizer_E.step()
                optimizer_R.step()

                for _ in range(self.parameters["n_critic"]):
                    # Training Critic
                    z = torch.randn(batch.size(dim=0), self.parameters["seq_length"], self.parameters["latent_dim"]).float()
                    h_fake = self.generator(z)
                    real_score = self.discriminator(h_real.detach())
                    fake_score = self.discriminator(h_fake.detach())
                    loss_d = -torch.mean(torch.log(real_score + 1e-8) + torch.log(1 - fake_score + 1e-8))
                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()

                # Training Generator
                z_g = torch.randn(batch.size(dim=0), self.parameters["seq_length"], self.parameters["latent_dim"]).float()
                h_fake_g = self.generator(z_g)
                fake_score_g = self.discriminator(h_fake_g)
                loss_g = -torch.mean(torch.log(fake_score_g + 1e-8))
                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()

            if verbose:
                grad_g_list.append(
                    torch.mean(torch.stack([p.grad.abs().mean() for p in self.generator.parameters() if p.grad is not None])).item())
                grad_d_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.discriminator.parameters() if p.grad is not None])).item())
                grad_e_list.append(
                    torch.mean(torch.stack([p.grad.abs().mean() for p in self.embedder.parameters() if p.grad is not None])).item())
                grad_r_list.append(
                    torch.mean(torch.stack([p.grad.abs().mean() for p in self.recovery.parameters() if p.grad is not None])).item())

                # Compute Wasserstein distance on validation
                for metric in self.metrics:
                    m = self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"])
                    metrics[metric].append(m)
                loss_g_list.append(loss_g.item())
                loss_d_list.append(loss_d.item())
                loss_ae_list.append(loss_ae.item())

                print(
                    f"Epoch [{epoch + 1}/{self.parameters["epochs"]}] Loss AE: {loss_ae.item():.4f}, Loss D: {loss_d.item():.4f}, "
                    f"Loss G: {loss_g.item():.4f}, " + ", ".join([f"{metric.capitalize()}: {metrics[metric][-1]:.4f}" for metric in self.metrics]))

        losses = {"Discriminator Loss": loss_d_list, "Generator Loss": loss_g_list, "AutoEncoder Loss": loss_ae_list}
        gradients = {"Discriminator Gradient": grad_d_list, "Generator Gradient": grad_g_list, "Embedder Gradient": grad_e_list, "Recovery Gradient": grad_r_list}

        return losses, gradients, metrics
