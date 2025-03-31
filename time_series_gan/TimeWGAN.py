from .ModelTimeGAN import *


class TimeWGAN(ModelTimeGAN):

    def __init__(self):
        super().__init__()
        self.parameters = {"lr_g": 1e-5, "lr_c": 1e-5, "lr_e": 5e-3, "lr_r": 5e-3, "epochs": 100, "batch_size": 32, "latent_dim": 100, "hidden_dim": 128, "seq_length": 10, "n_critic": 2, "lambda_gp": 0.1, "offset": 5}

    def set_architecture(self):
        super().set_architecture()
        self.critic = Critic(self.parameters["hidden_dim"])
        self.critic.apply(self.weights_init)

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1).expand_as(real_samples).float()
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
        optimizer_E = optim.Adam(self.embedder.parameters(), lr=self.parameters["lr_e"])
        optimizer_R = optim.Adam(self.recovery.parameters(), lr=self.parameters["lr_r"])
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.parameters["lr_g"], betas=(0.5, 0.999))
        optimizer_C = optim.Adam(self.critic.parameters(), lr=self.parameters["lr_c"], betas=(0.5, 0.999))
        loss_fn = nn.MSELoss()

        metrics = {metric: [] for metric in self.metrics}
        loss_g_list = []
        loss_c_list = []
        loss_ae_list = []
        grad_g_list = [1.0]
        grad_c_list = []
        grad_e_list = []
        grad_r_list = []

        for epoch in range(self.parameters["epochs"]):
            for i, batch in enumerate(self.train_loader):
                real_data = batch.float()

                # Pre-training Embedder/Recovery
                h_real = self.embedder(real_data)
                x_tilde = self.recovery(h_real)
                loss_ae = loss_fn(real_data, x_tilde)
                optimizer_E.zero_grad()
                optimizer_R.zero_grad()
                loss_ae.backward(retain_graph=True)
                optimizer_E.step()
                optimizer_R.step()

                for _ in range(self.parameters["n_critic"]):
                    # Training Critic
                    z = torch.randn(batch.size(dim=0), self.parameters["seq_length"], self.parameters["latent_dim"]).float()
                    h_fake = self.generator(z)
                    real_score = self.critic(h_real.detach())
                    fake_score = self.critic(h_fake.detach())
                    gp = self.gradient_penalty(h_real.detach(), h_fake.detach())
                    loss_c = -torch.mean(real_score) + torch.mean(fake_score) + self.parameters["lambda_gp"] * gp
                    optimizer_C.zero_grad()
                    loss_c.backward()
                    optimizer_C.step()

                # Training Generator
                z_g = torch.randn(batch.size(dim=0), self.parameters["seq_length"], self.parameters["latent_dim"]).float()
                h_fake_g = self.generator(z_g)
                fake_score_g = self.critic(h_fake_g)
                loss_g = -torch.mean(fake_score_g)
                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()

            if verbose:
                # Store gradient norms
                grad_g_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.generator.parameters() if p.grad is not None])).item())
                grad_c_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.critic.parameters() if p.grad is not None])).item())
                grad_e_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.embedder.parameters() if p.grad is not None])).item())
                grad_r_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.recovery.parameters() if p.grad is not None])).item())

                # Compute Wasserstein distance on validation
                for metric in self.metrics:
                    m = self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"])
                    metrics[metric].append(m)
                loss_g_list.append(loss_g.item())
                loss_c_list.append(loss_c.item())
                loss_ae_list.append(loss_ae.item())

                print(
                    f"Epoch [{epoch + 1}/{self.parameters["epochs"]}] Loss AE: {loss_ae.item():.4f}, Loss D: {loss_c.item():.4f}, "
                    f"Loss G: {loss_g.item():.4f}, " + ", ".join(
                        [f"{metric.capitalize()}: {metrics[metric][-1]:.4f}" for metric in self.metrics]))

        losses = {"Critic Loss": loss_c_list, "Generator Loss": loss_g_list, "AutoEncoder Loss": loss_ae_list}
        gradients = {"Critic Gradient": grad_c_list, "Generator Gradient": grad_g_list,
                     "Embedder Gradient": grad_e_list, "Recovery Gradient": grad_r_list}

        return losses, gradients, metrics
