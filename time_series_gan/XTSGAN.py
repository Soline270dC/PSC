import torch

from .WTSGAN import *


class XTSGAN(WTSGAN):

    def __init__(self):
        super().__init__()
        self.parameters["var_coeff"] = 1
        self.parameters["lr_s"] = 1e-3

    def set_architecture(self):
        super().set_architecture()
        self.sequencer = Sequencer(self.output_dim*self.parameters["seq_length"]//2)
        self.sequencer.apply(self.weights_init)

    def train_sequencer(self, seq_size, batch_size, real_tuples, loss_fn, optimizer_S):
        indices = torch.randint(0, seq_size, (batch_size, ))

        # Gather the selected time steps
        seq_combined = torch.stack([
            real_tuples[torch.arange(batch_size), indices[:]],
            real_tuples[torch.arange(batch_size), indices[:]+seq_size]
        ], dim=1)

        # Generate fake sequences for the second batch
        z_fake_1 = torch.randn(batch_size, self.parameters["latent_dim"]).float()
        z_fake_2 = torch.randn(batch_size, self.parameters["latent_dim"]).float()
        seq_fake_1 = self.generator(z_fake_1).reshape(batch_size, self.parameters["seq_length"], self.output_dim)[:, -seq_size//2+1:, :]
        seq_fake_2 = self.generator(z_fake_2).reshape(batch_size, self.parameters["seq_length"], self.output_dim)[:, :seq_size-seq_size//2, :]

        # Concatenate the fake sequences and apply slicing
        seq_fake = torch.cat((seq_fake_1, seq_fake_2), dim=1).reshape(batch_size, seq_size, self.output_dim)

        # Labels for real (1) and fake (0) sequences
        labels = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)))

        # Combine the data and shuffle
        data = torch.cat((seq_combined, seq_fake), dim=0)
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_labels = labels[indices]

        # Predict and compute the loss
        predict = self.sequencer(shuffled_data.reshape(2*batch_size, seq_size * self.output_dim)).reshape(2*batch_size)
        loss_s = loss_fn(shuffled_labels, predict)

        # Backpropagation
        optimizer_S.zero_grad()
        loss_s.backward()
        optimizer_S.step()

        return loss_s

    def train(self, verbose=False):
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.parameters["lr_g"], betas=(0.5, 0.999))
        optimizer_C = optim.Adam(self.critic.parameters(), lr=self.parameters["lr_c"], betas=(0.5, 0.999))
        optimizer_S = optim.Adam(self.sequencer.parameters(), lr=self.parameters["lr_s"])
        loss_fn = nn.MSELoss()

        seq_size = self.parameters["seq_length"]//2

        metrics = {metric: [] for metric in self.metrics}
        loss_g_list = []
        loss_c_list = []
        loss_s_list = []
        grad_g_list = [1.0]
        grad_c_list = []

        for epoch in range(self.parameters["epochs"]):
            for i, (batch,) in enumerate(self.train_loader):
                batch_size = batch.shape[0]
                real_tuples = batch.float()

                loss_s = self.train_sequencer(seq_size, batch_size, real_tuples, loss_fn, optimizer_S)

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
                z_g_1 = torch.randn(batch.size(dim=0), self.parameters["latent_dim"]).float()
                h_fake_g = self.generator(z_g)
                fake_2 = self.generator(z_g_1)[:, :self.output_dim*seq_size]
                fake_1 = h_fake_g[:, -self.output_dim*seq_size:]
                seq_fake = torch.cat((fake_1, fake_2), dim=1)
                score_sequence = self.sequencer(seq_fake)
                fake_score_g = self.critic(h_fake_g)
                loss_g = -torch.mean(fake_score_g) - self.parameters["var_coeff"]*torch.mean(score_sequence)
                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()

            if verbose:
                # Store gradient norms
                grad_g_list.append(torch.mean(
                    torch.stack(
                        [p.grad.abs().mean() for p in self.generator.parameters() if p.grad is not None])).item())
                grad_c_list.append(torch.mean(
                    torch.stack([p.grad.abs().mean() for p in self.critic.parameters() if p.grad is not None])).item())

                # Compute Wasserstein distance on validation
                for metric in self.metrics:
                    m = self.compute_val_metric(self.metrics[metric]["function"], self.metrics[metric]["metric_args"])
                    metrics[metric].append(m)
                loss_g_list.append(loss_g.item())
                loss_c_list.append(loss_c.item())
                loss_s_list.append(loss_s.item())

                print(
                    f"Epoch [{epoch + 1}/{self.parameters["epochs"]}] Loss S : {loss_s.item():.4f}, Loss D: {loss_c.item():.4f}, "
                    f"Loss G: {loss_g.item():.4f}, " + ", ".join(
                        [f"{metric.capitalize()}: {metrics[metric][-1]:.4f}" for metric in self.metrics]))

        losses = {"Critic Loss": loss_c_list, "Generator Loss": loss_g_list, "Sequencer Loss": loss_s_list}
        gradients = {"Critic Gradient": grad_c_list, "Generator Gradient": grad_g_list}

        return losses, gradients, metrics
