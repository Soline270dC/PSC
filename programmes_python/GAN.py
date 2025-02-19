import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy

import torch
import torch.nn as nn
import torch.optim as optim


from initialisation import *



#ARCHITECTURE________________________________________________________________________________________
class Architecture():
    def __init__(self, lr, couches_gene, couches_discri, fct_transi_gene, fct_transi_discri, latent_dim):
        self.lr = lr
        self.couches_gene = couches_gene  # Une liste indiquant le nb de neuronnes à chaque couche
        self.couches_discri = couches_discri
        self.fct_transi_gene = fct_transi_gene  # Une liste indiquant les fonctions de transi sous la forme [nn.reLU(), nn. , ...]
        self.fct_transi_discri = fct_transi_discri
        self.latent_dim = latent_dim

    def print_archi(self):
        print(self.lr)
        print(self.latent_dim)
        print(self.couches_gene)
        print(self.couches_discri)
        print(self.fct_transi_gene)
        print(self.fct_transi_discri)
        print("")
        print("")

# Générateur
class Generator(nn.Module):
    def __init__(self, archi):
        #archi est un objet de classe Architecture
        super(Generator, self).__init__()
        
        n=len(archi.couches_gene)
        if n==0:
            layers=[nn.Linear(archi.latent_dim, data_dim)]
        else:
            layers=[]
            layers.append(nn.Linear(archi.latent_dim, archi.couches_gene[0]))
            for i in range (n-1):
                layers.append(archi.fct_transi_gene[i])
                layers.append(nn.Linear(archi.couches_gene[i], archi.couches_gene[i+1]))
            layers.append(nn.Linear(archi.couches_gene[n-1], data_dim))

        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)

# Discriminateur
class Discriminator(nn.Module):
    def __init__(self, archi):
        super(Discriminator, self).__init__()

        n=len(archi.couches_discri)
        if n==0:
            layers=[nn.Linear(data_dim,1 )]
        else:
            layers=[]
            layers.append(nn.Linear(data_dim, archi.couches_discri[0]))
            for i in range (n-1):
                layers.append(archi.fct_transi_discri[i])
                layers.append(nn.Linear(archi.couches_discri[i], archi.couches_discri[i+1]))
            layers.append(nn.Linear(archi.couches_discri[n-1], 1))

        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    
    
    def forward(self, x):
        return self.model(x)

# Fonction de perte
criterion = nn.BCELoss()

#ENTRAINEMENT___________________________________________________________________________________________
def entrainement(generator,discriminator, data, num_epochs, learning_rate):
    latent_dim = generator.model[0].in_features

    generator_local = copy.deepcopy(generator)
    discriminator_local = copy.deepcopy(discriminator)

    optimizer_G_local = optim.Adam(generator_local.parameters(), lr=learning_rate)
    optimizer_D_local = optim.Adam(discriminator_local.parameters(), lr=learning_rate)

    l=[]
    for epoch in range(num_epochs):
        testr=0
        for real_data in data.train_dataloader:
            # Données réelles (vecteurs de taille 4)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Entraînement du Discriminateur
            outputs = discriminator_local(real_data)
            d_loss_real = criterion(outputs, real_labels)

            # Données générées
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator_local(z)
            outputs = discriminator_local(fake_data.detach())
            # if testr ==0:
            #     print(discriminator(fake_data.detach()) == discriminator(fake_data))
            #     # print(discriminator(fake_data))
            #     testr=1
            d_loss_fake = criterion(outputs, fake_labels)

            # Total Discriminateur
            d_loss = d_loss_real + d_loss_fake
            optimizer_D_local.zero_grad()
            d_loss.backward()
            optimizer_D_local.step()

            # Entraînement du Générateur
            outputs = discriminator_local(fake_data)
            g_loss = criterion(outputs, real_labels)
            optimizer_G_local.zero_grad()
            g_loss.backward()
            optimizer_G_local.step()

        #print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
    return generator_local, discriminator_local







