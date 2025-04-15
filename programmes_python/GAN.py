import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy

import torch
import torch.nn as nn
import torch.optim as optim


from initialisation import *



#ARCHITECTURE________________________________________________________________________________________
class Architecture():
    def __init__(self, lr=0.0001, couches_gene=[], couches_discri=[], fct_transi_gene=[], fct_transi_discri=[], latent_dim=10, data_dim=4, nombre_epochs=50):
        self.parameters={}
        self.reseaux={}

        self.reseaux["gene"]={"couches":couches_gene, "fct_transi":fct_transi_gene}
        self.reseaux["discri"]={"couches":couches_discri, "fct_transi":fct_transi_discri}
        self.parameters["lr"]=lr
        self.parameters["latent_dim"]=latent_dim
        self.parameters	["nombre_epochs"]=nombre_epochs

        self.data_dim = data_dim
        #batch-size
        #nombre d'epochs


        
    def print_archi(self):
        print("learning rate : ", str(self.parameters["lr"]))
        print("latent_dim : ", str(self.parameters["latent_dim"]))
        print(self.reseaux["gene"]["couches"])
        print(self.reseaux["gene"]["fct_transi"])
        print(self.reseaux["discri"]["couches"])
        print(self.reseaux["discri"]["fct_transi"])
        print("")
        print("")

# Générateur
class Generator(nn.Module):
    def __init__(self, archi):
        #archi est un objet de classe Architecture
        super(Generator, self).__init__()
        
        n=len(archi.reseaux["gene"]["couches"])
        if n==0:
            layers=[nn.Linear(archi.parameters["latent_dim"], archi.data_dim)]
        else:
            layers=[]
            layers.append(nn.Linear(archi.parameters["latent_dim"], archi.reseaux["gene"]["couches"][0]))
            layers.append(archi.reseaux["gene"]["fct_transi"][0])
            for i in range (n-1):
                layers.append(nn.Linear(archi.reseaux["gene"]["couches"][i], archi.reseaux["gene"]["couches"][i+1]))
                layers.append(archi.reseaux["gene"]["fct_transi"][i+1])
            layers.append(nn.Linear(archi.reseaux["gene"]["couches"][n-1], archi.data_dim))


        self.model = nn.Sequential(*layers)

        # Produit scalaire avec un vecteur appris (taille = data_dim)
        self.weight = nn.Parameter(torch.ones(archi.data_dim))  # ou init à autre chose
        self.bias = nn.Parameter(torch.zeros(archi.data_dim))   # vecteur constant ajouté

    def forward(self, z):
        #return self.model(z)
        out = self.model(z)  # [batch_size, data_dim]
        return out * self.weight + self.bias  # transformation affine élément par élément
    


# Discriminateur
class Discriminator(nn.Module):
    def __init__(self, archi):
        super(Discriminator, self).__init__()

        n=len(archi.reseaux["discri"]["couches"])
        if n==0:
            layers=[nn.Linear(archi.data_dim,1 )]
        else:
            layers=[]
            layers.append(nn.Linear(archi.data_dim, archi.reseaux["discri"]["couches"][0]))
            layers.append(archi.reseaux["discri"]["fct_transi"][0])
            for i in range (n-1):
                layers.append(nn.Linear(archi.reseaux["discri"]["couches"][i], archi.reseaux["discri"]["couches"][i+1]))
                layers.append(archi.reseaux["discri"]["fct_transi"][i+1])
            layers.append(nn.Linear(archi.reseaux["discri"]["couches"][n-1], 1))

        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    
    
    def forward(self, x):
        return self.model(x)

class GAN():
    def __init__(self, data, archi=Architecture()):
        self.archi = archi
        self.generator = Generator(archi)
        self.discriminator = Discriminator(archi)
        self.data =  data

    def entrainer(self, batch_size=None):
        criterion = nn.BCELoss()

        latent_dim = self.archi.parameters["latent_dim"]
        learning_rate = self.archi.parameters["lr"]

        optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        for epoch in range(self.archi.parameters["nombre_epochs"]):
            for real_data in self.data.train_dataloader:
                if batch_size is None:
                    batch_size = real_data.size(0)

                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Discriminateur
                outputs = self.discriminator(real_data)
                d_loss_real = criterion(outputs, real_labels)

                z = torch.randn(batch_size, latent_dim)
                fake_data = self.generator(z)
                outputs = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # Générateur
                outputs = self.discriminator(fake_data)
                g_loss = criterion(outputs, real_labels)
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

            #print(f"[{epoch+1}/{num_epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")







