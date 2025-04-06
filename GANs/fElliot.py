#IMPORTATIONS_______________________________________________________________________________________
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wasserstein_distance

# #HYPERPARAMETRES_____________________________________________________________________________________
# batch_size = 50
# num_epochs = 60

# latent_dim = 10  # Taille du bruit d'entrée
data_dim = 4    

#IMPORT DES DONNEES_________________________________________________________________________________
def init_data():
    """Retourne les données filtrées et combinées sous forme de dataframe pandas"""
    dico_station={ 
                    "station_40": {"file":"data/3-station_40.csv", "threshold":6.4897},
                    "station_49": {"file":"data/1-station_49.csv", "threshold":3.3241},
                    "station_63": {"file":"data/4-station_63.csv", "threshold":7.1301},
                    "station_80": {"file":"data/2-station_80.csv", "threshold":5.1292}
    }

    #Génération des dataframes propres pour chaque station
    for station, info in dico_station.items():
        #On lit le csv original
        df = pd.read_csv(info["file"])

        #On filtre les lignes
        threshold = info["threshold"]
        df = df[df["W_13"] + df["W_14"] + df["W_15"] <= threshold]
        
        #On renomme la colonne YIELD    
        df = df.rename(columns={"YIELD": f"YIELD_{station}"})

        #On ne garde que les colonnes year et YIELD
        df= df[["YEAR", f"YIELD_{station}"]]

        #On range le résultat dans info
        info["df"] = df

    #Combinaison des dataframes
    combined_df = pd.DataFrame()
    for station, info in dico_station.items():
        df = info["df"]
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on="YEAR", how="inner")
    
    return combined_df

class YieldDataset(Dataset):
    def __init__(self, df):
        # On extrait uniquement les colonnes YIELD des 4 stations
        self.data = df[["YIELD_station_49", "YIELD_station_80", "YIELD_station_40", "YIELD_station_63"]].values #tableau numpy
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retourne un vecteur de taille 4
        return torch.tensor(self.data[idx], dtype=torch.float32)

def getDalatoader(batch_size) :
    df = init_data()    
    yield_dataset, valid_data = random_split(YieldDataset(df), [0.8, 0.2]) #Crée un objet de format YieldDataset à partir du dataframe initial
    valid_data = valid_data.dataset.data[valid_data.indices]
    dataloader = DataLoader(yield_dataset, batch_size=batch_size, shuffle=True)

    # all_data = torch.cat([batch for batch in dataloader], dim=0)  # Combine tous les batches
    # all_data_np = all_data.numpy()                               #Cobine en tableau np
    return dataloader, valid_data

# dataloader, all_data_np = getDalatoader(batch_size)

#ARCHITECTURE________________________________________________________________________________________

class Architecture():
    def __init__(self, lr, couches_gene, couches_discri, fct_transi_gene, fct_transi_discri):
        self.lr=lr
        self.couches_gene=couches_gene #Une liste indiquant le nb de neuronnes à chaque couche
        self.couches_discri=couches_discri
        self.fct_transi_gene= fct_transi_gene #Une liste indiquant les fonctions de transi sous la forme [nn.reLU(), nn. , ...]
        self.fct_transi_discri= fct_transi_discri


# Générateur
class Generator(nn.Module):
    def __init__(self, archi: 'Architecture', latent_dim):
        #archi est un objet de classe Architecture
        super(Generator, self).__init__()
        assert isinstance(archi, Architecture), 'archi not Architecture !?'
        
        n=len(archi.couches_gene)
        if n==0:
            layers=[nn.Linear(latent_dim, data_dim)]
        else:
            layers=[]
            layers.append(nn.Linear(latent_dim, archi.couches_gene[0]))
            for i in range (n-1):
                layers.append(archi.fct_transi_gene[i])
                layers.append(nn.Linear(archi.couches_gene[i], archi.couches_gene[i+1]))
            layers.append(nn.Linear(archi.couches_gene[n-1], data_dim))

        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)

# Discriminateur
class Discriminator(nn.Module):
    def __init__(self, archi: 'Architecture'):
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
def entrainement(generator, discriminator, num_epochs, learning_rate, dataloader, latent_dim, batch_size):

    generator_local = copy.deepcopy(generator)
    discriminator_local = copy.deepcopy(discriminator)

    optimizer_G_local = optim.Adam(generator_local.parameters(), lr=learning_rate)
    optimizer_D_local = optim.Adam(discriminator_local.parameters(), lr=learning_rate)

    l=[]
    for epoch in range(num_epochs):
        testr=0
        for real_data in dataloader:
            # Données réelles (vecteurs de taille 4)
            size = len(real_data)
            real_labels = torch.ones(size, 1)
            fake_labels = torch.zeros(size, 1)

            # Entraînement du Discriminateur
            outputs = discriminator_local(real_data)
            d_loss_real = criterion(outputs, real_labels)

            # Données générées
            z = torch.randn(size, latent_dim)
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




#DISTANCE DE WASSERSTEIN____________________________________________________________________________

def random_sph():
    """génère un vecteur aléatoire sur la sphère de dimension 4"""
    v= np.random.normal(loc=0, scale =1, size= 4 )
    return v / np.linalg.norm(v)

def Wassertstein_esti(data_reelle, data_simu, ite=1000):
    s=0
    for i in range(ite):
        v_proj= random_sph()
        dr= np.dot(data_reelle,v_proj)
        ds= np.dot(data_simu,v_proj)
        s+= wasserstein_distance(dr, ds)
    return s / ite




#ESPACE DE TEST____________________________________________________________________________________
archi = Architecture(0.0008, [28], [45, 49, 43, 19, 36, 36, 10, 10, 13, 21], [nn.Sigmoid()], [nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.ReLU()])

def esti_modele(generator, all_data_np, latent_dim, nb_points = 1000):
    z = torch.randn(nb_points, latent_dim)
    fake_data = generator(z)
    fake_data_np = fake_data.detach().numpy() 
    return Wassertstein_esti(all_data_np, fake_data_np)

def test_architecture(batch_size, latent_dim, num_epochs, lr, archi = archi, nb_ite=1):
    # Initialisation des modèles
    s=0
    dataloader, all_data_np = getDalatoader(batch_size)
    for _ in range(nb_ite):
        generator, discriminator = Generator(archi, latent_dim) , Discriminator(archi)
        generator, discriminator = entrainement(generator, discriminator, num_epochs, lr, dataloader, latent_dim, batch_size)
        x = esti_modele(generator, all_data_np, latent_dim)
        s+= x
    return s / nb_ite

# print(test_architecture(batch_size, latent_dim, num_epochs, 0.0008, archi))