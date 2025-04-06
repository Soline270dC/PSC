#IMPORTATIONS_______________________________________________________________________________________
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wasserstein_distance

#HYPERPARAMETRES_____________________________________________________________________________________
batch_size = 50
num_epochs = 50
#learning_rate = 0.0002 

latent_dim = 10  # Taille du bruit d'entrée
data_dim = 4    

#IMPORT DES DONNEES_________________________________________________________________________________
def init_data():
    """Retourne les données filtrées et combinées sous forme de dataframe pandas"""
    dico_station={ 
                    "station_40": {"file":"station_40.csv", "threshold":6.4897},
                    "station_49": {"file":"station_49.csv", "threshold":3.3241},
                    "station_63": {"file":"station_63.csv", "threshold":7.1301},
                    "station_80": {"file":"station_80.csv", "threshold":5.1292}
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

df = init_data()    
yield_dataset = YieldDataset(df) #Crée un objet de format YieldDataset à partir du dataframe initial
dataloader = torch.utils.data.DataLoader(yield_dataset, batch_size=batch_size, shuffle=True)

all_data = torch.cat([batch for batch in dataloader], dim=0)  # Combine tous les batches
all_data_np = all_data.numpy()                                  #Cobine en tableau np

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
    def __init__(self, archi):
        #archi est un objet de classe Architecture
        super(Generator, self).__init__()
        
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
def entrainement(generator,discriminator, num_epochs, learning_rate):

    generator_local = copy.deepcopy(generator)
    discriminator_local = copy.deepcopy(discriminator)

    optimizer_G_local = optim.Adam(generator_local.parameters(), lr=learning_rate)
    optimizer_D_local = optim.Adam(discriminator_local.parameters(), lr=learning_rate)

    l=[]
    for epoch in range(num_epochs):
        testr=0
        for real_data in dataloader:
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




#DISTANCE DE WASSERSTEIN____________________________________________________________________________

def random_sph():
    """génère un vecteur aléatoire sur la sphère de dimension 4"""
    v= np.random.normal(loc=0, scale =1, size= 4 )
    return v / np.linalg.norm(v)

def Wassertstein_esti (data_reelle, data_simu, ite=1000):
    s=0
    for i in range(ite):
        v_proj= random_sph()
        dr= np.dot(data_reelle,v_proj)
        ds= np.dot(data_simu,v_proj)
        s+= wasserstein_distance(dr, ds)
    return s / ite




#ESPACE DE TEST____________________________________________________________________________________
def esti_modele(generator, nb_points = 1000):
    z = torch.randn(nb_points, latent_dim)
    fake_data = generator(z)
    fake_data_np = fake_data.detach().numpy() 
    return Wassertstein_esti (all_data_np, fake_data_np)

def test_architecture(archi, nb_ite=1):
    # Initialisation des modèles
    s=0
    for i in range(nb_ite):
        generator,discriminator = Generator(archi) , Discriminator(archi)
        generator, discriminator = entrainement(generator,discriminator, num_epochs, archi.lr)
        x=esti_modele(generator)
        s+= x
    return s / nb_ite



def archi_adj(archi):
    """Renvoie une architecture <<adjacente>> à archi"""
    liste_fct_transi = [nn.ReLU(),nn.Tanh(),nn.Sigmoid()]

    def supprimer_couche_gene(archi):
        n= len(archi.couches_gene)
        k= random.randint(0,n-1)
        new_couches=[]
        new_fct_transi=[]
        for i in range(n):
            if i !=k:
                new_couches.append(archi.couches_gene[i])
                new_fct_transi.append(archi.fct_transi_gene[i])
        return Architecture(archi.lr, new_couches, archi.couches_discri, new_fct_transi, archi.fct_transi_discri)

    def supprimer_couche_discri(archi):
        n= len(archi.couches_discri)
        k= random.randint(0,n-1)
        new_couches=[]
        new_fct_transi=[]
        for i in range(n):
            if i !=k:
                new_couches.append(archi.couches_discri[i])
                new_fct_transi.append(archi.fct_transi_discri[i])
        return Architecture(archi.lr, archi.couches_gene , new_couches, archi.fct_transi_gene, new_fct_transi)

    def ajouter_couche_gene(archi):
        n= len(archi.couches_gene)
        k= random.randint(0,n)
        new_couches=[]
        new_fct_transi=[]
        for i in range(n):
            if i ==k:
                new_couches.append(random.randint(10,50))
                new_fct_transi.append(random.choice(liste_fct_transi))
            new_couches.append(archi.couches_gene[i])
            new_fct_transi.append(archi.fct_transi_gene[i])
        if k ==n:
            new_couches.append(random.randint(10,50))
            new_fct_transi.append(random.choice(liste_fct_transi))
        return Architecture(archi.lr, new_couches, archi.couches_discri, new_fct_transi, archi.fct_transi_discri)

    def ajouter_couche_discri(archi):
        n= len(archi.couches_discri)
        k= random.randint(0,n)
        new_couches=[]
        new_fct_transi=[]
        for i in range(n):
            if i ==k:
                new_couches.append(random.randint(10,50))
                new_fct_transi.append(random.choice(liste_fct_transi))
            new_couches.append(archi.couches_discri[i])
            new_fct_transi.append(archi.fct_transi_discri[i])
        if k ==n:
            new_couches.append(random.randint(10,50))
            new_fct_transi.append(random.choice(liste_fct_transi))
        return Architecture(archi.lr, archi.couches_gene , new_couches, archi.fct_transi_gene, new_fct_transi)

    def changer_lr(archi):
        if random.random()<0.5:
            return Architecture(archi.lr/2, archi.couches_gene ,  archi.couches_discri, archi.fct_transi_gene, archi.fct_transi_discri)
        else:
            return Architecture(archi.lr*2, archi.couches_gene ,  archi.couches_discri, archi.fct_transi_gene, archi.fct_transi_discri)
        

    epsilon = random.random()
    if epsilon < 2/6:
        print("modification du Learning rate")
        return changer_lr(archi)
    elif epsilon < 3/6 and len(archi.couches_gene)!=0:
        print("suppression d'une couche du générateur")
        return supprimer_couche_gene(archi)
    elif epsilon < 4/6:
        print("ajout d'une couche au générateur")
        return ajouter_couche_gene(archi)
    elif epsilon < 5/6 and len(archi.couches_discri)!=0:
        print("suppression d'une couche du discriminateur")
        return supprimer_couche_discri(archi)
    else:
        print("ajout d'une couche au discriminateur")
        return ajouter_couche_discri(archi)




def Metropolis_Hasting(beta, ite = 10):
    archi =Architecture (0.0002,[], [],[] ,[])
    qualite = test_architecture(archi)

    for i in range(ite):
        print("MH étape : " +str(i))
        archi_test = archi_adj(archi)
        qualite_test = test_architecture(archi_test)

        if random.random() < np.exp((qualite-qualite_test)/beta):

            print("succes")
            print(str(qualite) + " ->  " + str (qualite_test))
            print("")
            archi = archi_test
            qualite=qualite_test
            print(archi.lr)
            print(archi.couches_gene)
            print(archi.couches_discri)
            print(archi.fct_transi_gene)
            print(archi.fct_transi_discri)
            print("")
            print("")
        else:
            print("echec")
            print("")
            print("")
    

#ZONE DE TEST________________________________________________________________________________________

"""
archi =Architecture (0.0002,[26,12,64,25,32], [26, 25 ,43],[nn.ReLU(),nn.Tanh(),nn.Sigmoid(),nn.ReLU(),nn.ReLU()] ,[nn.ReLU(),nn.Sigmoid(),nn.Tanh()])
generator,discriminator = Generator(archi) , Discriminator(archi)
generator, discriminator = entrainement(generator,discriminator, num_epochs, archi.lr)

list_res=[]
for i in range(30):
    x=esti_modele(generator)
    list_res.append(x)
print(list_res)
"""
Metropolis_Hasting(0.1, ite = 500)
#print(test_architecture(archi))
