#IMPORTATIONS_______________________________________________________________________________________
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import pandas as pd


import numpy as np

from sklearn.model_selection import train_test_split

#HYPERPARAMETRES_____________________________________________________________________________________
batch_size = 50
num_epochs = 50
#learning_rate = 0.0002 

latent_dim = 10  # Taille du bruit d'entrée
data_dim = 4    

#IMPORT DES DONNEES_________________________________________________________________________________
def prep_data():
    """Retourne les données filtrées et combinées sous forme de dataframe pandas"""
    dico_station={ 
                    "station_40": {"file":"data/station_40.csv", "threshold":6.4897},
                    "station_49": {"file":"data/station_49.csv", "threshold":3.3241},
                    "station_63": {"file":"data/station_63.csv", "threshold":7.1301},
                    "station_80": {"file":"data/station_80.csv", "threshold":5.1292}
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


class Data:
    def __init__(self, df):
        self.df=df

        # Diviser les données en ensembles d'entraînement (80%) et de validation (20%)
        self.df_train, self.df_val = train_test_split(df, test_size=0.2, random_state=42)

        # Créer les datasets correspondants
        self.train_dataset = YieldDataset(self.df_train)
        self.val_dataset = YieldDataset(self.df_val)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)

        self.all_data_val = torch.cat([batch for batch in self.val_dataloader], dim=0)  # Combine tous les batches
        self.all_data_val_np = self.all_data_val.numpy()  

        self.batch_size = batch_size
        self.data_dim = 4 




def init_data():
    df = prep_data()
    return  Data(df)



