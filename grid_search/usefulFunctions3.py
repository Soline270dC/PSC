import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import pandas as pd

import numpy as np

'''
hyperparamètres :
- batch_size : [2;100]
- num_epoch : [10;50]
- nz : [1;50]
- lr : [1e-4;1e-5]
- beta1 : [0;1]
'''

def getGrid3(batch_sizes, num_epochs) :
    return [[(batch_size, num_epoch) for batch_size in batch_sizes] for num_epoch in num_epochs]

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
        return torch.tensor(self.data[idx].reshape(-1, 1), dtype=torch.float32)


def getData3(batch_size) :
    df = init_data()    
    yield_dataset = YieldDataset(df) #Crée un objet de format YieldDataset à partir du dataframe initial
    dataloader = DataLoader(yield_dataset, batch_size=batch_size, shuffle=True)

    # all_data = torch.cat([batch for batch in dataloader], dim=0)  # Combine tous les batches
    # all_data_np = all_data.numpy()
    return dataloader