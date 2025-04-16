

from time_series_gan import *
from time_series_gan import score


import copy
import random
import math
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn


def test (type_model, architectures, params, data, metrique, nom_de_la_métrique):
    model = type_model()
    model.set_metrics({nom_de_la_métrique:{"function":metrique, "metric_args":{}}})

    model.set_data(data)
    return model.fit(params=params, architectures=architectures, verbose=False, save=False)[nom_de_la_métrique][1]


def adjacent(archi, params):
    liste_fct_transi = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]
    new_archi = copy.deepcopy(archi)
    new_params = copy.deepcopy(params)

    def supprimer_couche(reseau):
        n = len(new_archi[reseau]["layer_sizes"])
        if n <=1:
            return ajouter_couche(reseau)
        k = random.randint(0, n - 2) #On ne peut pas supprimer la dernière couche
        new_layer_sizes = []
        for i in range(n):
            if i != k:
                new_layer_sizes.append(new_archi[reseau]["layer_sizes"][i])
        return new_layer_sizes

    def ajouter_couche(reseau):
        n = len(new_archi[reseau]["layer_sizes"])
        k = random.randint(0, n-1)#On ne peut pas ajouter la dernière couche
        new_layer_sizes = []
        for i in range(n):
            if i == k:
                new_layer_sizes.append(random.randint(10, 100))
            new_layer_sizes.append(new_archi[reseau]["layer_sizes"][i])
        return new_layer_sizes
    
    def modifier_fct_activation(reseau):
        fct_a_exclure = archi[reseau]["activation"]
        liste_filtrée = [fct for fct in liste_fct_transi if type(fct) != type(fct_a_exclure)]
        fct_choisie = random.choice(liste_filtrée)
        return fct_choisie


    def modifier_1_param(param):
        def round_sig(x, sig=3):
            if x == 0:
                return 0
            return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


        variation = np.exp(random.uniform(-1., 1.))
        
        if type(params[param]) == int:
            candidat= int(params[param] * variation) 
        else:
            x = params[param] * variation
            candidat= round_sig(x, sig=4)  # arrondi à 3 chiffres significatifs
        
        if candidat==0:
            candidat = params[param]

        if param == "latent_dim":
            return min (candidat,50)
        elif param == "epochs":
            return max(20, min (candidat, 150))
        elif param == "hidden_dim":
            candidat = min(candidat, 50)
            new_archi["generator"]["layer_sizes"][-1] = candidat
            return candidat
        elif param == "seq_length":
            return params[param]
        else:
            return candidat


    epsilon = random.random()
    if epsilon < 0.3:
        reseau = random.choice(list(archi.keys()))
        new_archi[reseau]["layer_sizes"] = supprimer_couche(reseau)
        print("suppression d'une couche du " + reseau)
    elif epsilon < 0.5:
        reseau = random.choice(list(archi.keys()))
        new_archi[reseau]["layer_sizes"] = ajouter_couche(reseau)
        print("ajout d'une couche au " + reseau)
    elif epsilon < 0.55:
        reseau = random.choice(list(archi.keys()))
        new_archi[reseau]["activation"] = modifier_fct_activation(reseau)
        print("modification de la fonction d'activation du " + reseau)
    else:
        
        for param in list(params.keys()):
            new_params[param] = modifier_1_param(param)
        print("modification des paramètres")


    return new_archi, new_params



def get_archi(model_de_base, data):
    archi = {}

    if "lr_g" in model_de_base.parameters:
        archi["generator"] = {
            "architecture": "MLP",
            "layer_sizes": [data.shape[1]],
            "activation": "ReLU"
        }
        if "hidden_dim" in model_de_base.parameters:  
            archi["generator"]["layer_sizes"][-1]= model_de_base.parameters["hidden_dim"]
        
        if isinstance(model_de_base, XTSGAN):
            archi["generator"]["layer_sizes"][-1]= model_de_base.parameters["seq_length"]*data.shape[1]

    if "lr_c" in model_de_base.parameters:
        archi["critic"] = {
            "architecture": "MLP",
            "layer_sizes": [1],
            "activation": "ReLU"
        }
    if "lr_d" in model_de_base.parameters:
        archi["discriminator"] = {
            "architecture": "MLP",
            "layer_sizes": [1],
            "activation": "ReLU"
        }
    return archi





def Metropolis_Hasting(beta, data, type_model, ite = 10, analyse= False, metrique = score, nom_de_la_métrique="score"):

    model_de_base=type_model()
    params = model_de_base.parameters
    if "lr_g" in params:
        params["lr_g"] = 0.001
    if "lr_c" in params:
        params["lr_c"] = 0.001
    if "lr_d" in params:
        params["lr_d"] = 0.001
    if "lr_e" in params:
        params["lr_e"] = 0.001
    if "lr_r" in params:
        params["lr_r"] = 0.001
    if "hidden_dim" in params:
        params["hidden_dim"] = 20
    if "latent_dim" in params:
        params["latent_dim"] = 20

    archi=get_archi(model_de_base, data)
    qualite = test(type_model, archi, params, data,metrique,nom_de_la_métrique)

    resultats=[]


    for i in range(ite):
        print("MH étape : " +str(i))
        archi_test, params_test = adjacent(archi, params)
        qualite_test = test(type_model,archi_test, params_test, data, metrique,nom_de_la_métrique)

        if random.random() < np.exp((qualite-qualite_test)/beta):

            print("succes")
            print(str(qualite) + " ->  " + str (qualite_test))
            print("")
            archi = archi_test
            params = params_test
            qualite=qualite_test
            print(params)
            for reseau in archi.keys():
                print(reseau + " : " + str(archi[reseau]["layer_sizes"]) + "        "+ str(archi[reseau]["activation"]))
            print("")
            print("")

            if analyse:
                resultats.append((i,qualite, archi, params))

        else:
            print("echec")
            print("")
            print("")

    if analyse:
        return resultats
    else:
        return archi



def generer_resultats(beta, data,model,  ite, nom_fichier, metrique=score, nom_de_la_metrique="score"):
    results = Metropolis_Hasting(beta, data, model, ite, analyse= True)
    with open(nom_fichier, "wb") as f:
        pickle.dump(results, f)

def charger_resultats(nom_fichier):
    with open(nom_fichier, "rb") as f:
        results = pickle.load(f)
    return results


def reestimer(model, archi, param, data, metrique, nom_de_la_metrique):
    esti=[]
    for i in range (15):
        esti.append(test(model, archi, param, data, metrique, nom_de_la_metrique))
    return np.mean(sorted(esti)[:12])



#data=prep_data()[["YIELD_station_49", "YIELD_station_80", "YIELD_station_40", "YIELD_station_63"]]
data=pd.read_csv("data/synthetic_data.csv")
#resultats = Metropolis_Hasting(0.1, data,XTSGAN, ite = 100, analyse=True)
generer_resultats(3., data, TimeGAN, ite =  1000, nom_fichier="results_pkl\\results_TimeGAN_GDP.pkl", metrique=score ,nom_de_la_metrique="score" )



results=charger_resultats("results_pkl\\results_GAN_SYNTHE.pkl")
sorted_results = sorted(results, key=lambda x: x[1]) 
   
for i in range (15):
    print(sorted_results[i][0]) # numero de l'itération
    print(sorted_results[i][1]) # score
    archi=sorted_results[i][2] 
    params=sorted_results[i][3]
    #print("nouvelle estimation : " + str(reestimer(GAN, archi, params, data, score, "score")))
    print(params)
    for reseau in archi.keys():
        print(reseau + " : " + str(archi[reseau]["layer_sizes"]) + "     " + str(archi[reseau]["activation"]))
    print("")





"""
----------------------------------------------------------------------------------------------
RESULTATS
----------------------------------------------------------------------------------------------


WGAN
500 ite

{'lr_g': 0.009446, 'lr_c': 0.03663, 'epochs': 100, 'batch_size': 234, 'latent_dim': 47, 'n_critic': 1, 'lambda_gp': 4.822}
generator : [4]     TanH()
critic : [49, 11, 37, 22, 41, 38, 24, 27, 14, 1]     Sigmoid()
=> 0.20 (DOUBLE CHECK ~0.4-0.5)

{'lr_g': 0.004592, 'lr_c': 0.05486, 'epochs': 92, 'batch_size': 36, 'latent_dim': 26, 'n_critic': 1, 'lambda_gp': 4.118}
generator : [4]      TanH()
critic : [22, 41, 13, 1]        TanH()
=> 0.25 (DOUBLE CHECK ~0.25 - 0.3)

        _______________________________________________________

TIMEGAN
25 ite
(PB un fit prend ~3min sur ma machine. => Demander au SMAT pour des résultats probants)
{'lr_g': 0.001097, 'lr_d': 0.001871, 'lr_e': 0.001838, 'lr_r': 0.0007295, 'epochs': 100, 'batch_size': 27, 'latent_dim': 28, 'hidden_dim': 17, 'seq_length': 7, 'n_critic': 2}
generator : [11, 17]
discriminator : [25, 15, 1]
=> 0.28



        ________________________________________________________

TIMEWGAN
50 ite (fit entre 30 sec et 3 min)
{'lr_g': 0.001351, 'lr_c': 0.005018, 'lr_e': 0.003824, 'lr_r': 0.001581, 'epochs': 100, 'batch_size': 76, 'latent_dim': 7, 'hidden_dim': 11, 'seq_length': 25, 'n_critic': 10, 'lambda_gp': 0.1156}
generator : [43, 11]
critic : [33, 37, 1]
=> 0.48

XTSGAN
120 ite (fit jusque 10 min)
{'lr_g': 0.0003576, 'lr_c': 0.003743, 'epochs': 100, 'batch_size': 2, 'latent_dim': 27, 'seq_length': 10, 'n_critic': 5, 'lambda_gp': 0.7225}
generator : [40]
critic : [42, 1]
=> 0.15  

"""