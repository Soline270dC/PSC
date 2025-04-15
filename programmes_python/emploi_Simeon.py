
from initialisation import *
from time_series_gan import *
from estimation_modele import Wassertstein_esti
import copy
import random
import math
import pickle

data=prep_data()[["YIELD_station_49", "YIELD_station_80", "YIELD_station_40", "YIELD_station_63"]]
metrics = {
    "Wasserstein": {
        "function": Wassertstein_esti,
        "metric_args": {}
    }
}


def test (type_model, architectures, params, data):
    model = type_model()
    model.set_metrics(metrics)

    model.set_data(data)
    return model.fit(params=params, architectures=architectures, verbose=False, save=False)['Wasserstein'][1]


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
                new_layer_sizes.append(random.randint(10, 50))
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


        variation = random.uniform(0.5, 2.)
        
        if type(params[param]) == int:
            candidat= int(params[param] * variation) 
        else:
            x = params[param] * variation
            candidat= round_sig(x, sig=4)  # arrondi à 3 chiffres significatifs
        
        if param == "latent_dim":
            return min (candidat,50)
        elif param == "epochs":
            return min (candidat, 100)
        elif param == "hidden_dim":
            candidat = min(candidat, 50)
            new_archi["generator"]["layer_sizes"][-1] = candidat
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

def Metropolis_Hasting(beta, data, type_model, ite = 10, analyse= False):

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
    qualite = test(type_model, archi, params, data)

    resultats=[]


    for i in range(ite):
        print("MH étape : " +str(i))
        archi_test, params_test = adjacent(archi, params)
        qualite_test = test(type_model,archi_test, params_test, data)

        if random.random() < np.exp((qualite-qualite_test)/beta):

            print("succes")
            print(str(qualite) + " ->  " + str (qualite_test))
            print("")
            archi = archi_test
            params = params_test
            qualite=qualite_test
            print(params)
            for reseau in archi.keys():
                print(reseau + " : " + str(archi[reseau]["layer_sizes"]))
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



def generer_resultats(beta, data,model,  ite, nom_fichier):
    results = Metropolis_Hasting(beta, data, model, ite, analyse= True)
    with open(nom_fichier, "wb") as f:
        pickle.dump(results, f)

def charger_resultats(nom_fichier):
    with open(nom_fichier, "rb") as f:
        results = pickle.load(f)
    return results

Metropolis_Hasting(0.1, data,TimeGAN, ite = 100, analyse=False)
#generer_resultats(0.1,data, WGAN, ite = 1000, nom_fichier="results_pkl\\resultsWGAN2.pkl")
"""
results=charger_resultats("results_pkl\\resultsWGAN2.pkl")
sorted_results = sorted(results, key=lambda x: x[1])

for i in range (15):
    print(sorted_results[i][0])
    print(sorted_results[i][1])
    archi=sorted_results[i][2]
    params=sorted_results[i][3]
    print("nouvelle estimation : " + str(test(WGAN, archi, params, data)))
    print(params)
    for reseau in archi.keys():
        print(reseau + " : " + str(archi[reseau]["layer_sizes"]) + "     " + str(archi[reseau]["activation"]))
    print("")

"""