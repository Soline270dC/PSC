
import random
import copy

from initialisation import *
from GAN import *
from estimation_modele import *


def archi_adj(archi):
    """Renvoie une architecture <<adjacente>> à archi"""
    liste_fct_transi = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]
    new_archi = copy.deepcopy(archi)

    def supprimer_couche(reseau):
        n = len(new_archi.reseaux[reseau]["couches"])
        if n == 0:
            return [], []
        k = random.randint(0, n - 1)
        new_couches = []
        new_fct_transi = []
        for i in range(n):
            if i != k:
                new_couches.append(new_archi.reseaux[reseau]["couches"][i])
                new_fct_transi.append(new_archi.reseaux[reseau]["fct_transi"][i])
        return new_couches, new_fct_transi

    def ajouter_couche(reseau):
        n = len(new_archi.reseaux[reseau]["couches"])
        k = random.randint(0, n)
        new_couches = []
        new_fct_transi = []
        for i in range(n):
            if i == k:
                new_couches.append(random.randint(10, 50))
                new_fct_transi.append(random.choice(liste_fct_transi))
            new_couches.append(new_archi.reseaux[reseau]["couches"][i])
            new_fct_transi.append(new_archi.reseaux[reseau]["fct_transi"][i])
        if k == n:
            new_couches.append(random.randint(10, 50))
            new_fct_transi.append(random.choice(liste_fct_transi))
        return new_couches, new_fct_transi


    def modifier_1_param(param):
   
        variation = random.uniform(0.5, 2.)
        
        if type(new_archi.parameters[param]) == int:
            return int(new_archi.parameters[param] * variation) 
        else:
            return new_archi.parameters[param] * variation 


    epsilon = random.random()
    if epsilon < 1 / 4:
        reseau = random.choice(list(archi.reseaux.keys()))
        new_archi.reseaux[reseau]["couches"], new_archi.reseaux[reseau]["fct_transi"] = supprimer_couche(reseau)
        print("suppression d'une couche du " + reseau)
    elif epsilon < 2 / 4:
        reseau = random.choice(list(archi.reseaux.keys()))
        new_archi.reseaux[reseau]["couches"], new_archi.reseaux[reseau]["fct_transi"] = ajouter_couche(reseau)
        print("ajout d'une couche au " + reseau)
    else:
        param = random.choice(list(new_archi.parameters.keys()))
        new_archi.parameters[param] = modifier_1_param(param)
        print("modification de " + param)

    

    return new_archi



def Metropolis_Hasting(beta, data, ite = 10, analyse= False):
    archi =Architecture ()
    qualite = test_architecture(archi, data)

    resultats=[]


    for i in range(ite):
        print("MH étape : " +str(i))
        archi_test = archi_adj(archi)
        qualite_test = test_architecture(archi_test, data)

        if random.random() < np.exp((qualite-qualite_test)/beta):

            print("succes")
            print(str(qualite) + " ->  " + str (qualite_test))
            print("")
            archi = archi_test
            qualite=qualite_test
            archi.print_archi()
            if analyse:
                resultats.append((i,qualite, archi))
        else:
            print("echec")
            print("")
            print("")

    if analyse:
        return resultats
    else:
        return archi