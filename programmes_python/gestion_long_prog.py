import pickle

from Metropolis import *


def generer_resultats(beta, data, ite):
    results = Metropolis_Hasting(beta, data, ite, analyse= True)
    with open("results_pkl\\results3.pkl", "wb") as f:
        pickle.dump(results, f)

def charger_resultats():
    with open("results_pkl\\results2.pkl", "rb") as f:
        results = pickle.load(f)

    #print("Résultats chargés :", results)
    return results