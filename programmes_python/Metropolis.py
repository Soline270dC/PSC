
import random

from initialisation import *
from GAN import *
from estimation_modele import *


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
        return Architecture(archi.lr, new_couches, archi.couches_discri, new_fct_transi, archi.fct_transi_discri, archi.latent_dim)

    def supprimer_couche_discri(archi):
        n= len(archi.couches_discri)
        k= random.randint(0,n-1)
        new_couches=[]
        new_fct_transi=[]
        for i in range(n):
            if i !=k:
                new_couches.append(archi.couches_discri[i])
                new_fct_transi.append(archi.fct_transi_discri[i])
        return Architecture(archi.lr, archi.couches_gene , new_couches, archi.fct_transi_gene, new_fct_transi, archi.latent_dim)

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
        return Architecture(archi.lr, new_couches, archi.couches_discri, new_fct_transi, archi.fct_transi_discri, archi.latent_dim)

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
        return Architecture(archi.lr, archi.couches_gene , new_couches, archi.fct_transi_gene, new_fct_transi, archi.latent_dim)

    def changer_lr(archi):
        if random.random()<0.5:
            return Architecture(archi.lr/2, archi.couches_gene ,  archi.couches_discri, archi.fct_transi_gene, archi.fct_transi_discri, archi.latent_dim)
        else:
            return Architecture(archi.lr*2, archi.couches_gene ,  archi.couches_discri, archi.fct_transi_gene, archi.fct_transi_discri, archi.latent_dim)
    
    def changer_latent_dim(archi):
        new_latent_dim = random.randint(10,100)
        return Architecture(archi.lr, archi.couches_gene ,  archi.couches_discri, archi.fct_transi_gene, archi.fct_transi_discri, new_latent_dim)

    epsilon = random.random()
    if epsilon < 1/6:
        print("modification du Learning rate")
        return changer_lr(archi)
    elif epsilon < 2/6:
        print("modification de Latent dim")
        return changer_latent_dim(archi)
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



def Metropolis_Hasting(beta, data, ite = 10, analyse= False):
    archi =Architecture (0.0002,[], [],[] ,[], 20)
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
