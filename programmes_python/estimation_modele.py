import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np

from scipy.stats import wasserstein_distance

from initialisation import *
from GAN import *

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


#ESTIMATION ET OPTIMISATION____________________________________________________________________________________
def esti_modele(mon_gan,  nb_points = 1000):
    latent_dim = mon_gan.generator.model[0].in_features
    z = torch.randn(nb_points, latent_dim)
    fake_data = mon_gan.generator(z)
    fake_data_np = fake_data.detach().numpy() 
    return Wassertstein_esti (mon_gan.data.all_data_val_np, fake_data_np)

def test_architecture(archi, data, nb_ite=1):
    # Initialisation des modèles
    s=0
    for i in range(nb_ite):
        mon_gan = GAN( data=data, archi=archi)
        mon_gan.entrainer()
        x=esti_modele(mon_gan)
        s+= x
    return s / nb_ite