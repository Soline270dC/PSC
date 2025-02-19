import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from initialisation import *
from GAN import *
from estimation_modele import *
from Metropolis import *
from gestion_long_prog import *


#PRESENTATION DES RESULTATS__________________________________________________________________________
def courbes_discri(generator, data, latent_dim):
    z = torch.randn(10000, latent_dim)
    fake_data = generator(z)
    fake_data_np = fake_data.detach().numpy()
    fig, axes = plt.subplots(4, 2, figsize=(12, 16)) 
    for i in range(4):
        if i ==0:
            v_proj= [1,0,0,0]
        elif i==1:
            v_proj= [0,1,0,0]
        elif i ==2:
            v_proj= [0,0,1,0]
        else:
            v_proj= [0,0,0,1]
        dr= np.dot(data.all_data_val_np,v_proj)
        ds= np.dot(fake_data_np,v_proj)
        sns.kdeplot(dr, fill=True, ax=axes[i,0])
        axes[i,0].set_title("Réelles")
        axes[i,0].set_xlim(-5, 20)
        sns.kdeplot(ds, fill=True, ax=axes[i,1])
        axes[i,1].set_title("Simulées")
        axes[i,1].set_xlim(-5, 20)
    plt.tight_layout()
    plt.show()

#ZONE DE TEST________________________________________________________________________________________

data = init_data()
"""
archi =Architecture (0.0008,[28], [45, 49, 43, 19, 36, 36, 10, 10, 13, 21],[nn.Sigmoid()] ,[nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.ReLU()],10)
generator,discriminator = Generator(archi) , Discriminator(archi)
generator, discriminator = entrainement(generator,discriminator, num_epochs, archi.lr)


courbes_discri(generator, archi.latent_dim)
"""
"""
archi =Architecture (0.0002,[128,256,128], [128,64],[nn.ReLU(),nn.ReLU(),nn.ReLU()] ,[nn.LeakyReLU(0.2), nn.LeakyReLU(0.2)],10)
generator,discriminator = Generator(archi) , Discriminator(archi)

list_res=[]
for i in range(10):
    generator, discriminator = entrainement(generator,discriminator, num_epochs, archi.lr)
    x=esti_modele(generator)
    list_res.append(x)
print(list_res)
"""


#Metropolis_Hasting(0.1, data, ite = 500)
#generer_resultats(0.1, ite = 10000)
"""
results=charger_resultats()
sorted_results = sorted(results, key=lambda x: x[1])
data = init_data()


for i in range(15):
    print(sorted_results[i][0])
    print(sorted_results[i][1])
    archi=sorted_results[i][2]
    print("nouvelle estimation : " + str(test_architecture(archi, data, nb_ite=10)))
    archi.print_archi()
    print("")
"""

"""
# Préparer les points pour le plot
x = [0,results[0][0]]
y = [0,0]

for i in range(len(results)):
    start = results[i][0]  # Début de l'intervalle
    value = len(results[i][2].couches_discri)  # Valeur dans l'intervalle
    print(value)
    if i > 0:
        # Ajouter un point au début du nouvel intervalle pour continuité
        x.append(start)
        y.append (len(results[i][2].couches_discri))
    
    # Ajouter le point final de l'intervalle
    x.append(start)
    y.append(value)

# Ajouter un point final pour la dernière valeur
x.append(3000)  # Allonger l'intervalle artificiellement
y.append(value)

# Plot
plt.step(x, y, where="post")
plt.xlabel("itération")
plt.ylabel("Nombre de couches")
plt.grid(True)
plt.legend()
plt.show()
"""