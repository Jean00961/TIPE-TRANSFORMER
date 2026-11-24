"""BONUS B3 : deux spirales imbriquées, un problème bien plus dur que XOR.

On compare trois architectures :
- 2-16-1 : une petite couche cachée (65 paramètres) ;
- 2-300-1 : une seule couche cachée, très large (1201 paramètres) ;
- 2-32-32-1 : deux couches cachées (1185 paramètres, presque autant que 2-300-1).
Une seule chose change : l'architecture. La comparaison 2-300-1 contre 2-32-32-1
est la plus intéressante, car le nombre de paramètres est presque le même.

Lancer depuis S3/ :  python experiences/s3_spirale.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DOSSIER_S3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DOSSIER_S3, "src"))
sys.path.insert(0, os.path.join(DOSSIER_S3, "experiences"))
# from mlp import MLP, entrainer
# from s3_xor import tracer_frontiere


def generer_spirale(n_par_bras, bruit=0.02, graine=0):
    """Deux spirales : la classe 0 et, tournée d'un demi-tour, la classe 1.

    Reçoit : n_par_bras (points par spirale), bruit (écart-type du bruit), graine.
    Renvoie : X (2 * n_par_bras, 2) et Y (2 * n_par_bras, 1).

    Étapes :
    1. rng = np.random.default_rng(graine)
    2. Angles : t = np.linspace(0.25, 3 * np.pi, n_par_bras) (un tour et demi).
       Rayons : r = t / (3 * np.pi) (de presque 0 à 1 : on s'éloigne du centre).
    3. Bras 0 : les points (r cos t, r sin t), empilés en (n_par_bras, 2)
       avec np.column_stack.
    4. Bras 1 : l'opposé du bras 0 (symétrie de centre O = rotation d'un demi-tour).
    5. X = np.vstack des deux bras, puis ajoutez rng.normal(0, bruit, size=X.shape).
    6. Y : n_par_bras zéros puis n_par_bras uns, en forme (2 * n_par_bras, 1).

    Valeurs de contrôle (n_par_bras = 100) : X.shape == (200, 2),
    Y.shape == (200, 1), Y.sum() == 100, et np.abs(X).max() proche de 1.
    Tracez les points une fois avec plt.scatter pour vérifier l'allure.
    """
    pass


def comparer_architectures():
    """Entraîne 2-16-1, 2-300-1 et 2-32-32-1 sur la même spirale, trace les trois frontières.

    Étapes :
    1. X, Y = generer_spirale(100)  (UNE fois, avant la boucle)
    2. fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    3. Pour chaque architecture (et son axe) :
       a. mlp = MLP(architecture, graine=0)
       b. mesurez le temps d'entraînement : import time en haut du fichier,
          debut = time.perf_counter() avant, time.perf_counter() - debut après ;
       c. couts = entrainer(mlp, X, Y, eta=0.5, n_iterations=5000)
       d. precision = proportion de mlp.predire(X) égale à Y (np.mean de la comparaison)
       e. tracer_frontiere(mlp, X, Y, ax, titre) avec dans le titre
          l'architecture, le nombre de paramètres et la précision.
       f. print de l'architecture, du nombre de paramètres, du coût final, de
          la précision et du temps.
    4. Enregistrez figures/s3_spirale.png.

    Valeurs de contrôle (référence, graine 0 ; les temps dépendent du poste) :
    - 2-16-1    :   65 paramètres, précision 0.775, coût final ≈ 0.51,   ≈ 0.3 s
    - 2-300-1   : 1201 paramètres, précision 1.0,   coût final ≈ 0.020,  ≈ 2 s
    - 2-32-32-1 : 1185 paramètres, précision 1.0,   coût final ≈ 0.0006, ≈ 0.6 s

    À noter dans le journal : à nombre de paramètres presque égal, le réseau
    profond atteint un coût 30 fois plus bas, 3 fois plus vite. Pourquoi ?
    (chaque couche « replie » l'espace produit par la précédente : les plis se
    composent au lieu de s'additionner). Et pourquoi 2-300-1 est-il plus lent
    alors qu'il a autant de paramètres ? (comptez les activations cachées par
    exemple : 300 contre 32 + 32 = 64 ; la ReLU, son masque et les produits terme
    à terme se font sur chacune d'elles)
    """
    pass


if __name__ == "__main__":
    comparer_architectures()
    plt.show()
