"""Expérience S3 : le MLP résout XOR.

1. figure_xor : courbe de coût + frontière de décision -> figures/s3_xor.png (tâche 6)
2. figure_taux_de_reussite : combien de graines réussissent selon la taille de la
   couche cachée -> figures/s3_taux_reussite.png et experiences/s3_taux_reussite.csv
   (BONUS B2)

Lancer depuis S3/ :  python experiences/s3_xor.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Rend src/ visible quand on lance ce script directement (pytest.ini ne sert
# qu'à pytest). Ces 3 lignes sont fournies : pas besoin d'y toucher.
DOSSIER_S3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DOSSIER_S3, "src"))
# from mlp import MLP, entrainer


def donnees_xor():
    """Les 4 points de XOR : X (4, 2) et Y (4, 1).

    Valeur de contrôle : X.shape == (4, 2), Y.shape == (4, 1), Y.ravel() == [0, 1, 1, 0].
    Piège : Y de forme (4,) ferait échouer l'assert de la perte (et c'est tant mieux).
    """
    pass


def tracer_frontiere(mlp, X, Y, ax, titre=""):
    """Dessine sur ax la probabilité prédite par mlp sur une grille, et les points.

    Reçoit : mlp entraîné, X (n, 2), Y (n, 1), ax (un axe matplotlib), titre.
    Ne renvoie rien. Cette fonction resservira pour la spirale (bonus B3).

    Étapes :
    1. Bornes de la grille : de X[:, 0].min() - 0.5 à X[:, 0].max() + 0.5, pareil
       pour la deuxième coordonnée. Pour XOR : le carré [-0.5, 1.5]².
    2. xs = np.linspace(xmin, xmax, 200) et ys de même ;
       A, B = np.meshgrid(xs, ys) : deux matrices (200, 200).
    3. Les 40 000 points de la grille en une seule matrice (40000, 2) :
       points = np.column_stack([A.ravel(), B.ravel()])
       C'est la force de la vectorisation : UN SEUL appel à mlp.avant.
    4. P = mlp.avant(points).reshape(A.shape)  -> (200, 200)
    5. ax.contourf(A, B, P, levels=20, cmap="RdBu_r", alpha=0.8) : la couleur
       donne la probabilité. Ajoutez ax.contour(A, B, P, levels=[0.5],
       colors="k") : la frontière de décision, là où P = 0.5.
    6. Les points : ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="RdBu_r",
       edgecolors="k", s=80).
    7. ax.set_title(titre), ax.set_aspect("equal").

    Valeur de contrôle : points.shape == (40000, 2) et P.shape == (200, 200).

    Comparez avec la figure de XOR de S1 (frontière droite, toujours fausse).
    """
    pass


def figure_xor():
    """Entraîne un 2-4-1 sur XOR et enregistre figures/s3_xor.png (deux panneaux).

    Étapes :
    1. X, Y = donnees_xor() ; mlp = MLP([2, 4, 1], graine=0)
    2. couts = entrainer(mlp, X, Y, eta=0.5, n_iterations=2000)
    3. fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    4. À gauche : ax1.plot(couts), ax1.set_yscale("log"), titres des axes
       (« itération », « coût »). Pourquoi l'échelle log ? Essayez sans.
    5. À droite : tracer_frontiere(mlp, X, Y, ax2, "Frontière du MLP 2-4-1").
    6. fig.tight_layout() ; enregistrez dans figures/s3_xor.png :
           chemin = os.path.join(DOSSIER_S3, "figures", "s3_xor.png")
           fig.savefig(chemin, dpi=150)
       puis print(chemin).

    Valeurs de contrôle (référence) : couts[0] ≈ 0.7742, couts[-1] ≈ 0.0020.

    À observer (journal, ligne Étonnement) : combien de « morceaux de droite »
    forment la frontière ? Chaque neurone caché ReLU ajoute au plus un pli.
    """
    pass


def taux_de_reussite(n_caches, n_graines=20, eta=0.5, n_iterations=2000):
    """BONUS B2. Proportion des graines 0 .. n_graines - 1 pour lesquelles MLP([2, n_caches, 1]) résout XOR.

    Reçoit : n_caches (nombre de neurones cachés), n_graines, eta, n_iterations.
    Renvoie : un float entre 0 et 1.

    Étapes :
    1. X, Y = donnees_xor() ; compteur = 0
    2. Pour graine de 0 à n_graines - 1 : nouveau MLP avec cette graine,
       entraîner, puis ajouter 1 si predire(X) redonne exactement Y.
    3. Renvoyer compteur / n_graines.

    Une seule variable change d'un appel à l'autre (n_caches) : mêmes données,
    mêmes graines, même eta, même nombre d'itérations.

    AVANT de lancer, écrivez votre prédiction dans le journal.
    """
    pass


def figure_taux_de_reussite():
    """BONUS B2. Taux de réussite selon n_caches dans [2, 3, 4, 8, 16].

    Étapes :
    1. Pour chaque n_caches, calculez taux_de_reussite(n_caches) (environ 6 s
       au total : affichez la progression avec print).
    2. Enregistrez un CSV experiences/s3_taux_reussite.csv avec deux colonnes,
       n_caches et taux (une ligne d'en-tête, puis une ligne par valeur ; un
       simple open(..., "w") et f.write suffisent).
    3. Tracez un diagramme en barres (ax.bar) ou une courbe avec marqueurs,
       axe x : neurones cachés, axe y : taux de réussite entre 0 et 1.
       Titre : préciser n_graines, eta et n_iterations (sinon la figure ne se
       reproduit pas).
    4. Enregistrez figures/s3_taux_reussite.png.

    Comparez à votre prédiction, puis seulement ensuite aux valeurs de la
    référence (GUIDE.md, bonus B2).
    """
    pass


if __name__ == "__main__":
    figure_xor()
    # Décommentez pour le bonus B2 :
    # figure_taux_de_reussite()
    plt.show()
