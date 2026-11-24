"""Expérience S2 : effet du pas eta sur la descente de gradient.

Premier réflexe expérimental du TIPE : on ne change QU'UNE variable (eta).
Mêmes données, même point de départ (w = 0), même nombre d'itérations.

Lancer depuis S2/ :  python experiences/s2_couts_eta.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Rend src/ visible quand on lance ce script directement (pytest.ini ne sert
# qu'à pytest). Ces 3 lignes sont fournies : pas besoin d'y toucher.
DOSSIER_S2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DOSSIER_S2, "src"))
# from regression_logistique import entrainer


def generer_nuages(n_par_classe, graine=0):
    """Deux nuages gaussiens en 2D : la classe 0 et la classe 1.

    Reçoit : n_par_classe (nombre de points par nuage), graine (pour le hasard).
    Renvoie : X de forme (2 * n_par_classe, 2) et y de forme (2 * n_par_classe,).

    Étapes :
    1. rng = np.random.default_rng(graine)
    2. Classe 0 : rng.normal(loc=..., scale=1.0, size=(n_par_classe, 2)),
       centrée par exemple en (-1, -1). Classe 1 : centrée en (1, 1).
    3. Empilez les deux nuages l'un sous l'autre (np.vstack) pour obtenir X.
    4. y : n_par_classe zéros, puis n_par_classe uns (np.concatenate).

    Valeur de contrôle : avec n_par_classe = 100, X.shape == (200, 2),
    y.shape == (200,) et y.sum() == 100.

    Pour voir les données, tracez-les une fois avec plt.scatter(X[:, 0], X[:, 1], c=y)
    et plt.show(). Les nuages se chevauchent un peu : c'est voulu, le coût ne
    pourra pas descendre à 0.
    """
    pass


def tracer_couts_selon_eta():
    """Trace sur UNE figure les courbes de coût pour eta dans {0.01, 0.1, 1, 10}.

    Étapes :
    1. Générez les données UNE seule fois (avant la boucle) : sinon on change
       deux variables à la fois.
    2. fig, ax = plt.subplots(figsize=(7, 4))
    3. Pour chaque eta de la liste :
       a. w, b, couts = entrainer(X, y, eta, n_iterations=200)
       b. ax.plot(couts, label=f"eta = {eta}")
    4. Titre, ax.set_xlabel("itération"), ax.set_ylabel("coût"), ax.legend().
       Essayez aussi ax.set_yscale("log") : quelle version se lit le mieux ?
    5. Enregistrez dans figures/s2_couts_eta.png :
           chemin = os.path.join(DOSSIER_S2, "figures", "s2_couts_eta.png")
           fig.savefig(chemin, dpi=150)
       puis affichez le chemin avec print.

    À observer et à noter dans le journal (ligne Étonnement) :
    - Quel eta descend le plus vite ? Lequel est trop lent ?
    - eta = 10 diverge-t-il ? Si non, essayez 50 ou 100 : à partir de quand
      la courbe devient-elle irrégulière ?
    - Comparez avec le bonus de S1, où eta ne changeait rien. Pourquoi ici si ?
    """
    pass


if __name__ == "__main__":
    tracer_couts_selon_eta()
