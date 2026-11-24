"""Frontières de décision du perceptron sur ET, OU et XOR.

Lancer depuis S1/ :  python experiences/s1_frontieres.py
La figure est enregistrée dans figures/s1_frontieres.png.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Rend src/ visible quand on lance ce script directement (pytest.ini ne sert qu'à pytest).
DOSSIER_S1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DOSSIER_S1, "src"))

from neurone import neurone_batch  # noqa: E402
from perceptron import entrainer_perceptron  # noqa: E402

# Les 4 entrées possibles, communes aux trois fonctions logiques : forme (4, 2).
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Les sorties attendues : forme (4,).
FONCTIONS = {
    "ET": np.array([0, 0, 0, 1]),
    "OU": np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}

ETA = 0.1
N_EPOQUES = 100


def grille(debut=-0.5, fin=1.5, pas=0.01):
    """Construit la grille de points du plan [debut ; fin]^2.

    Renvoie xx et yy de forme (m, m), et points de forme (m*m, 2) : une ligne
    par point de la grille, comme les lignes de X, pour pouvoir appeler neurone_batch.
    """
    valeurs = np.arange(debut, fin + pas, pas)
    xx, yy = np.meshgrid(valeurs, valeurs)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, points


def tracer_frontiere(ax, w, b, y, titre):
    """Trace sur ax la zone prédite 0 / 1 par le neurone (w, b), et les 4 points.

    - fond coloré : sortie du neurone en chaque point de la grille (contourf) ;
    - trait noir : la droite w1 x1 + w2 x2 + b = 0, c'est-à-dire la frontière ;
    - points : la vraie classe y (rond blanc = 0, rond noir = 1).
    """
    xx, yy, points = grille()
    # neurone_batch renvoie (m*m,) : on le remet en (m, m) pour contourf.
    sorties = neurone_batch(points, w, b).reshape(xx.shape)

    ax.contourf(xx, yy, sorties, levels=[-0.5, 0.5, 1.5], colors=["#dbe7f5", "#f7d9c4"])
    # La frontière : là où la sortie passe de 0 à 1.
    ax.contour(xx, yy, sorties, levels=[0.5], colors="black", linewidths=1.5)

    for classe, couleur in [(0, "white"), (1, "black")]:
        masque = y == classe
        ax.scatter(
            X[masque, 0], X[masque, 1],
            s=120, c=couleur, edgecolors="black", linewidths=1.5, zorder=3,
            label=f"classe {classe}",
        )

    ax.set_title(titre)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")


def tracer_frontieres():
    """Entraîne le perceptron sur ET, OU et XOR, puis trace côte à côte les trois
    frontières : grille np.meshgrid sur [-0,5 ; 1,5]^2, plt.contourf de
    neurone_batch sur la grille, et les 4 points colorés selon leur classe.
    Enregistre la figure dans figures/s1_frontieres.png."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (nom, y) in zip(axes, FONCTIONS.items()):
        w, b, erreurs = entrainer_perceptron(X, y, ETA, N_EPOQUES)
        # erreurs[-1] compte les erreurs PENDANT la dernière époque (les poids bougent
        # encore) ; ici on compte les points mal classés avec les poids FINAUX.
        mal_classes = int(np.sum(neurone_batch(X, w, b) != y))
        print(f"{nom:>3} : w = {w}, b = {b:.2f}, {len(erreurs)} époques, "
              f"{erreurs[-1]} erreur(s) pendant la dernière, {mal_classes} point(s) mal classé(s) à la fin")
        titre = f"{nom} : {mal_classes} point(s) mal classé(s) sur 4\naprès {len(erreurs)} époque(s)"
        tracer_frontiere(ax, w, b, y, titre)

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Perceptron : un neurone = une droite. ET et OU sont séparables, XOR ne l'est pas.")
    fig.tight_layout()

    chemin = os.path.join(DOSSIER_S1, "figures", "s1_frontieres.png")
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    print(f"Figure enregistrée : {chemin}")


if __name__ == "__main__":
    tracer_frontieres()
