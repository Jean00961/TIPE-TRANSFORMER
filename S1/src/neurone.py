"""Brique 1 : le neurone y = phi(w.x + b)."""
import numpy as np


def heaviside(z):
    """Fonction échelon : renvoie 1 si z >= 0, sinon 0.
    Doit marcher aussi bien sur un nombre que sur un tableau (np.where)."""
    return np.where(z >= 0, 1, 0)  # CORRIGÉ : le if/else plantait sur un tableau, np.where marche sur un nombre et un tableau


def neurone(x, w, b):
    """Calcule la sortie du neurone pour UN seul exemple x (taille d).
    Utilise une boucle explicite sur les d coordonnées, sans l'opérateur @,
    puis applique heaviside."""

    n = len(x)
    somme = 0  
    for k in range(n):
        somme += x[k] * w[k]  

    return heaviside(somme + b)  


def neurone_batch(X, w, b):
    """Calcule la sortie du neurone pour les n exemples d'un coup.
    X de forme (n, d), w de forme (d,), b scalaire : utilise X @ w + b,
    puis heaviside. Renvoie un tableau de forme (n,)."""

    res = X @ w + b

    return heaviside(res)  
