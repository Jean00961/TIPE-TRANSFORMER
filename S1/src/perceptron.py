"""Brique 2 : le perceptron et sa règle d'apprentissage."""
import numpy as np
from neurone import neurone


def entrainer_perceptron(X, y, eta, n_epoques):
    """Entraîne un perceptron avec la règle w <- w + eta * (y - y_chapeau) * x.
    Part de w = 0 et b = 0, parcourt les exemples époque après époque, et
    s'arrête dès qu'une époque se termine sans erreur.
    Renvoie w, b et la liste erreurs (nombre d'erreurs à chaque époque)."""

    n,d = X.shape
    w=np.zeros(d) ## crée une liste de 0 de dimension d

    b = 0.0
    erreurs = []

    for epoque in range(n_epoques):

        nb_erreurs = 0

        for i in range(n):

            y_predit = neurone(X[i],w,b)
            ecart = y[i] - y_predit
            if ecart != 0:
                w = w + eta * ecart * X[i]
                b = b + eta*ecart
                nb_erreurs += 1

        erreurs.append(nb_erreurs)

        if nb_erreurs == 0:  # époque sans erreur : on a convergé
            break

    return w,b,erreurs
