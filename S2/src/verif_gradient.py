"""Brique 4 : vérifier un gradient par différences finies centrées.

Cet outil est GÉNÉRIQUE : il ne connaît rien à la régression logistique.
Il prend n'importe quelle fonction f qui reçoit un vecteur et renvoie un nombre.
On s'en servira jusqu'en S10 pour vérifier tous les gradients à la main.
"""
import numpy as np


def gradient_numerique(f, w, eps=1e-5):
    """Approche le gradient de f en w par différences finies centrées.

    Pour chaque coordonnée i :
        g[i] = ( f(w + eps * e_i) - f(w - eps * e_i) ) / (2 * eps)
    où e_i vaut 1 en position i et 0 ailleurs.

    Reçoit :
    - f : une fonction qui prend un tableau de même forme que w et renvoie un float ;
    - w : le point où on calcule le gradient (tableau de forme (d,)) ;
    - eps : le petit décalage.
    Renvoie : g, un tableau de même forme que w.

    Étapes :
    1. Créez g, un tableau de zéros de même forme que w (np.zeros_like).
       Attention : si w contient des entiers, zeros_like donne des entiers.
       Convertissez w en float au début.
    2. Pour chaque indice i :
       a. faites une COPIE de w et ajoutez eps en position i -> w_plus ;
       b. faites une autre copie et retirez eps en position i -> w_moins ;
       c. g[i] = (f(w_plus) - f(w_moins)) / (2 * eps).
    3. Renvoyez g.

    Valeurs de contrôle :
    - f = lambda v: np.sum(v ** 2), w = np.array([1., 2., 3.])
      -> environ [2., 4., 6.] (le gradient de la somme des carrés est 2w)
    - Après l'appel, w doit valoir encore exactement [1., 2., 3.].

    Pièges :
    - w_plus = w ne fait PAS de copie : les deux noms désignent le même tableau,
      et modifier w_plus modifie w. Utilisez w.copy().
    - Ne prenez pas eps trop petit (1e-12) : les erreurs d'arrondi dominent.
    """
    pass


def erreur_relative(g1, g2):
    """Erreur relative entre deux gradients, pour savoir s'ils sont « égaux ».

    err = ||g1 - g2|| / (||g1|| + ||g2||)

    Reçoit : deux tableaux de même forme.
    Renvoie : un float entre 0 et 1.

    Étapes :
    1. Normes avec np.linalg.norm.
    2. Si le dénominateur vaut 0 (les deux gradients sont nuls), renvoyez 0.0
       pour ne pas diviser par zéro.

    Valeurs de contrôle :
    - erreur_relative(np.array([1., 2.]), np.array([1., 2.])) -> 0.0
    - erreur_relative(np.array([1., 0.]), np.array([0., 1.])) -> 0.7071...

    Lecture : < 1e-6 très bon ; entre 1e-6 et 1e-3 suspect ; > 1e-3 il y a une
    erreur dans le gradient à la main.
    """
    pass
