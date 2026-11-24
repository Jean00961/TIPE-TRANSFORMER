"""Brique 4 (version 2) : vérifier un gradient par différences finies centrées.

Même outil qu'en S2, mais il doit maintenant accepter des MATRICES (les poids W
d'une couche sont de forme (d_entree, d_sortie)). À partir de S3, c'est cette
version-ci qui fait référence ; celle de S2/ reste comme trace.

Tâche 1 du GUIDE : recopiez le corps de vos fonctions de S2/src/verif_gradient.py,
puis généralisez gradient_numerique avec np.ndindex.
"""
import numpy as np


def gradient_numerique(f, w, eps=1e-5):
    """Approche le gradient de f en w par différences finies centrées.

    Pour chaque indice (une case du tableau, quelle que soit sa forme) :
        g[indice] = ( f(w + eps * e_indice) - f(w - eps * e_indice) ) / (2 * eps)
    où e_indice vaut 1 dans cette case et 0 partout ailleurs.

    Reçoit :
    - f : une fonction qui prend un tableau de même forme que w et renvoie un float ;
    - w : le point où on calcule le gradient, de forme QUELCONQUE :
      (d,) pour un vecteur, (d_entree, d_sortie) pour une matrice de poids ;
    - eps : le petit décalage (1e-5).
    Renvoie : g, un tableau de même forme que w.

    Étapes :
    1. Convertissez w en tableau de flottants (np.asarray(w, dtype=float)).
    2. Créez g = np.zeros_like(w).
    3. Parcourez tous les indices avec « for indice in np.ndindex(w.shape): ».
       Pour un vecteur (3,), indice vaut (0,), (1,), (2,).
       Pour une matrice (2, 3), indice vaut (0, 0), (0, 1), ..., (1, 2).
       Dans les deux cas, w_plus[indice] désigne UNE case.
    4. Pour chaque indice : w_plus = copie de w, avec + eps dans la case ;
       w_moins = autre copie, avec - eps dans la case ;
       g[indice] = (f(w_plus) - f(w_moins)) / (2 * eps).
    5. Renvoyez g.

    Valeurs de contrôle :
    - f = lambda v: np.sum(v ** 2), w = np.array([1., 2., 3.])
      -> environ [2., 4., 6.] (comme en S2 : rien ne doit avoir cassé)
    - f = lambda M: np.sum(M ** 2), M = np.array([[1., 2., 3.], [4., 5., 6.]])
      -> environ [[2., 4., 6.], [8., 10., 12.]], de forme (2, 3)
    - Après l'appel, w et M n'ont pas changé.

    Pièges :
    - « for i in range(len(w)) » ne marche pas pour une matrice : len(M) vaut 2
      (le nombre de lignes) et M[i] est une ligne entière, pas une case.
    - w_plus = w ne fait PAS de copie (piège de S2, toujours valable) : .copy().
    - Cette fonction appelle f 2 fois par case : sur une matrice (100, 100),
      cela fait 20 000 appels. On ne s'en sert que sur de PETITES matrices.
    """
    pass


def erreur_relative(g1, g2):
    """Erreur relative entre deux gradients : ||g1 - g2|| / (||g1|| + ||g2||).

    Identique à S2 : recopiez votre version. Elle marche déjà pour des matrices,
    car np.linalg.norm d'une matrice est la racine de la somme des carrés de
    tous ses coefficients (norme de Frobenius).

    Reçoit : deux tableaux de même forme (vecteurs ou matrices).
    Renvoie : un float entre 0 et 1 (0.0 si les deux tableaux sont nuls).

    Valeurs de contrôle :
    - erreur_relative(np.array([1., 2.]), np.array([1., 2.])) -> 0.0
    - erreur_relative(np.array([1., 0.]), np.array([0., 1.])) -> 0.7071...
    - erreur_relative(np.eye(2), np.eye(2)) -> 0.0

    Lecture : < 1e-6 très bon ; entre 1e-6 et 1e-3 suspect ; > 1e-3 il y a une
    erreur dans le gradient à la main. En S3, avec un backward juste, on obtient
    environ 1e-11.
    """
    pass
