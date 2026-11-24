"""Tests de verif_gradient.py. Lancer depuis S2/ : python -m pytest -v

C'est le test le plus important de la séance : il valide le gradient à la main
de la régression logistique. Le même schéma servira pour le MLP (S3) et pour
le Transformer (S10).
"""
import numpy as np

# Décommentez au fur et à mesure que les fonctions existent :
# from verif_gradient import gradient_numerique, erreur_relative
# from regression_logistique import cout, gradient


def test_gradient_numerique_somme_des_carres():
    """Sur f(v) = somme des v_i², le gradient est 2v.

    Étapes :
    1. f = lambda v: np.sum(v ** 2)
    2. w = np.array([1., 2., 3.])
    3. np.allclose(gradient_numerique(f, w), 2 * w)
    """
    pass


def test_gradient_numerique_ne_modifie_pas_w():
    """Après l'appel, w n'a pas changé (le piège de la copie).

    Étapes : gardez une copie avant (w_avant = w.copy()), appelez
    gradient_numerique, puis np.array_equal(w, w_avant).
    """
    pass


def test_erreur_relative():
    """erreur_relative vaut 0 pour deux tableaux égaux et 0.7071... pour [1, 0] et [0, 1]."""
    pass


def test_gradient_regression_logistique_en_w():
    """Le gradient à la main en w coïncide avec le gradient numérique.

    Étapes :
    1. rng = np.random.default_rng(0) ; X aléatoire (30, 4) ; y de 0 et 1 ;
       w aléatoire de taille 4 (PAS zéro : en zéro, certaines erreurs passent
       inaperçues) ; b = 0.3.
    2. grad_w, grad_b = gradient(X, y, w, b)
    3. f = lambda v: cout(X, y, v, b)  (on fait varier w, b est fixé)
    4. g_num = gradient_numerique(f, w)
    5. assert erreur_relative(grad_w, g_num) < 1e-6
    """
    pass


def test_gradient_regression_logistique_en_b():
    """Même chose pour b.

    Astuce : gradient_numerique attend un tableau. Faites varier b à travers un
    tableau d'une case : f = lambda v: cout(X, y, w, v[0]), au point np.array([b]).
    Comparez g_num[0] avec grad_b (np.isclose ou erreur_relative).
    """
    pass
