"""Tests de verif_gradient.py (version 2, pour les matrices). Lancer depuis S3/ :
python -m pytest -v tests/test_verif_gradient.py

Les deux premiers tests sont ceux de S2 : recopiez-les. Les deux suivants
vérifient la nouveauté de S3 (les matrices).
Rappel : pour comparer des flottants, np.isclose (un nombre) ou np.allclose
(des tableaux), jamais ==.
"""
import numpy as np

# Décommentez quand les fonctions existent :
# from verif_gradient import gradient_numerique, erreur_relative


def test_vecteur_somme_des_carres():
    """Comme en S2 : sur f(v) = somme des v_i², le gradient est 2v.

    Étapes : f = lambda v: np.sum(v ** 2) ; w = np.array([1., 2., 3.]) ;
    assert np.allclose(gradient_numerique(f, w), 2 * w).
    """
    pass


def test_erreur_relative():
    """Comme en S2 : 0 pour deux tableaux égaux, 0.7071... pour [1, 0] et [0, 1].

    Ajoutez un cas avec des matrices : erreur_relative(np.eye(2), np.eye(2)) == 0.
    """
    pass


def test_matrice_somme_des_carres():
    """Sur f(M) = somme des M_ij², le gradient est 2M, et il a la forme de M.

    Étapes :
    1. M = np.array([[1., 2., 3.], [4., 5., 6.]])
    2. g = gradient_numerique(lambda A: np.sum(A ** 2), M)
    3. assert g.shape == (2, 3)
    4. assert np.allclose(g, 2 * M)
    """
    pass


def test_matrice_non_modifiee_et_produit():
    """Deux vérifications sur une fonction moins symétrique.

    f(M) = somme des C_ij * M_ij avec C une matrice fixe : le gradient vaut C
    exactement (f est linéaire en M). C'est l'astuce qui servira pour tester
    les couches (GUIDE.md, partie 3.9).

    Étapes :
    1. rng = np.random.default_rng(0) ; M = rng.normal(size=(3, 2)) ;
       C = rng.normal(size=(3, 2)) ; M_avant = M.copy()
    2. g = gradient_numerique(lambda A: np.sum(C * A), M)
    3. assert np.allclose(g, C)
    4. assert np.array_equal(M, M_avant)   (M n'a pas été modifiée)
    """
    pass
