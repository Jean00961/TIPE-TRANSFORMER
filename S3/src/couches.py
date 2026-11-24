"""Brique 5 : les couches d'un réseau de neurones, chacune avec son forward et son backward.

Toutes les couches suivent le même contrat (c'est ce qui permet de les empiler) :
- avant(X)   : calcule la sortie ET met en cache ce dont arriere aura besoin ;
- arriere(G) : reçoit G = dL/d(sortie), de même forme que la sortie,
               renvoie dL/d(entrée), de même forme que l'entrée.
               Une couche qui a des paramètres stocke aussi leurs gradients
               (self.dW, self.db).

Convention de formes (la même qu'en S1 et S2) : une LIGNE par exemple.
X (n, d_entree), W (d_entree, d_sortie), b (d_sortie,), Z (n, d_sortie).

Voir GUIDE.md, parties 3.2 à 3.5, pour la théorie et les dérivations.
"""
import numpy as np


def sigmoide(z):
    """Sigmoïde stable numériquement : recopiez votre version de S2.

    Reçoit : un nombre ou un tableau de n'importe quelle forme.
    Renvoie : un tableau de même forme, à valeurs dans [0, 1].

    Valeurs de contrôle (les mêmes qu'en S2) :
    - sigmoide(0) -> 0.5
    - sigmoide(2) -> 0.8807970...
    - sigmoide(np.array([-1000., 0., 1000.])) -> [0, 0.5, 1], SANS warning

    Elle doit maintenant marcher aussi sur une matrice (n, 1) : vérifiez
    sigmoide(np.zeros((3, 1))).shape == (3, 1).
    """
    pass


class Lineaire:
    """Couche linéaire (dite « dense » ou « entièrement connectée ») : Z = X @ W + b.

    C'est d_sortie neurones de S1 côte à côte : la colonne j de W est le vecteur
    de poids du neurone j.

    Attributs après __init__ : self.W (d_entree, d_sortie), self.b (d_sortie,).
    Attribut après avant     : self.X (le cache).
    Attributs après arriere  : self.dW (même forme que W), self.db (même forme que b).
    """

    def __init__(self, d_entree, d_sortie, rng):
        """Initialisation de He : W ~ N(0, 2 / d_entree), b = 0.

        Reçoit : les deux dimensions et rng, un générateur np.random.default_rng.
        Le MLP crée UN SEUL rng et le passe à toutes ses couches : c'est ce qui
        rend l'entraînement reproductible.

        Étapes :
        1. self.W = rng.normal(0.0, np.sqrt(2.0 / d_entree), size=(d_entree, d_sortie))
           Écrivez EXACTEMENT cet appel (même ordre des arguments, même forme) :
           sinon vos poids diffèrent de ceux de la solution de référence et les
           valeurs de contrôle de mlp.py ne correspondront pas.
        2. self.b = vecteur de d_sortie zéros.

        Valeurs de contrôle :
        - Lineaire(3, 2, np.random.default_rng(0)).W.shape -> (3, 2), b.shape -> (2,)
        - Lineaire(1000, 500, rng).W.std() -> environ 0.0447 (= racine de 2/1000)

        Pièges :
        - rng.normal prend l'ÉCART-TYPE, pas la variance : np.sqrt(2.0 / d_entree).
        - Ne pas créer un nouveau rng ici avec une graine fixe : toutes les
          couches de même taille auraient alors les mêmes poids.
        """
        pass

    def avant(self, X):
        """Forward : renvoie Z = X @ W + b, et garde X en cache.

        Reçoit : X (n, d_entree).
        Renvoie : Z (n, d_sortie).

        Étapes :
        1. self.X = X (le cache : arriere en aura besoin pour dW).
        2. Calculez Z. b, de forme (d_sortie,), est ajouté à chaque ligne de
           X @ W grâce au broadcasting.

        Valeur de contrôle (exemple de GUIDE.md, partie 3.3) :
        - X = np.ones((4, 3)), W = [[1, 2], [3, 4], [5, 6]], b = [0.5, -0.5]
          (on remplace couche.W et couche.b à la main après la création)
          -> chaque ligne de Z vaut [9.5, 11.5]
        """
        pass

    def arriere(self, G):
        """Backward : stocke dW et db, renvoie dX.

        Reçoit : G = dL/dZ, de forme (n, d_sortie).
        Renvoie : dL/dX, de forme (n, d_entree).

        Formules (GUIDE.md, partie 3.3) :
        - self.dW = X^T G        (d_entree, d_sortie)
        - self.db = somme des lignes de G   (d_sortie,)
        - renvoyer G W^T         (n, d_entree)

        Étapes :
        1. Récupérez X dans le cache (self.X).
        2. Calculez et STOCKEZ self.dW et self.db.
        3. Calculez et RENVOYEZ dX.
        4. La première fois, affichez les trois formes avec print et comparez à
           la colonne de droite ci-dessus. Puis retirez les print.

        Valeurs de contrôle (suite de l'exemple de avant, avec G = np.ones((4, 2))) :
        - dW vaut 4 partout, forme (3, 2)
        - db = [4., 4.]
        - chaque ligne de dX vaut [3., 7., 11.]

        Pièges :
        - G.sum(axis=0), pas G.sum() : on veut un vecteur, pas un nombre.
        - Pas de division par n ici : le 1/n est déjà dans G (il vient de la perte).
        - G.T @ X a la forme (d_sortie, d_entree) : c'est la transposée de dW.
        """
        pass


class ReLU:
    """Activation ReLU, coefficient par coefficient : A = max(0, Z).

    Pas de paramètre, donc pas de dW ni de db.
    Attribut après avant : self.masque (tableau de booléens, même forme que Z).
    """

    def avant(self, Z):
        """Forward : renvoie max(0, Z) et garde le masque Z > 0 en cache.

        Reçoit : Z de forme quelconque. Renvoie : un tableau de même forme.

        Étapes :
        1. self.masque = Z > 0
        2. Renvoyez Z * self.masque (True compte pour 1, False pour 0),
           ou np.maximum(0, Z) : les deux donnent la même chose.

        Valeur de contrôle :
        - Z = np.array([[-1., 2.], [0., -3.]]) -> [[0., 2.], [0., 0.]]

        Piège : np.max(0, Z) n'est pas np.maximum(0, Z). np.max cherche le plus
        grand élément d'UN tableau ; np.maximum compare deux tableaux case par case.
        """
        pass

    def arriere(self, G):
        """Backward : le gradient passe là où le neurone était actif, il est coupé ailleurs.

        Reçoit : G = dL/dA, même forme que Z. Renvoie : dL/dZ = G * masque.

        Valeur de contrôle (suite de l'exemple ci-dessus) :
        - G = np.ones((2, 2)) -> [[0., 1.], [0., 0.]]
          (en Z = 0, la dérivée vaut 0 par convention : c'est ce que donne Z > 0)

        Piège : G @ masque est un produit MATRICIEL, faux ici. L'activation agit
        case par case, son backward aussi : produit terme à terme, donc *.
        """
        pass


class Sigmoide:
    """Activation sigmoïde, coefficient par coefficient : P = sigma(Z).

    Utilisée en sortie du MLP pour obtenir une probabilité.
    Attribut après avant : self.P (la sortie, même forme que Z).
    """

    def avant(self, Z):
        """Forward : renvoie P = sigmoide(Z) et garde P en cache.

        Valeur de contrôle : Z = np.zeros((2, 1)) -> [[0.5], [0.5]]

        Pourquoi mettre en cache P et pas Z ? Parce que sigma'(z) = P (1 - P) :
        avec P, le backward n'a pas besoin de recalculer d'exponentielle.
        """
        pass

    def arriere(self, G):
        """Backward : dL/dZ = G * P * (1 - P).

        Valeur de contrôle (suite) : G = np.ones((2, 1)) -> [[0.25], [0.25]]

        Piège : utilisez le P du cache, ne rappelez pas sigmoide.
        """
        pass


class EntropieCroiseeBinaire:
    """Perte d'entropie croisée binaire, moyennée sur les n exemples.

    L = - moyenne( Y ln P + (1 - Y) ln(1 - P) )

    C'est la fin du graphe de calcul : avant renvoie un NOMBRE, et arriere ne
    reçoit rien (dL/dL = 1) et renvoie dL/dP.

    Attributs après avant : self.P (bornée) et self.Y.
    """

    def avant(self, P, Y):
        """Calcule la perte (un float).

        Reçoit : P (n, 1), les probabilités prédites ; Y (n, 1), les étiquettes 0/1.
        Renvoie : un float positif.

        Étapes :
        1. Vérifiez que P et Y ont la même forme :
               assert P.shape == Y.shape, f"formes différentes : {P.shape} et {Y.shape}"
           C'est LA protection contre le piège n°1 de la séance (GUIDE.md, 3.5).
        2. Bornez P dans [1e-12, 1 - 1e-12] avec np.clip, gardez-la dans self.P.
        3. Gardez Y dans self.Y.
        4. Calculez la moyenne et renvoyez-la avec float(...).

        Valeurs de contrôle :
        - P = [[0.5], [0.5]], Y = [[0.], [1.]] -> ln 2 = 0.693147...
        - P = [[0.9], [0.2]], Y = [[1.], [0.]] -> environ 0.1643
          (vérifiez à la main : -(ln 0.9 + ln 0.8) / 2)
        - P de forme (2, 1) et Y de forme (2,) -> AssertionError
        """
        pass

    def arriere(self):
        """Renvoie dL/dP = (P - Y) / (P (1 - P)) / n, de forme (n, 1).

        Étapes :
        1. Récupérez P, Y dans le cache et n = nombre de lignes de Y.
        2. Appliquez la formule (dérivée en GUIDE.md, partie 3.5).

        Valeurs de contrôle :
        - après avant(P = [[0.5], [0.5]], Y = [[0.], [1.]]) -> [[1.], [-1.]]
          (0.5 / 0.25 / 2 = 1)
        - Composé avec Sigmoide.arriere, on doit retrouver (P - Y) / n :
          c'est le résultat clé de S2, retrouvé par composition. Voir le test
          test_sigmoide_puis_perte_redonne_p_moins_y.

        Remarque : diviser par P (1 - P) est fragile quand P est proche de 0 ou 1
        (d'où le np.clip dans avant). En S4, on fusionnera activation et perte en
        une seule brique pour éviter ce problème.
        """
        pass
