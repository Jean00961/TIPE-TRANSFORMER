"""Brique 6 : le perceptron multicouche (MLP), un simple empilement de couches.

Le MLP ne contient AUCUNE formule de dérivée : toutes sont dans couches.py.
Son seul travail est d'appeler les couches dans le bon ordre :
- avant   : de la première à la dernière ;
- arriere : de la dernière à la première ;
- maj     : descendre le gradient de chaque couche qui a des paramètres.

Exemple : MLP([2, 4, 1]) = Lineaire(2, 4) -> ReLU -> Lineaire(4, 1) -> Sigmoide.
Voir GUIDE.md, partie 3.6, pour le graphe de calcul et toutes les formes.
"""
import numpy as np

from couches import EntropieCroiseeBinaire, Lineaire, ReLU, Sigmoide


class MLP:
    """Réseau de couches linéaires séparées par des ReLU, avec une sigmoïde en sortie.

    Attribut : self.couches, la liste des couches dans l'ordre du forward.
    """

    def __init__(self, tailles, graine=0):
        """Construit la liste des couches.

        Reçoit :
        - tailles : la liste des dimensions, par exemple [2, 4, 1] (2 entrées,
          4 neurones cachés, 1 sortie) ou [2, 32, 32, 1] (deux couches cachées) ;
        - graine : pour np.random.default_rng.

        Étapes :
        1. rng = np.random.default_rng(graine), UNE seule fois, ici.
        2. self.couches = liste vide.
        3. Pour i de 0 à len(tailles) - 2 :
           a. ajoutez Lineaire(tailles[i], tailles[i + 1], rng) ;
           b. si ce n'est PAS la dernière couche linéaire, ajoutez ReLU() ;
              sinon, ajoutez Sigmoide().
           Combien y a-t-il de couches linéaires pour tailles = [2, 4, 1] ?
           Et de couches au total ? (réponse : 2 et 4)

        Valeurs de contrôle, avec MLP([2, 4, 1], graine=0) :
        - [type(c).__name__ for c in mlp.couches]
          -> ['Lineaire', 'ReLU', 'Lineaire', 'Sigmoide']
        - mlp.couches[0].W[0] -> environ [0.1257, -0.1321, 0.6404, 0.1049]
        - mlp.couches[2].W[0] -> environ [-0.4976]

        Piège : créer les couches linéaires dans un autre ordre (ou recréer rng)
        donne d'autres poids : le réseau marchera, mais les valeurs de contrôle
        ne correspondront plus.
        """
        pass

    def avant(self, X):
        """Forward : fait passer X dans chaque couche, dans l'ordre.

        Reçoit : X (n, tailles[0]).
        Renvoie : P (n, tailles[-1]), les probabilités de sortie.

        Étapes : une boucle sur self.couches ; la sortie d'une couche est
        l'entrée de la suivante.

        Valeur de contrôle (graine 0, X = les 4 points de XOR, voir test_mlp.py) :
        -> environ [[0.5], [0.2951], [0.4154], [0.2628]]
        Le premier vaut exactement 0.5 : pourquoi ? (x = (0, 0) et b = 0 au départ)
        """
        pass

    def arriere(self, G):
        """Backward : fait remonter G dans chaque couche, dans l'ordre INVERSE.

        Reçoit : G = dL/dP, de même forme que la sortie (n, tailles[-1]).
        Renvoie : dL/dX (n, tailles[0]) (inutile pour l'entraînement, mais pratique
        pour tester).

        Étapes : une boucle sur reversed(self.couches) ; ce que renvoie l'arriere
        d'une couche est le G de la couche d'avant.

        Après l'appel, chaque couche linéaire a ses gradients dans .dW et .db.

        Piège : passer le G de départ à toutes les couches au lieu de faire
        circuler le résultat. Les tests de couches isolées passent, mais le test
        de gradient du MLP entier échoue.
        """
        pass

    def maj(self, eta):
        """Un pas de descente de gradient sur tous les paramètres.

        Étapes : pour chaque couche de self.couches, si c'est une Lineaire
        (isinstance(couche, Lineaire)), faites W <- W - eta dW et b <- b - eta db.

        Piège : le signe. On DESCEND la pente : moins.
        """
        pass

    def nombre_parametres(self):
        """Nombre total de poids et de biais (un entier).

        Étapes : additionnez W.size + b.size pour chaque couche linéaire.

        Valeurs de contrôle :
        - MLP([2, 4, 1]) -> 17
        - MLP([2, 32, 32, 1]) -> 1185
        - MLP([64, 128, 10]) -> 9610 (le réseau de S4 ; la dernière activation
          sera alors un softmax, mais le nombre de paramètres ne change pas)
        """
        pass

    def predire(self, X):
        """Classe prédite (0 ou 1) pour chaque exemple : 1 si P > 0.5.

        Reçoit : X (n, tailles[0]).
        Renvoie : un tableau d'entiers de forme (n, 1).

        Étapes : P = self.avant(X), puis (P > 0.5).astype(int).
        """
        pass


def entrainer(mlp, X, Y, eta, n_iterations):
    """Entraîne mlp par descente de gradient sur tout le jeu (X, Y) à chaque pas.

    Reçoit : mlp (un MLP), X (n, d), Y (n, 1), eta (le pas), n_iterations.
    Renvoie : couts, la liste des n_iterations valeurs de la perte, chacune
    calculée AVANT la mise à jour (comme en S2).

    Étapes :
    1. perte = EntropieCroiseeBinaire(), couts = liste vide.
    2. À chaque itération :
       a. P = mlp.avant(X)
       b. ajoutez perte.avant(P, Y) à couts
       c. mlp.arriere(perte.arriere())
       d. mlp.maj(eta)
    3. Renvoyez couts.

    Valeurs de contrôle (MLP([2, 4, 1], graine=0), XOR, eta = 0.5, 2000 itérations) :
    - couts[0] -> environ 0.7742
    - couts[100] -> environ 0.2019
    - couts[500] -> environ 0.0118
    - couts[-1] -> environ 0.0020
    - mlp.avant(X) ensuite -> environ [[0.006], [0.999], [0.999], [0.001]]

    Pièges :
    - Si les coûts restent vers 0.693 (ln 2) : vérifiez que Y est de forme
      (n, 1) et non (n,) (l'assert de la perte doit l'attraper).
    - Si les coûts augmentent : signe de la mise à jour dans maj.
    - Certaines graines n'arrivent pas à résoudre XOR (bonus B2 du GUIDE) :
      ce n'est pas forcément un bug. Testez d'abord avec graine = 0.
    """
    pass
