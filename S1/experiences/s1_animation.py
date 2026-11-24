"""Animation : on regarde la droite du perceptron bouger pendant l'apprentissage.

Lancer depuis S1/ :
    python experiences/s1_animation.py              -> ET, OU et XOR côte à côte
    python experiences/s1_animation.py XOR          -> XOR seul
    python experiences/s1_animation.py ET --intervalle 800    -> plus lent (ms par image)
    python experiences/s1_animation.py --gif        -> enregistre figures/s1_animation.gif

Une image = un exemple présenté au perceptron. On voit :
- le fond coloré : ce que le neurone prédit en chaque point du plan (bleu = 0, orange = 1) ;
- le trait noir : la droite w1 x1 + w2 x2 + b = 0 ;
- la flèche : le vecteur w, toujours perpendiculaire à la droite, tourné vers la zone 1 ;
- le cercle autour du point présenté : vert si bien classé, rouge s'il provoque une correction.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

# Rend src/ visible quand on lance ce script directement (pytest.ini ne sert qu'à pytest).
DOSSIER_S1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DOSSIER_S1, "src"))

from neurone import neurone, neurone_batch  # noqa: E402

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
FONCTIONS = {
    "ET": np.array([0, 0, 0, 1]),
    "OU": np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}
ETA = 0.1
BORNES = (-0.5, 1.5)


def etapes_perceptron(X, y, eta, n_epoques):
    """Même règle que entrainer_perceptron (src/perceptron.py), mais on garde une photo
    de l'état après CHAQUE exemple, au lieu de ne renvoyer que les poids finaux.

    Renvoie une liste de dictionnaires : époque, indice i de l'exemple (None pour l'état
    initial), prédiction, écart, w, b, erreurs de l'époque en cours, convergé ou non.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    etapes = [dict(epoque=0, i=None, y_predit=None, ecart=0, w=w.copy(), b=b, nb_erreurs=0, converge=False)]

    for epoque in range(1, n_epoques + 1):
        nb_erreurs = 0
        for i in range(n):
            y_predit = neurone(X[i], w, b)
            ecart = y[i] - y_predit
            if ecart != 0:
                w = w + eta * ecart * X[i]
                b = b + eta * ecart
                nb_erreurs += 1
            # w.copy() : sinon toutes les photos pointeraient vers le même tableau.
            etapes.append(dict(epoque=epoque, i=i, y_predit=int(y_predit), ecart=int(ecart),
                               w=w.copy(), b=b, nb_erreurs=nb_erreurs, converge=False))
        if nb_erreurs == 0:
            etapes[-1]["converge"] = True
            break

    return etapes


def extremites_droite(w, b):
    """Deux points de la droite w1 x1 + w2 x2 + b = 0, ou None si w = 0 (pas de droite)."""
    lo, hi = BORNES
    if abs(w[1]) > 1e-12:  # on peut écrire x2 en fonction de x1
        x1 = np.array([lo, hi])
        return x1, -(w[0] * x1 + b) / w[1]
    if abs(w[0]) > 1e-12:  # droite verticale x1 = -b / w1
        return np.array([-b / w[0]] * 2), np.array([lo, hi])
    return None


class Panneau:
    """Un sous-graphique animé pour une fonction logique."""

    def __init__(self, ax, nom, y, n_epoques):
        self.ax, self.nom, self.y = ax, nom, y
        self.etapes = etapes_perceptron(X, y, ETA, n_epoques)

        # Grille du plan, une fois pour toutes : points de forme (m*m, 2).
        valeurs = np.linspace(*BORNES, 200)
        xx, yy = np.meshgrid(valeurs, valeurs)
        self.forme = xx.shape
        self.points = np.column_stack([xx.ravel(), yy.ravel()])

        self.fond = ax.imshow(np.zeros(self.forme), extent=(*BORNES, *BORNES), origin="lower",
                              cmap=ListedColormap(["#dbe7f5", "#f7d9c4"]), vmin=0, vmax=1)
        (self.droite,) = ax.plot([], [], color="black", linewidth=2)
        self.fleche = None
        for classe, couleur in [(0, "white"), (1, "black")]:
            m = y == classe
            ax.scatter(X[m, 0], X[m, 1], s=140, c=couleur, edgecolors="black", linewidths=1.5, zorder=3)
        (self.anneau,) = ax.plot([], [], "o", markersize=24, markerfacecolor="none", markeredgewidth=3, zorder=4)
        self.texte = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9, family="monospace",
                             va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

        ax.set_xlim(*BORNES)
        ax.set_ylim(*BORNES)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    def afficher(self, k):
        """Dessine l'étape k (la dernière si le panneau a déjà fini)."""
        e = self.etapes[min(k, len(self.etapes) - 1)]
        w, b = e["w"], e["b"]

        self.fond.set_data(neurone_batch(self.points, w, b).reshape(self.forme))

        extremites = extremites_droite(w, b)
        self.droite.set_data(*(extremites if extremites else ([], [])))

        # Flèche w, partant du point de la droite le plus proche du centre (0,5 ; 0,5).
        if self.fleche is not None:
            self.fleche.remove()
            self.fleche = None
        norme2 = w @ w
        if norme2 > 0:
            c = np.array([0.5, 0.5])
            pied = c - (w @ c + b) / norme2 * w  # projection orthogonale de c sur la droite
            u = 0.3 * w / np.sqrt(norme2)
            self.fleche = self.ax.annotate("", xy=pied + u, xytext=pied,
                                           arrowprops=dict(arrowstyle="->", color="#b4471f", lw=2))

        if e["i"] is None:
            self.anneau.set_data([], [])
            detail = "état initial : w = 0, b = 0"
        else:
            self.anneau.set_data([X[e["i"], 0]], [X[e["i"], 1]])
            self.anneau.set_markeredgecolor("#d62728" if e["ecart"] else "#2ca02c")
            action = "correction" if e["ecart"] else "rien à corriger"
            detail = f"exemple {tuple(X[e['i']])} : attendu {self.y[e['i']]}, prédit {e['y_predit']} -> {action}"

        if k >= len(self.etapes) - 1 and e["converge"]:
            statut = "CONVERGÉ : une époque sans erreur"
        elif k >= len(self.etapes) - 1:
            statut = "ARRÊT : nombre maximal d'époques atteint"
        else:
            statut = f"erreurs dans l'époque : {e['nb_erreurs']}"

        self.ax.set_title(f"{self.nom} · époque {e['epoque']}")
        self.texte.set_text(f"{detail}\nw = ({w[0]:+.1f}, {w[1]:+.1f})   b = {b:+.1f}\n{statut}")


def main():
    parser = argparse.ArgumentParser(description="Anime la droite du perceptron pendant l'apprentissage.")
    # Pas de choices= ici : avec nargs="*", certaines versions de Python refusent la liste vide.
    parser.add_argument("fonctions", nargs="*", help="ET, OU et/ou XOR (par défaut : les trois)")
    parser.add_argument("--epoques", type=int, default=15, help="nombre maximal d'époques (15 par défaut)")
    parser.add_argument("--intervalle", type=int, default=400, help="durée d'une image en ms (400 par défaut)")
    parser.add_argument("--gif", action="store_true", help="enregistre figures/s1_animation.gif au lieu d'afficher")
    args = parser.parse_args()

    noms = [nom.upper() for nom in args.fonctions] or list(FONCTIONS)  # rien de demandé : les trois
    inconnus = [nom for nom in noms if nom not in FONCTIONS]
    if inconnus:
        parser.error(f"fonction inconnue : {', '.join(inconnus)} (au choix : ET, OU, XOR)")
    fig, axes = plt.subplots(1, len(noms), figsize=(5.2 * len(noms), 5.6), squeeze=False)
    panneaux = [Panneau(ax, nom, FONCTIONS[nom], args.epoques) for ax, nom in zip(axes[0], noms)]
    fig.suptitle("Règle du perceptron : w ← w + η (y − ŷ) x  et  b ← b + η (y − ŷ)")
    fig.tight_layout()

    # On anime jusqu'à ce que le plus long ait fini, puis on fige 8 images avant de reboucler.
    n_images = max(len(p.etapes) for p in panneaux) + 8

    def image(k):
        for p in panneaux:
            p.afficher(k)

    animation = FuncAnimation(fig, image, frames=n_images, interval=args.intervalle, repeat=True)

    if args.gif:
        chemin = os.path.join(DOSSIER_S1, "figures", "s1_animation.gif")
        animation.save(chemin, writer=PillowWriter(fps=1000 / args.intervalle))
        print(f"Animation enregistrée : {chemin}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
