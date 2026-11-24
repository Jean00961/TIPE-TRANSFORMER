"""Animation fluide : un réseau de neurones apprend une frontière non linéaire, en direct.

Lancer depuis la racine du dépôt :
    python demos/animation_reseau.py                          -> spirale, réseau 2-16-16-1
    python demos/animation_reseau.py --donnees lunes          -> lunes, cercles, xor ou spirale
    python demos/animation_reseau.py --donnees cercles --couches 3   -> 3 neurones cachés : trop peu ?
    python demos/animation_reseau.py --vitesse 1              -> 1 itération par image : plus lent, plus doux

Pendant l'animation :
    Espace        pause / reprise
    N             nouveau départ (poids tirés à nouveau)
    + / -         plus ou moins d'itérations par image
    D             afficher / masquer les droites des neurones de la 1re couche
    clic gauche   ajoute un point bleu (classe 0) ; clic droit : un point orange (classe 1)

Ce qu'on voit :
- à gauche, le plan : la couleur donne la probabilité p(classe 1) prédite par le réseau, par
  tranches de 0,1 (courbes de niveau de p), et le trait noir est la frontière p = 0,5. Les traits gris sont les droites w.x + b = 0 des neurones de la
  1re couche cachée : chacun est un perceptron de S1 (« un neurone = un hyperplan »). Le réseau
  combine ces droites pour fabriquer une frontière courbe ;
- en haut à droite, le coût (entropie croisée) à chaque itération, en échelle log ;
- en bas à droite, le réseau : une arête par poids, épaisseur = |poids|, orange si positif,
  bleu si négatif.

ATTENTION, S3 : ce fichier contient une rétropropagation complète, écrite pour la démonstration.
Ce n'est pas votre code : en S3, on écrit la sienne sans regarder celle-ci.

Pourquoi c'est fluide : on ne redessine pas toute la figure à chaque image (le plus lent), seulement
ce qui bouge (« blitting »), et l'apprentissage avance de quelques petits pas de gradient par image,
si bien que la frontière se déforme continûment au lieu de sauter. Le nombre d'images par seconde
réellement obtenu est affiché en bas du plan. S'il est bas : --resolution 60, ou moins de points.
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

BORNES = (-1.4, 1.4)
BLEU, ORANGE = "#2856a3", "#d9692c"
FOND = LinearSegmentedColormap.from_list("fond", ["#9dbbe3", "#f7f7f7", "#f5b58a"])


# ---------------------------------------------------------------- données


def generer_donnees(nom, n, rng):
    """Renvoie X (n, 2), à peu près dans [-1 ; 1]^2, et y (n,) en 0/1."""
    if nom == "lunes":
        t = rng.uniform(0, np.pi, n)
        y = rng.integers(0, 2, n)
        x1 = np.where(y == 0, np.cos(t), 1 - np.cos(t))
        x2 = np.where(y == 0, np.sin(t), 0.5 - np.sin(t))
        X = (np.column_stack([x1, x2]) + rng.normal(0, 0.1, (n, 2)) - [0.5, 0.25]) / 1.2
    elif nom == "cercles":
        y = rng.integers(0, 2, n)
        r = np.where(y == 1, 0.45, 0.95) + rng.normal(0, 0.08, n)
        t = rng.uniform(0, 2 * np.pi, n)
        X = np.column_stack([r * np.cos(t), r * np.sin(t)])
    elif nom == "xor":
        quadrant = rng.integers(0, 4, n)
        centres = np.array([[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]])
        X = centres[quadrant] + rng.normal(0, 0.18, (n, 2))
        y = (quadrant >= 2).astype(int)  # classe 1 : quadrants où x1 et x2 sont de signes opposés
    else:  # spirale : deux bras enroulés l'un dans l'autre
        y = rng.integers(0, 2, n)
        r = rng.uniform(0.05, 1, n)
        t = 3 * np.pi * r + np.pi * y
        X = np.column_stack([r * np.cos(t), r * np.sin(t)]) + rng.normal(0, 0.03, (n, 2))
    return X, y


# ---------------------------------------------------------------- le réseau (NumPy seulement)


def initialiser(tailles, rng):
    """Une liste de couches (W, b) : W de forme (entrées, sorties), b de forme (sorties,).
    Poids tirés selon N(0, 1/entrées) pour que tanh ne sature pas dès le départ."""
    return [(rng.normal(0, 1 / np.sqrt(a), (a, s)), np.zeros(s)) for a, s in zip(tailles[:-1], tailles[1:])]


def propager(couches, X):
    """Passe avant. Renvoie la liste des activations [A0 = X, A1, ..., AL = p].
    tanh pour les couches cachées, sigmoïde pour la sortie : AL de forme (n, 1)."""
    A = [X]
    for k, (W, b) in enumerate(couches):
        Z = A[-1] @ W + b
        A.append(np.tanh(Z) if k < len(couches) - 1 else 1 / (1 + np.exp(-Z)))
    return A


def pas_de_gradient(couches, X, Y, eta):
    """Une itération de descente de gradient sur tout le jeu de données. Renvoie le coût.

    Coût : entropie croisée moyenne. Avec la sigmoïde en sortie, dL/dZ_L = (p - Y) / n.
    Puis, en remontant : dW = A^T dZ, db = somme des lignes de dZ, et
    dZ_(k-1) = (dZ_k W_k^T) * (1 - A_(k-1)^2), car tanh' = 1 - tanh^2.
    """
    A = propager(couches, X)
    p = np.clip(A[-1], 1e-12, 1 - 1e-12)
    cout = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    dZ = (A[-1] - Y) / len(X)
    for k in range(len(couches) - 1, -1, -1):
        W, b = couches[k]
        dW = A[k].T @ dZ
        db = dZ.sum(axis=0)
        if k > 0:
            dZ = (dZ @ W.T) * (1 - A[k] ** 2)  # avec l'ancien W, avant la mise à jour
        couches[k] = (W - eta * dW, b - eta * db)
    return cout


# ---------------------------------------------------------------- petits outils de dessin


def segment_droite(w, b):
    """Deux points de la droite w . x + b = 0 qui traversent la fenêtre, ou None."""
    t = np.array(BORNES)
    if abs(w[1]) >= abs(w[0]) and abs(w[1]) > 1e-9:
        return np.column_stack([t, -(w[0] * t + b) / w[1]])
    if abs(w[0]) > 1e-9:
        return np.column_stack([-(w[1] * t + b) / w[0], t])
    return None


def morceaux(contour):
    """Les objets à dessiner d'une courbe de niveau. Selon la version de matplotlib, c'est
    le ContourSet lui-même (>= 3.8) ou sa liste .collections (versions anciennes)."""
    return [contour] if isinstance(contour, Artist) else list(contour.collections)


def positions_reseau(tailles):
    """Coordonnées (x, y) de chaque neurone pour le schéma du réseau, couche par couche."""
    positions = []
    for c, n in enumerate(tailles):
        ys = np.linspace(-1, 1, n) * min(1, n / 10) if n > 1 else np.zeros(1)
        positions.append(np.column_stack([np.full(n, float(c)), ys]))
    return positions


# ---------------------------------------------------------------- l'application


class Animation:
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(args.graine)
        self.X, self.y = generer_donnees(args.donnees, args.points, self.rng)
        self.tailles = [2] + args.couches + [1]
        self.vitesse = args.vitesse
        self.pause = False
        self.droites_visibles = True
        self.nouveau_depart()

        # Grille du plan, calculée une fois : points de forme (m*m, 2).
        valeurs = np.linspace(*BORNES, args.resolution)
        self.xx, self.yy = np.meshgrid(valeurs, valeurs)
        self.points_grille = np.column_stack([self.xx.ravel(), self.yy.ravel()])

        self.construire_figure()

    def nouveau_depart(self):
        self.couches = initialiser(self.tailles, self.rng)
        self.iteration = 0
        self.fin = self.args.iterations
        self.couts = []
        self.t_image = None
        self.ips = 0.0  # images par seconde, lissées

    # ------------------------------------------------------------ figure

    def construire_figure(self):
        a = self.args
        self.fig = plt.figure(figsize=(13, 7))
        g = self.fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.42, wspace=0.15,
                                  left=0.05, right=0.97, top=0.9, bottom=0.11)
        self.ax = self.fig.add_subplot(g[:, 0])
        self.ax_cout = self.fig.add_subplot(g[0, 1])
        self.ax_res = self.fig.add_subplot(g[1, 1])

        archi = "-".join(str(t) for t in self.tailles)
        self.fig.suptitle(f"Réseau {archi} · {a.donnees} · {len(self.X)} points · descente de gradient, η = {a.eta}",
                          fontsize=13)
        self.fig.text(0.5, 0.01, "Espace : pause  ·  N : nouveau départ  ·  + / - : vitesse  ·  "
                                 "D : droites de la 1re couche  ·  clic gauche / droit : ajouter un point bleu / orange",
                      ha="center", fontsize=9, color="#555555")

        # --- le plan
        ax = self.ax
        # Fond et frontière : recréés à chaque image (contourf est bien plus rapide à dessiner
        # qu'une image agrandie au pixel près).
        self.fond = None
        self.contour = None
        self.droites = LineCollection([], colors="#444444", linewidths=0.9, alpha=0.45, zorder=1)
        ax.add_collection(self.droites)
        self.nuage = ax.scatter(self.X[:, 0], self.X[:, 1], s=20, c=self.couleurs_points(),
                                edgecolors="white", linewidths=0.6, zorder=3)
        # Le texte est redessiné à chaque image, et le texte coûte cher : une seule ligne courte.
        self.texte = ax.text(0.015, 0.015, "", transform=ax.transAxes, fontsize=10,
                             va="bottom", zorder=5, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        # La légende est hors du plan : dessinée une seule fois, pas à chaque image.
        self.fig.legend(handles=[
            Line2D([], [], color="black", lw=2, label="frontière (p = 0,5)"),
            Line2D([], [], color="#444444", lw=0.9, alpha=0.6, label="droites des neurones de la 1re couche"),
            Line2D([], [], ls="", marker="o", color=BLEU, label="classe 0"),
            Line2D([], [], ls="", marker="o", color=ORANGE, label="classe 1"),
        ], loc="lower left", bbox_to_anchor=(0.05, 0.035), ncol=4, fontsize=8, frameon=False)
        ax.set_xlim(*BORNES)
        ax.set_ylim(*BORNES)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # --- la courbe de coût
        c = self.ax_cout
        (self.courbe,) = c.plot([], [], color="#c0392b", lw=1.6)
        c.set_xlim(0, self.fin)
        c.set_yscale("log")
        c.set_ylim(5e-3, 2)
        c.set_title("Coût (entropie croisée)", fontsize=10)
        c.set_xlabel("itération")
        c.grid(True, which="major", alpha=0.3)

        # --- le schéma du réseau
        r = self.ax_res
        self.pos = positions_reseau(self.tailles)
        segments = []
        for k in range(len(self.tailles) - 1):
            for i in range(self.tailles[k]):
                for j in range(self.tailles[k + 1]):
                    segments.append([self.pos[k][i], self.pos[k + 1][j]])
        self.aretes = LineCollection(segments, zorder=1)
        r.add_collection(self.aretes)
        tous = np.vstack(self.pos)
        self.noeuds = r.scatter(tous[:, 0], tous[:, 1], s=70, c="white", edgecolors="#333333", linewidths=1.2, zorder=2)
        r.set_xlim(-0.4, len(self.tailles) - 0.6)
        r.set_ylim(-1.15, 1.15)
        r.set_xticks(range(len(self.tailles)))
        r.set_xticklabels(["entrée"] + [f"cachée {k + 1}" for k in range(len(self.args.couches))] + ["sortie"])
        r.set_yticks([])
        for cote in ("top", "right", "left"):
            r.spines[cote].set_visible(False)
        r.set_title("Les poids : épaisseur = |w|, orange > 0, bleu < 0", fontsize=10)

        # Tout ce qui bouge est « animé » : jamais dessiné dans l'arrière-plan figé, seulement par-dessus.
        for a in (self.droites, self.nuage, self.texte, self.courbe, self.aretes, self.noeuds):
            a.set_animated(True)

        self.fig.canvas.mpl_connect("key_press_event", self.touche)
        self.fig.canvas.mpl_connect("button_press_event", self.clic)

    def couleurs_points(self):
        return np.where(self.y[:, None] == 1, [[0.85, 0.41, 0.17, 1]], [[0.16, 0.34, 0.64, 1]])

    # ------------------------------------------------------------ interaction

    def touche(self, e):
        if e.key in (" ", "space"):
            self.pause = not self.pause
        elif e.key == "n":
            self.nouveau_depart()
            self.reajuster_axe_cout()
        elif e.key in ("+", "="):
            self.vitesse = min(50, self.vitesse + 1)
        elif e.key == "-":
            self.vitesse = max(1, self.vitesse - 1)
        elif e.key == "d":
            self.droites_visibles = not self.droites_visibles

    def clic(self, e):
        if e.inaxes is not self.ax or e.button not in (1, 3):
            return
        barre = getattr(self.fig.canvas.manager, "toolbar", None)
        if barre is not None and getattr(barre, "mode", ""):  # zoom ou déplacement en cours
            return
        self.X = np.vstack([self.X, [e.xdata, e.ydata]])
        self.y = np.append(self.y, 1 if e.button == 3 else 0)
        self.nuage.set_offsets(self.X)
        self.nuage.set_facecolors(self.couleurs_points())
        if self.iteration >= self.fin:  # l'apprentissage était fini : on le relance
            self.fin = self.iteration + self.args.iterations // 2
            self.reajuster_axe_cout()

    def reajuster_axe_cout(self):
        """Change l'échelle de la courbe de coût. Il faut alors tout redessiner une fois
        (ce que le blitting évite d'habitude), pour mettre à jour les graduations."""
        self.ax_cout.set_xlim(0, self.fin)
        self.fig.canvas.draw()

    # ------------------------------------------------------------ une image

    def image_suivante(self, _):
        # 1. Apprendre : quelques petits pas de gradient.
        if not self.pause:
            Y = self.y[:, None]
            for _ in range(min(self.vitesse, self.fin - self.iteration)):
                self.couts.append(pas_de_gradient(self.couches, self.X, Y, self.args.eta))
                self.iteration += 1

        # 2. Le fond : p(classe 1) en chaque point de la grille, et la frontière p = 0,5.
        p = propager(self.couches, self.points_grille)[-1].reshape(self.xx.shape)
        for ancien in (self.fond, self.contour):
            if ancien is not None:
                for m in morceaux(ancien):
                    m.remove()
        self.fond = self.ax.contourf(self.xx, self.yy, p, levels=np.linspace(0, 1, 11), cmap=FOND, zorder=0)
        self.contour = None
        if p.min() < 0.5 < p.max():
            self.contour = self.ax.contour(self.xx, self.yy, p, levels=[0.5], colors="black", linewidths=2, zorder=2)
        for nouveau in (self.fond, self.contour):
            if nouveau is not None:
                for m in morceaux(nouveau):
                    m.set_animated(True)

        # 3. Les droites des neurones de la 1re couche : colonne j de W1 et b1[j].
        W1, b1 = self.couches[0]
        segments = [] if not self.droites_visibles else \
            [s for s in (segment_droite(W1[:, j], b1[j]) for j in range(W1.shape[1])) if s is not None]
        self.droites.set_segments(segments)

        # 4. La courbe de coût.
        self.courbe.set_data(np.arange(1, len(self.couts) + 1), self.couts)

        # 5. Les arêtes du réseau.
        poids = np.concatenate([W.ravel() for W, _ in self.couches])
        echelle = np.abs(poids) / max(np.abs(poids).max(), 1e-9)
        couleurs = np.where(poids[:, None] > 0, [[0.85, 0.41, 0.17, 1.0]], [[0.23, 0.44, 0.71, 1.0]])
        couleurs[:, 3] = 0.15 + 0.85 * echelle
        self.aretes.set_color(couleurs)
        self.aretes.set_linewidths(0.3 + 3 * echelle)

        # 6. Le texte, avec les images par seconde réellement obtenues.
        maintenant = time.perf_counter()
        if self.t_image is not None:
            self.ips = 0.9 * self.ips + 0.1 / max(maintenant - self.t_image, 1e-6)
        self.t_image = maintenant
        precision = np.mean((propager(self.couches, self.X)[-1][:, 0] >= 0.5) == self.y)
        etat = "PAUSE" if self.pause else ("terminé" if self.iteration >= self.fin else f"{self.vitesse} it./im.")
        cout = f"{self.couts[-1]:.4f}" if self.couts else "-"
        self.texte.set_text(f"it. {self.iteration} · coût {cout} · {precision:.1%} bien classés · {etat} · {self.ips:.0f} im/s")

        return self.artistes()

    def artistes(self):
        """Tout ce qui bouge : seuls ces objets sont redessinés à chaque image."""
        fond = morceaux(self.fond) if self.fond is not None else []
        contour = morceaux(self.contour) if self.contour is not None else []
        return [*fond, *contour, self.droites, self.nuage, self.texte, self.courbe, self.aretes, self.noeuds]

    def lancer(self):
        self.animation = FuncAnimation(self.fig, self.image_suivante, init_func=self.artistes,
                                       interval=self.args.intervalle, blit=True, cache_frame_data=False)
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Un réseau de neurones apprend en direct, animation fluide.")
    p.add_argument("--donnees", choices=["spirale", "lunes", "cercles", "xor"], default="spirale")
    p.add_argument("--couches", type=int, nargs="+", default=None,
                   help="neurones par couche cachée (défaut : 16 16 pour la spirale, 8 sinon)")
    p.add_argument("--eta", type=float, default=0.5, help="pas d'apprentissage (0.5)")
    p.add_argument("--iterations", type=int, default=3000, help="nombre d'itérations (3000)")
    p.add_argument("--vitesse", type=int, default=3, help="itérations par image (3)")
    p.add_argument("--points", type=int, default=300, help="nombre de points (300)")
    p.add_argument("--graine", type=int, default=0, help="graine du hasard (0)")
    p.add_argument("--resolution", type=int, default=80, help="finesse de la grille du fond (80)")
    # Le minuteur n'est relancé qu'une fois l'image dessinée : l'attente s'ajoute au temps de dessin.
    # 1 ms revient donc à « aussi vite que possible ».
    p.add_argument("--intervalle", type=int, default=1, help="pause en ms entre deux images (1 : au plus vite)")
    args = p.parse_args()
    if args.couches is None:
        args.couches = [16, 16] if args.donnees == "spirale" else [8]
    Animation(args).lancer()


if __name__ == "__main__":
    main()
