"""Animation : le perceptron sur un grand nuage de points linéairement séparable.

Lancer depuis S1/ :
    python experiences/s1_animation_nuage.py                    -> 300 points, marge 0,02, centre (2, 2)
    python experiences/s1_animation_nuage.py --centre 0         -> nuage centré en 0 : converge très vite
    python experiences/s1_animation_nuage.py --points 800 --marge 0.01   -> plus dur, plus long
    python experiences/s1_animation_nuage.py --bruit 0.05       -> 5 % d'étiquettes fausses : plus séparable
    python experiences/s1_animation_nuage.py --graine 7         -> un autre tirage
    python experiences/s1_animation_nuage.py --intervalle 15    -> encore plus rapide (ms par image)
    python experiences/s1_animation_nuage.py --gif              -> enregistre figures/s1_animation_nuage.gif

Ce qu'on voit :
- à gauche, le plan : fond = prédiction du neurone, trait noir = droite actuelle, traits gris =
  les dernières droites (la traînée), pointillés verts = la droite cachée qui a servi à fabriquer
  les données, flèche = vecteur w. Les points mal classés sont grossis et cerclés de noir ;
  le cercle rouge montre le point qui vient de provoquer la correction ;
- en haut à droite : le nombre de points mal classés après chaque correction ;
- en bas à droite : l'angle entre (w, b) et la vraie direction (w*, b*).

Une image = une correction (les exemples bien classés ne changent rien, on les saute).

Pourquoi un nuage loin de l'origine converge lentement : b est corrigé du même pas eta que
w, comme une 3e coordonnée de w associée à une entrée constante 1. Loin de l'origine, les
x sont grands devant ce 1 : chaque correction fait surtout tourner la droite autour de
l'origine, et il en faut beaucoup pour la translater jusqu'au nuage. En chiffres, R grandit
donc la borne (R / gamma)^2 aussi. Comparer avec --centre 0.

Théorème de Novikoff (1962) : si les données sont séparables avec une marge gamma, et si
R = max ||(x, 1)||, alors le perceptron fait au plus (R / gamma)^2 corrections, quel que soit
eta et l'ordre des exemples. Le titre affiche cette borne pour la comparer au nombre réel.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

DOSSIER_S1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DOSSIER_S1, "src"))

from neurone import neurone, neurone_batch  # noqa: E402

BLEU, ORANGE = "#3b6fb6", "#d9692c"
LONGUEUR_TRAINEE = 25


# ---------------------------------------------------------------- données


def generer_nuage(n_points, marge, bruit, centre, rng):
    """Tire n_points uniformément dans le carré [centre - 1 ; centre + 1]^2 et les étiquette
    avec une droite cachée qui traverse ce carré.

    La droite cachée : w* = (cos t, sin t), et b* choisi pour qu'elle passe à une distance
    c (tirée au hasard, |c| <= 0,4) du centre du carré.
    On écarte les points à moins de `marge` de la droite : il reste une bande vide,
    c'est la marge gamma du théorème de Novikoff. Puis on retourne une proportion
    `bruit` des étiquettes (si bruit > 0, le nuage n'est plus séparable).

    Renvoie X (n, 2), y (n,) en 0/1, w_vrai (2,), b_vrai.
    """
    t = rng.uniform(0, 2 * np.pi)
    w_vrai = np.array([np.cos(t), np.sin(t)])  # ||w*|| = 1 : w*.x + b* est une distance signée
    milieu = np.array([centre, centre])
    b_vrai = -rng.uniform(-0.4, 0.4) - w_vrai @ milieu

    X = np.empty((0, 2))
    while len(X) < n_points:  # on retire jusqu'à avoir assez de points hors de la bande
        candidats = milieu + rng.uniform(-1, 1, size=(2 * n_points, 2))
        loin = np.abs(candidats @ w_vrai + b_vrai) >= marge
        X = np.vstack([X, candidats[loin]])
    X = X[:n_points]

    y = (X @ w_vrai + b_vrai >= 0).astype(int)
    if bruit > 0:
        a_retourner = rng.random(n_points) < bruit
        y[a_retourner] = 1 - y[a_retourner]
    return X, y, w_vrai, b_vrai


def borne_novikoff(X, y, w_vrai, b_vrai):
    """(R / gamma)^2 avec des vecteurs augmentés x~ = (x, 1) et u = (w*, b*) / ||(w*, b*)||.

    gamma = min_i s_i (u . x~_i), où s_i = +1 pour la classe 1 et -1 pour la classe 0.
    Renvoie None si gamma <= 0 (données non séparables par la droite cachée, avec du bruit).
    """
    X_aug = np.column_stack([X, np.ones(len(X))])
    u = np.append(w_vrai, b_vrai)
    u = u / np.linalg.norm(u)
    signes = 2 * y - 1
    gamma = np.min(signes * (X_aug @ u))
    if gamma <= 0:
        return None
    R = np.max(np.linalg.norm(X_aug, axis=1))
    return (R / gamma) ** 2


# ---------------------------------------------------------------- apprentissage


def corrections_perceptron(X, y, eta, n_epoques, rng):
    """Règle de entrainer_perceptron (src/perceptron.py), exemples mélangés à chaque époque.

    On ne garde une photo qu'après chaque CORRECTION. Chaque photo contient w, b,
    l'indice du point qui a provoqué la correction et l'époque.
    Renvoie la liste des photos et un booléen « convergé ».
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    photos = [dict(w=w.copy(), b=b, i=None, epoque=0)]

    for epoque in range(1, n_epoques + 1):
        nb_erreurs = 0
        for i in rng.permutation(n):
            ecart = y[i] - neurone(X[i], w, b)
            if ecart != 0:
                w = w + eta * ecart * X[i]
                b = b + eta * ecart
                nb_erreurs += 1
                photos.append(dict(w=w.copy(), b=b, i=i, epoque=epoque))
        if nb_erreurs == 0:
            return photos, True
    return photos, False


def angle_degres(w, b, w_vrai, b_vrai):
    """Angle entre (w, b) et (w*, b*), en degrés. nan si w = 0 et b = 0."""
    v, u = np.append(w, b), np.append(w_vrai, b_vrai)
    norme = np.linalg.norm(v) * np.linalg.norm(u)
    if norme == 0:
        return np.nan
    return np.degrees(np.arccos(np.clip(v @ u / norme, -1, 1)))


# ---------------------------------------------------------------- dessin


def segment_droite(w, b, bornes):
    """Deux points de la droite w . x + b = 0 qui traversent toute la fenêtre, ou None si w = 0.

    On paramètre par la coordonnée qui varie le plus (évite les droites presque verticales).
    """
    t = np.array(bornes)
    if abs(w[1]) >= abs(w[0]) and abs(w[1]) > 1e-12:
        return t, -(w[0] * t + b) / w[1]
    if abs(w[0]) > 1e-12:
        return -(w[1] * t + b) / w[0], t
    return None


def main():
    p = argparse.ArgumentParser(description="Perceptron sur un grand nuage de points, animé.")
    p.add_argument("--points", type=int, default=300, help="nombre de points (300)")
    p.add_argument("--marge", type=float, default=0.02, help="demi-largeur de la bande vide autour de la droite cachée (0.02)")
    p.add_argument("--centre", type=float, default=2.0, help="le nuage est centré en (centre, centre) (2)")
    p.add_argument("--bruit", type=float, default=0.0, help="proportion d'étiquettes retournées (0)")
    p.add_argument("--eta", type=float, default=0.1, help="pas d'apprentissage (0.1)")
    p.add_argument("--epoques", type=int, default=200, help="nombre maximal d'époques (200)")
    p.add_argument("--graine", type=int, default=0, help="graine du hasard (0)")
    p.add_argument("--intervalle", type=int, default=30, help="durée d'une image en ms (30, soit ~33 images/s)")
    p.add_argument("--max-images", type=int, default=1500, help="au-delà, on saute des corrections pour rester rapide (1500)")
    p.add_argument("--gif", action="store_true", help="enregistre figures/s1_animation_nuage.gif au lieu d'afficher")
    args = p.parse_args()

    rng = np.random.default_rng(args.graine)
    X, y, w_vrai, b_vrai = generer_nuage(args.points, args.marge, args.bruit, args.centre, rng)
    BORNES = (args.centre - 1.2, args.centre + 1.2)  # la fenêtre suit le nuage
    borne = borne_novikoff(X, y, w_vrai, b_vrai)
    photos, converge = corrections_perceptron(X, y, args.eta, args.epoques, rng)
    n_corr = len(photos) - 1

    # Courbes calculées d'avance pour toutes les corrections.
    mal_classes = np.array([np.sum(neurone_batch(X, ph["w"], ph["b"]) != y) for ph in photos])
    angles = np.array([angle_degres(ph["w"], ph["b"], w_vrai, b_vrai) for ph in photos])

    # Trop de corrections (bruit, petite marge) : on n'affiche qu'une photo sur `saut`.
    saut = max(1, int(np.ceil(len(photos) / args.max_images)))
    indices = list(range(0, len(photos), saut))
    if indices[-1] != len(photos) - 1:
        indices.append(len(photos) - 1)
    print(f"{args.points} points, {n_corr} corrections, "
          f"{'convergé' if converge else 'pas convergé'} en {photos[-1]['epoque']} époque(s), "
          f"borne de Novikoff : {'%.0f' % borne if borne else 'sans objet (bruit)'}"
          f"{f', une correction sur {saut} affichée' if saut > 1 else ''}")

    # --- mise en page : le plan à gauche, deux courbes à droite
    fig = plt.figure(figsize=(14, 7))
    grille = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.35, wspace=0.18)
    ax = fig.add_subplot(grille[:, 0])
    ax_err = fig.add_subplot(grille[0, 1])
    ax_ang = fig.add_subplot(grille[1, 1])

    valeurs = np.linspace(*BORNES, 160)
    xx, yy = np.meshgrid(valeurs, valeurs)
    points_grille = np.column_stack([xx.ravel(), yy.ravel()])
    fond = ax.imshow(np.zeros(xx.shape), extent=(*BORNES, *BORNES), origin="lower",
                     cmap=ListedColormap(["#dfe8f5", "#f9e0cf"]), vmin=0, vmax=1)

    vraie = segment_droite(w_vrai, b_vrai, BORNES)
    if vraie is not None:
        ax.plot(*vraie, "--", color="#2a9d4b", lw=1.5, label="droite cachée (w*, b*)")

    trainee = [ax.plot([], [], color="#555555", lw=1)[0] for _ in range(LONGUEUR_TRAINEE)]
    (droite,) = ax.plot([], [], color="black", lw=2.5, label="droite du perceptron")
    nuage = ax.scatter(X[:, 0], X[:, 1], s=22, c=np.where(y == 1, ORANGE, BLEU),
                       edgecolors="none", linewidths=1.6, zorder=3)
    (anneau,) = ax.plot([], [], "o", ms=16, mfc="none", mec="#d62728", mew=2.5, zorder=4)
    fleche = [None]

    ax.set_xlim(*BORNES)
    ax.set_ylim(*BORNES)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Courbes : la trace complète en pâle, la partie déjà parcourue en foncé.
    x_corr = np.arange(len(photos))
    ax_err.plot(x_corr, mal_classes, color="#cccccc", lw=1)
    (courbe_err,) = ax_err.plot([], [], color="#d62728", lw=1.8)
    ax_err.set_xlim(0, max(1, n_corr))
    ax_err.set_ylim(0, max(1, mal_classes.max()) * 1.05)
    ax_err.set_title("Points mal classés", fontsize=10)
    ax_err.set_xlabel("nombre de corrections")

    ax_ang.plot(x_corr, angles, color="#cccccc", lw=1)
    (courbe_ang,) = ax_ang.plot([], [], color="#2a9d4b", lw=1.8)
    ax_ang.set_xlim(0, max(1, n_corr))
    ax_ang.set_ylim(0, 180)
    ax_ang.set_title("Angle entre (w, b) et (w*, b*), en degrés", fontsize=10)
    ax_ang.set_xlabel("nombre de corrections")
    if borne and borne <= n_corr * 1.5:
        for a in (ax_err, ax_ang):
            a.axvline(borne, color="#7a4cc2", ls=":", lw=1.2)
        ax_err.text(borne, ax_err.get_ylim()[1] * 0.9, " borne de Novikoff", color="#7a4cc2", fontsize=8)

    texte_borne = f"borne de Novikoff : {borne:.0f}" if borne else "pas de borne : données non séparables"
    fig.suptitle(f"Perceptron sur {args.points} points · centre ({args.centre:g}, {args.centre:g}) · η = {args.eta} · marge = {args.marge}"
                 f"{f' · bruit = {args.bruit:.0%}' if args.bruit else ''}", fontsize=13)

    def image(k):
        j = indices[min(k, len(indices) - 1)]  # indice de la photo affichée
        ph = photos[j]
        w, b = ph["w"], ph["b"]

        fond.set_data(neurone_batch(points_grille, w, b).reshape(xx.shape))
        seg = segment_droite(w, b, BORNES)
        droite.set_data(*(seg if seg is not None else ([], [])))

        # Traînée : les LONGUEUR_TRAINEE photos précédentes, de plus en plus pâles.
        for r, trait in enumerate(trainee, start=1):
            q = j - r * saut
            seg_q = segment_droite(photos[q]["w"], photos[q]["b"], BORNES) if q >= 0 else None
            trait.set_data(*(seg_q if seg_q is not None else ([], [])))
            trait.set_alpha(0.5 * (1 - r / (LONGUEUR_TRAINEE + 1)))

        # Points mal classés : grossis et cerclés de noir.
        faux = neurone_batch(X, w, b) != y
        nuage.set_edgecolors(np.where(faux[:, None], [[0, 0, 0, 1]], [[0, 0, 0, 0]]))
        nuage.set_sizes(np.where(faux, 45, 18))

        if ph["i"] is None:
            anneau.set_data([], [])
        else:
            anneau.set_data([X[ph["i"], 0]], [X[ph["i"], 1]])

        if fleche[0] is not None:
            fleche[0].remove()
            fleche[0] = None
        n2 = w @ w
        if n2 > 0:
            c = np.array([args.centre, args.centre])
            pied = c - (w @ c + b) / n2 * w  # point de la droite le plus proche du centre
            if np.all((pied > BORNES[0]) & (pied < BORNES[1])):
                fleche[0] = ax.annotate("", xy=pied + 0.25 * w / np.sqrt(n2), xytext=pied,
                                        arrowprops=dict(arrowstyle="->", color="black", lw=2))

        courbe_err.set_data(x_corr[: j + 1], mal_classes[: j + 1])
        courbe_ang.set_data(x_corr[: j + 1], angles[: j + 1])

        fini = k >= len(indices) - 1
        if fini and converge:
            statut = f"CONVERGÉ en {ph['epoque']} époque(s)"
        elif fini:
            statut = f"ARRÊT après {ph['epoque']} époques, sans convergence"
        else:
            statut = f"époque {ph['epoque']}"
        ax.set_title(f"{statut} · correction {j} / {n_corr} ({texte_borne})\n"
                     f"w = ({w[0]:+.2f}, {w[1]:+.2f})   b = {b:+.2f}   mal classés : {mal_classes[j]}",
                     fontsize=10)

    n_images = len(indices) + 40  # on fige la fin un peu plus d'une seconde avant de reboucler
    animation = FuncAnimation(fig, image, frames=n_images, interval=args.intervalle, repeat=True)

    if args.gif:
        chemin = os.path.join(DOSSIER_S1, "figures", "s1_animation_nuage.gif")
        animation.save(chemin, writer=PillowWriter(fps=1000 / args.intervalle))
        print(f"Animation enregistrée : {chemin}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
