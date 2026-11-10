# TIPE-TRANSFORMER

Optimisation des réseaux de neurones : implémentation « from scratch » d'un MLP puis d'un Transformer en Python avec NumPy uniquement, rétropropagation écrite à la main, et étude expérimentale de l'élagage (pruning). Projet de TIPE en binôme, classe de MP.

L'objectif n'est pas d'obtenir un modèle performant, mais de comprendre, de savoir justifier chaque opération et de mesurer ce que l'on peut retirer à un réseau sans dégrader ses performances.

Le déroulé séance par séance est dans [docs/PLAN_DE_TRAVAIL.md](docs/PLAN_DE_TRAVAIL.md).

## Le projet

### Objectifs

1. **Comprendre** : reconstruire chaque opération à partir des équations, forward et backward, sans bibliothèque de haut niveau.
2. **Implémenter** : un MLP entraîné sur des chiffres manuscrits, puis un mini-GPT entraîné sur un petit corpus de texte français.
3. **Optimiser et mesurer** : élaguer les deux réseaux et confronter les mesures (précision, nombre de paramètres, temps d'inférence) aux prédictions théoriques. Cette partie expérimentale constitue l'apport personnel du TIPE.

### Démarche en quatre phases

| Phase | Séances | Contenu | Jalon de sortie |
| --- | --- | --- | --- |
| 1. Fondations | S1–S4 | Neurone, perceptron, descente de gradient, MLP, rétropropagation, softmax | MLP ≥ 95 % de précision en test, gradients vérifiés numériquement |
| 2. Optimisation du MLP | S5–S7 | Élagage par magnitude, itératif, structuré ; protocole expérimental | Problématique v1, banc d'essai reproductible |
| 3. Transformer | S8–S11 | Embeddings, self-attention, multi-head, bloc, backward, entraînement | Tous les gradients vérifiés, texte généré |
| 4. Expériences et oral | S12–S14 | Élagage des têtes d'attention et des MLP, analyse, mise en forme | Figures finales, diaporama, DOT |

Chaque brique est écrite puis testée séparément avant de passer à la suivante, avec des dimensions minuscules au départ. Chaque gradient écrit à la main est comparé à un gradient calculé par différences finies.

### Modèle mathématique

**MLP.** Pour une couche linéaire `Z = XW + b`, avec `X` de taille `(n, d_in)` et le gradient amont `G = ∂L/∂Z` :

- `∂L/∂W = Xᵀ G`, `∂L/∂b = somme des lignes de G`, `∂L/∂X = G Wᵀ`
- Softmax suivie de l'entropie croisée : `∂L/∂Z = (P − Y) / n`

**Élagage.** On applique un masque binaire `M` aux poids, `W ← W ⊙ M`, et aussi au gradient pendant le réentraînement :

- non structuré : on retire les poids de plus petite valeur absolue (même nombre d'opérations, mais moins de paramètres non nuls) ;
- structuré : on retire des neurones entiers ou des têtes d'attention entières (les matrices rétrécissent, donc le calcul aussi).

**Transformer.** Pour une séquence de `T` vecteurs `X` de dimension `d_model` :

- Projections : `Q = X Wq`, `K = X Wk`, `V = X Wv`
- Attention : `A = softmax(Q Kᵀ / √dk)`, de taille `T × T`, puis `Y = A V`
- Multi-head : `h` têtes de dimension `dk = d_model / h`, concaténées puis projetées par `Wo`
- MLP appliqué à chaque position : `W2 ReLU(W1 x + b1) + b2`
- Bloc : attention et MLP, chacun avec connexion résiduelle et LayerNorm. On choisira entre la variante post-norm de l'article original et la variante pre-norm de GPT-2, en justifiant le choix.
- Décodeur de type GPT : masque causal qui empêche chaque position de regarder les positions futures
- Nombre de paramètres ≈ `12 · L · d_model² + |V| · d_model` pour `L` blocs

### Problématique et axes d'étude

La problématique sera fixée à partir de la séance 5, une fois les premières mesures d'élagage obtenues. Elle sera ensuite affinée en S7 (version pour la MCOT), puis en S11 (version finale). Pistes envisagées :

- Quelle proportion des poids d'un réseau peut-on retirer sans dégrader ses performances, et quel critère d'élagage est le plus efficace ?
- L'élagage réduit-il réellement le coût de calcul, ou seulement le nombre de paramètres ?
- Toutes les têtes d'attention d'un Transformer sont-elles utiles ?
- Le coût de l'attention croît-il bien en `O(T²)` avec la longueur de séquence ?
- Quel est le rôle de la normalisation par `√dk` dans la stabilité de l'entraînement ?

### Limites assumées

Les modèles restent petits et tournent sur CPU. Un MLP qui reconnaît les chiffres 8×8 et un mini-GPT qui génère des caractères ressemblant au texte d'entraînement suffisent pour mener les expériences.

### Organisation

Le travail est mené à deux. Hors séance, l'un s'occupe plutôt des données et de l'architecture, l'autre plutôt des mécanismes neuronaux. En séance, les rôles de pilote et de copilote alternent, et chacun doit pouvoir expliquer l'ensemble du code au jury. Un journal de bord (`journal.md`) garde la trace de chaque séance et servira à rédiger le DOT.

## Structure

```
TIPE-TRANSFORMER/
├── src/
│   ├── data.py            # tokenisation par caractères (fait)
│   ├── neurone.py         # neurone et perceptron (S1, à venir)
│   ├── verif_gradient.py  # gradient par différences finies (S2, à venir)
│   ├── couches.py         # Lineaire, ReLU, softmax + cross-entropy (S3–S4, à venir)
│   ├── mlp.py             # MLP et boucle d'entraînement (S3–S4, à venir)
│   ├── elagage.py         # masques, élagage non structuré et structuré (S5–S6, à venir)
│   ├── attention.py       # self-attention, multi-head (S8–S9, à venir)
│   ├── transformer.py     # bloc et mini-GPT (S9–S11, à venir)
│   └── optimiseurs.py     # SGD, Adam (S11, à venir)
├── tests/                 # un fichier de tests par brique
├── experiences/           # banc d'essai et résultats CSV
├── figures/
├── notebooks/
│   └── experiments.ipynb
├── docs/
│   └── PLAN_DE_TRAVAIL.md # plan détaillé des 14 séances
├── tableau-de-bord/       # site des fiches de séance (déployé sur Vercel)
│   ├── seances/           # index.json + une fiche SXX.json par séance
│   └── verifier_fiches.py # vérifie les fiches avant un commit
├── journal.md             # journal de bord des séances (à venir)
├── CLAUDE.md
└── README.md
```

## Séances et fiches de séance

Chaque séance (le mardi, 2 h) a sa fiche sur **https://tipe-transformer.vercel.app**. Le site s'ouvre sur la prochaine séance ; les autres sont accessibles depuis la frise en haut de page.

**Pendant la séance :**

1. Ouvrir la fiche et cliquer sur « Démarrer la séance ». Le chrono suit le déroulé (rappel actif, relance, théorie, code, explication croisée, clôture) et met en avant le créneau en cours.
2. Rappel actif : répondre de tête, puis découvrir les réponses. Le « Mode rappel » de la barre du haut masque aussi la théorie, pour se tester avant de la relire.
3. Cocher les tâches au fur et à mesure. Le panneau « Reste à coder » indique le temps restant et signale un retard de plus de 10 min (voir alors « Si on prend du retard »).
4. Clôture : compléter le bloc de journal en bas de page (il est pré-rempli avec les tâches cochées), le coller dans `journal.md`, puis commit et push.

Les cases cochées et le chrono restent dans le navigateur de chacun. **La trace qui compte est `journal.md`** : c'est d'après lui que la fiche suivante est préparée.

**Préparation automatique, le lundi soir :** une tâche planifiée de Claude, lancée depuis l'interface de Claude, s'exécute dans le cloud tous les lundis à 20 h (heure de Paris). Elle clone le dépôt, lit le journal, les commits et le résultat des tests, puis écrit la fiche du mardi en adaptant le plan à l'état réel : reliquats en premier, rattrapage si une séance bloquante n'est pas validée. Elle vérifie la fiche, la pousse sur `main`, et Vercel la met en ligne. La procédure suivie est décrite dans `CLAUDE.md`, section « Fiches de séance ».

Aperçu local du site : dans `tableau-de-bord/`, lancer `python -m http.server 8000`, puis ouvrir `http://localhost:8000`.

## Avancement

**Phase 1 : Fondations**
- [ ] Neurone vectorisé et perceptron (ET, OU, échec sur XOR)
- [ ] Régression logistique et vérification numérique des gradients
- [ ] Couches `Lineaire` et `ReLU` avec backward, MLP qui résout XOR
- [ ] Softmax, cross-entropy, mini-batch : MLP de référence ≥ 95 % sur les chiffres 8×8

**Phase 2 : Optimisation du MLP**
- [ ] Élagage par magnitude en une fois : courbe précision / taux d'élagage
- [ ] Élagage itératif avec réentraînement, élagage structuré, mesure du temps
- [ ] Banc d'essai reproductible (graines, CSV) et problématique v1

**Phase 3 : Transformer**
- [x] Tokenisation par caractères (`stoi`, `itos`, `encoder`, `decoder`)
- [ ] Embedding et self-attention avec masque causal
- [ ] Multi-head, LayerNorm, bloc et mini-GPT (forward)
- [ ] Backward de toutes les briques, vérifié par différences finies
- [ ] Entraînement avec Adam et génération de texte

**Phase 4 : Expériences et oral**
- [ ] Élagage des têtes d'attention et des MLP du Transformer
- [ ] Campagne d'expériences finale et analyse
- [ ] Diaporama et DOT

## Utilisation

Prérequis : Python 3 et NumPy (matplotlib pour les figures, pytest pour les tests).

```
python src/data.py
```

Affiche le vocabulaire et les tokens de « bonjour le monde », puis vérifie que le décodage redonne le texte d'origine.

## Références

- Vaswani et al., *Attention Is All You Need*, 2017 : [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- LeCun, Denker, Solla, *Optimal Brain Damage*, NeurIPS 1990
- Han et al., *Learning both Weights and Connections for Efficient Neural Networks*, 2015 : [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- Frankle & Carbin, *The Lottery Ticket Hypothesis*, 2019 : [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)
- Michel, Levy, Neubig, *Are Sixteen Heads Really Better than One?*, 2019 : [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) et [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
