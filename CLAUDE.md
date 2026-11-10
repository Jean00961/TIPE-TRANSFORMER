# TIPE-TRANSFORMER

Projet de TIPE (MP) : implémentation d'un Transformer en Python/NumPy, brique par brique. Le dépôt est https://github.com/Jean00961/TIPE-TRANSFORMER.

## Git

- Travailler directement sur la branche `main` : pas de branche de fonctionnalité, pas de pull request.
- Toujours pousser sur `origin main` après un commit, sans attendre qu'on le demande.
- Exception : si l'utilisateur demande explicitement de ne pas pousser sur `main` (branche séparée, PR, pas de push), suivre sa demande pour ce cas.
- Cette règle est temporaire ("pour l'instant") et peut être changée par l'utilisateur.
- Ne jamais forcer un push (`--force`) sans demande explicite.

## Organisation

- `src/` : code du modèle (`data.py`, `attention.py`, `mlp.py`, `transformer.py`).
- `notebooks/` : expériences et mesures (`experiments.ipynb`).
- `docs/PLAN_DE_TRAVAIL.md` : le plan des 14 séances (référence fixe).
- `tableau-de-bord/` : le site des fiches de séance, en React + TypeScript (voir plus bas).
- `journal.md` (créé en S1) : une entrée par séance, écrite à la clôture.
- Lancer les scripts depuis la racine du dépôt, par exemple `python src/data.py`.

## Conventions

- Langue : français pour les commentaires, noms de fonctions, README et échanges.
- NumPy uniquement pour le calcul (pas de PyTorch pour le forward pass) : l'objectif est de comprendre et reconstruire les opérations.
- Chaque brique (tokenisation, attention, multi-head, MLP, bloc, modèle) doit être testée séparément avant de passer à la suivante.

## Fiches de séance (tableau de bord)

Les séances de TIPE ont lieu le **mardi**, en 2 h. Avant chacune, une fiche détaillée est écrite le lundi soir. Le site l'affiche avec le déroulé minuté, un chrono, les tâches à cocher, la théorie (formules KaTeX, mode rappel qui masque les notions) et un bloc de journal à copier.

- Application React + TypeScript (Vite) dans `tableau-de-bord/` : code dans `src/` (`types.ts` décrit la forme des fiches, `composants/` l'affichage, `progression.ts` les calculs). Vercel la reconstruit à chaque push sur `main` (projet `tipe-transformer`, dossier racine `tableau-de-bord`, réglages dans `vercel.json`). Adresse : https://tipe-transformer.vercel.app (une séance précise : `?s=S01`).
- Données : `tableau-de-bord/seances/index.json` (les 14 séances, leur statut, la séance courante) et un fichier `SXX.json` par fiche écrite. Elles sont intégrées au build. **On ne modifie que ces JSON** pour préparer une séance, jamais `src/` du site.
- Si la forme d'une fiche change, mettre à jour ensemble `src/types.ts`, `verifier_fiches.py` et cette section.
- Vérification obligatoire avant tout commit de fiche : `python tableau-de-bord/verifier_fiches.py` doit afficher « Fiches valides ». Si Node est disponible, lancer aussi dans `tableau-de-bord/` : `npm ci`, `npm run build` et `npm test`.
- Aperçu local : dans `tableau-de-bord/`, `npm install` puis `npm run dev`. Sur un lecteur réseau comme `U:`, Vite échoue (chemin UNC) : travailler sur une copie du dossier sur `C:`.
- Les cases cochées et le chrono restent dans le navigateur. La trace qui fait foi est `journal.md`.

### Préparer la fiche de la séance de demain (tâche du lundi soir)

Cette tâche est lancée par une tâche planifiée de Claude, dans le cloud, tous les lundis à 20 h (heure de Paris). Personne n'est là pour répondre : ne poser aucune question, faire le choix le plus raisonnable et le signaler dans le résumé final. Dans cet environnement :

- La commande Python peut s'appeler `python3`. Si `numpy` ou `pytest` manquent, les installer (`pip install numpy pytest matplotlib`). Si les tests ne peuvent quand même pas tourner, l'écrire dans `etat.alertes` au lieu de bloquer.
- Pousser directement sur `main`, comme le veut la section Git : pas de branche `claude/…`, pas de pull request. Si le push sur `main` est refusé, le dire clairement dans le résumé final.
- Les dates s'entendent en heure de Paris : « demain » est le mardi qui suit.

Quand on demande de « préparer la fiche de la séance de demain », suivre ces étapes dans l'ordre :

1. `git pull`, puis lire `tableau-de-bord/seances/index.json` pour trouver la dernière fiche écrite (`seance_courante`).
2. **Établir où on en est réellement**, sans se fier au plan seul :
   - la dernière entrée de `journal.md` (lignes *Fait*, *Compris*, *Bloqué*, *Étonnement*, *Première action de la prochaine séance*) ;
   - `git log` depuis la date de la dernière fiche, et les fichiers présents dans `src/`, `tests/`, `figures/`, `docs/` ;
   - le résultat de `python -m pytest -q` (noter les tests rouges) ;
   - les livrables de la séance précédente : présents ou non.
3. **Choisir la séance N à préparer** :
   - si `journal.md` contient une entrée pour la séance courante, alors N = séance courante + 1 ;
   - sinon, la séance n'a pas eu lieu (vacances, absence) : on garde la même séance, on met à jour la date et on le signale dans `etat.resume` ;
   - si la séance précédente est **bloquante** (S4, S7, S10) et que son jalon n'est pas atteint, on ne passe pas à la suite : la fiche N reprend ce qui manque, avec une alerte.
4. Lire dans `docs/PLAN_DE_TRAVAIL.md` la section de la séance N, ainsi que le déroulé type, les règles de rattrapage et le fil rouge de la problématique.
5. **Écrire `tableau-de-bord/seances/SNN.json`** en adaptant le plan à l'état réel. Prendre `S01.json` comme modèle de structure et de ton. Points à respecter :
   - `date` : la date du mardi (le lendemain). `genere_le` : la date du jour. `pilote` et `copilote` : alterner à chaque séance, à partir des noms de `index.json` → `binome`.
   - `etat.resume` : 2 à 4 phrases factuelles sur ce qui est fait et ce qui manque, d'après l'étape 2. `etat.reliquats` : ce qui n'a pas été fini la dernière fois ; ces reliquats deviennent les **premières** tâches. `etat.alertes` : tests rouges, jalon en danger, échéance (MCOT, SCEI), retard cumulé.
   - `deroule` : les 6 créneaux du déroulé type, contigus de 0 à 120 min, avec un créneau dont le titre contient « Code ».
   - `rappel_actif` : 2 à 4 questions **sur la séance précédente** (équation clé, dimensions, piège), avec leur réponse. Si le journal signale une notion mal comprise, elle doit y figurer.
   - `theorie` : les notions du plan, environ 25 min au total. Pour chacune : contenu rigoureux, formule clé, dimensions des matrices, piège. Mettre `au_tableau: true` pour les dérivations à faire à la main.
   - `taches` : les reliquats d'abord, puis les tâches du plan. Chaque tâche a des étapes concrètes (noms de fichiers et de fonctions en français) et le **test** à écrire. La somme des durées hors bonus ne doit pas dépasser le créneau Code. Si on est en retard, couper d'abord ce que le plan marque *bonus*.
   - `plan_b`, `livrables`, `questions_jury` (3 à 5, avec une piste de réponse), `ancrage`, `etonnement`, `suite` (la séance N+1 et ce qu'il faut relire la veille).
6. Mettre à jour `index.json` : la séance précédente passe en `"faite"` si elle a eu lieu, N passe en `"prochaine"` avec `"fiche": true`, et `seance_courante` devient N.
7. Lancer `python tableau-de-bord/verifier_fiches.py` et corriger jusqu'à « Fiches valides ». Si Node est disponible, lancer aussi `npm ci && npm run build && npm test` dans `tableau-de-bord/` : un échec bloquerait le déploiement Vercel.
8. Commit « Fiche SNN (date) », puis push sur `origin main`. Vercel redéploie tout seul.
9. Terminer par un résumé de 5 lignes : la séance préparée, l'état constaté, les reliquats, les alertes et le lien du site.

Règles de rédaction des fiches :

- Ne jamais toucher à `src/`, `tests/` ni au plan pendant cette tâche : la fiche s'adapte au plan, elle ne le réécrit pas. Si un jalon glisse, le dire dans `etat.alertes`.
- Ne pas écrire le code de la séance dans la fiche : donner les signatures, les étapes et les tests, pas les solutions. Le binôme doit pouvoir expliquer chaque ligne au jury.
- Formatage des textes : `$…$` et `$$…$$` pour les formules (KaTeX), `` `…` `` pour le code, `**gras**`, `*italique*`, des listes `- ` ou `1. `, des blocs de code entre lignes « ``` ». Dans le JSON, **doubler chaque antislash LaTeX** (`\\frac`, `\\eta`, `\\sigma`) : un `\f` ou un `\t` non doublé devient un caractère invisible, et le vérificateur le signale.
- Français, phrases courtes, vouvoiement évité (« on »).
