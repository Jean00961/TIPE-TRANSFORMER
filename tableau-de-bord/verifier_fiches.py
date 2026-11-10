"""Vérifie la cohérence des fiches de séance avant de les pousser.

Usage, depuis la racine du dépôt : python tableau-de-bord/verifier_fiches.py
Code de sortie 0 si tout est valide, 1 sinon (avec la liste des problèmes).
Uniquement la bibliothèque standard (compatible Python 3.6).
"""

import json
import os
import re
import sys

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seances")

CHAMPS_FICHE = {
    "id": str, "numero": int, "titre": str, "date": str, "phase": int,
    "pilote": str, "copilote": str, "genere_le": str, "objectif": str,
    "etat": dict, "acquis": list, "deroule": list, "rappel_actif": list,
    "theorie": list, "taches": list, "plan_b": list, "livrables": list,
    "questions_jury": list, "ancrage": dict,
}

# Caractères de contrôle produits par un antislash LaTeX mal échappé en JSON :
# "\frac" -> saut de page, "\beta" -> retour arrière, "\theta" -> tabulation, "\rho" -> retour chariot.
CONTROLE = re.compile(r"[\x00-\x09\x0b-\x1f]")
# "\nabla" ou "\neq" non doublés deviennent un saut de ligne : on le repère à l'intérieur d'une formule.
FORMULE = re.compile(r"\$\$[^$]*\$\$|\$[^$]*\$")


def charger(chemin, problemes):
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        problemes.append("{} : illisible ({})".format(os.path.basename(chemin), e))
        return None


def parcourir_textes(objet, chemin=""):
    """Renvoie toutes les chaînes de l'objet avec leur chemin, pour les contrôles de texte."""
    if isinstance(objet, str):
        yield chemin, objet
    elif isinstance(objet, dict):
        for k, v in objet.items():
            for x in parcourir_textes(v, chemin + "." + k):
                yield x
    elif isinstance(objet, list):
        for i, v in enumerate(objet):
            for x in parcourir_textes(v, "{}[{}]".format(chemin, i)):
                yield x


def verifier_fiche(fiche, attendu, problemes):
    nom = attendu["id"] + ".json"
    p = lambda msg: problemes.append(nom + " : " + msg)

    for champ, type_ in CHAMPS_FICHE.items():
        if champ not in fiche:
            p("champ manquant « {} »".format(champ))
        elif not isinstance(fiche[champ], type_):
            p("« {} » devrait être de type {}".format(champ, type_.__name__))
    if fiche.get("id") != attendu["id"] or fiche.get("numero") != attendu["numero"]:
        p("id/numero ne correspondent pas à index.json")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(fiche.get("date", ""))):
        p("date au format AAAA-MM-JJ attendue")

    # Déroulé : créneaux contigus de 0 à 120 min, dont un créneau « Code ».
    deroule = fiche.get("deroule") or []
    t = 0
    for i, c in enumerate(deroule):
        for k in ("debut", "fin", "titre", "qui", "contenu"):
            if k not in c:
                p("deroule[{}] : champ « {} » manquant".format(i, k))
        if c.get("debut") != t:
            p("deroule[{}] : commence à {} au lieu de {}".format(i, c.get("debut"), t))
        t = c.get("fin", t)
    if deroule and t != 120:
        p("le déroulé finit à {} min au lieu de 120".format(t))
    code = [c for c in deroule if re.search("code", c.get("titre", ""), re.I)]

    # Tâches : identifiants uniques, durée totale hors bonus compatible avec le créneau Code.
    ids = set()
    total = 0
    for i, tache in enumerate(fiche.get("taches") or []):
        for k in ("id", "titre", "duree", "etapes"):
            if k not in tache:
                p("taches[{}] : champ « {} » manquant".format(i, k))
        if tache.get("id") in ids:
            p("taches : id « {} » en double".format(tache.get("id")))
        ids.add(tache.get("id"))
        if not tache.get("bonus"):
            total += tache.get("duree", 0)
    if code:
        dispo = code[0]["fin"] - code[0]["debut"]
        if total > dispo:
            p("tâches hors bonus : {} min pour un créneau Code de {} min".format(total, dispo))

    # Théorie : identifiants uniques et champs présents.
    ids = set()
    for i, n in enumerate(fiche.get("theorie") or []):
        for k in ("id", "titre", "duree", "contenu"):
            if k not in n:
                p("theorie[{}] : champ « {} » manquant".format(i, k))
        if n.get("id") in ids:
            p("theorie : id « {} » en double".format(n.get("id")))
        ids.add(n.get("id"))

    for i, q in enumerate(fiche.get("rappel_actif") or []):
        if "question" not in q or "reponse" not in q:
            p("rappel_actif[{}] : « question » et « reponse » attendus".format(i))
    for i, q in enumerate(fiche.get("questions_jury") or []):
        if "question" not in q or "piste" not in q:
            p("questions_jury[{}] : « question » et « piste » attendus".format(i))

    # Textes : antislashs LaTeX mal échappés et nombre impair de « $ ».
    for chemin, texte in parcourir_textes(fiche):
        if CONTROLE.search(texte):
            p("{} : caractère de contrôle (antislash LaTeX non doublé ? écrire \\\\frac et non \\frac)".format(chemin))
        sans_code = re.sub(r"`[^`]*`", "", texte)
        if sans_code.count("$") % 2:
            p("{} : nombre impair de « $ »".format(chemin))
        elif any("\n" in m for m in FORMULE.findall(sans_code)):
            p("{} : saut de ligne dans une formule (\\nabla ou \\neq non doublé ?)".format(chemin))


def main():
    problemes = []
    index = charger(os.path.join(DOSSIER, "index.json"), problemes)
    if index is None:
        print("\n".join(problemes))
        return 1

    seances = index.get("seances", [])
    ids = [s.get("id") for s in seances]
    if len(set(ids)) != len(ids):
        problemes.append("index.json : ids de séance en double")
    if index.get("seance_courante") not in ids:
        problemes.append("index.json : seance_courante inconnue")
    statuts = {"faite", "prochaine", "a_venir"}
    for s in seances:
        if s.get("statut") not in statuts:
            problemes.append("index.json : {} a un statut invalide".format(s.get("id")))
    if sum(1 for s in seances if s.get("statut") == "prochaine") > 1:
        problemes.append("index.json : une seule séance peut être « prochaine »")

    for s in seances:
        chemin = os.path.join(DOSSIER, s["id"] + ".json")
        existe = os.path.exists(chemin)
        if s.get("fiche") and not existe:
            problemes.append("index.json : {} marquée fiche=true mais {} absent".format(s["id"], s["id"] + ".json"))
        if existe and not s.get("fiche"):
            problemes.append("index.json : {}.json existe mais fiche=false".format(s["id"]))
        if existe:
            fiche = charger(chemin, problemes)
            if fiche is not None:
                verifier_fiche(fiche, s, problemes)

    if problemes:
        print("Fiches invalides :")
        for pb in problemes:
            print(" - " + pb)
        return 1
    print("Fiches valides ({} fiche(s)).".format(sum(1 for s in seances if s.get("fiche"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
