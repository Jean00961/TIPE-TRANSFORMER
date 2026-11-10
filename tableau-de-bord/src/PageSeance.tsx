import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Cloture } from "./composants/Cloture";
import { ListeTexte, QuestionRevelable } from "./composants/communs";
import { Deroule } from "./composants/Deroule";
import { Icone } from "./composants/Icone";
import { Lateral, type Categorie } from "./composants/Lateral";
import { PanneauSeance } from "./composants/PanneauSeance";
import { Taches } from "./composants/Taches";
import { Theorie } from "./composants/Theorie";
import { useCategorieDansAdresse, useHorloge, useStockageLocal } from "./hooks";
import { dateLongue, formaterMinutes, indiceCreneau, minutesDeCode } from "./progression";
import { texteBrut } from "./texte/analyse";
import { Blocs, EnLigne } from "./texte/Texte";
import type { EtatSeance, Fiche, IndexSeances } from "./types";

const ETAT_VIDE: EtatSeance = { coches: {}, livrables: {} };

/** Accepte l'état stocké s'il a la bonne forme, sinon repart de zéro. */
function normaliserEtat(brut: unknown): EtatSeance {
  if (typeof brut !== "object" || brut === null) return ETAT_VIDE;
  const b = brut as Partial<EtatSeance>;
  return {
    coches: typeof b.coches === "object" && b.coches !== null ? b.coches : {},
    livrables: typeof b.livrables === "object" && b.livrables !== null ? b.livrables : {},
    debut: typeof b.debut === "number" ? b.debut : undefined,
    journal: typeof b.journal === "string" ? b.journal : undefined,
  };
}

/** Sous-titre à l'intérieur d'une catégorie. */
function Bloc({ titre, precision, children }: { titre: string; precision?: string; children: ReactNode }) {
  return (
    <section className="bloc">
      <h2 className="bloc__titre">
        {titre}
        {precision && <span className="discret">{precision}</span>}
      </h2>
      {children}
    </section>
  );
}

export function PageSeance({
  fiche,
  index,
  onChoisir,
}: {
  fiche: Fiche;
  index: IndexSeances;
  onChoisir: (id: string) => void;
}) {
  const [etat, setEtat] = useStockageLocal(`tipe-fiche-${fiche.id}`, ETAT_VIDE, normaliserEtat);
  const [modeRappel, setModeRappel] = useStockageLocal("tipe-mode-rappel", false, (b) => b === true);
  const [menuOuvert, setMenuOuvert] = useState(false);
  const maintenant = useHorloge(5000);

  const minutesEcoulees = etat.debut === undefined ? null : (maintenant - etat.debut) / 60000;
  const phase = index.phases.find((p) => p.numero === fiche.phase);
  const resume = index.seances.find((s) => s.id === fiche.id);
  const code = minutesDeCode(fiche, etat.coches);
  const dureeTheorie = fiche.theorie.reduce((s, n) => s + n.duree, 0);
  const livrablesFaits = fiche.livrables.filter((_, k) => etat.livrables[k]).length;

  const categories = useMemo(() => {
    const liste: (Categorie | false)[] = [
      { id: "accueil", libelle: "Vue d'ensemble", icone: "accueil" },
      { id: "deroule", libelle: "Déroulé", icone: "deroule", badge: "2 h" },
      fiche.rappel_actif.length > 0 && {
        id: "rappel",
        libelle: "Rappel actif",
        icone: "rappel",
        badge: String(fiche.rappel_actif.length),
      },
      { id: "theorie", libelle: "Théorie", icone: "theorie", badge: `${dureeTheorie} min` },
      { id: "code", libelle: "Code", icone: "code", badge: `${code.nombreFaites}/${code.nombreTotal}` },
      { id: "livrables", libelle: "Livrables", icone: "livrables", badge: `${livrablesFaits}/${fiche.livrables.length}` },
      fiche.questions_jury.length > 0 && {
        id: "jury",
        libelle: "Questions de jury",
        icone: "jury",
        badge: String(fiche.questions_jury.length),
      },
      { id: "cloture", libelle: "Clôture et journal", icone: "cloture" },
    ];
    return liste.filter((c): c is Categorie => c !== false);
  }, [fiche, dureeTheorie, code.nombreFaites, code.nombreTotal, livrablesFaits]);

  const ids = useMemo(() => categories.map((c) => c.id), [categories]);
  const [active, allerA] = useCategorieDansAdresse(ids);
  const rang = ids.indexOf(active);
  const categorie = categories[rang];
  const precedente = categories[rang - 1];
  const suivante = categories[rang + 1];

  useEffect(() => {
    document.title = `S${fiche.numero} · ${texteBrut(fiche.titre)}`;
  }, [fiche]);

  // Échap ferme le menu sur téléphone.
  useEffect(() => {
    if (!menuOuvert) return;
    const surTouche = (e: KeyboardEvent) => e.key === "Escape" && setMenuOuvert(false);
    window.addEventListener("keydown", surTouche);
    return () => window.removeEventListener("keydown", surTouche);
  }, [menuOuvert]);

  const modifier = (changement: Partial<EtatSeance>) => setEtat((e) => ({ ...e, ...changement }));
  const demarrer = () => modifier({ debut: Date.now() });

  const ouvrirCategorie = (id: string) => {
    setMenuOuvert(false);
    allerA(id);
  };

  let chrono: { temps: string; creneau: string } | null = null;
  if (minutesEcoulees !== null) {
    const creneau = fiche.deroule[indiceCreneau(fiche, minutesEcoulees)];
    chrono = { temps: formaterMinutes(minutesEcoulees), creneau: creneau ? texteBrut(creneau.titre) : "Clôture" };
  }

  const contenus: Record<string, () => ReactNode> = {
    accueil: () => (
      <>
        <div className="entete__meta">
          <span>{dateLongue(fiche.date)}</span>
          <span className="puce">
            Pilote <b>{fiche.pilote}</b>
          </span>
          <span className="puce">
            Copilote <b>{fiche.copilote}</b>
          </span>
          {resume?.bloquante && <span className="badge badge--alerte">séance bloquante</span>}
        </div>
        <p className="entete__objectif">
          <EnLigne texte={fiche.objectif} />
        </p>
        {phase && (
          <p className="discret">
            Phase {phase.numero} : {phase.titre}. Jalon de fin de phase : <EnLigne texte={phase.jalon} />
          </p>
        )}

        <PanneauSeance
          fiche={fiche}
          coches={etat.coches}
          minutesEcoulees={minutesEcoulees}
          onDemarrer={demarrer}
          onRemettreAZero={() => modifier({ debut: undefined })}
        />

        <Bloc titre="Où on en est">
          <div className="carte">
            <Blocs texte={fiche.etat.resume} />
            {fiche.etat.reliquats.length > 0 && (
              <>
                <p className="sous-titre">Reliquats de la séance précédente, à faire en premier</p>
                <ListeTexte items={fiche.etat.reliquats} />
              </>
            )}
          </div>
          {fiche.etat.alertes.map((a, k) => (
            <div key={k} className="alerte">
              <strong>À surveiller · </strong>
              <EnLigne texte={a} />
            </div>
          ))}
        </Bloc>

        <Bloc titre="À la fin de la séance, on sait…">
          <div className="carte">
            <ListeTexte items={fiche.acquis} className="acquis" />
          </div>
        </Bloc>
      </>
    ),

    deroule: () => <Deroule fiche={fiche} minutesEcoulees={minutesEcoulees} />,

    rappel: () => (
      <>
        <p className="intro">De tête, sans notes : on répond à voix haute, puis on vérifie.</p>
        <div className="pile">
          {fiche.rappel_actif.map((q, k) => (
            <div key={k} className="carte">
              <QuestionRevelable question={q.question} reponse={q.reponse} libelle="Voir la réponse" />
            </div>
          ))}
        </div>
      </>
    ),

    theorie: () => (
      <>
        <label className="interrupteur intro" title="Masque la théorie pour se tester avant de la relire">
          <input type="checkbox" checked={modeRappel} onChange={(e) => setModeRappel(e.target.checked)} />
          Mode rappel : masquer les notions pour se tester avant de les relire
        </label>
        <Theorie notions={fiche.theorie} modeRappel={modeRappel} />
      </>
    ),

    code: () => (
      <>
        <p className="intro">
          {code.total} min hors bonus · {code.nombreFaites} / {code.nombreTotal} tâches faites. Une tâche cochée se replie.
        </p>
        <Taches
          taches={fiche.taches}
          coches={etat.coches}
          onCocher={(id, fait) => modifier({ coches: { ...etat.coches, [id]: fait } })}
        />
        {fiche.plan_b.length > 0 && (
          <Bloc titre="Si on prend du retard">
            <div className="carte">
              <ListeTexte items={fiche.plan_b} />
            </div>
          </Bloc>
        )}
      </>
    ),

    livrables: () => (
      <div className="carte">
        <ul className="cases">
          {fiche.livrables.map((l, k) => (
            <li key={k}>
              <label>
                <input
                  type="checkbox"
                  checked={!!etat.livrables[k]}
                  onChange={(e) => modifier({ livrables: { ...etat.livrables, [k]: e.target.checked } })}
                />
                <span>
                  <EnLigne texte={l} />
                </span>
              </label>
            </li>
          ))}
        </ul>
      </div>
    ),

    jury: () => (
      <div className="pile">
        {fiche.questions_jury.map((q, k) => (
          <div key={k} className="carte">
            <QuestionRevelable question={q.question} reponse={q.piste} libelle="Voir une piste de réponse" />
          </div>
        ))}
      </div>
    ),

    cloture: () => (
      <Cloture fiche={fiche} coches={etat.coches} journal={etat.journal} onJournal={(journal) => modifier({ journal })} />
    ),
  };

  return (
    <div className="appli">
      <Lateral
        categories={categories}
        active={active}
        onCategorie={ouvrirCategorie}
        index={index}
        affichee={fiche.id}
        onSeance={(id) => {
          setMenuOuvert(false);
          onChoisir(id);
        }}
        chrono={chrono}
        onDemarrer={demarrer}
        ouvert={menuOuvert}
        onFermer={() => setMenuOuvert(false)}
      />

      <div className="principal">
        <header className="barre-mobile">
          <button type="button" className="bouton-icone" onClick={() => setMenuOuvert(true)} aria-label="Ouvrir le menu">
            <Icone nom="menu" />
          </button>
          <span className="barre-mobile__titre">{categorie?.libelle}</span>
          {chrono && <span className="pastille-chrono">{chrono.temps}</span>}
        </header>

        <main className="contenu">
          <div className="contenu__surtitre">
            Séance {fiche.numero} / {index.seances.length} · <EnLigne texte={fiche.titre} />
          </div>
          <h1 className="contenu__titre">
            {categorie && <Icone nom={categorie.icone} />}
            {active === "accueil" ? <EnLigne texte={fiche.titre} /> : categorie?.libelle}
          </h1>

          {contenus[active]?.()}

          <nav className="pagination" aria-label="Catégorie précédente ou suivante">
            {precedente ? (
              <a
                href={`#${precedente.id}`}
                className="pagination__lien"
                onClick={(e) => {
                  e.preventDefault();
                  ouvrirCategorie(precedente.id);
                }}
              >
                <span className="discret">Précédent</span>
                <span>← {precedente.libelle}</span>
              </a>
            ) : (
              <span />
            )}
            {suivante && (
              <a
                href={`#${suivante.id}`}
                className="pagination__lien pagination__lien--suivant"
                onClick={(e) => {
                  e.preventDefault();
                  ouvrirCategorie(suivante.id);
                }}
              >
                <span className="discret">Suivant</span>
                <span>{suivante.libelle} →</span>
              </a>
            )}
          </nav>

          <footer className="pied discret">
            Fiche préparée le {dateLongue(fiche.genere_le)}. Les cases cochées et le chrono restent dans ce navigateur.
          </footer>
        </main>
      </div>
    </div>
  );
}
