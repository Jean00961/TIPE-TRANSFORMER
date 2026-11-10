import { useEffect } from "react";
import { BarreHaut } from "./composants/BarreHaut";
import { Cloture } from "./composants/Cloture";
import { ListeTexte, QuestionRevelable, Section } from "./composants/communs";
import { Deroule } from "./composants/Deroule";
import { Frise } from "./composants/Frise";
import { PanneauSeance } from "./composants/PanneauSeance";
import { Taches } from "./composants/Taches";
import { Theorie } from "./composants/Theorie";
import { useHorloge, useStockageLocal } from "./hooks";
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
  const maintenant = useHorloge(5000);

  const minutesEcoulees = etat.debut === undefined ? null : (maintenant - etat.debut) / 60000;
  const phase = index.phases.find((p) => p.numero === fiche.phase);
  const resume = index.seances.find((s) => s.id === fiche.id);
  const code = minutesDeCode(fiche, etat.coches);
  const dureeTheorie = fiche.theorie.reduce((s, n) => s + n.duree, 0);

  useEffect(() => {
    document.title = `S${fiche.numero} · ${texteBrut(fiche.titre)}`;
  }, [fiche]);

  const modifier = (changement: Partial<EtatSeance>) => setEtat((e) => ({ ...e, ...changement }));

  let chrono: string | null = null;
  if (minutesEcoulees !== null) {
    const creneau = fiche.deroule[indiceCreneau(fiche, minutesEcoulees)];
    chrono = `${formaterMinutes(minutesEcoulees)} · ${creneau ? texteBrut(creneau.titre) : "Clôture"}`;
  }

  return (
    <>
      <BarreHaut chrono={chrono} modeRappel={modeRappel} onModeRappel={setModeRappel} />

      <div className={`page ${modeRappel ? "page--rappel" : ""}`}>
        <Frise index={index} affichee={fiche.id} onChoisir={onChoisir} />

        <main>
          <header className="entete">
            <div className="entete__surtitre">
              Séance {fiche.numero} / {index.seances.length} · Phase {fiche.phase} : {phase?.titre}
              {resume?.bloquante && " · séance bloquante"}
            </div>
            <h1>
              <EnLigne texte={fiche.titre} />
            </h1>
            <div className="entete__meta">
              <span>{dateLongue(fiche.date)}</span>
              <span className="puce">
                Pilote <b>{fiche.pilote}</b>
              </span>
              <span className="puce">
                Copilote <b>{fiche.copilote}</b>
              </span>
            </div>
            <p className="entete__objectif">
              <EnLigne texte={fiche.objectif} />
            </p>
            {phase && (
              <p className="discret">
                Jalon de fin de phase : <EnLigne texte={phase.jalon} />
              </p>
            )}
          </header>

          <PanneauSeance
            fiche={fiche}
            coches={etat.coches}
            minutesEcoulees={minutesEcoulees}
            onDemarrer={() => modifier({ debut: Date.now() })}
            onRemettreAZero={() => modifier({ debut: undefined })}
          />

          <Section id="etat" titre="Où on en est">
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
          </Section>

          <Section id="acquis" titre="À la fin de la séance, on sait…">
            <div className="carte">
              <ListeTexte items={fiche.acquis} className="acquis" />
            </div>
          </Section>

          <Section id="deroule" titre="Déroulé" precision="2 h">
            <Deroule fiche={fiche} minutesEcoulees={minutesEcoulees} />
          </Section>

          {fiche.rappel_actif.length > 0 && (
            <Section id="rappel" titre="Rappel actif" precision="de tête, sans notes">
              <div className="pile">
                {fiche.rappel_actif.map((q, k) => (
                  <div key={k} className="carte">
                    <QuestionRevelable question={q.question} reponse={q.reponse} libelle="Voir la réponse" />
                  </div>
                ))}
              </div>
            </Section>
          )}

          <Section id="theorie" titre="Théorie" precision={`${dureeTheorie} min`}>
            <Theorie notions={fiche.theorie} modeRappel={modeRappel} />
          </Section>

          <Section id="taches" titre="Code" precision={`${code.total} min hors bonus`}>
            <Taches
              taches={fiche.taches}
              coches={etat.coches}
              onCocher={(id, fait) => modifier({ coches: { ...etat.coches, [id]: fait } })}
            />
          </Section>

          {fiche.plan_b.length > 0 && (
            <Section id="plan-b" titre="Si on prend du retard">
              <div className="carte">
                <ListeTexte items={fiche.plan_b} />
              </div>
            </Section>
          )}

          <Section id="livrables" titre="Livrables">
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
          </Section>

          {fiche.questions_jury.length > 0 && (
            <Section id="jury" titre="Questions de jury possibles">
              <div className="pile">
                {fiche.questions_jury.map((q, k) => (
                  <div key={k} className="carte">
                    <QuestionRevelable question={q.question} reponse={q.piste} libelle="Voir une piste de réponse" />
                  </div>
                ))}
              </div>
            </Section>
          )}

          <Section id="cloture" titre="Clôture">
            <Cloture
              fiche={fiche}
              coches={etat.coches}
              journal={etat.journal}
              onJournal={(journal) => modifier({ journal })}
            />
          </Section>

          <footer className="pied discret">
            Fiche préparée le {dateLongue(fiche.genere_le)}. Les cases cochées et le chrono restent dans ce navigateur.
          </footer>
        </main>
      </div>
    </>
  );
}
