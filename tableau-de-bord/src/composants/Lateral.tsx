import type { MouseEvent } from "react";
import type { IndexSeances } from "../types";
import { Icone, type NomIcone } from "./Icone";

export interface Categorie {
  id: string;
  libelle: string;
  icone: NomIcone;
  /** Petit compteur affiché à droite du libellé (« 2/5 », « 25 min »). */
  badge?: string;
}

/** Le menu de gauche : les catégories de la fiche, puis les 14 séances. */
export function Lateral({
  categories,
  active,
  onCategorie,
  index,
  affichee,
  onSeance,
  chrono,
  onDemarrer,
  ouvert,
  onFermer,
}: {
  categories: Categorie[];
  active: string;
  onCategorie: (id: string) => void;
  index: IndexSeances;
  affichee: string;
  onSeance: (id: string) => void;
  chrono: { temps: string; creneau: string } | null;
  onDemarrer: () => void;
  ouvert: boolean;
  onFermer: () => void;
}) {
  const choisirCategorie = (e: MouseEvent, id: string) => {
    e.preventDefault();
    onCategorie(id);
  };
  const choisirSeance = (e: MouseEvent, id: string) => {
    e.preventDefault();
    onSeance(id);
  };

  return (
    <>
      <div className={`voile ${ouvert ? "voile--visible" : ""}`} onClick={onFermer} aria-hidden="true" />
      <aside className={`lateral ${ouvert ? "lateral--ouvert" : ""}`}>
        <div className="lateral__haut">
          <a className="lateral__marque" href="./">
            <span className="lateral__logo" aria-hidden="true">
              T
            </span>
            <span>
              TIPE Transformer
              <span className="lateral__sous-marque">Fiches de séance</span>
            </span>
          </a>
          <button type="button" className="bouton-icone lateral__fermer" onClick={onFermer} aria-label="Fermer le menu">
            <Icone nom="fermer" />
          </button>
        </div>

        <div className="lateral__chrono">
          {chrono ? (
            <>
              <span className="lateral__temps">{chrono.temps}</span>
              <span className="lateral__creneau">{chrono.creneau}</span>
            </>
          ) : (
            <button type="button" className="bouton bouton--principal bouton--large" onClick={onDemarrer}>
              Démarrer la séance
            </button>
          )}
        </div>

        <nav aria-label="Catégories de la fiche">
          <div className="lateral__groupe">Cette séance</div>
          <ul className="lateral__liste">
            {categories.map((c) => (
              <li key={c.id}>
                <a
                  href={`#${c.id}`}
                  className={`lien-cat ${c.id === active ? "lien-cat--actif" : ""}`}
                  aria-current={c.id === active ? "page" : undefined}
                  onClick={(e) => choisirCategorie(e, c.id)}
                >
                  <Icone nom={c.icone} />
                  <span className="lien-cat__libelle">{c.libelle}</span>
                  {c.badge && <span className="lien-cat__badge">{c.badge}</span>}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <nav aria-label="Toutes les séances">
          <div className="lateral__groupe">Toutes les séances</div>
          {index.phases.map((phase) => (
            <div key={phase.numero} className={`lateral__phase phase-${phase.numero}`}>
              <div className="lateral__phase-titre" title={`Jalon : ${phase.jalon}`}>
                {phase.numero}. {phase.titre}
              </div>
              <ul className="lateral__liste">
                {index.seances
                  .filter((s) => s.phase === phase.numero)
                  .map((s) => {
                    const contenu = (
                      <>
                        <span className={`statut statut--${s.statut}`} aria-hidden="true" />
                        <span className="lien-seance__num">S{s.numero}</span>
                        <span className="lien-seance__titre">{s.titre}</span>
                        {s.bloquante && (
                          <span className="lien-seance__bloq" title="Séance bloquante : son jalon doit être atteint">
                            !
                          </span>
                        )}
                      </>
                    );
                    const bulle = `S${s.numero} : ${s.titre}${s.bloquante ? " (bloquante)" : ""}`;
                    if (!s.fiche) {
                      return (
                        <li key={s.id}>
                          <span className="lien-seance lien-seance--sans-fiche" title={`${bulle} · fiche pas encore écrite`}>
                            {contenu}
                          </span>
                        </li>
                      );
                    }
                    return (
                      <li key={s.id}>
                        <a
                          href={`?s=${s.id}`}
                          className={`lien-seance ${s.id === affichee ? "lien-seance--actif" : ""}`}
                          title={bulle}
                          aria-current={s.id === affichee ? "page" : undefined}
                          onClick={(e) => choisirSeance(e, s.id)}
                        >
                          {contenu}
                        </a>
                      </li>
                    );
                  })}
              </ul>
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}
