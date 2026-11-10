import { EnLigne } from "../texte/Texte";
import type { Tache } from "../types";
import { ListeTexte, QuestionRevelable } from "./communs";

export function Taches({
  taches,
  coches,
  onCocher,
}: {
  taches: Tache[];
  coches: Record<string, boolean>;
  onCocher: (id: string, fait: boolean) => void;
}) {
  return (
    <ol className="pile taches">
      {taches.map((t) => {
        const faite = !!coches[t.id];
        return (
          <li key={t.id} className={`carte tache ${faite ? "tache--faite" : ""}`}>
            <label className="tache__tete">
              <input type="checkbox" checked={faite} onChange={(e) => onCocher(t.id, e.target.checked)} />
              <span className="tache__titre">
                <EnLigne texte={t.titre} />
                <span className="tache__infos">
                  <span className="discret">{t.duree} min</span>
                  {t.fichier && <span className="mono discret">{t.fichier}</span>}
                  {t.bonus && <span className="badge badge--alerte">bonus</span>}
                </span>
              </span>
            </label>
            {!faite && (
              <div className="tache__corps">
                <ListeTexte items={t.etapes} />
                {t.test && (
                  <div className="tache__test">
                    <strong>Test · </strong>
                    <EnLigne texte={t.test} />
                  </div>
                )}
                {t.solution && <QuestionRevelable reponse={t.solution} libelle="Voir la réponse attendue" />}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
