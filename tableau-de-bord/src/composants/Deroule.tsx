import { formaterMinutes, indiceCreneau } from "../progression";
import { EnLigne } from "../texte/Texte";
import type { Fiche } from "../types";

export function Deroule({ fiche, minutesEcoulees }: { fiche: Fiche; minutesEcoulees: number | null }) {
  const actuel = minutesEcoulees === null ? null : indiceCreneau(fiche, minutesEcoulees);

  const etatCreneau = (k: number) => {
    if (actuel === null) return "";
    if (actuel === -1 || k < actuel) return "creneau--passe";
    return k === actuel ? "creneau--actuel" : "";
  };

  return (
    <ol className="carte deroule">
      {fiche.deroule.map((c, k) => (
        <li key={k} className={`creneau ${etatCreneau(k)}`} aria-current={k === actuel ? "step" : undefined}>
          <div className="creneau__heure">
            {formaterMinutes(c.debut)}–{formaterMinutes(c.fin)}
          </div>
          <div>
            <div className="creneau__titre">
              <EnLigne texte={c.titre} /> <span className="discret">· {c.qui}</span>
            </div>
            <p>
              <EnLigne texte={c.contenu} />
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}
