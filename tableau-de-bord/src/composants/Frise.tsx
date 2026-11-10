import type { MouseEvent } from "react";
import type { IndexSeances } from "../types";

/** Les 14 séances, groupées par phase, avec leur statut. */
export function Frise({
  index,
  affichee,
  onChoisir,
}: {
  index: IndexSeances;
  affichee: string;
  onChoisir: (id: string) => void;
}) {
  return (
    <nav className="frise" aria-label="Les séances">
      {index.phases.map((phase) => (
        <div key={phase.numero} className={`frise__phase phase-${phase.numero}`}>
          <div className="frise__titre" title={`Jalon : ${phase.jalon}`}>
            {phase.numero}. {phase.titre}
          </div>
          <div className="frise__seances">
            {index.seances
              .filter((s) => s.phase === phase.numero)
              .map((s) => {
                const classes = [
                  "pastille",
                  `pastille--${s.statut}`,
                  s.bloquante ? "pastille--bloquante" : "",
                  s.id === affichee ? "pastille--affichee" : "",
                ].join(" ");
                const bulle = `S${s.numero} : ${s.titre}${s.bloquante ? " (bloquante)" : ""}`;
                if (!s.fiche) {
                  return (
                    <span key={s.id} className={`${classes} pastille--sans-fiche`} title={`${bulle} · fiche pas encore écrite`}>
                      S{s.numero}
                    </span>
                  );
                }
                const ouvrir = (e: MouseEvent) => {
                  e.preventDefault();
                  onChoisir(s.id);
                };
                return (
                  <a
                    key={s.id}
                    className={classes}
                    href={`?s=${s.id}`}
                    title={bulle}
                    onClick={ouvrir}
                    aria-current={s.id === affichee ? "page" : undefined}
                  >
                    S{s.numero}
                  </a>
                );
              })}
          </div>
        </div>
      ))}
    </nav>
  );
}
