const SECTIONS = [
  { ancre: "deroule", libelle: "Déroulé" },
  { ancre: "theorie", libelle: "Théorie" },
  { ancre: "taches", libelle: "Code" },
  { ancre: "cloture", libelle: "Clôture" },
];

export function BarreHaut({
  chrono,
  modeRappel,
  onModeRappel,
}: {
  chrono: string | null;
  modeRappel: boolean;
  onModeRappel: (actif: boolean) => void;
}) {
  return (
    <header className="barre">
      <div className="barre__interne">
        <a className="barre__marque" href="./">
          TIPE · Fiches de séance
        </a>
        <nav className="barre__sections" aria-label="Sections de la fiche">
          {SECTIONS.map((s) => (
            <a key={s.ancre} href={`#${s.ancre}`}>
              {s.libelle}
            </a>
          ))}
        </nav>
        <div className="barre__droite">
          {chrono && <span className="barre__chrono">{chrono}</span>}
          <label className="interrupteur" title="Masque la théorie pour se tester avant de la relire">
            <input type="checkbox" checked={modeRappel} onChange={(e) => onModeRappel(e.target.checked)} />
            Mode rappel
          </label>
        </div>
      </div>
    </header>
  );
}
