import { useState, type ReactNode } from "react";
import { Blocs, EnLigne } from "../texte/Texte";

export function Section({
  id,
  titre,
  precision,
  children,
}: {
  id: string;
  titre: string;
  precision?: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="section">
      <h2 className="section__titre">
        {titre}
        {precision && <span className="discret">{precision}</span>}
      </h2>
      {children}
    </section>
  );
}

/** Liste à puces simple dans une carte. */
export function ListeTexte({ items, className }: { items: string[]; className?: string }) {
  return (
    <ul className={className}>
      {items.map((item, k) => (
        <li key={k}>
          <EnLigne texte={item} />
        </li>
      ))}
    </ul>
  );
}

/** Une question dont la réponse reste cachée jusqu'au clic. */
export function QuestionRevelable({
  question,
  reponse,
  libelle,
}: {
  question?: string;
  reponse: string;
  libelle: string;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="revelable">
      {question && (
        <div className="revelable__question">
          <EnLigne texte={question} />
        </div>
      )}
      {visible ? (
        <div className="revelable__reponse">
          <Blocs texte={reponse} />
        </div>
      ) : (
        <button type="button" className="bouton bouton--lien" onClick={() => setVisible(true)}>
          {libelle} ↓
        </button>
      )}
    </div>
  );
}

export function Jauge({ ratio, ton = "accent" }: { ratio: number; ton?: "accent" | "ok" }) {
  const pourcentage = Math.round(Math.min(1, Math.max(0, ratio)) * 100);
  return (
    <div className={`jauge jauge--${ton}`} role="progressbar" aria-valuenow={pourcentage} aria-valuemin={0} aria-valuemax={100}>
      <div className="jauge__barre" style={{ width: `${pourcentage}%` }} />
    </div>
  );
}
