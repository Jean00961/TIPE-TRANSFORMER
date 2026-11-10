import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";
import { decouperBlocs, decouperEnLigne } from "./analyse";

export function Formule({ tex, bloc = false }: { tex: string; bloc?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: bloc, throwOnError: false, strict: false }),
    [tex, bloc],
  );
  return <span className="formule" dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Texte sur une ligne : code, formules, gras, italique. */
export function EnLigne({ texte }: { texte: string }) {
  return (
    <>
      {decouperEnLigne(texte).map((s, k) => {
        switch (s.type) {
          case "texte":
            return s.valeur;
          case "code":
            return <code key={k}>{s.valeur}</code>;
          case "formule":
            return <Formule key={k} tex={s.valeur} bloc={s.bloc} />;
          case "gras":
            return (
              <strong key={k}>
                <EnLigne texte={s.valeur} />
              </strong>
            );
          case "italique":
            return (
              <em key={k}>
                <EnLigne texte={s.valeur} />
              </em>
            );
        }
      })}
    </>
  );
}

/** Texte en blocs : paragraphes, listes, blocs de code. */
export function Blocs({ texte }: { texte: string }) {
  return (
    <>
      {decouperBlocs(texte).map((b, k) => {
        if (b.type === "code") {
          return (
            <pre key={k}>
              <code>{b.texte}</code>
            </pre>
          );
        }
        if (b.type === "liste") {
          const Balise = b.ordonnee ? "ol" : "ul";
          return (
            <Balise key={k}>
              {b.items.map((item, j) => (
                <li key={j}>
                  <EnLigne texte={item} />
                </li>
              ))}
            </Balise>
          );
        }
        return (
          <p key={k}>
            {b.lignes.map((l, j) => (
              <span key={j}>
                {j > 0 && <br />}
                <EnLigne texte={l} />
              </span>
            ))}
          </p>
        );
      })}
    </>
  );
}
