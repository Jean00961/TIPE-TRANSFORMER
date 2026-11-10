// Markdown réduit utilisé dans les fiches : `code`, $formule$, $$formule$$, **gras**, *italique*,
// listes « - » et « 1. », blocs de code entre lignes ```.

export type Segment =
  | { type: "texte"; valeur: string }
  | { type: "code"; valeur: string }
  | { type: "formule"; valeur: string; bloc: boolean }
  | { type: "gras"; valeur: string }
  | { type: "italique"; valeur: string };

export type Bloc =
  | { type: "paragraphe"; lignes: string[] }
  | { type: "liste"; ordonnee: boolean; items: string[] }
  | { type: "code"; texte: string };

const MOTIF_EN_LIGNE = /`([^`]+)`|\$\$([^$]+)\$\$|\$([^$]+)\$|\*\*([^*]+)\*\*|\*([^*\s][^*]*)\*/g;

export function decouperEnLigne(texte: string): Segment[] {
  const segments: Segment[] = [];
  let curseur = 0;
  for (const m of texte.matchAll(MOTIF_EN_LIGNE)) {
    const debut = m.index ?? 0;
    if (debut > curseur) segments.push({ type: "texte", valeur: texte.slice(curseur, debut) });
    const [, code, formuleBloc, formule, gras, italique] = m;
    if (code !== undefined) segments.push({ type: "code", valeur: code });
    else if (formuleBloc !== undefined) segments.push({ type: "formule", valeur: formuleBloc, bloc: true });
    else if (formule !== undefined) segments.push({ type: "formule", valeur: formule, bloc: false });
    else if (gras !== undefined) segments.push({ type: "gras", valeur: gras });
    else if (italique !== undefined) segments.push({ type: "italique", valeur: italique });
    curseur = debut + m[0].length;
  }
  if (curseur < texte.length) segments.push({ type: "texte", valeur: texte.slice(curseur) });
  return segments;
}

const PUCE = /^\s*- /;
const NUMERO = /^\s*\d+\. /;
const CLOTURE_CODE = /^\s*```/;

export function decouperBlocs(texte: string): Bloc[] {
  const lignes = texte.split("\n");
  const blocs: Bloc[] = [];
  let i = 0;
  const ligne = (k: number) => lignes[k] ?? "";

  while (i < lignes.length) {
    const courante = ligne(i);
    if (CLOTURE_CODE.test(courante)) {
      const code: string[] = [];
      i++;
      while (i < lignes.length && !CLOTURE_CODE.test(ligne(i))) code.push(ligne(i++));
      i++;
      blocs.push({ type: "code", texte: code.join("\n") });
    } else if (PUCE.test(courante) || NUMERO.test(courante)) {
      const motif = PUCE.test(courante) ? PUCE : NUMERO;
      const items: string[] = [];
      while (i < lignes.length && motif.test(ligne(i))) items.push(ligne(i++).replace(motif, ""));
      blocs.push({ type: "liste", ordonnee: motif === NUMERO, items });
    } else if (courante.trim() === "") {
      i++;
    } else {
      const paragraphe: string[] = [];
      while (
        i < lignes.length &&
        ligne(i).trim() !== "" &&
        !CLOTURE_CODE.test(ligne(i)) &&
        !PUCE.test(ligne(i)) &&
        !NUMERO.test(ligne(i))
      ) {
        paragraphe.push(ligne(i++));
      }
      blocs.push({ type: "paragraphe", lignes: paragraphe });
    }
  }
  return blocs;
}

/** Version sans balisage, pour le titre de l'onglet ou le journal. */
export function texteBrut(texte: string): string {
  return texte.replace(/[`*]/g, "").replace(/\$([^$]+)\$/g, "$1");
}
