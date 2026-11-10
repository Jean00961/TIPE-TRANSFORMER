/** Petites icônes au trait, dessinées à la main pour éviter une dépendance. */
const TRACES = {
  accueil: (
    <>
      <path d="M3 10.5 12 3l9 7.5" />
      <path d="M5 9v11h5v-6h4v6h5V9" />
    </>
  ),
  deroule: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  rappel: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.5 9.5a2.5 2.5 0 1 1 3.4 2.3c-.6.3-.9.8-.9 1.4v.3" />
      <path d="M12 17h.01" />
    </>
  ),
  theorie: (
    <>
      <path d="M3 5h6a3 3 0 0 1 3 3v12a2 2 0 0 0-2-2H3z" />
      <path d="M21 5h-6a3 3 0 0 0-3 3v12a2 2 0 0 1 2-2h7z" />
    </>
  ),
  code: (
    <>
      <path d="m8 8-4 4 4 4" />
      <path d="m16 8 4 4-4 4" />
      <path d="m14 5-4 14" />
    </>
  ),
  livrables: (
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="3" />
      <path d="m8 12 3 3 5-6" />
    </>
  ),
  jury: (
    <>
      <path d="M20 12a8 8 0 0 1-11.5 7.2L4 20l.9-4.2A8 8 0 1 1 20 12z" />
    </>
  ),
  cloture: (
    <>
      <path d="M5 21V4" />
      <path d="M5 4h12l-2.5 4L17 12H5" />
    </>
  ),
  menu: (
    <>
      <path d="M4 7h16" />
      <path d="M4 12h16" />
      <path d="M4 17h16" />
    </>
  ),
  fermer: (
    <>
      <path d="m6 6 12 12" />
      <path d="M18 6 6 18" />
    </>
  ),
} as const;

export type NomIcone = keyof typeof TRACES;

export function Icone({ nom }: { nom: NomIcone }) {
  return (
    <svg
      className="icone"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {TRACES[nom]}
    </svg>
  );
}
