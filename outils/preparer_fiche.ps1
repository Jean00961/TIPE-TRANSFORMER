# Prépare la fiche de la séance de TIPE du lendemain (lancé le lundi soir par le Planificateur de tâches).
# Claude suit la section « Fiches de séance » de CLAUDE.md : état réel, fiche JSON, vérification, commit, push.
# Lancement manuel : powershell -ExecutionPolicy Bypass -File outils\preparer_fiche.ps1

$ErrorActionPreference = "Stop"
$depot = Split-Path -Parent $PSScriptRoot
Set-Location $depot

# Journal d'exécution (ignoré par git)
$dossierLogs = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $dossierLogs | Out-Null
$log = Join-Path $dossierLogs ("fiche-" + (Get-Date -Format "yyyy-MM-dd_HH-mm") + ".log")

function Ecrire($message) {
    $ligne = (Get-Date -Format "HH:mm:ss") + "  " + $message
    $ligne | Out-File -FilePath $log -Append -Encoding utf8
}

# Trouver Claude Code : d'abord dans le PATH, sinon la version la plus récente fournie par l'extension VS Code.
$claude = $null
$commande = Get-Command claude -ErrorAction SilentlyContinue
if ($commande) {
    $claude = $commande.Source
} else {
    $candidats = Get-ChildItem -Path (Join-Path $env:USERPROFILE ".vscode\extensions") -Directory -Filter "anthropic.claude-code-*" -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "resources\native-binary\claude.exe" } |
        Where-Object { Test-Path $_ } |
        Sort-Object { (Get-Item $_).LastWriteTime } -Descending
    if ($candidats) { $claude = @($candidats)[0] }
}
if (-not $claude) {
    Ecrire "ERREUR : Claude Code introuvable (ni dans le PATH, ni dans les extensions VS Code)."
    exit 1
}
Ecrire "Claude : $claude"
Ecrire "Dépôt : $depot"

$consigne = @'
Prépare la fiche de la séance de TIPE de demain en suivant la section « Fiches de séance » de CLAUDE.md, étapes 1 à 9.
Tu es lancé automatiquement, sans personne pour répondre : ne pose aucune question, fais le choix le plus raisonnable et signale-le dans le résumé final.
'@

$outils = @(
    "Read", "Glob", "Grep", "Edit", "Write",
    "Bash(git:*)", "Bash(python:*)", "Bash(ls:*)", "Bash(cat:*)",
    "PowerShell(git:*)", "PowerShell(python:*)"
)

Ecrire "Lancement de Claude…"
$sortie = & $claude -p $consigne --permission-mode acceptEdits --allowedTools $outils 2>&1
$code = $LASTEXITCODE
$sortie | Out-File -FilePath $log -Append -Encoding utf8
Ecrire "Terminé (code $code)."

# Notification Windows simple avec le résumé de Claude
try {
    Add-Type -AssemblyName System.Windows.Forms
    $resume = ($sortie | Out-String).Trim()
    if ($resume.Length -gt 600) { $resume = $resume.Substring($resume.Length - 600) }
    $titre = if ($code -eq 0) { "Fiche TIPE prête" } else { "Fiche TIPE : échec (voir $log)" }
    [System.Windows.Forms.MessageBox]::Show($resume, $titre) | Out-Null
} catch { }

exit $code
