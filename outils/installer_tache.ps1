# Programme preparer_fiche.ps1 tous les lundis à 20 h dans le Planificateur de tâches Windows.
# À lancer une seule fois : powershell -ExecutionPolicy Bypass -File outils\installer_tache.ps1
# Pour supprimer la tâche : Unregister-ScheduledTask -TaskName "TIPE - fiche du mardi"

$script = Join-Path $PSScriptRoot "preparer_fiche.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "' + $script + '"')
$declencheur = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "20:00"
# StartWhenAvailable : si le PC est éteint à 20 h, la tâche part dès qu'il est rallumé.
$reglages = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName "TIPE - fiche du mardi" -Action $action -Trigger $declencheur `
    -Settings $reglages -Description "Prépare la fiche de la séance de TIPE du mardi (voir CLAUDE.md)" -Force | Out-Null

Write-Output "Tâche « TIPE - fiche du mardi » programmée tous les lundis à 20:00."
Write-Output "Test immédiat : Start-ScheduledTask -TaskName 'TIPE - fiche du mardi'"
