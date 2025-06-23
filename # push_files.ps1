# push_files.ps1
Write-Host "🚀 Push progressif des fichiers" -ForegroundColor Green
Write-Host "=" * 50

# Étape 1: Reset des fichiers déjà stagés
Write-Host "🔄 Reset des fichiers déjà stagés..." -ForegroundColor Cyan
git reset
Write-Host "✅ Zone de staging nettoyée" -ForegroundColor Green

# Vérifier le statut
Write-Host "`n📊 Statut Git actuel:" -ForegroundColor Cyan
git status --short

$files = @(
    "app.py",
    "semantic_search.py", 
    "requirements.txt",
    "README.md",
    ".gitignore"
)

$folders = @(
    ".streamlit",
    "data/icd10"
)

Write-Host "`n� Début du push progressif..." -ForegroundColor Green

# Push fichiers individuels
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "📁 Push: $file" -ForegroundColor Yellow
        git add $file
        git commit -m "📦 Add $file"
        git push origin main
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Erreur avec $file" -ForegroundColor Red
            exit 1
        }
        Start-Sleep -Seconds 2
    }
}

# Push dossiers
foreach ($folder in $folders) {
    if (Test-Path $folder) {
        Write-Host "📂 Push dossier: $folder" -ForegroundColor Yellow
        git add "$folder/"
        git commit -m "📁 Add $folder directory"
        git push origin main
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Erreur avec $folder" -ForegroundColor Red
            exit 1
        }
        Start-Sleep -Seconds 5
    }
}

Write-Host "`n✅ Push terminé!" -ForegroundColor Green

# Afficher le statut final
Write-Host "`n📊 Statut final:" -ForegroundColor Cyan
git status

# Vérifier les fichiers LFS si présents
if (Test-Path ".gitattributes") {
    Write-Host "`n📁 Fichiers Git LFS:" -ForegroundColor Cyan
    git lfs ls-files
}