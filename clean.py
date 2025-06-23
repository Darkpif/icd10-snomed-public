#!/usr/bin/env python3
"""
🧹 Script de nettoyage automatique du projet ICD-10/SNOMED
Nettoie les fichiers temporaires, caches et prépare pour la production.
"""

import os
import shutil
import glob
from pathlib import Path

def clean_python_cache():
    """Supprimer tous les caches Python."""
    print("🐍 Nettoyage des caches Python...")
    
    # Supprimer __pycache__
    for pycache_dir in glob.glob("**/__pycache__", recursive=True):
        if os.path.exists(pycache_dir):
            shutil.rmtree(pycache_dir)
            print(f"   ✅ Supprimé: {pycache_dir}")
    
    # Supprimer fichiers .pyc
    for pyc_file in glob.glob("**/*.pyc", recursive=True):
        if os.path.exists(pyc_file):
            os.remove(pyc_file)
            print(f"   ✅ Supprimé: {pyc_file}")

def clean_test_files():
    """Supprimer les fichiers de test."""
    print("🧪 Nettoyage des fichiers de test...")
    
    test_patterns = [
        "test_*.py",
        "*_test.py",
        "launch_fixed.py",
        "run_fixed.bat"
    ]
    
    for pattern in test_patterns:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                os.remove(file)
                print(f"   ✅ Supprimé: {file}")

def clean_logs_and_temp():
    """Supprimer logs et fichiers temporaires."""
    print("📊 Nettoyage des logs et fichiers temporaires...")
    
    temp_patterns = [
        "*.log",
        "temp/*",
        "tmp/*",
        ".streamlit/secrets.toml"
    ]
    
    for pattern in temp_patterns:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)
                print(f"   ✅ Supprimé: {file}")

def optimize_scripts():
    """Optimiser les scripts de lancement."""
    print("🚀 Optimisation des scripts...")
    
    # Créer un script de lancement simple et propre
    launch_script = """@echo off
REM 🚀 Lancement optimisé de l'application ICD-10/SNOMED
echo 🔍 Démarrage de l'application ICD-10/SNOMED Search...
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou pas dans le PATH
    pause
    exit /b 1
)

REM Vérifier les dépendances
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo 📦 Installation des dépendances...
    pip install -r requirements.txt
)

REM Lancer l'application
echo ✅ Lancement de Streamlit...
streamlit run app.py

pause
"""
    
    with open("run.bat", "w", encoding="utf-8") as f:
        f.write(launch_script)
    print("   ✅ Script run.bat optimisé")

def create_production_structure():
    """Créer la structure pour la production."""
    print("📁 Vérification de la structure de production...")
    
    required_dirs = [
        "data/faiss",
        "data/icd10",
        ".streamlit"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ Créé: {dir_path}")

def verify_essential_files():
    """Vérifier que les fichiers essentiels sont présents."""
    print("🔍 Vérification des fichiers essentiels...")
    
    essential_files = [
        "app.py",
        "semantic_search.py", 
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"   ✅ {file}")
    
    if missing_files:
        print(f"   ⚠️  Fichiers manquants: {missing_files}")
        return False
    
    # Vérifier les données
    faiss_files = glob.glob("data/faiss/*.faiss")
    icd_files = glob.glob("data/icd10/*.xml")
    
    print(f"   📊 Index FAISS: {len(faiss_files)} fichiers")
    print(f"   📋 Fichiers ICD-10: {len(icd_files)} fichiers")
    
    return True

def main():
    """Fonction principale de nettoyage."""
    print("🧹 NETTOYAGE DU PROJET ICD-10/SNOMED")
    print("=" * 50)
    
    # Changer vers le répertoire du script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Étapes de nettoyage
        clean_python_cache()
        clean_test_files()
        clean_logs_and_temp()
        optimize_scripts()
        create_production_structure()
        
        print("\n🔍 VÉRIFICATION FINALE")
        print("=" * 30)
        
        if verify_essential_files():
            print("\n🎉 NETTOYAGE TERMINÉ AVEC SUCCÈS!")
            print("✅ Le projet est prêt pour la production")
            print("✅ Tous les fichiers essentiels sont présents")
            print("\n🚀 Pour déployer:")
            print("   1. Testez localement: run.bat")
            print("   2. Committez les changements: git add . && git commit")
            print("   3. Déployez sur Streamlit Cloud")
        else:
            print("\n⚠️  Nettoyage terminé avec des avertissements")
            print("   Vérifiez les fichiers manquants ci-dessus")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant le nettoyage: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
