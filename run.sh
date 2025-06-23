#!/bin/bash

echo "🚀 Lancement de l'application ICD-10/SNOMED Search"
echo "=================================================="

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

echo "✅ Python détecté: $(python3 --version)"

# Vérifier que pip est installé
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 n'est pas installé"
    exit 1
fi

echo "✅ pip détecté"

# Installer les dépendances si nécessaire
echo ""
echo "📦 Vérification des dépendances..."
pip3 install -r requirements.txt

# Lancer les tests
echo ""
echo "🧪 Tests de l'application..."
python3 test_app.py

# Si les tests passent, lancer l'application
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Lancement de l'application Streamlit..."
    echo "📱 L'application sera disponible sur: http://localhost:8501"
    echo ""
    streamlit run app.py
else
    echo "❌ Les tests ont échoué. Vérifiez la configuration."
    exit 1
fi
