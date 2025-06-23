# 🚀 Guide de déploiement Streamlit Cloud

Ce document explique comment déployer l'application ICD-10/SNOMED Search sur Streamlit Cloud.

## 📋 Prérequis

1. **Compte GitHub** avec un repository public
2. **Compte Streamlit Cloud** (gratuit sur [share.streamlit.io](https://share.streamlit.io))
3. **Données préparées** : Ce repository contient déjà tous les fichiers nécessaires

## 📂 Structure du repository

```
icd10-snomed-search/
├── 🎨 app.py                     # Application Streamlit principale
├── 🧠 semantic_search.py         # Module de recherche IA avec cache
├── 📦 requirements.txt           # Dépendances Python optimisées
├── 📚 README.md                  # Documentation publique
├── 📜 LICENSE                    # Licence MIT
├── 🧪 test_app.py               # Script de test
├── 🚀 run.bat / run.sh          # Scripts de lancement local
├── ⚙️ .streamlit/
│   └── config.toml              # Configuration Streamlit
├── 📂 data/
│   ├── 🗂️ icd10/                # Fichiers XML ICD-10 (4 langues)
│   └── 🔍 faiss/                # Index FAISS et métadonnées pré-construits
└── 🙈 .gitignore                # Exclusions Git (sans données sensibles)
```

## 🛠️ Étapes de déploiement

### 1. 📤 Publier sur GitHub

```bash
# Initialiser le repository Git
cd icd10-snomed-public
git init

# Ajouter tous les fichiers
git add .

# Premier commit
git commit -m "🎉 Initial release: ICD-10/SNOMED semantic search app"

# Lier au repository GitHub (remplacer par votre URL)
git remote add origin https://github.com/VOTRE-USERNAME/icd10-snomed-search.git

# Pousser vers GitHub
git branch -M main
git push -u origin main
```

### 2. 🌐 Déployer sur Streamlit Cloud

1. **Aller sur [share.streamlit.io](https://share.streamlit.io)**
2. **Se connecter avec GitHub**
3. **Cliquer sur "New app"**
4. **Configurer le déploiement :**
   - **Repository** : `VOTRE-USERNAME/icd10-snomed-search`
   - **Branch** : `main`
   - **Main file path** : `app.py`
   - **App name** : `icd10-snomed-search` (ou personnalisé)

5. **Cliquer sur "Deploy!"**

### 3. ⏱️ Temps de déploiement

- **Premier déploiement** : ~5-10 minutes (téléchargement des modèles IA)
- **Déploiements suivants** : ~2-3 minutes (cache Streamlit)
- **Démarrage à froid** : ~30 secondes (cache des modèles)

## 🎯 Configuration optimisée

### ✨ Nouvelles fonctionnalités UX

- **🔍 Recherche par Entrée** : Appuyez sur **Entrée** dans le champ de saisie pour lancer la recherche
- **👁️ Concept ID cachés** : Les identifiants techniques SNOMED sont masqués par défaut
- **🚀 Préchargement optimisé** : Tous les modèles IA sont chargés une seule fois au démarrage
- **💡 Interface intuitive** : Hints visuels et instructions claires pour l'utilisateur

### Cache Streamlit
L'application utilise `@st.cache_resource` pour :
- ✅ **Modèles IA** : Chargés une seule fois
- ✅ **Index FAISS** : Mise en cache persistante
- ✅ **Descriptions ICD-10** : Chargement optimisé

### Modèles robustes
- **Généraliste** : `paraphrase-multilingual-MiniLM-L12-v2` (compatible cloud)
- **Médical** : Modèles spécialisés avec fallbacks
- **Multilingue** : Support FR, EN, DE, IT

## 🔧 Variables d'environnement (optionnel)

Aucune variable requise ! L'application fonctionne sans configuration.

Pour personnaliser (dans Streamlit Cloud > Settings > Secrets) :
```toml
# secrets.toml (optionnel)
[general]
app_title = "Mon App ICD-10"
contact_email = "votre-email@exemple.com"
```

## 📊 Monitoring et performance

### Métriques Streamlit Cloud
- **RAM** : ~1-2 GB (modèles IA en mémoire)
- **CPU** : Faible (FAISS optimisé)
- **Stockage** : ~500 MB (données + modèles)
- **Temps de réponse** : <2 secondes

### Logs utiles
```bash
# Streamlit affiche automatiquement :
✅ Modèle généraliste chargé pour fr
✅ Index généraliste chargé pour fr-ch  
✅ Métadonnées chargées pour fr-ch
```

## 🚨 Résolution de problèmes

### Erreur de mémoire
- **Cause** : Modèles trop volumineux
- **Solution** : Les modèles sont déjà optimisés pour le cloud

### Timeout au démarrage
- **Cause** : Premier chargement des modèles
- **Solution** : Normal, ~30 secondes la première fois

### Erreur import tiktoken
- **Cause** : Conflit avec CamemBERT  
- **Solution** : Déjà résolu dans `semantic_search.py`

### Fichiers manquants
- **Cause** : .gitignore trop restrictif
- **Solution** : Vérifier que les `.faiss` et `.xml` sont inclus

## 🔄 Mise à jour

```bash
# Modifier le code localement
git add .
git commit -m "✨ Nouvelle fonctionnalité"
git push

# Streamlit Cloud redéploie automatiquement !
```

## 📈 Optimisations avancées

### 1. Pré-chauffage du cache
Ajouter dans `app.py` :
```python
# Pré-charger au démarrage (optionnel)
if "models_loaded" not in st.session_state:
    load_models()
    st.session_state.models_loaded = True
```

### 2. Monitoring personnalisé
```python
import time
start_time = time.time()
# ... recherche ...
st.sidebar.metric("Temps de réponse", f"{time.time() - start_time:.2f}s")
```

## 🎉 Post-déploiement

1. **Tester l'application** sur l'URL fournie
2. **Mettre à jour le README** avec le lien de démo
3. **Partager** avec la communauté médicale !

---

## 📞 Support

- **Issues GitHub** : Signaler les bugs
- **Discussions** : Proposer des améliorations  
- **Streamlit Community** : [forum.streamlit.io](https://forum.streamlit.io)

---

<div align="center">

**🚀 Votre application sera bientôt disponible sur :**  
**`https://votre-app-name.streamlit.app`**

</div>
