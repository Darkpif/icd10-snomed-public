import streamlit as st
import pandas as pd
from semantic_search import semantic_search_multilingual, preload_all_data, model_mapping_generalist, load_icd10_descriptions
import fasttext
import os

# Configuration de la page
st.set_page_config(
    page_title="Recherche ICD-10 via SNOMED",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du modèle FastText pour la détection de langue
fasttext_model_path = "data/fasttext/lid.176.bin"  # Chemin vers le modèle pré-entraîné
if not os.path.exists(fasttext_model_path):
    st.error("❌ Modèle FastText introuvable. Veuillez le télécharger depuis https://fasttext.cc/docs/en/language-identification.html")
    st.stop()

fasttext_model = fasttext.load_model(fasttext_model_path)

def detect_language_fasttext(text):
    """Détecte la langue d'un texte donné en utilisant FastText."""
    try:
        predictions = fasttext_model.predict(text, k=1)  # Prédiction de la langue avec le score le plus élevé
        lang_code = predictions[0][0].replace("__label__", "")
        return lang_code
    except Exception as e:
        st.warning(f"⚠️ Erreur lors de la détection de la langue avec FastText: {e}. Utilisation de l'anglais par défaut.")
        return 'en-int'

# Préchargement des modèles et données au démarrage
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    st.title("🔍 Recherche augmentée de codes ICD-10 via SNOMED")
    st.markdown("---")

    st.markdown("### 🚀 Initialisation de l'application")
    st.info("Premier lancement : chargement complet des modèles et des index de recherche pour toutes les langues disponibles...")

    preloaded_data = preload_all_data()
    icd10_descriptions = load_icd10_descriptions()
    if preloaded_data is None or icd10_descriptions is None:
        st.error("❌ Erreur lors du chargement des données. Veuillez vérifier les logs et réessayer.")
        st.stop()

    st.session_state.preloaded_data = preloaded_data
    st.session_state.icd10_descriptions = icd10_descriptions
    st.session_state.data_loaded = True

    st.success("🎉 Application prête ! Rechargez la page pour commencer.")
    st.rerun()

# Interface principale (ne s'affiche qu'après le chargement)
if st.session_state.data_loaded:
    st.title("🔍 Recherche augmentée de codes ICD-10 via SNOMED")
    st.markdown("---")

    # Sidebar avec informations
    with st.sidebar:
        st.header("ℹ️ À propos")
        st.markdown(
            """
            Cette application utilise l'intelligence artificielle pour mapper 
            des termes médicaux en langage naturel vers les codes ICD-10 
            officiels via la terminologie SNOMED CT.

            **Fonctionnalités:**
            - 🌍 Support multilingue (FR, EN, DE, IT)
            - 📊 Recherche sémantique avancée
            - 🎯 Mapping automatique vers ICD-10
            """
        )

        if 'preloaded_data' in st.session_state:
            st.success("✅ Toutes les données sont chargées")
            st.markdown("- Modèles: CamemBERT pour FR-CH, All-MPNET pour EN-INT, Paraphrase-MiniLM pour DE-CH, UmBERTo pour IT-CH")
            st.markdown("- Index de recherche: 4 langues")

        st.header("🚀 Instructions")
        st.markdown(
            """
            1. Entrez un terme médical
            2. Choisissez la langue
            3. Ajustez le nombre de résultats
            4. Appuyez sur Entrée ou cliquez sur "Rechercher"
            """
        )

        st.header("🔧 Options avancées")
        show_concept_id = st.checkbox("Afficher les Concept ID SNOMED", value=False, help="Affiche les identifiants techniques SNOMED dans les résultats")

    # Interface principale avec formulaire
    with st.form("search_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            query = st.text_input(
                "Entrez votre terme médical :",
                placeholder="Ex: diabète type 2, fracture du fémur, pneumonie...",
                help="Saisissez un terme médical en langage naturel et appuyez sur Entrée ou cliquez sur Rechercher"
            )

        with col2:
            lang = st.selectbox(
                "Langue :",
                ["auto", "fr-ch", "en-int", "de-ch", "it-ch"],
                format_func=lambda x: {
                    "auto": "🌐 Automatique",
                    "fr-ch": "🇫🇷 Français",
                    "en-int": "🇬🇧 Anglais",
                    "de-ch": "🇩🇪 Allemand",
                    "it-ch": "🇮🇹 Italien"
                }[x]
            )
            top_k = st.slider("Nombre de résultats :", 1, 10, 5)

        search_submitted = st.form_submit_button("🔍 Rechercher")

    if search_submitted:
        if not query.strip():
            st.warning("⚠️ Veuillez entrer un terme de recherche.")
        else:
            with st.spinner("🔄 Recherche en cours..."):
                try:
                    preloaded_data = st.session_state.get('preloaded_data', None)

                    if preloaded_data:
                        # Modification pour afficher uniquement la langue détectée par FastText
                        if lang == "auto":
                            detected_lang = detect_language_fasttext(query.strip())
                            lang_showed = detected_lang
                            lang_mapping = {
                                'fr': 'fr-ch',
                                'en': 'en-int',
                                'de': 'de-ch',
                                'it': 'it-ch'
                            }
                            detected_lang = lang_mapping.get(detected_lang, 'en-int')
                            st.info(f"🌐 Langue détectée par FastText: {lang_showed}")

                            lang = detected_lang if detected_lang in ["fr-ch", "en-int", "de-ch", "it-ch"] else "en-int"

                        results = semantic_search_multilingual(
                            query=query.strip(),
                            lang=lang,
                            model_type="generalist",
                            top_k=top_k,
                            preloaded_models=preloaded_data['models'],
                            preloaded_data=preloaded_data['faiss_data']
                        )

                        if results:
                            st.success(f"✅ {len(results)} résultat(s) trouvé(s)")

                            model_name = model_mapping_generalist.get(lang, "Inconnu")
                            st.info(f"**Modèle utilisé:** {model_name} | **Langue:** {lang.upper()}")

                            table_data = [
                                {
                                    "🏆 Rang": i + 1,
                                    "🔬 Terme SNOMED": result['term'],
                                    "📊 Score": result['score'],
                                    "📋 Code ICD-10": result.get('ICD10Code', "Non mappé"),
                                    "📝 Description ICD-10": result.get('ICD10Description', "Aucune description disponible"),
                                    **({"🆔 Concept ID": result['conceptId']} if show_concept_id else {})
                                }
                                for i, result in enumerate(results)
                            ]

                            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                        else:
                            st.error("❌ Aucun résultat trouvé. Essayez avec d'autres termes.")
                    else:
                        st.error("❌ Les données préchargées sont introuvables.")

                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche: {str(e)}")

# Footer simplifié
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🔬 Application de recherche sémantique médicale basée sur SNOMED CT et ICD-10<br>
        Développée avec Streamlit et des modèles d'IA avancés
    </div>
    """,
    unsafe_allow_html=True
)
