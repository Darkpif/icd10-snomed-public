import streamlit as st
import pandas as pd
from semantic_search import semantic_search_multilingual, preload_all_data

# Configuration de la page
st.set_page_config(
    page_title="Recherche ICD-10 via SNOMED",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Préchargement des modèles et données au démarrage
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    with st.container():
        st.title("🔍 Recherche augmentée de codes ICD-10 via SNOMED")
        st.markdown("---")
        
        # Préchargement avec indicateur de progression
        st.markdown("### 🚀 Initialisation de l'application")
        st.info("Premier lancement : chargement complet des modèles IA et des index de recherche...")
        
        # Précharger toutes les données (modèles + FAISS)
        preloaded_data = preload_all_data()
        st.session_state.preloaded_data = preloaded_data
        st.session_state.data_loaded = True
        
        st.success("🎉 Application prête ! Rechargez la page pour commencer.")
        st.rerun()

# Interface principale (ne s'affiche qu'après le chargement)
if st.session_state.data_loaded:    # Titre principal
    st.title("🔍 Recherche augmentée de codes ICD-10 via SNOMED")
    st.markdown("---")
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("ℹ️ À propos")
        st.markdown("""
        Cette application utilise l'intelligence artificielle pour mapper 
        des termes médicaux en langage naturel vers les codes ICD-10 
        officiels via la terminologie SNOMED CT.
        
        **Fonctionnalités:**
        - 🌍 Support multilingue (FR, EN, DE, IT)
        - 🤖 2 types de modèles IA (généraliste et médical)
        - 📊 Recherche sémantique avancée
        - 🎯 Mapping automatique vers ICD-10
          **Statut des modèles:**
        """)
        
        # Affichage du statut des données préchargées
        if 'preloaded_data' in st.session_state:
            data = st.session_state.preloaded_data
            st.success("✅ Toutes les données sont chargées")
            st.markdown(f"- Modèles généralistes: {len(data['models']['generalist'])}")
            st.markdown(f"- Modèles médicaux: {len(data['models']['medical'])}")
            st.markdown(f"- Index de recherche: {len(data['faiss_data'])} langues")
        
        st.header("🚀 Instructions")
        st.markdown("""
        1. Entrez un terme médical
        2. Choisissez la langue
        3. Sélectionnez le type de modèle
        4. Ajustez le nombre de résultats
        5. Appuyez sur Entrée ou cliquez sur "Rechercher"
        """)
        
        st.header("🔧 Options avancées")
        show_concept_id = st.checkbox("Afficher les Concept ID SNOMED", value=False, help="Affiche les identifiants techniques SNOMED dans les résultats")

    # Interface principale avec formulaire pour la touche Entrée
    with st.form("search_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🔍 Recherche")
            query = st.text_input(
                "Entrez votre terme médical :",
                placeholder="Ex: diabète type 2, fracture du fémur, pneumonie...",
                help="Saisissez un terme médical en langage naturel et appuyez sur Entrée ou cliquez sur Rechercher"
            )
            st.caption("💡 Astuce : Appuyez sur **Entrée** pour lancer la recherche rapidement !")

        with col2:
            st.subheader("⚙️ Paramètres")
            lang = st.selectbox(
                "Langue :",
                ["fr", "en", "de", "it"],
                format_func=lambda x: {"fr": "🇫🇷 Français", "en": "🇬🇧 Anglais", "de": "🇩🇪 Allemand", "it": "🇮🇹 Italien"}[x]
            )
            
            model_type = st.selectbox(
                "Type de modèle :",
                ["generalist", "medical"],
                format_func=lambda x: {"generalist": "🌐 Généraliste", "medical": "⚕️ Médical"}[x],
                help="Le modèle médical est spécialisé pour les termes médicaux"
            )
            
            top_k = st.slider("Nombre de résultats :", 1, 10, 5)

        # Bouton de soumission du formulaire
        search_submitted = st.form_submit_button("🔍 Rechercher", type="primary", use_container_width=True)

    # Traitement de la recherche (que ce soit par bouton ou par Entrée)
    if search_submitted:
        if not query.strip():
            st.warning("⚠️ Veuillez entrer un terme de recherche.")
        else:
            with st.spinner("🔄 Recherche en cours..."):
                try:                    # Utiliser les données préchargées
                    preloaded_data = st.session_state.get('preloaded_data', None)
                    preloaded_models = None
                    preloaded_faiss_data = None
                    
                    if preloaded_data:
                        preloaded_models = preloaded_data['models']
                        preloaded_faiss_data = preloaded_data['faiss_data']
                    
                    results = semantic_search_multilingual(
                        query=query.strip(),
                        lang=lang or "fr",
                        model_type=model_type or "generalist",
                        top_k=top_k,
                        preloaded_models=preloaded_models,
                        preloaded_data=preloaded_faiss_data
                    )
                    if results:
                        st.success(f"✅ {len(results)} résultat(s) trouvé(s)")
                        
                        # Informations contextuelles
                        model_name = (model_type or "generalist").capitalize()
                        lang_name = (lang or "fr").upper()
                        st.info(f"**Modèle utilisé:** {model_name} | **Langue:** {lang_name}")                        # Préparation des données pour le tableau
                        table_data = []
                        for i, result in enumerate(results, 1):
                            icd_code = result['ICD10Code'] if result['ICD10Code'] else "Non mappé"
                            icd_desc = result['ICD10Description'] if result['ICD10Description'] else "Aucune description disponible"
                            
                            row_data = {
                                "🏆 Rang": i,
                                "🔬 Terme SNOMED": result['term'],
                                "📊 Score": result['score'],
                                "📋 Code ICD-10": icd_code,
                                "📝 Description ICD-10": icd_desc
                            }
                            
                            # Ajouter Concept ID seulement si demandé
                            if show_concept_id:
                                row_data["🆔 Concept ID"] = result['conceptId']
                            
                            table_data.append(row_data)
                        
                        # Configuration des colonnes
                        column_config = {
                            "🏆 Rang": st.column_config.NumberColumn(width="small"),
                            "🔬 Terme SNOMED": st.column_config.TextColumn(width="medium"),
                            "📊 Score": st.column_config.NumberColumn(format="%.4f", width="small"),
                            "📋 Code ICD-10": st.column_config.TextColumn(width="medium"),
                            "📝 Description ICD-10": st.column_config.TextColumn(width="large")
                        }
                        
                        # Ajouter configuration Concept ID si affiché
                        if show_concept_id:
                            column_config["🆔 Concept ID"] = st.column_config.TextColumn(width="medium")
                        
                        # Affichage du tableau
                        df = pd.DataFrame(table_data)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config=column_config
                        )
                            
                    else:
                        st.error("❌ Aucun résultat trouvé. Essayez avec d'autres termes.")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche: {str(e)}")

# Footer
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
