# semantic_search.py - Module de recherche sémantique pour déploiement Streamlit

# Configuration Hugging Face Token
import streamlit as st
if "huggingface" in st.secrets:
    import os
    os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["huggingface"]["token"]
    os.environ["HF_TOKEN"] = st.secrets["huggingface"]["token"]

# Workaround pour le conflit PyTorch/Streamlit
import os
import warnings
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# Imports avec gestion d'erreur pour PyTorch
try:
    import torch
    import torch.nn.functional as F
    # Vérification de la version PyTorch
    torch_version = torch.__version__.split('+')[0]
    print(f"PyTorch version: {torch_version} (CPU: {not torch.cuda.is_available()})")
    
    # Patch pour éviter l'erreur __path__._path avec les anciennes versions
    if hasattr(torch, 'classes'):
        classes_attr = getattr(torch, 'classes', None)
        if classes_attr and hasattr(classes_attr, '__path__'):
            delattr(classes_attr, '__path__')
            
except ImportError as e:
    print(f"Erreur d'importation: {e}")
except RuntimeError as e:
    print(f"Erreur d'exécution: {e}")
except ValueError as e:
    print(f"Erreur de valeur: {e}")

import pandas as pd
import faiss
from sklearn.preprocessing import normalize
import xml.etree.ElementTree as ET
from transformers import AutoModel, AutoTokenizer

# Configuration des modèles (optimisés pour le déploiement)
model_mapping_generalist = {
    'fr': 'dangvantuan/sentence-camembert-large',  # Modèle français spécialisé
    'de': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'it': 'Musixmatch/umberto-commoncrawl-cased-v1',
    'en': 'sentence-transformers/all-mpnet-base-v2'
}

model_mapping_medical = {
    'fr': 'Dr-BERT/DrBERT-7GB',
    'de': 'GerMedBERT/medbert-512',
    'it': 'Musixmatch/umberto-commoncrawl-cased-v1',  # Fine-tuned
    'en': 'emilyalsentzer/Bio_ClinicalBERT'
}

def mean_pooling(model_output, attention_mask):
    """Effectue un mean pooling sur les embeddings de tokens."""
    token_embeddings = model_output[0]  # Premier élément contient tous les embeddings de tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text(text, model, tokenizer, max_length=512):
    """Encode un texte en utilisant AutoModel et AutoTokenizer."""
    # Tokeniser le texte
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    
    # Générer les embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Appliquer le mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normaliser les embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings.numpy()

@st.cache_resource
def load_icd10_descriptions():
    """Charge les descriptions ICD-10 multilingues avec cache Streamlit."""
    descriptions_by_lang = {}
    xml_files = {
        'fr': 'data/icd10/icd10_fr.xml',
        'en': 'data/icd10/icd10_en.xml',
        'it': 'data/icd10/icd10_it.xml',
        'de': 'data/icd10/icd10_de.xml'
    }
    
    for lang, file_path in xml_files.items():
        if not os.path.exists(file_path):
            st.warning(f"Fichier ICD-10 manquant pour {lang}: {file_path}")
            continue
            
        tree = ET.parse(file_path)
        root = tree.getroot()
        descriptions = {}
        
        for class_element in root.findall('.//Class'):
            code = class_element.get('code')
            preferred_label = None
            parent_label = None

            # Extraction du label préféré long
            for rubric in class_element.findall("Rubric[@kind='preferredLong']"):
                label = rubric.find('Label')
                if label is not None and label.get('{http://www.w3.org/XML/1998/namespace}lang') == lang:
                    preferred_label = ''.join(label.itertext()).strip()
                    break

            # Extraction du label préféré si le label préféré long est manquant
            if not preferred_label:
                for rubric in class_element.findall("Rubric[@kind='preferred']"):
                    label = rubric.find('Label')
                    if label is not None and label.get('{http://www.w3.org/XML/1998/namespace}lang') == lang:
                        preferred_label = ''.join(label.itertext()).strip()
                        break

            # Extraction du label parent si le label préféré est manquant
            if not preferred_label:
                parent_code_value = code.split('.')[0] if code and '.' in code else None
                if parent_code_value:
                    parent_class = root.find(f".//Class[@code='{parent_code_value}']")
                    if parent_class is not None:
                        for rubric in parent_class.findall("Rubric[@kind='preferred']"):
                            label = rubric.find('Label')
                            if label is not None and label.get('{http://www.w3.org/XML/1998/namespace}lang') == lang:
                                parent_label = ''.join(label.itertext()).strip()
                                break
                if parent_label:
                    preferred_label = f"{parent_label} ({code})"

            if code and preferred_label:
                descriptions[code] = preferred_label
        descriptions_by_lang[lang] = descriptions
    
    return descriptions_by_lang

@st.cache_resource
def preload_all_data():
    """Précharge tous les modèles et données FAISS disponibles."""
    st.info("🚀 Chargement initial complet en cours...")
    
    # Conteneurs pour tous les modèles et données
    all_data = {
        'models': {
            'generalist': {},
            'medical': {}
        },
        'faiss_data': {}
    }
    
    # Total : modèles + langues pour les données FAISS
    total_models = len(model_mapping_generalist) + len(model_mapping_medical)
    
    # Compter les fichiers FAISS pour estimer le nombre total d'opérations
    faiss_dir = "data/faiss"
    faiss_files_count = 0
    if os.path.exists(faiss_dir):
        faiss_files_count = len([f for f in os.listdir(faiss_dir) if f.endswith('.faiss')])
    
    total_operations = total_models + faiss_files_count + 4  # +4 pour les métadonnées
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    current = 0    # 1. Chargement des modèles généralistes
    status_text.text("🤖 Chargement des modèles généralistes...")
    for lang, model_name in model_mapping_generalist.items():
        try:
            status_text.text(f"Chargement modèle généraliste {lang.upper()}...")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=(lang == 'de'))
            all_data['models']['generalist'][lang] = {'model': model, 'tokenizer': tokenizer}
            current += 1
            progress_bar.progress(current / total_operations)
            st.success(f"✅ Modèle généraliste {lang.upper()} chargé")
        except (OSError, ValueError, RuntimeError) as e:
            st.error(f"❌ Erreur modèle généraliste {lang}: {e}")
            current += 1
            progress_bar.progress(current / total_operations)
    
    # Intégration de la logique de chargement local dans preload_all_data
    for lang, model_name in model_mapping_generalist.items():
        use_fast = lang == 'de'  # Utiliser use_fast=True uniquement pour l'allemand
        local_path = f"models/{lang}/generalist"  # Chemin local hypothétique
        all_data['models']['generalist'][lang] = load_model_local_or_remote(model_name, local_path, use_fast=use_fast)

    # 2. Chargement des modèles médicaux
    status_text.text("⚕️ Chargement des modèles médicaux...")
    for lang, model_name in model_mapping_medical.items():
        try:
            status_text.text(f"Chargement modèle médical {lang.upper()}...")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            all_data['models']['medical'][lang] = {'model': model, 'tokenizer': tokenizer}
            current += 1
            progress_bar.progress(current / total_operations)
            st.success(f"✅ Modèle médical {lang.upper()} chargé")
        except (OSError, ValueError, RuntimeError) as e:
            st.error(f"❌ Erreur modèle médical {lang}: {e}")
            current += 1
            progress_bar.progress(current / total_operations)
    
    # Intégration de la logique de chargement local dans preload_all_data
    for lang, model_name in model_mapping_medical.items():
        local_path = f"models/{lang}/medical"  # Chemin local hypothétique
        all_data['models']['medical'][lang] = load_model_local_or_remote(model_name, local_path)

    # 3. Chargement des données FAISS
    status_text.text("📊 Chargement des index de recherche...")
    language_data = {}
    
    if not os.path.exists(faiss_dir):
        st.error(f"Répertoire FAISS '{faiss_dir}' introuvable.")
    else:
        faiss_files = [f for f in os.listdir(faiss_dir) if f.endswith('.faiss')]
        
        for faiss_file in faiss_files:
            if 'generalist_index_' in faiss_file:
                lang = faiss_file.replace('snomed_generalist_index_', '').replace('.faiss', '')
                if lang not in language_data:
                    language_data[lang] = {}
                
                try:
                    status_text.text(f"Chargement index généraliste {lang.upper()}...")
                    index_path = os.path.join(faiss_dir, faiss_file)
                    language_data[lang]['generalist_faiss_index'] = faiss.read_index(index_path)
                    current += 1
                    progress_bar.progress(current / total_operations)
                    st.success(f"✅ Index généraliste {lang.upper()} chargé")
                except (OSError, ValueError, RuntimeError) as e:
                    st.error(f"❌ Erreur index généraliste {lang}: {e}")
                    current += 1
                    progress_bar.progress(current / total_operations)
            
            elif 'medical_index_' in faiss_file:
                lang = faiss_file.replace('snomed_medical_index_', '').replace('.faiss', '')
                if lang not in language_data:
                    language_data[lang] = {}
                
                try:
                    status_text.text(f"Chargement index médical {lang.upper()}...")
                    index_path = os.path.join(faiss_dir, faiss_file)
                    language_data[lang]['medical_faiss_index'] = faiss.read_index(index_path)
                    current += 1
                    progress_bar.progress(current / total_operations)
                    st.success(f"✅ Index médical {lang.upper()} chargé")
                except (OSError, ValueError, RuntimeError) as e:
                    st.error(f"❌ Erreur index médical {lang}: {e}")
                    current += 1
                    progress_bar.progress(current / total_operations)
        
        # 4. Charger les métadonnées
        status_text.text("📋 Chargement des métadonnées...")
        for lang in language_data.keys():
            metadata_path = os.path.join(faiss_dir, f"terms_metadata_{lang}.parquet")
            if os.path.exists(metadata_path):
                try:
                    status_text.text(f"Chargement métadonnées {lang.upper()}...")
                    language_data[lang]['dataframe'] = pd.read_parquet(metadata_path)
                    current += 1
                    progress_bar.progress(current / total_operations)
                    st.success(f"✅ Métadonnées {lang.upper()} chargées")
                except (OSError, ValueError, RuntimeError) as e:
                    st.error(f"❌ Erreur métadonnées {lang}: {e}")
                    current += 1
                    progress_bar.progress(current / total_operations)
    
    all_data['faiss_data'] = language_data
    
    progress_bar.progress(1.0)
    status_text.text("✅ Chargement complet terminé !")
    
    return all_data

@st.cache_resource
def preload_all_models():
    """Précharge tous les modèles disponibles pour toutes les langues (version legacy)."""
    st.info("🚀 Chargement initial des modèles IA en cours...")
    
    # Conteneurs pour tous les modèles
    all_models = {
        'generalist': {},
        'medical': {}
    }
    
    total_models = len(model_mapping_generalist) + len(model_mapping_medical)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    current = 0
      # Chargement des modèles généralistes
    status_text.text("Chargement des modèles généralistes...")
    for lang, model_name in model_mapping_generalist.items():
        try:
            status_text.text(f"Chargement modèle généraliste {lang.upper()}...")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            all_models['generalist'][lang] = {'model': model, 'tokenizer': tokenizer}
            current += 1
            progress_bar.progress(current / total_models)
            st.success(f"✅ Modèle généraliste {lang.upper()} chargé")
        except (OSError, ValueError, RuntimeError) as e:
            st.error(f"❌ Erreur modèle généraliste {lang}: {e}")
            current += 1
            progress_bar.progress(current / total_models)
    
    # Intégration de la logique de chargement local dans preload_all_models
    for lang, model_name in model_mapping_generalist.items():
        use_fast = lang == 'de'  # Utiliser use_fast=True uniquement pour l'allemand
        local_path = f"models/{lang}/generalist"  # Chemin local hypothétique
        all_models['generalist'][lang] = load_model_local_or_remote(model_name, local_path, use_fast=use_fast)

      # Chargement des modèles médicaux
    status_text.text("Chargement des modèles médicaux...")
    for lang, model_name in model_mapping_medical.items():
        try:
            status_text.text(f"Chargement modèle médical {lang.upper()}...")
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            all_models['medical'][lang] = {'model': model, 'tokenizer': tokenizer}
            current += 1
            progress_bar.progress(current / total_models)
            st.success(f"✅ Modèle médical {lang.upper()} chargé")
        except (OSError, ValueError, RuntimeError) as e:
            st.error(f"❌ Erreur modèle médical {lang}: {e}")
            current += 1
            progress_bar.progress(current / total_models)
    
    # Intégration de la logique de chargement local dans preload_all_models
    for lang, model_name in model_mapping_medical.items():
        local_path = f"models/{lang}/medical"  # Chemin local hypothétique
        all_models['medical'][lang] = load_model_local_or_remote(model_name, local_path)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Tous les modèles sont chargés et prêts !")
    
    return all_models

@st.cache_resource
def load_models():
    """Charge les modèles avec cache Streamlit (version legacy pour compatibilité)."""
    models_generalist = {}
    models_medical = {}
    
    for lang, model_name in model_mapping_generalist.items():
        try:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            models_generalist[lang] = {'model': model, 'tokenizer': tokenizer}
        except Exception as e:
            st.error(f"❌ Erreur modèle généraliste {lang}: {e}")
    
    for lang, model_name in model_mapping_medical.items():
        try:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
            models_medical[lang] = {'model': model, 'tokenizer': tokenizer}
        except Exception as e:
            st.error(f"❌ Erreur modèle médical {lang}: {e}")
    
    return models_generalist, models_medical

@st.cache_resource
def load_faiss_data():
    """Charge les index FAISS et métadonnées avec cache Streamlit."""
    language_data = {}
    faiss_dir = "data/faiss"
    
    if not os.path.exists(faiss_dir):
        st.error(f"Répertoire FAISS '{faiss_dir}' introuvable.")
        return {}
    
    faiss_files = [f for f in os.listdir(faiss_dir) if f.endswith('.faiss')]
    
    for faiss_file in faiss_files:
        if 'generalist_index_' in faiss_file:
            lang = faiss_file.replace('snomed_generalist_index_', '').replace('.faiss', '')
            if lang not in language_data:
                language_data[lang] = {}
            
            try:
                index_path = os.path.join(faiss_dir, faiss_file)
                language_data[lang]['generalist_faiss_index'] = faiss.read_index(index_path)
            except Exception as e:
                st.error(f"❌ Erreur index généraliste {lang}: {e}")
        
        elif 'medical_index_' in faiss_file:
            lang = faiss_file.replace('snomed_medical_index_', '').replace('.faiss', '')
            if lang not in language_data:
                language_data[lang] = {}
            
            try:
                index_path = os.path.join(faiss_dir, faiss_file)
                language_data[lang]['medical_faiss_index'] = faiss.read_index(index_path)
            except Exception as e:
                st.error(f"❌ Erreur index médical {lang}: {e}")
    
    # Charger les métadonnées
    for lang in language_data.keys():
        metadata_path = os.path.join(faiss_dir, f"terms_metadata_{lang}.parquet")
        if os.path.exists(metadata_path):
            try:
                language_data[lang]['dataframe'] = pd.read_parquet(metadata_path)
            except Exception as e:
                st.error(f"❌ Erreur métadonnées {lang}: {e}")
    
    return language_data

def normalize_icd10_code(icd10_code):
    """Normalise les codes ICD-10."""
    if icd10_code and '.' in icd10_code:
        parts = icd10_code.split('.')
        if len(parts) == 2:
            if parts[1].strip('0') == '':
                return f"{parts[0]}.0"
            parts[1] = parts[1].rstrip('0')
            return '.'.join(parts)
    return icd10_code

def semantic_search_multilingual(query, lang, model_type='generalist', top_k=5, preloaded_models=None, preloaded_data=None):
    """Effectue une recherche sémantique multilingue avec modèles et données préchargés."""
    # Charger les données nécessaires
    icd10_descriptions = load_icd10_descriptions()
    
    # Utiliser les modèles préchargés ou charger les modèles
    if preloaded_models:
        models_generalist = preloaded_models['generalist']
        models_medical = preloaded_models['medical']
    else:
        models_generalist, models_medical = load_models()
    
    # Utiliser les données FAISS préchargées ou les charger
    if preloaded_data:
        language_data = preloaded_data
    else:
        language_data = load_faiss_data()
    
    # Trouver la langue correspondante
    matching_lang = next((key for key in language_data.keys() if key.startswith(lang)), None)
    if not matching_lang:
        st.error(f"Langue {lang} non prise en charge.")
        return []    # Sélectionner le modèle
    base_lang = lang.split('-')[0] if '-' in lang else lang
    if model_type == 'generalist':
        model_dict = models_generalist.get(base_lang)
        index = language_data.get(matching_lang, {}).get('generalist_faiss_index')
    else:
        model_dict = models_medical.get(base_lang)
        index = language_data.get(matching_lang, {}).get('medical_faiss_index')

    if not model_dict or not index:
        st.error(f"Modèle ou index manquant pour {lang}")
        return []

    # Générer l'embedding de la requête
    query_embedding = encode_text(query, model_dict['model'], model_dict['tokenizer'])
    query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2')

    # Effectuer la recherche
    try:
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
    except Exception as e:
        st.error(f"Erreur lors de la recherche: {e}")
        return []

    # Construire les résultats
    results = []
    dataframe = language_data.get(matching_lang, {}).get('dataframe', pd.DataFrame())
    
    for i, idx in enumerate(indices[0]):
        if idx >= len(dataframe):
            continue
            
        row = dataframe.iloc[idx]
        icd10_code = row.get('mapTarget', None)
        icd10_description = None
        
        if isinstance(icd10_code, str):
            icd10_code = normalize_icd10_code(icd10_code)
            icd10_description = icd10_descriptions.get(lang, {}).get(icd10_code, None)
            if not icd10_description and '.' in icd10_code:
                parent_code = icd10_code.split('.')[0]
                icd10_description = icd10_descriptions.get(lang, {}).get(parent_code, None)
                if icd10_description:
                    icd10_description += f" ({parent_code})"
        
        results.append({
            'conceptId': row.get('conceptId', None),
            'term': row.get('term', None),
            'score': distances[0][i],
            'ICD10Code': icd10_code,
            'ICD10Description': icd10_description
        })

    return results

def load_model_local_or_remote(model_name, local_path=None, use_fast=False):
    """Charge un modèle depuis un chemin local ou depuis Hugging Face."""
    try:
        if local_path and os.path.exists(local_path):
            model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, use_fast=use_fast)
            return {'model': model, 'tokenizer': tokenizer}
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=use_fast)
            return {'model': model, 'tokenizer': tokenizer}
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle {model_name}: {e}")
        return None
