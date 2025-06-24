# semantic_search.py - Module de recherche sémantique pour déploiement Streamlit

import os
import warnings
import streamlit as st
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
import xml.etree.ElementTree as ET
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

# Configuration Hugging Face Token
if "huggingface" in st.secrets:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["huggingface"]["token"]
    os.environ["HF_TOKEN"] = st.secrets["huggingface"]["token"]

# Workaround pour le conflit PyTorch/Streamlit
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# Configuration des modèles (optimisés pour le déploiement)
model_mapping_generalist = {
    'fr-ch': 'dangvantuan/sentence-camembert-large',
    'de-ch': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'it-ch': 'Musixmatch/umberto-commoncrawl-cased-v1',
    'en-int': 'sentence-transformers/all-mpnet-base-v2'
}

def mean_pooling(model_output, attention_mask):
    """Effectue un mean pooling sur les embeddings de tokens."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_text(text, model, tokenizer, max_length=512, lang=None):
    """Encode un texte en utilisant AutoModel et AutoTokenizer."""
    if lang == 'fr':
        # Utilisation spécifique du tokenizer CamemBERT pour le français
        tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        tokenized_input = tokenizer.batch_decode(tokenized_input['input_ids'], skip_special_tokens=True)
        encoded_input = tokenizer(tokenized_input, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    else:
        encoded_input = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(sentence_embeddings, p=2, dim=1).numpy()

@st.cache_resource
def load_icd10_descriptions():
    """Charge les descriptions ICD-10 multilingues avec cache Streamlit."""
    descriptions_by_lang = {}
    xml_files = {
        'fr-ch': 'data/icd10/icd10_fr.xml',
        'en-int': 'data/icd10/icd10_en.xml',
        'it-ch': 'data/icd10/icd10_it.xml',
        'de-ch': 'data/icd10/icd10_de.xml'
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

            for rubric in class_element.findall("Rubric[@kind='preferredLong']"):
                label = rubric.find('Label')
                if label is not None:
                    preferred_label = ''.join(label.itertext()).strip()
                    break

            if not preferred_label:
                for rubric in class_element.findall("Rubric[@kind='preferred']"):
                    label = rubric.find('Label')
                    if label is not None:
                        preferred_label = ''.join(label.itertext()).strip()
                        break

            if code and preferred_label:
                descriptions[code] = preferred_label

        descriptions_by_lang[lang] = descriptions

    return descriptions_by_lang

@st.cache_resource
def preload_all_data():
    """Précharge les modèles et les données FAISS pour toutes les langues disponibles."""
    st.info("🚀 Chargement initial complet en cours...")

    all_data = {
        'models': {
            'generalist': {}
        },
        'faiss_data': {},
        'metadata': {}
    }

    try:
        status_text = st.empty()
        progress_bar = st.progress(0)
        total_operations = len(model_mapping_generalist) * 3  # modèles + index + métadonnées
        current = 0

        for lang, model_name in model_mapping_generalist.items():
            status_text.text(f"🤖 Chargement du modèle pour {lang.upper()}...")
            try:
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
                all_data['models']['generalist'][lang] = {'model': model, 'tokenizer': tokenizer}
                st.success(f"✅ Modèle pour {lang.upper()} chargé")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du modèle pour {lang.upper()}: {e}")
                st.stop()

            current += 1
            progress_bar.progress(current / total_operations)

        faiss_dir = "data/faiss"
        for lang in model_mapping_generalist.keys():
            # Correction pour utiliser les variantes de langue comme 'fr-ch'
            index_path = os.path.join(faiss_dir, f"snomed_generalist_index_spec_{lang}.faiss")
            metadata_path = os.path.join(faiss_dir, f"terms_metadata_{lang}.parquet")

            if not os.path.exists(index_path):
                st.error(f"Index FAISS introuvable: {index_path}")
                st.stop()

            try:
                status_text.text(f"📊 Chargement de l'index FAISS pour {lang.upper()}...")
                all_data['faiss_data'][lang] = faiss.read_index(index_path)
                st.success(f"✅ Index FAISS pour {lang.upper()} chargé")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement de l'index FAISS pour {lang.upper()}: {e}")
                st.stop()

            current += 1
            progress_bar.progress(current / total_operations)

            if not os.path.exists(metadata_path):
                st.error(f"Fichier de métadonnées introuvable: {metadata_path}")
                st.stop()

            try:
                status_text.text(f"📋 Validation du fichier de métadonnées pour {lang.upper()}...")
                with open(metadata_path, 'rb') as f:
                    if f.read(4) != b'PAR1':
                        raise ValueError("Le fichier n'est pas un fichier Parquet valide.")

                status_text.text(f"📋 Chargement des métadonnées pour {lang.upper()}...")
                all_data['metadata'][lang] = pd.read_parquet(metadata_path)
                st.success(f"✅ Métadonnées pour {lang.upper()} chargées")
            except ValueError as ve:
                st.error(f"❌ Erreur de validation du fichier de métadonnées pour {lang.upper()}: {ve}")
                st.warning("Veuillez vérifier ou régénérer le fichier Parquet.")
                st.stop()
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Erreur lors du chargement des métadonnées pour {lang.upper()}: {error_msg}")
                if "Repetition level histogram size mismatch" in error_msg:
                    st.warning("⚠️ Ce fichier Parquet semble avoir été créé avec une version incompatible ou est corrompu.")
                    st.warning("🔧 Solutions recommandées:")
                    st.warning("   • Régénérer le fichier Parquet à partir des données sources")
                    st.warning("   • Vérifier la compatibilité des versions pandas/pyarrow")
                    st.warning("   • S'assurer que le fichier n'est pas corrompu")
                st.stop()

            current += 1
            progress_bar.progress(current / total_operations)

        progress_bar.progress(1.0)
        status_text.text("✅ Chargement complet terminé !")

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        st.stop()

    return all_data

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
    icd10_descriptions = load_icd10_descriptions()

    if preloaded_models is None or model_type not in preloaded_models:
        st.error(f"Modèle '{model_type}' non trouvé dans les données préchargées.")
        return []

    if preloaded_data is None or lang not in preloaded_data:
        st.error(f"Données FAISS pour la langue '{lang}' non trouvées.")
        return []

    model_dict = preloaded_models[model_type].get(lang)
    index = preloaded_data.get(lang)

    if not model_dict or not index:
        st.error(f"Modèle ou index manquant pour la langue '{lang}'.")
        return []

    query_embedding = encode_text(query, model_dict['model'], model_dict['tokenizer'], lang=lang)
    query_embedding = normalize(query_embedding.reshape(1, -1), norm='l2')

    try:
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
    except Exception as e:
        st.error(f"Erreur lors de la recherche: {e}")
        return []

    results = []
    # Utiliser les métadonnées préchargées
    metadata_df = st.session_state.get('preloaded_data', {}).get('metadata', {}).get(lang, pd.DataFrame())

    for i, idx in enumerate(indices[0]):
        if idx >= len(metadata_df):
            continue

        row = metadata_df.iloc[idx]
        icd10_code = row.get('mapTarget', None)
        icd10_description = icd10_descriptions.get(lang, {}).get(icd10_code, "Aucune description disponible")

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
