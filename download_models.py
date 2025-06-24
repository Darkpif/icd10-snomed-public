import os
from transformers import AutoModel, AutoTokenizer

# Mapping des modèles
model_mapping_generalist = {
    'fr': 'dangvantuan/sentence-camembert-large',
    'de': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'it': 'Musixmatch/umberto-commoncrawl-cased-v1',
    'en': 'sentence-transformers/all-mpnet-base-v2'
}

model_mapping_medical = {
    'fr': 'Dr-BERT/DrBERT-7GB',
    'de': 'GerMedBERT/medbert-512',
    'it': 'Musixmatch/umberto-commoncrawl-cased-v1',
    'en': 'emilyalsentzer/Bio_ClinicalBERT'
}

def download_and_save_model(model_name, local_path):
    """Télécharge et sauvegarde un modèle Hugging Face."""
    try:
        print(f"📥 Tentative de téléchargement du modèle: {model_name}")
        os.makedirs(local_path, exist_ok=True)
        print(f"📂 Chemin local créé: {local_path}")
        use_fast = model_name == 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # Utiliser use_fast=True pour ce modèle
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=use_fast)
        if model is None or tokenizer is None:
            raise ValueError(f"Le modèle ou le tokenizer est introuvable pour {model_name}")
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print(f"✅ Modèle sauvegardé: {local_path}")
    except ValueError as ve:
        print(f"❌ Erreur de validation: {ve}")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement du modèle {model_name}: {e}")

if __name__ == "__main__":
    base_dir = "models"

    # Télécharger les modèles généralistes
    for lang, model_name in model_mapping_generalist.items():
        local_path = os.path.join(base_dir, lang, "generalist")
        download_and_save_model(model_name, local_path)

    # Télécharger les modèles médicaux
    for lang, model_name in model_mapping_medical.items():
        local_path = os.path.join(base_dir, lang, "medical")
        download_and_save_model(model_name, local_path)
