# fine-tuning du bi-encoder bge-small avec CosineSimilarityLoss

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity

from src.data_utils import load_dataset_fit, group_split
from src.evaluation import classify, find_best_thresholds, compute_f1

# === Chargement du dataset principal ===
print("Chargement du dataset principal...")
df_ats = load_dataset_fit()
print(f"\nAperçu des labels:")
print(df_ats['label'].value_counts())

# on regarde à quoi ressemblent les colonnes
print(df_ats.head(3).to_string())

# trouver les colonnes CV, offre, score
# TODO: adapter les noms de colonnes selon ce qu'on trouve dans df_ats.columns

# === Préparation des données ===
# on normalise les scores entre 0 et 1, puis on fait un group split par CV
cv_col = 'resume_text'
job_col = 'job_description_text'

# mapper les labels en scores continus pour CosineSimilarityLoss
label_to_score = {'Good Fit': 1.0, 'Potential Fit': 0.5, 'No Fit': 0.0}
df_ats['score_norm'] = df_ats['label'].map(label_to_score)

print(f"Distribution des scores normalisés:")
print(df_ats['score_norm'].value_counts())
print(f"Score normalisé — moyenne: {df_ats['score_norm'].mean():.4f}")

# group split par CV pour éviter le data leakage
train_ats, test_ats = group_split(df_ats)

# === Préparation des InputExamples pour sentence-transformers ===
# créer les exemples d'entraînement
train_examples = [
    InputExample(
        texts=[row[cv_col], row[job_col]],
        label=float(row['score_norm'])
    )
    for _, row in train_ats.iterrows()
]

print(f"Exemples d'entraînement: {len(train_examples)}")
print(f"Exemple 0 — score: {train_examples[0].label:.4f}")

# === Fine-tuning avec CosineSimilarityLoss ===
# Hyperparamètres :
# - Learning rate : 2e-5
# - Batch size : 8 (réduit depuis 16 pour tenir en 4GB VRAM)
# - Epochs : 3
# - fp16 : activé pour tenir en 4GB VRAM

# vider le cache GPU avant de charger le modèle
import torch
torch.cuda.empty_cache()

# charger le modèle de base
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# dataloader — batch 8 pour tenir en 4GB VRAM
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# loss function
train_loss = losses.CosineSimilarityLoss(model)

# evaluator sur le test set ATS
test_examples = [
    InputExample(texts=[row[cv_col], row[job_col]], label=float(row['score_norm']))
    for _, row in test_ats.iterrows()
]
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='ats-test')

# fine-tuning
print("Début du fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    warmup_steps=100,
    output_path='models/bge-small-finetuned',
    use_amp=True,  # fp16
    show_progress_bar=True,
    evaluation_steps=500,
    save_best_model=True,
)

print("Fine-tuning terminé !")
print("Modèle sauvegardé dans models/bge-small-finetuned/")

# === Évaluation du modèle fine-tuné sur le dataset principal ===
# on charge le dataset principal et on compare le F1 avant/après fine-tuning

# charger le modèle fine-tuné
model_ft = SentenceTransformer('models/bge-small-finetuned')
print("Modèle fine-tuné chargé")

# charger le dataset principal
df_main = load_dataset_fit()
train_main, test_main = group_split(df_main)

# encoder avec le modèle fine-tuné
print("Encodage avec modèle fine-tuné...")
cv_ft = model_ft.encode(test_main['resume_text'].tolist(), show_progress_bar=True, batch_size=32)
job_ft = model_ft.encode(test_main['job_description_text'].tolist(), show_progress_bar=True, batch_size=32)

scores_ft = [
    cosine_similarity([cv_ft[i]], [job_ft[i]])[0][0]
    for i in range(len(test_main))
]

# calibrer les seuils sur le train
cv_train_ft = model_ft.encode(train_main['resume_text'].tolist(), show_progress_bar=True, batch_size=32)
job_train_ft = model_ft.encode(train_main['job_description_text'].tolist(), show_progress_bar=True, batch_size=32)
scores_train_ft = [
    cosine_similarity([cv_train_ft[i]], [job_train_ft[i]])[0][0]
    for i in range(len(train_main))
]

high_ft, low_ft, _ = find_best_thresholds(scores_train_ft, train_main['label'].tolist())
f1_ft = compute_f1(scores_ft, test_main['label'].tolist(), high_ft, low_ft)

print(f"\nF1 weighted — bge-small fine-tuné: {f1_ft:.4f}")
print(f"Seuils: Good Fit >= {high_ft:.2f}, Potential Fit >= {low_ft:.2f}")

# === Récapitulatif ===
print("=" * 50)
print("RÉSULTATS — FINE-TUNING bge-small")
print("=" * 50)
print(f"Dataset utilisé: {len(train_ats)} paires (train)")
print(f"Epochs: 3 | LR: 2e-5 | Batch: 8 | fp16: True")
print(f"\nF1 bge-small pré-entraîné (troncation): voir notebook 03")
print(f"F1 bge-small fine-tuné: {f1_ft:.4f}")
print("\nModèle sauvegardé: models/bge-small-finetuned/")
