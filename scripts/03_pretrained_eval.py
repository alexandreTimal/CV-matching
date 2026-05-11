# évaluation de 2 modèles pré-entraînés (MiniLM, bge-small) avec 3 stratégies de chunking

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_utils import load_dataset_fit, group_split
from src.chunking import embed_mean_chunks, score_max_chunks
from src.evaluation import classify, find_best_thresholds, compute_f1

# === Chargement et split des données ===
df = load_dataset_fit()
train_df, test_df = group_split(df)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# === Chargement des modèles ===
# all-MiniLM-L6-v2 : petit et rapide, 256 tokens max
# BAAI/bge-small-en-v1.5 : un peu plus grand, 512 tokens max
print("Chargement des modèles...")
model_minilm = SentenceTransformer('all-MiniLM-L6-v2')
model_bge = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("Modèles chargés !")

# === Stratégie 1 : Troncation simple ===
# le modèle tronque automatiquement les textes trop longs
cv_texts = test_df['resume_text'].tolist()
job_texts = test_df['job_description_text'].tolist()

print("Encodage MiniLM (troncation)...")
cv_emb_minilm = model_minilm.encode(cv_texts, show_progress_bar=True, batch_size=32)
job_emb_minilm = model_minilm.encode(job_texts, show_progress_bar=True, batch_size=32)

print("Encodage bge-small (troncation)...")
cv_emb_bge = model_bge.encode(cv_texts, show_progress_bar=True, batch_size=32)
job_emb_bge = model_bge.encode(job_texts, show_progress_bar=True, batch_size=32)

# scores cosinus
scores_minilm_trunc = [
    cosine_similarity([cv_emb_minilm[i]], [job_emb_minilm[i]])[0][0]
    for i in range(len(test_df))
]
scores_bge_trunc = [
    cosine_similarity([cv_emb_bge[i]], [job_emb_bge[i]])[0][0]
    for i in range(len(test_df))
]

print(f"Score moyen MiniLM (troncation): {np.mean(scores_minilm_trunc):.4f}")
print(f"Score moyen bge-small (troncation): {np.mean(scores_bge_trunc):.4f}")

# on encode aussi le train set pour calibrer les seuils
print("Encodage train set MiniLM...")
cv_train_minilm = model_minilm.encode(train_df['resume_text'].tolist(), show_progress_bar=True, batch_size=32)
job_train_minilm = model_minilm.encode(train_df['job_description_text'].tolist(), show_progress_bar=True, batch_size=32)

scores_train_minilm = [
    cosine_similarity([cv_train_minilm[i]], [job_train_minilm[i]])[0][0]
    for i in range(len(train_df))
]

print("Encodage train set bge-small...")
cv_train_bge = model_bge.encode(train_df['resume_text'].tolist(), show_progress_bar=True, batch_size=32)
job_train_bge = model_bge.encode(train_df['job_description_text'].tolist(), show_progress_bar=True, batch_size=32)

scores_train_bge = [
    cosine_similarity([cv_train_bge[i]], [job_train_bge[i]])[0][0]
    for i in range(len(train_df))
]

# calibration des seuils sur le train
high_minilm, low_minilm, _ = find_best_thresholds(scores_train_minilm, train_df['label'].tolist())
high_bge, low_bge, _ = find_best_thresholds(scores_train_bge, train_df['label'].tolist())

print(f"Seuils MiniLM: Good Fit >= {high_minilm:.2f}, Potential Fit >= {low_minilm:.2f}")
print(f"Seuils bge-small: Good Fit >= {high_bge:.2f}, Potential Fit >= {low_bge:.2f}")

f1_minilm_trunc = compute_f1(scores_minilm_trunc, test_df['label'].tolist(), high_minilm, low_minilm)
f1_bge_trunc = compute_f1(scores_bge_trunc, test_df['label'].tolist(), high_bge, low_bge)

print(f"F1 MiniLM (troncation): {f1_minilm_trunc:.4f}")
print(f"F1 bge-small (troncation): {f1_bge_trunc:.4f}")

# === Stratégie 2 : Mean pooling des chunks ===
# on découpe en morceaux de ~200 mots, on encode chaque morceau, et on fait la moyenne
print("Mean pooling MiniLM (test)...")
cv_mean_minilm = np.array([embed_mean_chunks(t, model_minilm) for t in cv_texts])
job_mean_minilm = np.array([embed_mean_chunks(t, model_minilm) for t in job_texts])

print("Mean pooling bge-small (test)...")
cv_mean_bge = np.array([embed_mean_chunks(t, model_bge) for t in cv_texts])
job_mean_bge = np.array([embed_mean_chunks(t, model_bge) for t in job_texts])

scores_minilm_mean = [
    cosine_similarity([cv_mean_minilm[i]], [job_mean_minilm[i]])[0][0]
    for i in range(len(test_df))
]
scores_bge_mean = [
    cosine_similarity([cv_mean_bge[i]], [job_mean_bge[i]])[0][0]
    for i in range(len(test_df))
]

# calibrer les seuils sur le train
print("Mean pooling MiniLM (train)...")
cv_train_mean_m = np.array([embed_mean_chunks(t, model_minilm) for t in train_df['resume_text'].tolist()])
job_train_mean_m = np.array([embed_mean_chunks(t, model_minilm) for t in train_df['job_description_text'].tolist()])
scores_train_mean_m = [cosine_similarity([cv_train_mean_m[i]], [job_train_mean_m[i]])[0][0] for i in range(len(train_df))]

print("Mean pooling bge-small (train)...")
cv_train_mean_b = np.array([embed_mean_chunks(t, model_bge) for t in train_df['resume_text'].tolist()])
job_train_mean_b = np.array([embed_mean_chunks(t, model_bge) for t in train_df['job_description_text'].tolist()])
scores_train_mean_b = [cosine_similarity([cv_train_mean_b[i]], [job_train_mean_b[i]])[0][0] for i in range(len(train_df))]

high_mm, low_mm, _ = find_best_thresholds(scores_train_mean_m, train_df['label'].tolist())
high_mb, low_mb, _ = find_best_thresholds(scores_train_mean_b, train_df['label'].tolist())

f1_minilm_mean = compute_f1(scores_minilm_mean, test_df['label'].tolist(), high_mm, low_mm)
f1_bge_mean = compute_f1(scores_bge_mean, test_df['label'].tolist(), high_mb, low_mb)

print(f"F1 MiniLM (mean pooling): {f1_minilm_mean:.4f}")
print(f"F1 bge-small (mean pooling): {f1_bge_mean:.4f}")

# === Stratégie 3 : Max similarity des chunks ===
# on découpe le CV en morceaux, on encode chaque morceau et l'offre entière, on garde le max
# l'idée : même si un CV est long, au moins un passage devrait être très pertinent
print("Max similarity MiniLM (test)...")
scores_minilm_max = [score_max_chunks(cv, job, model_minilm) for cv, job in zip(cv_texts, job_texts)]

print("Max similarity bge-small (test)...")
scores_bge_max = [score_max_chunks(cv, job, model_bge) for cv, job in zip(cv_texts, job_texts)]

# calibrer sur le train
print("Max similarity MiniLM (train)...")
scores_train_max_m = [score_max_chunks(cv, job, model_minilm) for cv, job in zip(train_df['resume_text'].tolist(), train_df['job_description_text'].tolist())]

print("Max similarity bge-small (train)...")
scores_train_max_b = [score_max_chunks(cv, job, model_bge) for cv, job in zip(train_df['resume_text'].tolist(), train_df['job_description_text'].tolist())]

high_xm, low_xm, _ = find_best_thresholds(scores_train_max_m, train_df['label'].tolist())
high_xb, low_xb, _ = find_best_thresholds(scores_train_max_b, train_df['label'].tolist())

f1_minilm_max = compute_f1(scores_minilm_max, test_df['label'].tolist(), high_xm, low_xm)
f1_bge_max = compute_f1(scores_bge_max, test_df['label'].tolist(), high_xb, low_xb)

print(f"F1 MiniLM (max similarity): {f1_minilm_max:.4f}")
print(f"F1 bge-small (max similarity): {f1_bge_max:.4f}")

# === Tableau comparatif ===
# le score TF-IDF de référence est dans le notebook 02
results = {
    'Modèle': ['MiniLM', 'MiniLM', 'MiniLM', 'bge-small', 'bge-small', 'bge-small'],
    'Chunking': ['Troncation', 'Mean pooling', 'Max similarity', 'Troncation', 'Mean pooling', 'Max similarity'],
    'F1 weighted': [
        f1_minilm_trunc, f1_minilm_mean, f1_minilm_max,
        f1_bge_trunc, f1_bge_mean, f1_bge_max
    ]
}

results_df = pd.DataFrame(results)
results_df['F1 weighted'] = results_df['F1 weighted'].round(4)
print(results_df.to_string(index=False))

best_idx = results_df['F1 weighted'].idxmax()
print(f"\nMeilleure combinaison: {results_df.loc[best_idx, 'Modèle']} + {results_df.loc[best_idx, 'Chunking']}")
print(f"F1: {results_df.loc[best_idx, 'F1 weighted']:.4f}")

# graphique des distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# distribution des scores MiniLM (troncation)
axes[0].hist(scores_minilm_trunc, bins=50, alpha=0.7, color='steelblue')
axes[0].set_title("Distribution scores MiniLM (troncation)")
axes[0].set_xlabel("Score cosinus")
axes[0].set_ylabel("Nombre de paires")

# distribution des scores bge-small (troncation)
axes[1].hist(scores_bge_trunc, bins=50, alpha=0.7, color='coral')
axes[1].set_title("Distribution scores bge-small (troncation)")
axes[1].set_xlabel("Score cosinus")

plt.tight_layout()
plt.savefig('figures/pretrained_score_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée dans figures/")
