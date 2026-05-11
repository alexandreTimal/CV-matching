# baseline TF-IDF + cosine similarity, score à battre avec le deep learning

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from src.data_utils import load_dataset_fit, group_split

df = load_dataset_fit()
train_df, test_df = group_split(df)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# === TF-IDF Vectorisation ===
# on fit le vectorizer sur tous les textes du train (CVs + offres ensemble)
all_train_texts = pd.concat([train_df['resume_text'], train_df['job_description_text']]).tolist()
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
tfidf.fit(all_train_texts)

print(f"Vocabulaire: {len(tfidf.vocabulary_)} termes")

# encoder les paires
train_cv_vectors = tfidf.transform(train_df['resume_text'])
train_job_vectors = tfidf.transform(train_df['job_description_text'])
test_cv_vectors = tfidf.transform(test_df['resume_text'])
test_job_vectors = tfidf.transform(test_df['job_description_text'])

# === Cosine Similarity ===
# calculer la similarité cosinus pour chaque paire
# attention: cosine_similarity retourne une matrice, on veut juste la diagonale
train_scores = np.array([
    cosine_similarity(train_cv_vectors[i], train_job_vectors[i])[0][0]
    for i in range(len(train_df))
])
test_scores = np.array([
    cosine_similarity(test_cv_vectors[i], test_job_vectors[i])[0][0]
    for i in range(len(test_df))
])

print(f"Scores train — min: {train_scores.min():.4f}, max: {train_scores.max():.4f}, mean: {train_scores.mean():.4f}")
print(f"Scores test  — min: {test_scores.min():.4f}, max: {test_scores.max():.4f}, mean: {test_scores.mean():.4f}")

# distribution des scores par label
fig, ax = plt.subplots(figsize=(10, 5))
for label in sorted(train_df['label'].unique()):
    mask = train_df['label'] == label
    ax.hist(train_scores[mask], bins=30, alpha=0.5, label=label)
ax.set_title('Distribution des scores TF-IDF par label (train)')
ax.set_xlabel('Cosine similarity')
ax.legend()
plt.tight_layout()
plt.savefig('figures/tfidf_score_distribution.png', dpi=150)
plt.show()

# === Calibration des seuils ===
# grid search simple pour trouver les meilleurs seuils sur le train set
# 2 seuils : high_thresh (au-dessus = Good Fit) et low_thresh (en-dessous = No Fit)
best_f1 = 0
best_thresholds = (0.5, 0.3)

for high in np.arange(0.1, 0.9, 0.02):
    for low in np.arange(0.05, high, 0.02):
        preds = []
        for s in train_scores:
            if s >= high:
                preds.append("Good Fit")
            elif s >= low:
                preds.append("Potential Fit")
            else:
                preds.append("No Fit")

        f1 = f1_score(train_df['label'], preds, average='weighted', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = (high, low)

high_thresh, low_thresh = best_thresholds
print(f"\nMeilleurs seuils trouvés sur le train:")
print(f"  Good Fit si score >= {high_thresh:.2f}")
print(f"  Potential Fit si score >= {low_thresh:.2f}")
print(f"  No Fit sinon")
print(f"  F1 train: {best_f1:.4f}")

# === Évaluation sur le test set ===
def classify_tfidf(scores, high_thresh, low_thresh):
    preds = []
    for s in scores:
        if s >= high_thresh:
            preds.append("Good Fit")
        elif s >= low_thresh:
            preds.append("Potential Fit")
        else:
            preds.append("No Fit")
    return preds

test_preds = classify_tfidf(test_scores, high_thresh, low_thresh)

f1 = f1_score(test_df['label'], test_preds, average='weighted')
acc = accuracy_score(test_df['label'], test_preds)

print(f"=== Résultats TF-IDF sur le TEST set ===")
print(f"F1 weighted: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"\nClassification report:")
print(classification_report(test_df['label'], test_preds))

# === Matrice de confusion ===
labels = ["Good Fit", "Potential Fit", "No Fit"]
cm = confusion_matrix(test_df['label'], test_preds, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Matrice de confusion, TF-IDF Baseline')
ax.set_ylabel('Vrai label')
ax.set_xlabel('Prédiction')
plt.tight_layout()
plt.savefig('figures/tfidf_confusion_matrix.png', dpi=150)
plt.show()

# === Récap ===
# tableau récapitulatif
print("=" * 50)
print("BASELINE TF-IDF — RÉSULTATS")
print("=" * 50)
print(f"Méthode: TF-IDF + Cosine Similarity")
print(f"Vocabulaire: {len(tfidf.vocabulary_)} termes")
print(f"Seuils: Good Fit >= {high_thresh:.2f}, Potential Fit >= {low_thresh:.2f}")
print(f"")
print(f"F1 weighted:  {f1:.4f}")
print(f"Accuracy:     {acc:.4f}")
print(f"")
print(f"C'est le score à battre avec les modèles deep learning.")
print(f"Prochaine étape: sentence-transformers pré-entraînés")
