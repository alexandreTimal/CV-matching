# exploration du dataset resume-job-description-fit

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_utils import load_dataset_fit, group_split

# === Imports et chargement ===
# charger le dataset
df = load_dataset_fit()
print(df.head())

# === Aperçu du dataset ===
print(f"Shape: {df.shape}")
print(f"\nColonnes: {list(df.columns)}")
print(f"\nTypes:\n{df.dtypes}")
print(f"\nValeurs manquantes:\n{df.isnull().sum()}")

# === Distribution des labels ===
print(f"Distribution:\n{df['label'].value_counts()}")

fig, ax = plt.subplots(figsize=(8, 5))
df['label'].value_counts().plot(kind='bar', ax=ax, color=['green', 'yellow', 'red'])
ax.set_title("Distribution des labels")
ax.set_ylabel("Nombre de paires")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/label_distribution.png', dpi=150)
plt.show()

# === Longueur des textes ===
# combien de mots dans les CVs et offres, pour savoir si on dépasse les limites des modèles
df['cv_len'] = df['resume_text'].str.split().str.len()
df['job_len'] = df['job_description_text'].str.split().str.len()

print(f"Longueur CVs (mots):")
print(df['cv_len'].describe())
print(f"\nLongueur offres (mots):")
print(df['job_len'].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['cv_len'], bins=50, color='steelblue', alpha=0.7)
axes[0].axvline(256, color='red', linestyle='--', label='256 tokens (~MiniLM)')
axes[0].axvline(512, color='orange', linestyle='--', label='512 tokens (~bge/mpnet)')
axes[0].set_title('Longueur des CVs (mots)')
axes[0].set_xlabel('Nombre de mots')
axes[0].legend()

axes[1].hist(df['job_len'], bins=50, color='coral', alpha=0.7)
axes[1].axvline(256, color='red', linestyle='--', label='256 tokens')
axes[1].axvline(512, color='orange', linestyle='--', label='512 tokens')
axes[1].set_title('Longueur des offres (mots)')
axes[1].set_xlabel('Nombre de mots')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/text_lengths.png', dpi=150)
plt.show()

# combien de textes dépassent les limites ?
print(f"\nCVs > 256 mots: {(df['cv_len'] > 256).sum()} ({(df['cv_len'] > 256).mean()*100:.1f}%)")
print(f"CVs > 512 mots: {(df['cv_len'] > 512).sum()} ({(df['cv_len'] > 512).mean()*100:.1f}%)")
print(f"Offres > 256 mots: {(df['job_len'] > 256).sum()} ({(df['job_len'] > 256).mean()*100:.1f}%)")
print(f"Offres > 512 mots: {(df['job_len'] > 512).sum()} ({(df['job_len'] > 512).mean()*100:.1f}%)")

# === Exemples de paires ===
# afficher 2 exemples par label
for label in df['label'].unique():
    print(f"\n{'='*80}")
    print(f"LABEL: {label}")
    print(f"{'='*80}")
    samples = df[df['label'] == label].sample(2, random_state=42)
    for i, row in samples.iterrows():
        print(f"\n--- CV (extrait, 200 premiers mots) ---")
        print(' '.join(row['resume_text'].split()[:200]))
        print(f"\n--- Offre (extrait, 200 premiers mots) ---")
        print(' '.join(row['job_description_text'].split()[:200]))
        print()

# === Duplication des CVs ===
# un même CV peut être associé à plusieurs offres : important pour le split train/test
n_unique_cvs = df['resume_text'].nunique()
n_unique_jobs = df['job_description_text'].nunique()
print(f"Paires totales: {len(df)}")
print(f"CVs uniques: {n_unique_cvs}")
print(f"Offres uniques: {n_unique_jobs}")
print(f"\nEn moyenne, chaque CV apparaît {len(df)/n_unique_cvs:.1f} fois")
print(f"En moyenne, chaque offre apparaît {len(df)/n_unique_jobs:.1f} fois")

# c'est pour ça qu'on a besoin du group split !

# === Group split ===
# on sépare train/test en s'assurant qu'un même CV n'apparaît pas dans les deux
train_df, test_df = group_split(df)

print(f"\nDistribution labels TRAIN:\n{train_df['label'].value_counts(normalize=True)}")
print(f"\nDistribution labels TEST:\n{test_df['label'].value_counts(normalize=True)}")

# vérifier visuellement
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_df['label'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[0].set_title('Train')
test_df['label'].value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[1].set_title('Test')
plt.tight_layout()
plt.savefig('figures/split_distribution.png', dpi=150)
plt.show()
