"""
Test script to verify the notebook code works correctly
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("="*80)
print("TESTING NOTEBOOK CODE")
print("="*80)

# Test 1: Load dataset
print("\n1. Loading dataset...")
url = 'https://raw.githubusercontent.com/marsgr6/estadistica-ux/main/data/words_ux.csv'
df = pd.read_csv(url)
print(f"   ✓ Loaded {len(df)} rows")
print(f"   ✓ Columns: {df.columns.tolist()}")

# Test 2: Check column name
print("\n2. Checking column structure...")
if 'Words' in df.columns:
    print("   ✓ 'Words' column found")
else:
    print("   ✗ ERROR: 'Words' column not found!")
    print(f"   Available columns: {df.columns.tolist()}")
    exit(1)

# Test 3: Clean data
print("\n3. Cleaning data...")
df_clean = df.dropna(subset=['Words'])
df_clean['Words'] = df_clean['Words'].str.lower().str.strip()
print(f"   ✓ Cleaned {len(df_clean)} documents")
print(f"   ✓ Sample: {df_clean['Words'].iloc[0][:50]}...")

# Test 4: Create documents list
print("\n4. Creating documents list...")
documents = df_clean['Words'].tolist()
print(f"   ✓ Created {len(documents)} documents")

# Test 5: Build DTM
print("\n5. Building Document-Term Matrix...")
vectorizer = CountVectorizer(binary=True, lowercase=True)
dtm = vectorizer.fit_transform(documents)
print(f"   ✓ DTM shape: {dtm.shape} (documents x words)")
print(f"   ✓ Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# Test 6: Calculate similarities
print("\n6. Calculating similarity matrices...")

# Cosine similarity
cosine_sim_matrix = cosine_similarity(dtm)
print(f"   ✓ Cosine similarity matrix: {cosine_sim_matrix.shape}")
print(f"   ✓ Mean cosine similarity: {cosine_sim_matrix[np.triu_indices_from(cosine_sim_matrix, k=1)].mean():.4f}")

# Jaccard similarity
def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    row_sums = matrix.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    union[union == 0] = 1
    return intersection / union

jaccard_sim_matrix = jaccard_similarity(dtm.toarray())
print(f"   ✓ Jaccard similarity matrix: {jaccard_sim_matrix.shape}")
print(f"   ✓ Mean Jaccard similarity: {jaccard_sim_matrix[np.triu_indices_from(jaccard_sim_matrix, k=1)].mean():.4f}")

# Test 7: Sample words
print("\n7. Top 10 most frequent words:")
dtm_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
word_freq = dtm_df.sum(axis=0).sort_values(ascending=False)
for i, (word, freq) in enumerate(word_freq.head(10).items(), 1):
    print(f"   {i}. {word}: {int(freq)} documents")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nThe notebook should work correctly now.")
