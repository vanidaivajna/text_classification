import fasttext
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

# Load the FastText model
model = fasttext.load_model('your_model.bin')  # Replace with the path to your FastText model

# Get the embeddings for your text data
embeddings = []
texts = []  # List of text data
for text in texts:
    embedding = model.get_sentence_vector(text)
    embeddings.append(embedding)
embeddings = np.array(embeddings)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)

# Create the interactive plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=10)

# Implement a search mechanism to find nearest neighbors
def find_nearest_neighbors(query_embedding, k=5):
    distances = cosine_distances(query_embedding, embeddings)
    nearest_indices = np.argsort(distances)[0][:k]
    return nearest_indices

# Interactive search interface
def search_nearest_neighbors(query_text):
    query_embedding = model.get_sentence_vector(query_text)
    nearest_indices = find_nearest_neighbors(query_embedding)
    
    # Highlight nearest neighbors on the t-SNE plot
    plt.scatter(embeddings_tsne[nearest_indices, 0], embeddings_tsne[nearest_indices, 1], color='r', s=50, label='Nearest Neighbors')
    plt.legend()
    plt.show()

# Example usage of the search interface
search_nearest_neighbors("your_query_text")
