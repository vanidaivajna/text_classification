import fasttext
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Prepare and load your labeled training data
train_data = 'train.txt'  # Replace with the path to your labeled training data
validation_data = 'validation.txt'  # Replace with the path to your validation data

# Train a supervised FastText model
model = fasttext.train_supervised(
    input=train_data,
    lr=0.1,
    epoch=10,
    wordNgrams=2,
    bucket=200000,
    dim=100,
    loss='softmax',
    minCount=1,
    thread=4
)

# Extract embeddings for visualization
embeddings = []
texts = []  # List of text samples
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
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    nearest_indices = np.argsort(distances)[:k]
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
#----------------------------------


import fasttext
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Prepare and load your labeled training data
train_data = 'train.txt'  # Replace with the path to your labeled training data
validation_data = 'validation.txt'  # Replace with the path to your validation data

# Train a supervised FastText model
model = fasttext.train_supervised(
    input=train_data,
    lr=0.1,
    epoch=10,
    wordNgrams=2,
    bucket=200000,
    dim=100,
    loss='softmax',
    minCount=1,
    thread=4
)

# Extract embeddings for visualization
embeddings = []
texts = []  # List of text samples
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
    distances = np.linalg.norm(embeddings - query_embedding.reshape(1, -1), axis=1)
    nearest_indices = np.argsort(distances)[:k]
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


#------------------------
import fasttext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load your pandas DataFrame with text and intent columns
data = pd.read_csv('your_data.csv')  # Replace with the path to your data file

# Extract the text and intent columns
texts = data['text'].tolist()
labels = data['intent'].tolist()

# Create the FastText training file
train_file = 'fasttext_train.txt'  # Path to the FastText training file
with open(train_file, 'w') as f:
    for text, label in zip(texts, labels):
        f.write(f'__label__{label} {text}\n')

# Train a supervised FastText model
model = fasttext.train_supervised(
    input=train_file,
    lr=0.1,
    epoch=10,
    wordNgrams=2,
    bucket=200000,
    dim=100,
    loss='softmax',
    minCount=1,
    thread=4
)

# Extract embeddings for visualization
embeddings = []
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
    distances = np.linalg.norm(embeddings - query_embedding.reshape(1, -1), axis=1)
    nearest_indices = np.argsort(distances)[:k]
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
#-----------------------------
# Get the nearest neighbors for a word
word = "example_word"
k = 10  # Number of nearest neighbors to retrieve
neighbors = model.get_nearest_neighbors(word, k)

# Extract the word vectors for the neighbors
word_vectors = []
for neighbor in neighbors:
    neighbor_word = neighbor[1]
    word_vector = model.get_word_vector(neighbor_word)
    word_vectors.append(word_vector)
word_vectors = np.array(word_vectors)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
word_vectors_tsne = tsne.fit_transform(word_vectors)

# Create the plot
plt.figure(figsize=(10, 8))
plt.scatter(word_vectors_tsne[:, 0], word_vectors_tsne[:, 1], s=50)
for i, neighbor in enumerate(neighbors):
    plt.annotate(neighbor[1], (word_vectors_tsne[i, 0], word_vectors_tsne[i, 1]))

plt.show()
In this example, you load your FastText model using fasttext.load_model(). Then, you use model.get_nearest_neighbors() to retrieve the nearest neighbors for a specific word. After that, you extract the word vectors for the neighbors using model.get_word_vector(). Next, t-SNE is applied for dimensionality reduction, and finally, the nearest neighbors are plotted using matplotlib.

Make sure to replace 'your_model.bin' with the path to your actual FastText model file. Also, modify 'example_word' with the word for which you want to find the nearest neighbors, and adjust the value of k to determine the number of nearest neighbors to retrieve.






Regenerate response

