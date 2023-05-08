import fasttext
import pandas as pd
import numpy as np

# Example data
data = pd.DataFrame({'humantext': ['The quick brown fox jumps over the lazy dog.',
                                 'The quick brown fox jumps over the lazy dog and runs away.',
                                 'A brown dog and a black dog are playing together.',
                                 'My sister is fond of chocolate cake.',
                                 'I enjoy reading books in my free time.'],
                    'intent': ['a', 'a', 'b', 'b', 'b']})

# Train FastText on the data
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# Generate embeddings for the humantext column
embeddings = np.zeros((len(data), 100)) # Set the embedding dimension to 100
for i, text in enumerate(data['humantext']):
    embeddings[i] = model.get_sentence_vector(text)

# Merge the embeddings with the intent column
data_with_embeddings = pd.concat([data['intent'], pd.DataFrame(embeddings)], axis=1)

# Use the data with embeddings to train machine learning models
# ...
import fasttext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Example data
data = pd.DataFrame({'humantext': ['The quick brown fox jumps over the lazy dog.',
                                 'The quick brown fox jumps over the lazy dog and runs away.',
                                 'A brown dog and a black dog are playing together.',
                                 'My sister is fond of chocolate cake.',
                                 'I enjoy reading books in my free time.'],
                    'intent': ['a', 'a', 'b', 'b', 'b']})

# Train FastText on the data
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# Generate embeddings for the humantext column
embeddings = np.zeros((len(data), 100)) # Set the embedding dimension to 100
for i, text in enumerate(data['humantext']):
    embeddings[i] = model.get_sentence_vector(text)

# Reduce the dimensionality of the embeddings to two dimensions using PCA
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

# Plot the embeddings on a scatter plot
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
plt.show()
import fasttext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Example data
data = pd.DataFrame({'humantext': ['The quick brown fox jumps over the lazy dog.',
                                 'The quick brown fox jumps over the lazy dog and runs away.',
                                 'A brown dog and a black dog are playing together.',
                                 'My sister is fond of chocolate cake.',
                                 'I enjoy reading books in my free time.'],
                    'intent': ['a', 'a', 'b', 'b', 'b']})

# Train FastText on the data
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# Generate embeddings for the humantext column
embeddings = np.zeros((len(data), 100)) # Set the embedding dimension to 100
for i, text in enumerate(data['humantext']):
    embeddings[i] = model.get_sentence_vector(text)

# Reduce the dimensionality of the embeddings to two dimensions using PCA
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)

# Plot the embeddings on a scatter plot
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
plt.show()


#----------------------------------------------
import fasttext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Example data
data = pd.DataFrame({'humantext': ['The quick brown fox jumps over the lazy dog.',
                                 'The quick brown fox jumps over the lazy dog and runs away.',
                                 'A brown dog and a black dog are playing together.',
                                 'My sister is fond of chocolate cake.',
                                 'I enjoy reading books in my free time.'],
                    'intent': ['a', 'a', 'b', 'b', 'b']})

# Train FastText on the data
model = fasttext.train_unsupervised('data.txt', model='skipgram')

# Generate embeddings for the humantext column
embeddings = np.zeros((len(data), 100)) # Set the embedding dimension to 100
for i, text in enumerate(data['humantext']):
    embeddings[i] = model.get_sentence_vector(text)

# Reduce the dimensionality of the embeddings to two dimensions using t-SNE
tsne = TSNE(n_components=2)
embeddings_tsne = tsne.fit_transform(embeddings)

# Plot the embeddings on a scatter plot
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
plt.show()
#--------------------------------------------------------------------------
import plotly.graph_objs as go

def plot_nearest_neighbors(model, word):
    # Get the nearest neighbors of the given word
    neighbors = model.get_nearest_neighbors(word)

    # Create the plotly figure
    fig = go.Figure()

    # Add the main scatter plot for the word embeddings
    scatter = fig.add_trace(go.Scatter(
        x=model[neighbors][:, 0], # X coordinate of the embeddings
        y=model[neighbors][:, 1], # Y coordinate of the embeddings
        mode='markers',
        marker=dict(color='blue'),
        text=neighbors,
        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}',
        name='Word Embeddings'
    ))

    # Add a trace for the queried word
    word_trace = fig.add_trace(go.Scatter(
        x=[model[word][0]],
        y=[model[word][1]],
        mode='markers',
        marker=dict(color='red', size=10),
        text=[word],
        hovertemplate='<b>%{text}</b>',
        name='Queried Word'
    ))

    # Add annotations for the queried word and each of the neighbors
    annotations = []
    annotations.append(dict(x=model[word][0], y=model[word][1], text=word, showarrow=False, font=dict(size=16), xshift=10, yshift=10))
    for neighbor in neighbors:
        annotations.append(dict(x=model[neighbor][0], y=model[neighbor][1], text=neighbor, showarrow=False, font=dict(size=16), xshift=10, yshift=10))

    fig.update_layout(
        title=f"Nearest Neighbors of '{word}'",
        xaxis=dict(title='Dimension 1'),
        yaxis=dict(title='Dimension 2'),
        annotations=annotations
    )

    fig.show()
plot_nearest_neighbors(model, 'dog')
#--------------------------------------------------------------------------------
import plotly.graph_objs as go

def plot_analogy(model, word1, word2, word3):
    # Compute the analogy using the given words
    analogy = model.get_analogies(word1, word2, word3)

    # Create the plotly figure
    fig = go.Figure()

    # Add the main scatter plot for the word embeddings
    scatter = fig.add_trace(go.Scatter(
        x=model[analogy][:, 0], # X coordinate of the embeddings
        y=model[analogy][:, 1], # Y coordinate of the embeddings
        mode='markers',
        marker=dict(color='blue'),
        text=analogy,
        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}',
        name='Word Embeddings'
    ))

    # Add a trace for the queried words
    query_trace = fig.add_trace(go.Scatter(
        x=[model[word][0] for word in [word1, word2, word3]],
        y=[model[word][1] for word in [word1, word2, word3]],
        mode='markers+text',
        marker=dict(color='red', size=15, line=dict(width=2, color='black')),
        text=[word1, word2, word3],
        textposition='top right',
        hoverinfo='none',
        name='Queried Words'
    ))

    # Set the title and axis labels
    fig.update_layout(title='Analogy Plot', xaxis_title='X', yaxis_title='Y')

    # Show the plot
    fig.show()
# Example usage
model = fasttext.load_model('cc.en.300.bin') # Load the FastText model
plot_analogy(model, 'man', 'woman', 'king') # Plot the analogy
