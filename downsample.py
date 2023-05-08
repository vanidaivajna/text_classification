import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# example data
data = pd.DataFrame({'sentences': ['The quick brown fox jumps over the lazy dog.',
                                 'The quick brown fox jumps over the lazy dog and runs away.',
                                 'A brown dog and a black dog are playing together.',
                                 'My sister is fond of chocolate cake.',
                                 'I enjoy reading books in my free time.']})

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute the TF-IDF matrix for the sentences
tfidf_matrix = vectorizer.fit_transform(data['sentences'])

# Compute the cosine similarity matrix between the sentences
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Set up the threshold distribution
thresholds = np.linspace(0.2, 0.8, num=7)

# Iterate over the upper triangle of the cosine similarity matrix
for i in range(len(data)):
    for j in range(i+1, len(data)):
        # Check if the cosine similarity is above any of the thresholds
        for threshold in thresholds:
            if cosine_sim_matrix[i][j] > threshold:
                # If the cosine similarity is above the threshold, drop the corresponding rows and columns from the DataFrame
                data = data.drop([j])
                data = data.reset_index(drop=True)
                break # break the loop if the threshold is exceeded

# Print the updated DataFrame
print(data)
