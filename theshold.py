import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('your_data.csv')  # Replace 'your_data.csv' with your actual file path
sentences = data['your_column_name']

vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)

cosine_sim = cosine_similarity(sentence_vectors)

threshold = 0.8

rows_to_drop = []
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] >= threshold:
            if j not in rows_to_drop:
                rows_to_drop.append(j)

filtered_data = data.drop(rows_to_drop)
