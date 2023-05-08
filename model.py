import pandas as pd
import numpy as np
import fasttext
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Example data
data = pd.DataFrame({'humantext': ['The quick brown fox jumps over the lazy dog.',
                                 'The quick brown fox jumps over the lazy dog and runs away.',
                                 'A brown dog and a black dog are playing together.',
                                 'My sister is fond of chocolate cake.',
                                 'I enjoy reading books in my free time.'],
                    'intent': ['a', 'a', 'b', 'b', 'b']})

# Split the data into train, validation, and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data['humantext'], data['intent'], test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Train FastText on the train data
model = fasttext.train_unsupervised('train.txt', model='skipgram')

# Generate embeddings for the train, validation, and test data
train_embeddings = np.zeros((len(train_data), 100))
for i, text in enumerate(train_data):
    train_embeddings[i] = model.get_sentence_vector(text)

val_embeddings = np.zeros((len(val_data), 100))
for i, text in enumerate(val_data):
    val_embeddings[i] = model.get_sentence_vector(text)

test_embeddings = np.zeros((len(test_data), 100))
for i, text in enumerate(test_data):
    test_embeddings[i] = model.get_sentence_vector(text)

# Train a Random Forest model with cross-validation
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf_model, train_embeddings, train_labels, cv=5, scoring='f1_macro')

# Print the cross-validation scores
print('Cross-validation scores:', cv_scores)
print('Average f1 score:', np.mean(cv_scores))

# Train the final model on the full training set and evaluate it on the test set
rf_model.fit(train_embeddings, train_labels)
test_pred = rf_model.predict(test_embeddings)

precision = precision_score(test_labels, test_pred, average='macro')
recall = recall_score(test_labels, test_pred, average='macro')
f1 = f1_score(test_labels, test_pred, average='macro')

print('Test precision:', precision)
print('Test recall:', recall)
print('Test f1 score:', f1)
#-----------------------------------
