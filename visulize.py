import fasttext
import joblib

# Load the FastText model
fasttext_model = fasttext.load_model('path/to/fasttext/model.bin')

# Load the Random Forest model
rf_model = joblib.load('path/to/random/forest/model.pkl')

# Define a function to make predictions on text input
def predict(text):
    # Generate the FastText embedding for the text input
    embedding = fasttext_model.get_sentence_vector(text)

    # Use the Random Forest model to predict the class label and score
    label = rf_model.predict([embedding])[0]
    score = rf_model.predict_proba([embedding])[0][label]

    # Return the predicted label and score
    return label, score
import fasttext
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import ipywidgets as widgets
from IPython.display import display

# Load data and train FastText model
data = pd.read_csv('data.csv')
model_ft = fasttext.train_unsupervised('data.txt', model='skipgram')

# Generate embeddings for the text data
embeddings = np.zeros((len(data), 100))
for i, text in enumerate(data['text']):
    embeddings[i] = model_ft.get_sentence_vector(text)

# Merge embeddings with labels
data_with_embeddings = pd.concat([data['label'], pd.DataFrame(embeddings)], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_with_embeddings.iloc[:, 1:], data_with_embeddings.iloc[:, 0], test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Define function to make predictions and display scores
def predict(text):
    # Generate FastText embedding for text
    embedding = model_ft.get_sentence_vector(text)

    # Use Random Forest to make prediction and get score
    rf_prediction = rf_model.predict([embedding])[0]
    rf_score = rf_model.predict_proba([embedding])[0][rf_model.classes_.tolist().index(rf_prediction)]

    # Use FastText to make prediction and get score
    ft_prediction, ft_score = model_ft.predict(text)

    # Display predictions and scores
    print(f"Random Forest: Prediction: {rf_prediction}, Score: {rf_score:.3f}")
    print(f"FastText: Prediction: {ft_prediction[0].replace('__label__', '')}, Score: {ft_score:.3f}")

# Create text input widget and button
text_input = widgets.Text(description='Enter Text:')
button = widgets.Button(description='Predict')

# Define function to handle button click event
def on_button_click(button):
    predict(text_input.value)

# Attach function to button click event
button.on_click(on_button_click)

# Display widgets
display(text_input)
display(button)
