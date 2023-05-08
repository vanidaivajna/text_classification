import string

def preprocess_text(df):
    # Remove punctuation
    df['preprocessed_text'] = df['humantext'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    # Remove numbers
    df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
    # Convert to lowercase
    df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: x.lower())
    return df
